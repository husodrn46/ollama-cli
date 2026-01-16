"""Chat engine module - Conversation, streaming, and summarization."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import requests
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from .models import ConfigModel, TokenStats
from .utils import estimate_message_tokens, get_model_prompt

if TYPE_CHECKING:
    from logging import Logger

PERSONAS = {
    "developer": {
        "name": "Yazilim Gelistirici",
        "icon": "\U0001f4bb",
        "prompt": (
            "Sen deneyimli bir yazilim gelistiricisin. Kod yazarken best "
            "practice'leri uygula, temiz ve okunabilir kod yaz. Hatalari detayli "
            "acikla ve cozum oner."
        ),
    },
    "teacher": {
        "name": "Ogretmen",
        "icon": "\U0001f468\u200d\U0001f3eb",
        "prompt": (
            "Sen sabirli ve anlayisli bir ogretmensin. Konulari basit ve anlasilir "
            "sekilde acikla. Ornekler ver, adim adim ilerle. Ogrencinin seviyesine "
            "gore uyarla."
        ),
    },
    "assistant": {
        "name": "Asistan",
        "icon": "\U0001f916",
        "prompt": (
            "Sen yardimci ve verimli bir asistansin. Kisa, oz ve dogrudan yanitlar "
            "ver. Gereksiz detaylardan kacin."
        ),
    },
    "creative": {
        "name": "Yaratici Yazar",
        "icon": "\u2728",
        "prompt": (
            "Sen yaratici bir yazarsin. Ozgun fikirler uret, ilgi cekici hikayeler "
            "anlat, siir ve metinler yaz. Hayal gucunu kullan."
        ),
    },
    "analyst": {
        "name": "Analist",
        "icon": "\U0001f4ca",
        "prompt": (
            "Sen detayci bir analistsin. Verileri incele, karsilastir, arti ve eksi "
            "leri listele. Objektif ve mantikli degerlendirmeler yap."
        ),
    },
    "debug": {
        "name": "Debug Uzmani",
        "icon": "\U0001f527",
        "prompt": (
            "Sen bir debug uzmanisin. Hatalari bul, nedenlerini acikla, cozum oner. "
            "Sistematik dusun, olasi tum senaryolari degerlendir."
        ),
    },
    "turkish": {
        "name": "Turkce Asistan",
        "icon": "\U0001f1f9\U0001f1f7",
        "prompt": (
            "Sen Turkce konusan yardimci bir asistansin. Her zaman Turkce yanit ver. "
            "Teknik terimleri de Turkce acikla."
        ),
    },
}

SUMMARY_PREFIX = "## Konusma Ozeti"
DEFAULT_SUMMARY_KEEP = 6


class ChatEngine:
    """Handles conversation flow, streaming, and summarization."""

    def __init__(
        self,
        config: ConfigModel,
        console: Console,
        logger: Logger,
        prompts: Dict,
        token_stats: TokenStats,
        get_theme: Callable[[], Dict[str, str]],
        on_autosave: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.console = console
        self.logger = logger
        self.prompts = prompts
        self.token_stats = token_stats
        self._get_theme = get_theme
        self._on_autosave = on_autosave

        # Conversation state
        self.messages: List[Dict[str, object]] = []
        self.summary: str = ""
        self.base_system_prompt: str = ""
        self.profile_prompt: str = ""
        self.current_persona: Optional[str] = None
        self.current_temperature: Optional[float] = None
        self.model: Optional[str] = None

    @property
    def theme(self) -> Dict[str, str]:
        """Get current theme colors."""
        return self._get_theme()

    def init_conversation(
        self,
        model_name: str,
        apply_profiles_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, object]]:
        """Initialize a new conversation with optional model profiles."""
        self.summary = ""
        self.model = model_name

        if apply_profiles_callback:
            apply_profiles_callback(model_name)

        prompt_info = get_model_prompt(model_name, self.prompts)
        self.base_system_prompt = prompt_info.get("system_prompt", "")
        combined_prompt = self.build_system_prompt()

        messages = []
        if combined_prompt:
            messages.append({"role": "system", "content": combined_prompt})

        self.messages = messages
        return messages

    def build_system_prompt(self) -> str:
        """Build combined system prompt from base, profile, and persona."""
        base = self.base_system_prompt or ""
        profile_prompt = self.profile_prompt or ""
        persona_prompt = ""

        if self.current_persona and self.current_persona in PERSONAS:
            persona_prompt = PERSONAS[self.current_persona]["prompt"]

        parts = [part for part in [base, profile_prompt, persona_prompt] if part]
        return "\n\n".join(parts)

    def update_system_message(self) -> None:
        """Update or insert the system message in conversation."""
        combined = self.build_system_prompt()
        base_idx = self._find_base_system_index()

        if combined:
            if base_idx is not None:
                self.messages[base_idx]["content"] = combined
            else:
                self.messages.insert(0, {"role": "system", "content": combined})
        elif base_idx is not None:
            self.messages.pop(base_idx)

        self.update_summary_message()

    def update_summary_message(self) -> None:
        """Update or insert the summary message in conversation."""
        summary_idx = self._find_summary_index()

        if not self.summary:
            if summary_idx is not None:
                self.messages.pop(summary_idx)
            return

        content = f"{SUMMARY_PREFIX}\n{self.summary}"

        if summary_idx is not None:
            self.messages[summary_idx]["content"] = content
            base_idx = self._find_base_system_index()
            if base_idx is not None and summary_idx < base_idx:
                summary_msg = self.messages.pop(summary_idx)
                if summary_idx < base_idx:
                    base_idx -= 1
                self.messages.insert(base_idx + 1, summary_msg)
            return

        base_idx = self._find_base_system_index()
        insert_at = 1 if base_idx == 0 else 0
        self.messages.insert(insert_at, {"role": "system", "content": content})

    def _find_summary_index(self) -> Optional[int]:
        """Find index of summary message in conversation."""
        for idx, msg in enumerate(self.messages):
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
                return idx
        return None

    def _find_base_system_index(self) -> Optional[int]:
        """Find index of base system message (not summary)."""
        for idx, msg in enumerate(self.messages):
            if msg.get("role") == "system" and not self._is_summary_message(msg):
                return idx
        return None

    def _is_summary_message(self, msg: Dict[str, object]) -> bool:
        """Check if message is a summary message."""
        content = msg.get("content", "")
        return isinstance(content, str) and content.startswith(SUMMARY_PREFIX)

    def estimate_context_tokens(self) -> int:
        """Estimate total tokens in current conversation."""
        return sum(estimate_message_tokens(msg) for msg in self.messages)

    def maybe_summarize(self, force: bool = False) -> bool:
        """Auto-summarize if approaching token limit."""
        if not force and not self.config.context_autosummarize:
            return False
        if self.config.context_token_budget <= 0:
            return False

        total = self.estimate_context_tokens()
        if not force and total <= self.config.context_token_budget:
            return False

        return self.summarize_messages()

    def summarize_messages(self) -> bool:
        """Trigger conversation summarization."""
        to_summarize, keep = self._split_messages_for_summary()
        if not to_summarize:
            return False

        self.console.print(f"[{self.theme['muted']}]Ozetleniyor...[/]")
        summary_text = self.request_summary(to_summarize)

        if not summary_text:
            self.console.print(f"[{self.theme['error']}]Ozetleme basarisiz[/]\n")
            return False

        self.summary = summary_text
        self.messages = keep
        self.update_system_message()
        self.update_summary_message()
        return True

    def _split_messages_for_summary(
        self,
    ) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """Partition messages for summarization."""
        conversation = [m for m in self.messages if m.get("role") != "system"]
        keep_last = max(1, self.config.context_keep_last or DEFAULT_SUMMARY_KEEP)

        if len(conversation) <= keep_last:
            return [], conversation

        to_summarize = conversation[:-keep_last]
        keep = conversation[-keep_last:]
        return to_summarize, keep

    def request_summary(self, messages: List[Dict[str, object]]) -> Optional[str]:
        """Send messages to model for summarization."""
        summary_model = self.config.summary_model or self.model
        if not summary_model:
            return None

        summary_input = self._build_summary_input(messages)
        payload = {
            "model": summary_model,
            "messages": [
                {"role": "system", "content": self.config.summary_prompt},
                {"role": "user", "content": summary_input},
            ],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.config.ollama_host}/api/chat",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            summary = data.get("message", {}).get("content", "").strip()
            return summary or None
        except Exception:
            self.logger.exception("Ozetleme istegi basarisiz")
            return None

    def _build_summary_input(self, messages: List[Dict[str, object]]) -> str:
        """Format messages for summarization request."""
        lines = []

        if self.summary:
            lines.append("Onceki ozet:")
            lines.append(self.summary)
            lines.append("")

        lines.append("Mesajlar:")
        for msg in messages:
            role = "Kullanici" if msg.get("role") == "user" else "Asistan"
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "[Gorsel]"
            lines.append(f"{role}: {content}")

        lines.append("")
        lines.append("Yeni, guncel bir ozet yaz.")
        return "\n".join(lines)

    def extract_summary(self, messages: List[Dict[str, object]]) -> str:
        """Extract summary text from message list."""
        for msg in messages:
            if self._is_summary_message(msg):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.replace(SUMMARY_PREFIX, "", 1).strip()
        return ""

    def _extract_base_system_prompt(self) -> str:
        """Get base system prompt from conversation."""
        for msg in self.messages:
            if msg.get("role") == "system" and not self._is_summary_message(msg):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, object]],
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Stream chat response from Ollama API."""
        host = self.config.ollama_host

        try:
            request_data = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                request_data["options"] = {"temperature": temperature}

            response = requests.post(
                f"{host}/api/chat",
                json=request_data,
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            self.console.print()
            header = Text()
            header.append("\u25c9 ", style=f"bold {self.theme['assistant']}")
            header.append(
                model.split(":")[0].upper(), style=f"bold {self.theme['assistant']}"
            )
            self.console.print(header)
            self.console.print(Rule(style=self.theme["muted"]))

            full_response = ""
            start_time = datetime.now()
            data = {}

            if self.config.render_markdown:
                full_response, data = self._stream_with_markdown(response)
            else:
                full_response, data = self._stream_without_markdown(response)

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            self.token_stats.prompt_tokens += prompt_tokens
            self.token_stats.completion_tokens += completion_tokens
            self.token_stats.total_tokens += prompt_tokens + completion_tokens

            if self.config.show_metrics:
                elapsed = (datetime.now() - start_time).total_seconds()
                tps = completion_tokens / elapsed if elapsed > 0 else 0
                self.console.print(
                    f"\n[{self.theme['muted']}]\u23f1 {elapsed:.1f}s  \u25c8 {completion_tokens} token  \u26a1 {tps:.1f} t/s[/]"
                )

            self.console.print()
            return full_response

        except KeyboardInterrupt:
            self.console.print(f"\n[{self.theme['accent']}]\u25fc Iptal[/]\n")
            return full_response if full_response else None
        except Exception as exc:
            self.logger.exception("Chat stream hatasi")
            self.console.print(f"\n[{self.theme['error']}]Hata: {exc}[/]\n")
            return None

    def _stream_with_markdown(self, response) -> tuple[str, Dict]:
        """Stream response with live markdown rendering."""
        full_response = ""
        data = {}
        last_update = time.monotonic()

        with Live(
            Markdown("", code_theme="monokai"),
            console=self.console,
            refresh_per_second=12,
        ) as live:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data:
                            content = data["message"].get("content", "")
                            full_response += content
                            now = time.monotonic()
                            if now - last_update > 0.1 or "\n" in content:
                                live.update(
                                    Markdown(full_response, code_theme="monokai")
                                )
                                last_update = now

                        if data.get("done"):
                            break
                    except Exception:
                        continue
            live.update(Markdown(full_response, code_theme="monokai"))

        return full_response, data

    def _stream_without_markdown(self, response) -> tuple[str, Dict]:
        """Stream response with code block detection."""
        full_response = ""
        data = {}
        in_code_block = False
        code_buffer = ""
        code_lang = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data:
                        content = data["message"].get("content", "")
                        full_response += content

                        for char in content:
                            if not in_code_block and full_response.endswith("```"):
                                in_code_block = True
                                code_buffer = ""
                                continue

                            if in_code_block:
                                code_buffer += char
                                if code_buffer.endswith("```"):
                                    in_code_block = False
                                    code_content = code_buffer[:-3]
                                    lines = code_content.split("\n", 1)
                                    if (
                                        lines[0].strip().isalnum()
                                        and len(lines[0].strip()) < 15
                                    ):
                                        code_lang = lines[0].strip()
                                        code_content = (
                                            lines[1] if len(lines) > 1 else ""
                                        )
                                    else:
                                        code_lang = "text"
                                    self.console.print()
                                    self.console.print(
                                        Panel(
                                            Syntax(
                                                code_content.strip(),
                                                code_lang,
                                                theme="monokai",
                                                line_numbers=True,
                                                word_wrap=True,
                                            ),
                                            title=f"[dim]{code_lang}[/]"
                                            if code_lang != "text"
                                            else None,
                                            border_style=self.theme["muted"],
                                            box=ROUNDED,
                                            padding=(0, 1),
                                        )
                                    )
                                    code_buffer = ""
                                    code_lang = ""
                            else:
                                print(char, end="", flush=True)

                    if data.get("done"):
                        break
                except Exception:
                    continue

        print()
        return full_response, data

    def render_response(self, text: str) -> None:
        """Render markdown/code blocks in response text."""
        if not text:
            return

        code_pattern = r"```(\w*)\n(.*?)```"
        parts = []
        last_end = 0

        for match in re.finditer(code_pattern, text, re.DOTALL):
            if match.start() > last_end:
                parts.append(("text", text[last_end : match.start()]))

            lang = match.group(1) or "text"
            code = match.group(2).strip()
            parts.append(("code", (lang, code)))
            last_end = match.end()

        if last_end < len(text):
            parts.append(("text", text[last_end:]))

        for part_type, content in parts:
            if part_type == "text" and content.strip():
                md = Markdown(content, code_theme="monokai")
                self.console.print(md)
            elif part_type == "code":
                lang, code = content
                self.console.print(
                    Panel(
                        Syntax(
                            code,
                            lang,
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True,
                        ),
                        title=f"[dim]{lang}[/]" if lang != "text" else None,
                        border_style=self.theme["muted"],
                        box=ROUNDED,
                        padding=(0, 1),
                    )
                )

    def handle_response(self, response: Optional[str]) -> None:
        """Process and store model response."""
        if response:
            self.messages.append({"role": "assistant", "content": response})
            if self._on_autosave:
                self._on_autosave()

    def send_user_message(
        self, content: str, images: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send user message and get response."""
        message: Dict[str, object] = {"role": "user", "content": content}
        if images:
            message["images"] = images

        self.messages.append(message)
        self.maybe_summarize()

        response = self.chat_stream(self.model, self.messages, self.current_temperature)
        self.handle_response(response)
        return response

    def set_persona(self, persona_name: Optional[str]) -> bool:
        """Set current persona."""
        if persona_name and persona_name not in PERSONAS:
            return False
        self.current_persona = persona_name
        self.update_system_message()
        return True

    def get_persona_info(self, persona_name: str) -> Optional[Dict]:
        """Get persona information."""
        return PERSONAS.get(persona_name)

    def list_personas(self) -> Dict:
        """List all available personas."""
        return PERSONAS.copy()
