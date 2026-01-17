from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .clipboard import ClipboardTracker
from .templates import generate_html_export as _generate_html_template
from .commands import CommandRegistry, CommandHandlers, SmartCompleter
from .logging_utils import set_log_level, setup_logging
from .models import TokenStats
from .security import (
    SecurityError,
    encrypt_text,
    get_encryption_key,
    mask_messages,
    mask_sensitive_text,
)
from .session_store import SessionMeta, SessionStore
from .storage import (
    load_config,
    load_favorites,
    load_prompts,
    migrate_history,
    resolve_paths,
)
from .utils import (
    estimate_message_tokens,
    format_size,
    get_model_prompt,
)

# Refactored modules
from .chat_engine import ChatEngine, SUMMARY_PREFIX
from .model_manager import ModelManager
from .ui_display import UIDisplay


DEFAULT_SUMMARY_KEEP = 6


class ChatApp:
    def __init__(self, diagnostic_override: bool = False) -> None:
        self.paths = resolve_paths()
        self.logger = setup_logging(self.paths.log_file, diagnostic_override)

        self.config = load_config(self.paths, self.logger)
        if diagnostic_override and not self.config.diagnostic:
            self.config.diagnostic = True

        set_log_level(self.logger, self.config.diagnostic)
        self._apply_env_overrides()

        self.prompts = load_prompts(self.paths, self.logger)
        self.favorites = load_favorites(self.paths, self.logger)
        migrate_history(self.paths, self.logger)
        self.session_store = SessionStore(self.paths, self.logger, self.config)

        self.console = Console()
        self.token_stats = TokenStats()
        self.chat_title: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_tags: List[str] = []
        self.session: Optional[PromptSession] = None
        self.clipboard_tracker = ClipboardTracker()

        # Initialize refactored modules
        self.model_manager = ModelManager(
            config=self.config,
            console=self.console,
            logger=self.logger,
            prompts=self.prompts,
            model_cache_file=self.paths.model_cache_file,
            benchmarks_file=self.paths.benchmarks_file,
            get_theme=lambda: self.theme,
        )

        self.chat_engine = ChatEngine(
            config=self.config,
            console=self.console,
            logger=self.logger,
            prompts=self.prompts,
            token_stats=self.token_stats,
            get_theme=lambda: self.theme,
        )

        self.ui_display = UIDisplay(
            config=self.config,
            console=self.console,
            logger=self.logger,
            favorites=self.favorites,
            prompts=self.prompts,
            token_stats=self.token_stats,
            get_theme=lambda: self.theme,
        )

        # Legacy state accessors (for backward compatibility during transition)
        self.model_cache = self.model_manager.model_cache
        self.models: List[Dict[str, object]] = []
        self.model: Optional[str] = None
        self.messages: List[Dict[str, object]] = []

        self.registry = CommandRegistry()
        self.cmd_handlers = CommandHandlers(self)
        self._register_commands()

    def _apply_env_overrides(self) -> None:
        env_host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_CLI_HOST")
        if env_host:
            self.logger.info("OLLAMA_HOST override bulundu: %s", env_host)
            self.config.ollama_host = env_host

    # ─────────────────────────────────────────────────────────────
    # Delegated Model Manager Methods
    # ─────────────────────────────────────────────────────────────

    def get_model_capabilities(
        self, model_name: str, refresh: bool = False
    ) -> Optional[Dict[str, object]]:
        """Delegate to model_manager."""
        return self.model_manager.get_model_capabilities(model_name, refresh)

    def supports_vision(self, model_name: str) -> bool:
        """Delegate to model_manager."""
        return self.model_manager.supports_vision(model_name)

    def apply_model_profiles(self, model_name: str) -> None:
        """Delegate to model_manager."""
        self.model_manager.apply_model_profiles(model_name)

    @property
    def profile_prompt(self) -> str:
        """Get profile prompt from model_manager."""
        return self.model_manager.profile_prompt

    @profile_prompt.setter
    def profile_prompt(self, value: str) -> None:
        """Set profile prompt on model_manager."""
        self.model_manager.profile_prompt = value

    @property
    def active_profile_name(self) -> Optional[str]:
        """Get active profile name from model_manager."""
        return self.model_manager.active_profile_name

    @active_profile_name.setter
    def active_profile_name(self, value: Optional[str]) -> None:
        """Set active profile name on model_manager."""
        self.model_manager.active_profile_name = value

    @property
    def current_temperature(self) -> Optional[float]:
        """Get current temperature from model_manager."""
        return self.model_manager.current_temperature

    @current_temperature.setter
    def current_temperature(self, value: Optional[float]) -> None:
        """Set current temperature on model_manager."""
        self.model_manager.current_temperature = value

    def _register_commands(self) -> None:
        """Register all commands using CommandHandlers."""
        self.cmd_handlers.register_all(self.registry)

    def run(self) -> int:
        self.console.clear()
        self.print_header()

        self.models = self.get_models()
        if not self.models:
            self.console.print("[yellow]Model bulunamadi. /pull <model> ile indir.[/]")
            return 1

        self.session = PromptSession(
            history=FileHistory(str(self.paths.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SmartCompleter(
                self.registry, self.favorites, self.models, self.config.profiles
            ),
            complete_while_typing=False,
        )

        self.model = self.select_model(self.models)
        self.console.print()
        self.show_model_info(self.model)
        self.console.print()

        self.messages = self.init_conversation(self.model)

        if self.base_system_prompt:
            self.console.print(f"[{self.theme['success']}]✓ Sistem promptu yuklendi[/]")
        if self.supports_vision(self.model):
            self.console.print(
                f"[{self.theme['accent']}]👁️ Vision model - /img ile resim gonder[/]"
            )

        self.console.print(
            f"[{self.theme['muted']}]↑↓ gecmis • Tab tamamla • /help yardim • /q cikis[/]"
        )
        self.console.print(Rule(style=self.theme["muted"]))
        self.console.print()

        while True:
            try:
                # Clipboard monitoring
                self._check_clipboard()

                prompt_text = HTML(
                    f'<style fg="{self.theme["user"]}" bold="true">◉ SEN: </style>'
                )
                user_input = self.session.prompt(prompt_text)

                if not user_input.strip():
                    continue

                cmd = user_input.strip()
                cmd_lower = cmd.lower()

                if cmd_lower == self.config.multiline_trigger:
                    multiline = self.get_multiline_input()
                    if multiline:
                        user_input = multiline
                        cmd_lower = ""
                    else:
                        continue

                if cmd_lower.startswith("/"):
                    should_continue = self.handle_command(cmd)
                    if not should_continue:
                        break
                    if should_continue:
                        continue

                self.send_user_message(user_input)

                # Auto-title generation
                self._maybe_generate_title()

            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                self.console.print(f"\n[{self.theme['accent']}]Gorusuruz! 👋[/]\n")
                break

        return 0

    def _check_clipboard(self) -> None:
        """Check for clipboard changes if monitoring is enabled."""
        if not self.config.clipboard_monitor:
            return

        change = self.clipboard_tracker.check_change(self.logger)
        if not change:
            return

        ctype, content = change
        if ctype == "text" and self.config.clipboard_notify:
            preview = content[:50] + "..." if len(content) > 50 else content
            self.console.print(
                f"[{self.theme['accent']}]📋 Panoda yeni metin:[/] {preview}"
            )
            self.console.print(f"[{self.theme['muted']}]/yapistir ile kullan[/]\n")
        elif ctype == "image" and self.config.clipboard_notify:
            self.console.print(f"[{self.theme['accent']}]🖼️ Panoda yeni resim var[/]")
            self.console.print(f"[{self.theme['muted']}]/paste ile kullan[/]\n")

    def _maybe_generate_title(self) -> None:
        """Generate auto-title if conditions are met."""
        if not self.config.auto_title:
            return
        if self.chat_title:  # Already has a title
            return

        # Count non-system messages
        conversation = [m for m in self.messages if m.get("role") != "system"]
        if len(conversation) < self.config.auto_title_after + 1:
            return

        title = self.chat_engine.generate_title(self.messages)
        if title:
            self.chat_title = title
            self.console.print(f"[{self.theme['muted']}]📝 Başlık: {title}[/]\n")

    @property
    def theme(self) -> Dict[str, str]:
        active = self.config.themes.get(self.config.theme)
        if active:
            return active.model_dump()
        fallback = next(iter(self.config.themes.values()))
        return fallback.model_dump()

    def handle_command(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        cmd_key = cmd_lower.split()[0]
        command = self.registry.get(cmd_key)
        if command:
            return command.handler(cmd)

        if self.handle_favorite_shortcut(cmd, cmd_lower):
            return True

        self.console.print(f"[{self.theme['error']}]Bilinmeyen komut: {cmd}[/]\n")
        return True

    def handle_favorite_shortcut(self, cmd: str, cmd_lower: str) -> bool:
        if not cmd_lower.startswith("/"):
            return False

        fav_shortcut = cmd_lower[1:].split()[0] if cmd_lower[1:] else ""
        favs = self.favorites.favorites
        if fav_shortcut in favs:
            extra = cmd[len(fav_shortcut) + 1 :].strip()
            user_input = f"{favs[fav_shortcut]} {extra}".strip()
            self.send_user_message(user_input)
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Delegated UI Display Methods
    # ─────────────────────────────────────────────────────────────

    def print_header(self) -> None:
        """Delegate to ui_display."""
        self.ui_display.print_header()

    # ─────────────────────────────────────────────────────────────
    # Delegated Model Manager Methods (continued)
    # ─────────────────────────────────────────────────────────────

    def get_models(self) -> List[Dict[str, object]]:
        """Delegate to model_manager."""
        models = self.model_manager.get_models()
        self.model_manager.models = models
        return models

    def select_model(self, models: List[Dict[str, object]]) -> str:
        """Delegate to model_manager."""
        # Ensure session is set on model_manager
        if self.session:
            self.model_manager.set_session(self.session)
        return self.model_manager.select_model(models)

    def show_model_info(self, model_name: str) -> None:
        """Delegate to model_manager."""
        # Update model_manager's models list for context
        self.model_manager.models = self.models
        self.model_manager.show_model_info(model_name)

    def show_help(self) -> None:
        sections = [
            (
                "Temel",
                [
                    ("/clear, /c", "Sohbeti temizle"),
                    ("/model, /m", "Model degistir"),
                    ("/save, /s", "Sohbeti kaydet"),
                    ("/load, /l", "Sohbet yukle"),
                    ("/sessions", "Kayitli sohbetleri listele"),
                    ("/session", "Oturum yonetimi (tag/sil)"),
                    ("/quit, /q", "Cikis"),
                ],
            ),
            (
                "Ozellikler",
                [
                    ("/prompt, /p", "Sistem promptu"),
                    ("/theme, /t", "Tema degistir"),
                    ("/default, /d", "Varsayilan modeli ayarla"),
                    ("/stats", "Yuklu modeller ve VRAM"),
                    ("/compare", "Modelleri karsilastir"),
                    ("/export", "Sohbeti disa aktar"),
                ],
            ),
            (
                "Kisayollar",
                [
                    ("/fav", "Favorileri listele"),
                    ("/fav add", "Favori ekle"),
                    ("/tpl", "Sablonlari listele"),
                    ("/model", "Model sec"),
                    ("/history, /h", "Mesaj gecmisi"),
                ],
            ),
            (
                "Gorsel",
                [
                    ("/img", "Resim gonder"),
                    ("/paste", "Panodan resim"),
                ],
            ),
            (
                "Diger",
                [
                    ("/retry", "Son yaniti yeniden olustur"),
                    ("/edit", "Son mesaji duzenle"),
                    ("/copy", "Son yaniti kopyala"),
                    ("/search", "Mesajlarda ara"),
                    ("/tokens", "Token kullanimi"),
                    ("/context", "Context durumunu goster"),
                    ("/summarize", "Konusma ozetle"),
                    ("/quick", "Hizli model degistir"),
                    ("/title", "Sohbete baslik"),
                    ("/persona", "Persona degistir"),
                    ("/profile", "Profil sec"),
                    ("/security", "Guvenlik ayarlari"),
                    ("/continue", "Yaniti devam ettir"),
                    ("/temp", "Sicaklik ayari"),
                    ("/diag", "Diagnostik mod"),
                ],
            ),
        ]

        for title, items in sections:
            table = Table(title=f"[bold]{title}[/]", box=ROUNDED, show_header=False)
            table.add_column("Komut", style=f"bold {self.theme['accent']}")
            table.add_column("Aciklama", style="white")
            for cmd, desc in items:
                table.add_row(cmd, desc)
            self.console.print(table)
            self.console.print()

    def show_favorites(self) -> None:
        favs = self.favorites.favorites
        if not favs:
            self.console.print(
                f"[{self.theme['muted']}]Favori yok. /fav add <isim> <prompt>[/]\n"
            )
            return

        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("Isim", style=f"bold {self.theme['accent']}")
        table.add_column("Prompt", style="white", max_width=50)

        for name, prompt in favs.items():
            table.add_row(name, prompt[:47] + "..." if len(prompt) > 50 else prompt)

        self.console.print(
            Panel(table, title="[bold]Favoriler[/]", border_style=self.theme["primary"])
        )
        self.console.print()

    def show_templates(self) -> None:
        templates = self.favorites.templates

        if not templates:
            self.console.print(f"[{self.theme['muted']}]Sablon yok.[/]\n")
            return

        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("Komut", style=f"bold {self.theme['accent']}")
        table.add_column("Isim", style="bold white")
        table.add_column("Degiskenler", style=self.theme["muted"])

        for key, tpl in templates.items():
            vars_found = re.findall(r"\{(\w+)\}", tpl.prompt)
            table.add_row(key, tpl.name or key, ", ".join(vars_found) or "-")

        self.console.print(
            Panel(table, title="[bold]Sablonlar[/]", border_style=self.theme["primary"])
        )
        self.console.print()

    def save_session(self, show_message: bool = True) -> Optional[SessionMeta]:
        if not self.messages or not self.model:
            return None
        title = self.chat_title or self._infer_title()
        token_stats = {
            "prompt_tokens": self.token_stats.prompt_tokens,
            "completion_tokens": self.token_stats.completion_tokens,
            "total_tokens": self.token_stats.total_tokens,
        }
        try:
            meta = self.session_store.save_session(
                session_id=self.session_id,
                title=title,
                model=self.model,
                messages=self.messages,
                token_stats=token_stats,
                tags=self.session_tags,
                summary=self.summary,
                show_log=not show_message,
            )
            self.session_id = meta.id
            self.session_tags = meta.tags
            self.session_store.prune_sessions([self.session_id])
            if show_message:
                self.console.print(
                    f"[{self.theme['success']}]✓ Kaydedildi: {meta.title}[/]\n"
                )
            return meta
        except SecurityError as exc:
            self.console.print(f"[{self.theme['error']}]Guvenlik hatasi: {exc}[/]\n")
            return None
        except Exception as exc:
            self.logger.exception("Session kaydedilemedi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return None

    def _infer_title(self) -> str:
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()[:60]
        return self.model or "Oturum"

    def list_sessions(self) -> List[SessionMeta]:
        sessions = self.session_store.list_sessions()
        if not sessions:
            self.console.print(f"[{self.theme['muted']}]Kayitli sohbet yok.[/]\n")
            return []

        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("#", style="bold cyan", width=4)
        table.add_column("Baslik", style="bold white", max_width=30)
        table.add_column("Model", style=self.theme["muted"], max_width=20)
        table.add_column("Etiket", style=self.theme["accent"], max_width=20)
        table.add_column("Guncel", style=self.theme["muted"], width=16)
        table.add_column("Token", style=self.theme["muted"], justify="right", width=8)

        for i, session in enumerate(sessions, 1):
            try:
                updated = datetime.fromisoformat(session.updated_at).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                updated = session.updated_at
            tags = ", ".join(session.tags) if session.tags else "-"
            table.add_row(
                str(i),
                session.title[:30],
                session.model,
                tags,
                updated,
                f"{session.token_total}",
            )

        self.console.print(
            Panel(
                table,
                title="[bold]Kayitli Sohbetler[/]",
                border_style=self.theme["primary"],
            )
        )
        self.console.print()
        return sessions

    def load_chat(self) -> None:
        sessions = self.list_sessions()
        if not sessions:
            return
        try:
            choice = (
                self.session.prompt(
                    HTML(f'<style fg="{self.theme["primary"]}">Sec (0=iptal): </style>')
                )
                or "0"
            )
            idx = int(choice)
            if idx == 0 or idx > len(sessions):
                return
            meta = sessions[idx - 1]
            self._load_session(meta)
        except ValueError:
            self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")

    def _load_session(self, meta: SessionMeta) -> None:
        try:
            data = self.session_store.load_session(meta.id)
        except SecurityError as exc:
            self.console.print(f"[{self.theme['error']}]Guvenlik hatasi: {exc}[/]\n")
            return

        if not data:
            self.console.print(f"[{self.theme['error']}]Session bulunamadi[/]\n")
            return

        self.session_id = meta.id
        self.session_tags = meta.tags
        self.chat_title = meta.title
        self.model = data.meta.model
        self.messages = data.messages
        self.summary = data.summary or self.extract_summary(self.messages)
        self.token_stats = TokenStats(**data.token_stats)

        self.models = self.get_models()
        self.session.completer = SmartCompleter(
            self.registry, self.favorites, self.models, self.config.profiles
        )
        self.apply_model_profiles(self.model)
        base_prompt = self._extract_base_system_prompt()
        if base_prompt:
            self.base_system_prompt = base_prompt
        else:
            prompt_info = get_model_prompt(self.model, self.prompts)
            self.base_system_prompt = prompt_info.get("system_prompt", "")
        self.update_system_message()
        self.update_summary_message()

        if self.model:
            self.show_model_info(self.model)
        self.console.print(f"[{self.theme['success']}]✓ Yuklendi: {meta.title}[/]\n")

    def show_stats(self) -> None:
        host = self.config.ollama_host
        try:
            response = requests.get(f"{host}/api/ps", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])

            if not models:
                self.console.print(f"[{self.theme['muted']}]Yuklu model yok.[/]\n")
                return

            table = Table(
                box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2)
            )
            table.add_column("Model", style="bold white")
            table.add_column("VRAM", style=self.theme["accent"])
            table.add_column("Sure", style=self.theme["muted"])

            for model in models:
                name = model.get("name", "?")
                vram = format_size(int(model.get("size_vram", 0)))
                expires = model.get("expires_at", "")
                try:
                    exp_time = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                    mins = int(
                        (exp_time - datetime.now(exp_time.tzinfo)).total_seconds() / 60
                    )
                    duration = f"{mins} dk"
                except Exception:
                    duration = "-"
                table.add_row(name, vram, duration)

            self.console.print(
                Panel(
                    table,
                    title="[bold]Yuklu Modeller[/]",
                    border_style=self.theme["primary"],
                )
            )
            self.console.print()
        except Exception as exc:
            self.logger.exception("Model durumlari alinmadi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")

    def pull_model(self, model_name: str) -> bool:
        host = self.config.ollama_host

        self.console.print(f"[{self.theme['accent']}]Indiriliyor: {model_name}[/]")
        try:
            response = requests.post(
                f"{host}/api/pull", json={"name": model_name}, stream=True, timeout=3600
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"[cyan]{model_name}", total=100)

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "pulling" in data.get("status", ""):
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            if total > 0:
                                progress.update(
                                    task, completed=(completed / total) * 100
                                )
                        if data.get("status") == "success":
                            progress.update(task, completed=100)
                            break

            self.console.print(
                f"[{self.theme['success']}]✓ Indirildi: {model_name}[/]\n"
            )
            return True
        except Exception as exc:
            self.logger.exception("Model indirilemedi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return False

    def delete_model(self, model_name: str) -> bool:
        host = self.config.ollama_host
        if not Confirm.ask(f"[{self.theme['error']}]{model_name} silinecek?[/]"):
            return False
        try:
            response = requests.delete(
                f"{host}/api/delete", json={"name": model_name}, timeout=30
            )
            if response.status_code == 200:
                self.console.print(
                    f"[{self.theme['success']}]✓ Silindi: {model_name}[/]\n"
                )
                return True
            self.console.print(f"[{self.theme['error']}]Hata: {response.text}[/]\n")
        except Exception as exc:
            self.logger.exception("Model silinemedi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
        return False

    def render_response(self, text: str) -> None:
        code_pattern = r"```(\w+)?\n(.*?)```"
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

    def chat_stream(
        self, model: str, messages, temperature: Optional[float] = None
    ) -> Optional[str]:
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
            header.append("◉ ", style=f"bold {self.theme['assistant']}")
            header.append(
                model.split(":")[0].upper(), style=f"bold {self.theme['assistant']}"
            )
            self.console.print(header)
            self.console.print(Rule(style=self.theme["muted"]))

            full_response = ""
            start_time = datetime.now()
            data = {}

            if self.config.render_markdown:
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
                                            Markdown(
                                                full_response, code_theme="monokai"
                                            )
                                        )
                                        last_update = now

                                if data.get("done"):
                                    break
                            except Exception:
                                continue
                    live.update(Markdown(full_response, code_theme="monokai"))
            else:
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
                                    if not in_code_block and full_response.endswith(
                                        "```"
                                    ):
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

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            self.token_stats.prompt_tokens += prompt_tokens
            self.token_stats.completion_tokens += completion_tokens
            self.token_stats.total_tokens += prompt_tokens + completion_tokens

            if self.config.show_metrics:
                elapsed = (datetime.now() - start_time).total_seconds()
                tps = completion_tokens / elapsed if elapsed > 0 else 0
                self.console.print(
                    f"\n[{self.theme['muted']}]⏱ {elapsed:.1f}s  ◈ {completion_tokens} token  ⚡ {tps:.1f} t/s[/]"
                )

            self.console.print()
            return full_response

        except KeyboardInterrupt:
            self.console.print(f"\n[{self.theme['accent']}]◼ Iptal[/]\n")
            return full_response if full_response else None
        except Exception as exc:
            self.logger.exception("Chat stream hatasi")
            self.console.print(f"\n[{self.theme['error']}]Hata: {exc}[/]\n")
            return None

    def init_conversation(self, model_name: str) -> List[Dict[str, object]]:
        """Initialize conversation, delegating to chat_engine for message creation."""
        self.session_id = None
        self.session_tags = []
        self.token_stats = TokenStats()
        self.chat_title = None

        # Delegate to chat_engine which handles message creation and state
        messages = self.chat_engine.init_conversation(
            model_name, apply_profiles_callback=self.apply_model_profiles
        )

        # Sync state between app and chat_engine
        self.summary = self.chat_engine.summary
        self.base_system_prompt = self.chat_engine.base_system_prompt

        return messages

    def build_system_prompt(self) -> str:
        """Build system prompt, delegating to chat_engine."""
        # Sync state to chat_engine
        self.chat_engine.base_system_prompt = self.base_system_prompt
        self.chat_engine.profile_prompt = self.profile_prompt
        return self.chat_engine.build_system_prompt()

    def update_system_message(self) -> None:
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
        for idx, msg in enumerate(self.messages):
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
                return idx
        return None

    def _find_base_system_index(self) -> Optional[int]:
        for idx, msg in enumerate(self.messages):
            if msg.get("role") == "system" and not self._is_summary_message(msg):
                return idx
        return None

    def _is_summary_message(self, msg: Dict[str, object]) -> bool:
        content = msg.get("content", "")
        return isinstance(content, str) and content.startswith(SUMMARY_PREFIX)

    def estimate_context_tokens(self) -> int:
        return sum(estimate_message_tokens(msg) for msg in self.messages)

    def maybe_summarize(self, force: bool = False) -> bool:
        if not force and not self.config.context_autosummarize:
            return False
        if self.config.context_token_budget <= 0:
            return False
        total = self.estimate_context_tokens()
        if not force and total <= self.config.context_token_budget:
            return False
        return self.summarize_messages()

    def summarize_messages(self) -> bool:
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
        conversation = [m for m in self.messages if m.get("role") != "system"]
        keep_last = max(1, self.config.context_keep_last or DEFAULT_SUMMARY_KEEP)
        if len(conversation) <= keep_last:
            return [], conversation
        to_summarize = conversation[:-keep_last]
        keep = conversation[-keep_last:]
        return to_summarize, keep

    def request_summary(self, messages: List[Dict[str, object]]) -> Optional[str]:
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
        for msg in messages:
            if self._is_summary_message(msg):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.replace(SUMMARY_PREFIX, "", 1).strip()
        return ""

    def _extract_base_system_prompt(self) -> str:
        for msg in self.messages:
            if msg.get("role") == "system" and not self._is_summary_message(msg):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def handle_response(self, response: Optional[str]) -> None:
        if response:
            self.messages.append({"role": "assistant", "content": response})
            self.maybe_autosave()

    def send_user_message(
        self, content: str, images: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send user message, delegating to chat_engine."""
        # Sync messages to chat_engine before sending
        self.chat_engine.messages = self.messages
        self.chat_engine.model = self.model
        self.chat_engine.current_temperature = self.current_temperature

        response = self.chat_engine.send_user_message(content, images)

        # Maybe autosave after response
        self.maybe_autosave()
        return response

    def maybe_autosave(self) -> None:
        if not self.config.auto_save:
            return
        self.save_session(show_message=False)

    def get_multiline_input(self) -> Optional[str]:
        self.console.print(
            f"[{self.theme['muted']}]Coklu satir modu. Bitirmek icin '{self.config.multiline_trigger}' yaz.[/]"
        )

        lines = []
        while True:
            try:
                line = self.session.prompt("... ")
                if line.strip() == self.config.multiline_trigger:
                    break
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                self.console.print(f"[{self.theme['accent']}]Iptal[/]")
                return None
        return "\n".join(lines)

    def search_messages(self, keyword: str) -> None:
        results = []
        keyword_lower = keyword.lower()

        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and keyword_lower in content.lower():
                results.append((i, msg))

        if not results:
            self.console.print(f"[{self.theme['muted']}]'{keyword}' bulunamadi[/]\n")
            return

        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("#", style="bold cyan", width=4)
        table.add_column("Rol", style=f"bold {self.theme['accent']}", width=10)
        table.add_column("Icerik", style="white")

        for idx, msg in results:
            content = msg.get("content", "")[:100]
            highlighted = re.sub(
                f"({re.escape(keyword)})",
                f"[bold {self.theme['accent']}]\\1[/]",
                content,
                flags=re.IGNORECASE,
            )
            role_color = (
                self.theme["user"] if msg["role"] == "user" else self.theme["assistant"]
            )
            table.add_row(
                str(idx), f"[{role_color}]{msg['role']}[/]", highlighted + "..."
            )

        self.console.print(
            Panel(
                table,
                title=f"[bold]Arama: '{keyword}'[/]",
                border_style=self.theme["primary"],
            )
        )
        self.console.print()

    def show_tokens(self) -> None:
        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("Tur", style=f"bold {self.theme['accent']}")
        table.add_column("Miktar", style="bold white", justify="right")

        table.add_row("Prompt Tokens", f"{self.token_stats.prompt_tokens:,}")
        table.add_row("Completion Tokens", f"{self.token_stats.completion_tokens:,}")
        table.add_row("Toplam", f"[bold]{self.token_stats.total_tokens:,}[/]")

        self.console.print(
            Panel(
                table,
                title="[bold]Token Kullanimi[/]",
                border_style=self.theme["primary"],
            )
        )
        self.console.print()

    def compare_models(self, question: str, model_names: List[str]) -> Dict[str, str]:
        host = self.config.ollama_host
        results: Dict[str, str] = {}

        self.console.print(
            f"\n[{self.theme['accent']}]🔀 {len(model_names)} model karsilastiriliyor...[/]\n"
        )

        def ask_model(model_name: str):
            try:
                prompt_info = get_model_prompt(model_name, self.prompts)
                msgs = []
                if prompt_info.get("system_prompt"):
                    msgs.append(
                        {"role": "system", "content": prompt_info["system_prompt"]}
                    )
                msgs.append({"role": "user", "content": question})

                response = requests.post(
                    f"{host}/api/chat",
                    json={"model": model_name, "messages": msgs, "stream": False},
                    timeout=120,
                )
                data = response.json()
                return model_name, data.get("message", {}).get("content", "Yanit yok")
            except Exception as exc:
                self.logger.exception("Model karsilastirma hatasi: %s", model_name)
                return model_name, f"Hata: {exc}"

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(ask_model, m): m for m in model_names}
            for future in as_completed(futures):
                model_name, response = future.result()
                results[model_name] = response

        for model_name, response in results.items():
            prompt_info = get_model_prompt(model_name, self.prompts)
            icon = prompt_info.get("icon", "🤖")
            self.console.print(
                Panel(
                    response[:500] + ("..." if len(response) > 500 else ""),
                    title=f"[bold]{icon} {model_name}[/]",
                    border_style=self.theme["primary"],
                    padding=(1, 2),
                )
            )

        return results

    def benchmark_model(
        self, model_name: str, prompt: str, runs: int
    ) -> Optional[Dict[str, object]]:
        host = self.config.ollama_host
        results = []

        self.console.print(
            f"[{self.theme['muted']}]Benchmark: {model_name} (x{runs})[/]"
        )

        for run in range(1, runs + 1):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.config.benchmark_temperature},
            }
            try:
                start = time.perf_counter()
                response = requests.post(
                    f"{host}/api/chat",
                    json=payload,
                    timeout=self.config.benchmark_timeout,
                )
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - start
                prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                completion_tokens = int(data.get("eval_count", 0) or 0)
                total_tokens = prompt_tokens + completion_tokens
                tps = completion_tokens / elapsed if elapsed > 0 else 0

                result = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": model_name,
                    "prompt": prompt,
                    "run": run,
                    "elapsed": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "tps": tps,
                    "temperature": self.config.benchmark_temperature,
                }
                results.append(result)
                self._save_benchmark_result(result)
            except Exception as exc:
                self.logger.exception("Benchmark hatasi: %s", model_name)
                self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]")
                return None

        avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
        avg_prompt = sum(r["prompt_tokens"] for r in results) / len(results)
        avg_completion = sum(r["completion_tokens"] for r in results) / len(results)
        avg_total = sum(r["total_tokens"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)

        return {
            "model": model_name,
            "runs": len(results),
            "avg_elapsed": avg_elapsed,
            "avg_prompt_tokens": avg_prompt,
            "avg_completion_tokens": avg_completion,
            "avg_total_tokens": avg_total,
            "avg_tps": avg_tps,
        }

    def export_chat(self, format_type: str) -> Optional[Path]:
        try:
            save_dir = Path(self.config.save_directory).expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = self.model.split(":")[0].replace("/", "-")
            title = self.chat_title or f"Chat with {self.model}"
            title_slug = title.replace(" ", "_")[:30]

            messages = self.messages
            if self.config.mask_sensitive:
                messages = mask_messages(messages, self.config.mask_patterns)
                title = mask_sensitive_text(title, self.config.mask_patterns)

            content = ""
            extension = format_type

            if format_type == "json":
                export_data = {
                    "title": title,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "messages": messages,
                    "token_stats": self.token_stats.__dict__,
                }
                content = json.dumps(export_data, indent=2, ensure_ascii=False)

            elif format_type == "txt":
                lines = [
                    f"Ollama Chat - {self.model}",
                    f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                if title:
                    lines.append(f"Baslik: {title}")
                lines.append("=" * 50)
                lines.append("")
                for msg in messages:
                    if msg["role"] == "system":
                        continue
                    role = "SEN" if msg["role"] == "user" else model_short.upper()
                    content_text = msg.get("content", "")
                    if isinstance(content_text, list):
                        content_text = "[Gorsel]"
                    lines.append(f"[{role}]")
                    lines.append(str(content_text))
                    lines.append("")
                content = "\n".join(lines)

            elif format_type == "html":
                content = self.generate_html_export(messages, title_override=title)

            else:
                extension = "md"
                lines = [
                    f"# {title}",
                    "",
                    f"**Model:** {self.model}  ",
                    f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "---",
                    "",
                ]
                for msg in messages:
                    if msg["role"] == "system":
                        continue
                    role = (
                        "🧑 Sen"
                        if msg["role"] == "user"
                        else f"🤖 {model_short.upper()}"
                    )
                    content_text = msg.get("content", "")
                    if isinstance(content_text, list):
                        content_text = "*[Gorsel]*"
                    lines.append(f"### {role}")
                    lines.append("")
                    lines.append(str(content_text))
                    lines.append("")
                    lines.append("---")
                    lines.append("")
                content = "\n".join(lines)

            if self.config.encrypt_exports:
                key = get_encryption_key(self.config)
                if not key:
                    raise SecurityError("Export sifreleme icin anahtar gerekli")
                content = encrypt_text(content, key)
                extension = f"{extension}.enc"

            filepath = save_dir / f"{timestamp}_{title_slug}.{extension}"
            filepath.write_text(content, encoding="utf-8")

            self.console.print(
                f"[{self.theme['success']}]✓ Disari aktarildi: {filepath}[/]\n"
            )
            return filepath
        except Exception as exc:
            self.logger.exception("Disa aktarma hatasi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return None

    def generate_html_export(
        self,
        messages: Optional[List[Dict[str, object]]] = None,
        title_override: Optional[str] = None,
    ) -> str:
        """Generate HTML export using shared template."""
        title = title_override or self.chat_title or f"Chat with {self.model}"
        msgs = messages or self.messages
        return _generate_html_template(
            messages=msgs,
            model=self.model,
            title=title,
            theme=self.theme,
            total_tokens=self.token_stats.total_tokens,
        )
