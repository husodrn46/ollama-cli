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
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.box import DOUBLE, ROUNDED
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

from .clipboard import copy_text
from .commands import Command, CommandRegistry
from .logging_utils import set_log_level, setup_logging
from .media import encode_image, paste_image_from_clipboard
from .models import FavoritesModel, ProfileModel, TokenStats
from .security import (
    SecurityError,
    encrypt_text,
    generate_key,
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
    read_json,
    resolve_paths,
    save_config,
    save_favorites,
    write_json,
)
from .utils import (
    estimate_message_tokens,
    format_size,
    get_model_prompt,
    is_vision_model,
)

# Refactored modules
from .chat_engine import ChatEngine, PERSONAS, SUMMARY_PREFIX
from .model_manager import ModelManager, MODEL_CACHE_TTL_SECONDS
from .ui_display import UIDisplay


DEFAULT_SUMMARY_KEEP = 6


class SmartCompleter(Completer):
    """Akilli tamamlayici - komutlar, favoriler, modeller."""

    def __init__(
        self,
        registry: CommandRegistry,
        favorites: FavoritesModel,
        models,
        profiles: Dict[str, ProfileModel],
    ):
        self.registry = registry
        self.favorites = favorites
        self.models = models
        self.profiles = profiles

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd in self.registry.command_strings():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))

            if text.startswith("/fav "):
                prefix = text[5:]
                for name in self.favorites.favorites.keys():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

            if text.startswith("/tpl ") or text.startswith("/template "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for name in self.favorites.templates.keys():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

            if text.startswith("/pull ") or text.startswith("/delete "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for model in self.models:
                    name = model.get("name", "")
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

            if text.startswith("/export "):
                prefix = text[8:]
                for fmt in ["html", "json", "txt", "md"]:
                    if fmt.startswith(prefix):
                        yield Completion(fmt, start_position=-len(prefix))

            if text.startswith("/persona "):
                prefix = text[9:]
                for name in PERSONAS.keys():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

            if text.startswith("/profile "):
                prefix = text[9:]
                for name in self.profiles.keys():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

            if text.startswith("/session "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for option in ["list", "open", "tag", "untag", "rename", "delete"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))

            if text.startswith("/security "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for option in ["mask", "encrypt", "export", "keygen", "key"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))

            if text.startswith("/markdown ") or text.startswith("/md "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for option in ["on", "off"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))

            if text.startswith("/bench "):
                prefix = text.split(" ", 1)[1] if " " in text else ""
                for option in ["all"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))


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
        self.registry.register(Command("/help", ("/?",), "Yardim", None, self.cmd_help))
        self.registry.register(
            Command("/quit", ("/q", "/exit"), "Cikis", None, self.cmd_quit)
        )
        self.registry.register(
            Command("/clear", ("/c",), "Sohbeti temizle", None, self.cmd_clear)
        )
        self.registry.register(
            Command("/model", ("/m",), "Model degistir", None, self.cmd_model)
        )
        self.registry.register(
            Command("/info", ("/i",), "Model bilgisi", None, self.cmd_info)
        )
        self.registry.register(
            Command("/prompt", ("/p",), "Sistem promptu", None, self.cmd_prompt)
        )
        self.registry.register(
            Command("/history", ("/h",), "Mesaj gecmisi", None, self.cmd_history)
        )
        self.registry.register(
            Command("/save", ("/s",), "Sohbeti kaydet", None, self.cmd_save)
        )
        self.registry.register(
            Command("/load", ("/l",), "Sohbet yukle", None, self.cmd_load)
        )
        self.registry.register(
            Command("/sessions", tuple(), "Kayitli sohbetler", None, self.cmd_sessions)
        )
        self.registry.register(
            Command("/session", tuple(), "Oturum yonetimi", None, self.cmd_session)
        )
        self.registry.register(
            Command("/theme", ("/t",), "Tema degistir", None, self.cmd_theme)
        )
        self.registry.register(
            Command("/default", ("/d",), "Varsayilan model", None, self.cmd_default)
        )
        self.registry.register(
            Command("/stats", tuple(), "Model durumlari", None, self.cmd_stats)
        )
        self.registry.register(
            Command("/fav", tuple(), "Favoriler", None, self.cmd_fav)
        )
        self.registry.register(
            Command("/template", ("/tpl",), "Sablonlar", None, self.cmd_template)
        )
        self.registry.register(
            Command("/pull", tuple(), "Model indir", None, self.cmd_pull)
        )
        self.registry.register(
            Command("/delete", tuple(), "Model sil", None, self.cmd_delete)
        )
        self.registry.register(
            Command("/img", tuple(), "Resim gonder", None, self.cmd_img)
        )
        self.registry.register(
            Command("/retry", tuple(), "Son yaniti yenile", None, self.cmd_retry)
        )
        self.registry.register(
            Command("/edit", tuple(), "Son mesaji duzenle", None, self.cmd_edit)
        )
        self.registry.register(
            Command("/copy", tuple(), "Son yaniti kopyala", None, self.cmd_copy)
        )
        self.registry.register(
            Command("/search", tuple(), "Mesajlarda ara", None, self.cmd_search)
        )
        self.registry.register(
            Command("/tokens", tuple(), "Token kullanimi", None, self.cmd_tokens)
        )
        self.registry.register(
            Command("/context", tuple(), "Context durumu", None, self.cmd_context)
        )
        self.registry.register(
            Command("/summarize", tuple(), "Konusma ozetle", None, self.cmd_summarize)
        )
        self.registry.register(
            Command("/quick", tuple(), "Hizli model degistir", None, self.cmd_quick)
        )
        self.registry.register(
            Command("/title", tuple(), "Baslik ata", None, self.cmd_title)
        )
        self.registry.register(
            Command(
                "/compare", tuple(), "Modelleri karsilastir", None, self.cmd_compare
            )
        )
        self.registry.register(
            Command("/bench", tuple(), "Benchmark calistir", None, self.cmd_bench)
        )
        self.registry.register(
            Command("/export", tuple(), "Disari aktar", None, self.cmd_export)
        )
        self.registry.register(
            Command("/paste", tuple(), "Panodan resim", None, self.cmd_paste)
        )
        self.registry.register(
            Command("/persona", tuple(), "Persona degistir", None, self.cmd_persona)
        )
        self.registry.register(
            Command("/profile", tuple(), "Profil sec", None, self.cmd_profile)
        )
        self.registry.register(
            Command("/security", tuple(), "Guvenlik ayarlari", None, self.cmd_security)
        )
        self.registry.register(
            Command("/continue", tuple(), "Yaniti devam ettir", None, self.cmd_continue)
        )
        self.registry.register(
            Command("/temp", tuple(), "Sicaklik ayari", None, self.cmd_temp)
        )
        self.registry.register(
            Command("/diag", tuple(), "Diagnostik mod", None, self.cmd_diag)
        )
        self.registry.register(
            Command("/markdown", ("/md",), "Markdown gorunumu", None, self.cmd_markdown)
        )

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

            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                self.console.print(f"\n[{self.theme['accent']}]Gorusuruz! 👋[/]\n")
                break

        return 0

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
        self.summary = ""
        self.session_id = None
        self.session_tags = []
        self.token_stats = TokenStats()
        self.chat_title = None
        self.apply_model_profiles(model_name)
        prompt_info = get_model_prompt(model_name, self.prompts)
        self.base_system_prompt = prompt_info.get("system_prompt", "")
        combined_prompt = self.build_system_prompt()
        messages = []
        if combined_prompt:
            messages.append({"role": "system", "content": combined_prompt})
        return messages

    def build_system_prompt(self) -> str:
        base = self.base_system_prompt or ""
        profile_prompt = self.profile_prompt or ""
        persona_prompt = ""
        if self.current_persona and self.current_persona in PERSONAS:
            persona_prompt = PERSONAS[self.current_persona]["prompt"]
        parts = [part for part in [base, profile_prompt, persona_prompt] if part]
        return "\n\n".join(parts)

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
        message: Dict[str, object] = {"role": "user", "content": content}
        if images:
            message["images"] = images
        self.messages.append(message)
        self.maybe_summarize()
        response = self.chat_stream(self.model, self.messages, self.current_temperature)
        self.handle_response(response)
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
                        "🧑 Sen" if msg["role"] == "user" else f"🤖 {model_short.upper()}"
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
        model_short = self.model.split(":")[0]
        title = title_override or self.chat_title or f"Chat with {self.model}"
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        messages = messages or self.messages

        html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, {self.theme['primary']}22, {self.theme['secondary']}22);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid {self.theme['primary']}44;
        }}
        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(135deg, {self.theme['primary']}, {self.theme['secondary']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header .meta {{
            color: {self.theme['muted']};
            font-size: 0.9rem;
        }}
        .header .model-badge {{
            display: inline-block;
            background: {self.theme['primary']}33;
            color: {self.theme['primary']};
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 1rem;
            border: 1px solid {self.theme['primary']}55;
        }}
        .message {{
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .message-user {{
            display: flex;
            justify-content: flex-end;
        }}
        .message-assistant {{
            display: flex;
            justify-content: flex-start;
        }}
        .bubble {{
            max-width: 80%;
            padding: 1rem 1.5rem;
            border-radius: 20px;
            position: relative;
        }}
        .bubble-user {{
            background: linear-gradient(135deg, {self.theme['user']}, {self.theme['user']}dd);
            border-bottom-right-radius: 5px;
            color: white;
        }}
        .bubble-assistant {{
            background: #1e293b;
            border: 1px solid #334155;
            border-bottom-left-radius: 5px;
        }}
        .role-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            opacity: 0.7;
        }}
        pre {{
            background: #0f172a;
            border-radius: 10px;
            padding: 1rem;
            overflow-x: auto;
            margin: 1rem 0;
            border: 1px solid #334155;
        }}
        code {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.9rem;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            padding: 2rem;
            background: #1e293b;
            border-radius: 15px;
            margin-top: 2rem;
            border: 1px solid #334155;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {self.theme['primary']};
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: {self.theme['muted']};
            text-transform: uppercase;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: {self.theme['muted']};
            font-size: 0.85rem;
        }}
        .footer a {{
            color: {self.theme['primary']};
            text-decoration: none;
        }}
        .code-container {{
            position: relative;
        }}
        .copy-btn {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: {self.theme['primary']}44;
            border: 1px solid {self.theme['primary']}66;
            color: {self.theme['primary']};
            padding: 4px 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        .code-container:hover .copy-btn {{
            opacity: 1;
        }}
        .copy-btn:hover {{
            background: {self.theme['primary']}66;
        }}
        .hljs {{
            background: transparent !important;
        }}
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💬 {title}</h1>
            <div class="meta">{date}</div>
            <div class="model-badge">🤖 {self.model}</div>
        </div>
        <div class="messages">
"""

        for msg in messages:
            if msg["role"] == "system":
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                content = "<em>[Gorsel icerik]</em>"
            else:
                code_pattern = r"```(\w*)?\n?(.*?)```"
                parts = []
                last_end = 0

                for match in re.finditer(code_pattern, content, re.DOTALL):
                    if match.start() > last_end:
                        text_part = content[last_end : match.start()]
                        text_part = (
                            text_part.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        text_part = re.sub(r"`([^`]+)`", r"<code>\1</code>", text_part)
                        text_part = text_part.replace("\n", "<br>")
                        parts.append(text_part)

                    lang = match.group(1) or ""
                    code = match.group(2)
                    code = (
                        code.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    parts.append(
                        f'<div class="code-container"><button class="copy-btn" onclick="copyCode(this)">Kopyala</button><pre><code class="language-{lang}">{code}</code></pre></div>'
                    )
                    last_end = match.end()

                if last_end < len(content):
                    text_part = content[last_end:]
                    text_part = (
                        text_part.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    text_part = re.sub(r"`([^`]+)`", r"<code>\1</code>", text_part)
                    text_part = text_part.replace("\n", "<br>")
                    parts.append(text_part)

                content = "".join(parts)

            if msg["role"] == "user":
                html += f"""
            <div class="message message-user">
                <div class="bubble bubble-user">
                    <div class="role-label">Sen</div>
                    <div class="content">{content}</div>
                </div>
            </div>
"""
            else:
                html += f"""
            <div class="message message-assistant">
                <div class="bubble bubble-assistant">
                    <div class="role-label">{model_short}</div>
                    <div class="content">{content}</div>
                </div>
            </div>
"""

        msg_count = len([m for m in messages if m["role"] != "system"])
        html += f"""
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{msg_count}</div>
                <div class="stat-label">Mesaj</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.token_stats.total_tokens:,}</div>
                <div class="stat-label">Token</div>
            </div>
        </div>
        <div class="footer">
            <p>Ollama CLI Pro v5.1 ile olusturuldu</p>
        </div>
    </div>
    <script>
        hljs.highlightAll();

        function copyCode(btn) {{
            const code = btn.nextElementSibling.querySelector('code');
            navigator.clipboard.writeText(code.textContent).then(() => {{
                btn.textContent = 'Kopyalandi!';
                setTimeout(() => btn.textContent = 'Kopyala', 2000);
            }});
        }}
    </script>
</body>
</html>
"""
        return html

    def cmd_help(self, _: str) -> bool:
        self.show_help()
        return True

    def cmd_quit(self, _: str) -> bool:
        if self.config.auto_save and len(self.messages) > 1:
            self.save_session(show_message=False)
        self.console.print(f"\n[{self.theme['accent']}]Gorusuruz! 👋[/]\n")
        return False

    def cmd_clear(self, _: str) -> bool:
        self.messages = self.init_conversation(self.model)
        self.console.clear()
        self.print_header()
        self.show_model_info(self.model)
        self.console.print(f"\n[{self.theme['success']}]✓ Temizlendi[/]\n")
        return True

    def cmd_model(self, _: str) -> bool:
        self.models = self.get_models()
        self.session.completer = SmartCompleter(
            self.registry, self.favorites, self.models, self.config.profiles
        )
        self.model = self.select_model(self.models)
        self.messages = self.init_conversation(self.model)
        self.console.print()
        self.show_model_info(self.model)
        if self.supports_vision(self.model):
            self.console.print(f"[{self.theme['accent']}]👁️ Vision model[/]")
        self.console.print()
        return True

    def cmd_info(self, _: str) -> bool:
        self.show_model_info(self.model)
        self.console.print()
        return True

    def cmd_prompt(self, _: str) -> bool:
        combined = self.build_system_prompt()
        self.console.print(
            Panel(
                f"[{self.theme['muted']}]{combined}[/]",
                title="[bold]Sistem Promptu[/]",
                border_style=self.theme["secondary"],
            )
        )
        self.console.print()
        return True

    def cmd_history(self, _: str) -> bool:
        for msg in self.messages:
            if msg["role"] == "system":
                continue
            content = msg.get("content", "")[:80]
            role_color = (
                self.theme["user"] if msg["role"] == "user" else self.theme["assistant"]
            )
            self.console.print(f"[{role_color}]{msg['role']}:[/] {content}...")
        self.console.print()
        return True

    def cmd_save(self, _: str) -> bool:
        self.save_session()
        return True

    def cmd_load(self, _: str) -> bool:
        self.load_chat()
        return True

    def cmd_sessions(self, _: str) -> bool:
        self.list_sessions()
        return True

    def cmd_session(self, cmd: str) -> bool:
        parts = cmd.split()
        if len(parts) == 1 or parts[1] == "list":
            self.list_sessions()
            return True

        action = parts[1]
        sessions = self.session_store.list_sessions()
        if not sessions:
            self.console.print(f"[{self.theme['muted']}]Kayitli sohbet yok.[/]\n")
            return True

        if action in ["open", "load"]:
            if len(parts) < 3:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /session open <no>[/]\n"
                )
                return True
            try:
                idx = int(parts[2])
            except ValueError:
                self.console.print(f"[{self.theme['error']}]Gecersiz numara[/]\n")
                return True
            if idx < 1 or idx > len(sessions):
                self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
                return True
            self._load_session(sessions[idx - 1])
            return True

        if action == "tag":
            if len(parts) < 4:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /session tag <no> <etiket>[/]\n"
                )
                return True
            try:
                idx = int(parts[2])
            except ValueError:
                self.console.print(f"[{self.theme['error']}]Gecersiz numara[/]\n")
                return True
            if idx < 1 or idx > len(sessions):
                self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
                return True
            tag = parts[3]
            meta = sessions[idx - 1]
            tags = set(meta.tags)
            tags.add(tag)
            self.session_store.update_tags(meta.id, list(tags))
            if self.session_id == meta.id:
                self.session_tags = list(tags)
            self.console.print(f"[{self.theme['success']}]✓ Etiket eklendi[/]\n")
            return True

        if action == "untag":
            if len(parts) < 4:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /session untag <no> <etiket>[/]\n"
                )
                return True
            try:
                idx = int(parts[2])
            except ValueError:
                self.console.print(f"[{self.theme['error']}]Gecersiz numara[/]\n")
                return True
            if idx < 1 or idx > len(sessions):
                self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
                return True
            tag = parts[3]
            meta = sessions[idx - 1]
            tags = [t for t in meta.tags if t != tag]
            self.session_store.update_tags(meta.id, tags)
            if self.session_id == meta.id:
                self.session_tags = tags
            self.console.print(f"[{self.theme['success']}]✓ Etiket kaldirildi[/]\n")
            return True

        if action == "delete":
            if len(parts) < 3:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /session delete <no>[/]\n"
                )
                return True
            try:
                idx = int(parts[2])
            except ValueError:
                self.console.print(f"[{self.theme['error']}]Gecersiz numara[/]\n")
                return True
            if idx < 1 or idx > len(sessions):
                self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
                return True
            meta = sessions[idx - 1]
            if not Confirm.ask(f"[{self.theme['error']}]{meta.title} silinsin mi?[/]"):
                return True
            self.session_store.delete_session(meta.id)
            if self.session_id == meta.id:
                self.session_id = None
                self.session_tags = []
            self.console.print(f"[{self.theme['success']}]✓ Silindi[/]\n")
            return True

        if action == "rename":
            if len(parts) < 4:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /session rename <no> <baslik>[/]\n"
                )
                return True
            try:
                idx = int(parts[2])
            except ValueError:
                self.console.print(f"[{self.theme['error']}]Gecersiz numara[/]\n")
                return True
            if idx < 1 or idx > len(sessions):
                self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
                return True
            new_title = " ".join(parts[3:]).strip()
            meta = sessions[idx - 1]
            self.session_store.update_title(meta.id, new_title)
            if self.session_id == meta.id:
                self.chat_title = new_title
            self.console.print(f"[{self.theme['success']}]✓ Baslik guncellendi[/]\n")
            return True

        self.console.print(
            f"[{self.theme['muted']}]Kullanim: /session list|open|tag|untag|rename|delete[/]\n"
        )
        return True

    def cmd_theme(self, _: str) -> bool:
        themes = list(self.config.themes.keys())
        current = self.config.theme
        self.console.print(
            f"[{self.theme['muted']}]Temalar: {', '.join(themes)} (aktif: {current})[/]"
        )
        new_theme = (
            self.session.prompt(
                HTML(f'<style fg="{self.theme["primary"]}">Tema: </style>')
            )
            or current
        )
        if new_theme in themes:
            self.config.theme = new_theme
            save_config(self.config, self.paths, self.logger)
            self.console.clear()
            self.print_header()
            self.show_model_info(self.model)
            self.console.print(f"\n[{self.theme['success']}]✓ Tema: {new_theme}[/]\n")
        return True

    def cmd_default(self, _: str) -> bool:
        self.config.default_model = self.model
        save_config(self.config, self.paths, self.logger)
        self.console.print(f"[{self.theme['success']}]✓ Varsayilan: {self.model}[/]\n")
        return True

    def cmd_stats(self, _: str) -> bool:
        self.show_stats()
        return True

    def cmd_fav(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        if cmd_lower == "/fav":
            self.show_favorites()
            return True

        if cmd_lower.startswith("/fav "):
            parts = cmd[5:].strip().split(maxsplit=1)
            fav_name = parts[0] if parts else ""

            if fav_name == "add" and len(parts) > 1:
                add_parts = parts[1].split(maxsplit=1)
                if len(add_parts) == 2:
                    self.favorites.favorites[add_parts[0]] = add_parts[1]
                    save_favorites(self.favorites, self.paths, self.logger)
                    self.console.print(
                        f"[{self.theme['success']}]✓ Eklendi: {add_parts[0]}[/]\n"
                    )
                return True

            favs = self.favorites.favorites
            if fav_name in favs:
                extra = parts[1] if len(parts) > 1 else ""
                user_input = f"{favs[fav_name]} {extra}".strip()
                self.send_user_message(user_input)
                return True

            self.console.print(f"[{self.theme['error']}]Favori yok: {fav_name}[/]\n")
            return True

        return True

    def cmd_template(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        if cmd_lower in ["/template", "/tpl"]:
            self.show_templates()
            return True

        if cmd_lower.startswith("/tpl ") and cmd_lower != "/tpl":
            parts = cmd[5:].strip().split(maxsplit=1)
            tpl_name = parts[0] if parts else ""
            templates = self.favorites.templates

            if tpl_name in templates:
                tpl = templates[tpl_name]
                prompt_template = tpl.prompt

                if len(parts) > 1:
                    for match in re.finditer(
                        r'(\w+)="([^"]+)"|(\w+)=([^\s]+)', parts[1]
                    ):
                        if match.group(1):
                            prompt_template = prompt_template.replace(
                                f"{{{match.group(1)}}}", match.group(2)
                            )
                        else:
                            prompt_template = prompt_template.replace(
                                f"{{{match.group(3)}}}", match.group(4)
                            )

                for var in re.findall(r"\{(\w+)\}", prompt_template):
                    val = self.session.prompt(
                        HTML(f'<style fg="{self.theme["accent"]}">{var}: </style>')
                    )
                    prompt_template = prompt_template.replace(f"{{{var}}}", val)

                user_input = prompt_template
                self.console.print(
                    f"[{self.theme['muted']}]Sablon: {tpl.name or tpl_name}[/]"
                )
                self.send_user_message(user_input)
                return True

            self.console.print(f"[{self.theme['error']}]Sablon yok: {tpl_name}[/]\n")
            return True

        return True

    def cmd_pull(self, cmd: str) -> bool:
        if cmd.lower().startswith("/pull "):
            model_to_pull = cmd[6:].strip()
            if model_to_pull and self.pull_model(model_to_pull):
                self.models = self.get_models()
                self.session.completer = SmartCompleter(
                    self.registry, self.favorites, self.models, self.config.profiles
                )
        return True

    def cmd_delete(self, cmd: str) -> bool:
        if cmd.lower().startswith("/delete "):
            model_to_delete = cmd[8:].strip()
            if model_to_delete and self.delete_model(model_to_delete):
                self.models = self.get_models()
                self.session.completer = SmartCompleter(
                    self.registry, self.favorites, self.models, self.config.profiles
                )
        return True

    def cmd_img(self, cmd: str) -> bool:
        if not self.supports_vision(self.model):
            self.console.print(f"[{self.theme['error']}]Vision model sec (/model)[/]\n")
            return True

        img_parts = cmd[5:].strip().split(maxsplit=1)
        if not img_parts:
            self.console.print(
                f"[{self.theme['error']}]Kullanim: /img <yol> [soru][/]\n"
            )
            return True

        img_path = img_parts[0]
        img_question = img_parts[1] if len(img_parts) > 1 else "Bu resimde ne var?"

        img_data, error = encode_image(img_path)
        if error:
            self.console.print(f"[{self.theme['error']}]{error}[/]\n")
            return True

        self.console.print(f"[{self.theme['success']}]✓ Resim: {img_path}[/]")
        self.send_user_message(img_question, images=[img_data])
        return True

    def cmd_retry(self, _: str) -> bool:
        if len(self.messages) >= 2:
            if self.messages[-1]["role"] == "assistant":
                self.messages.pop()
            if self.messages and self.messages[-1]["role"] == "user":
                self.console.print(
                    f"[{self.theme['accent']}]🔄 Yeniden olusturuluyor...[/]"
                )
                self.maybe_summarize()
                response = self.chat_stream(
                    self.model, self.messages, self.current_temperature
                )
                self.handle_response(response)
        else:
            self.console.print(
                f"[{self.theme['muted']}]Yeniden olusturulacak mesaj yok[/]\n"
            )
        return True

    def cmd_edit(self, _: str) -> bool:
        last_user_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            old_msg = self.messages[last_user_idx].get("content", "")
            self.console.print(f"[{self.theme['muted']}]Mevcut: {old_msg[:100]}...[/]")
            new_msg = self.session.prompt(
                HTML(f'<style fg="{self.theme["accent"]}">Yeni mesaj: </style>'),
                default=old_msg,
            )
            if new_msg and new_msg != old_msg:
                if len(self.messages) > last_user_idx + 1:
                    self.messages = self.messages[: last_user_idx + 1]
                self.messages[last_user_idx]["content"] = new_msg
                self.console.print(
                    f"[{self.theme['accent']}]✏️ Duzenlendi, yeniden gonderiliyor...[/]"
                )
                self.maybe_summarize()
                response = self.chat_stream(
                    self.model, self.messages, self.current_temperature
                )
                self.handle_response(response)
        else:
            self.console.print(f"[{self.theme['muted']}]Duzenlenecek mesaj yok[/]\n")
        return True

    def cmd_copy(self, _: str) -> bool:
        last_response = None
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                last_response = msg.get("content", "")
                break
        if last_response:
            if copy_text(last_response, self.logger):
                self.console.print(
                    f"[{self.theme['success']}]✓ Panoya kopyalandi ({len(last_response)} karakter)[/]\n"
                )
            else:
                self.console.print(f"[{self.theme['error']}]Kopyalama basarisiz[/]\n")
        else:
            self.console.print(f"[{self.theme['muted']}]Kopyalanacak yanit yok[/]\n")
        return True

    def cmd_search(self, cmd: str) -> bool:
        keyword = cmd[8:].strip()
        if keyword:
            self.search_messages(keyword)
        else:
            self.console.print(
                f"[{self.theme['error']}]Kullanim: /search <kelime>[/]\n"
            )
        return True

    def cmd_tokens(self, _: str) -> bool:
        self.show_tokens()
        return True

    def cmd_context(self, _: str) -> bool:
        total = self.estimate_context_tokens()
        summary_tokens = estimate_message_tokens({"content": self.summary})
        table = Table(box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2))
        table.add_column("Alan", style=f"bold {self.theme['accent']}")
        table.add_column("Deger", style="bold white", justify="right")

        table.add_row("Token (tahmini)", str(total))
        table.add_row("Butce", str(self.config.context_token_budget))
        table.add_row("Ozet token", str(summary_tokens))
        table.add_row(
            "Autosum", "acik" if self.config.context_autosummarize else "kapali"
        )
        table.add_row("Keep last", str(self.config.context_keep_last))
        if self.config.summary_model:
            table.add_row("Ozet model", self.config.summary_model)

        self.console.print(
            Panel(
                table,
                title="[bold]Context Durumu[/]",
                border_style=self.theme["primary"],
            )
        )
        self.console.print()
        return True

    def cmd_summarize(self, _: str) -> bool:
        if self.summarize_messages():
            self.console.print(f"[{self.theme['success']}]✓ Ozet guncellendi[/]\n")
        return True

    def cmd_quick(self, cmd: str) -> bool:
        new_model = cmd[7:].strip()
        available = [m["name"] for m in self.models]
        if new_model in available:
            self.model = new_model
            self.apply_model_profiles(self.model)
            prompt_info = get_model_prompt(self.model, self.prompts)
            self.base_system_prompt = prompt_info.get("system_prompt", "")
            self.update_system_message()
            self.console.print(
                f"[{self.theme['success']}]⚡ Model degisti: {self.model}[/]"
            )
            self.console.print(
                f"[{self.theme['muted']}]Sohbet gecmisi korundu ({len(self.messages)} mesaj)[/]\n"
            )
        else:
            self.console.print(
                f"[{self.theme['error']}]Model bulunamadi: {new_model}[/]\n"
            )
            self.console.print(
                f"[{self.theme['muted']}]Mevcut: {', '.join(available[:5])}...[/]\n"
            )
        return True

    def cmd_title(self, cmd: str) -> bool:
        title_arg = cmd[6:].strip() if len(cmd) > 6 else ""
        if title_arg:
            self.chat_title = title_arg
        else:
            self.chat_title = self.session.prompt(
                HTML(f'<style fg="{self.theme["accent"]}">Baslik: </style>')
            )
        if self.chat_title:
            if self.session_id:
                self.session_store.update_title(self.session_id, self.chat_title)
            self.console.print(
                f"[{self.theme['success']}]✓ Baslik: {self.chat_title}[/]\n"
            )
        return True

    def cmd_compare(self, _: str) -> bool:
        self.console.print(
            f"[{self.theme['muted']}]Karsilastirilacak modelleri sec (virgulle ayir):[/]"
        )
        available = [m["name"] for m in self.models]
        for i, model in enumerate(available, 1):
            self.console.print(f"  {i}. {model}")
        selection = self.session.prompt(
            HTML(f'<style fg="{self.theme["accent"]}">Modeller (orn: 1,3,5): </style>')
        )
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected_models = [available[i] for i in indices if 0 <= i < len(available)]
            if len(selected_models) >= 2:
                question = self.session.prompt(
                    HTML(f'<style fg="{self.theme["accent"]}">Soru: </style>')
                )
                if question:
                    self.compare_models(question, selected_models)
            else:
                self.console.print(f"[{self.theme['error']}]En az 2 model sec[/]\n")
        except Exception:
            self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
        return True

    def cmd_bench(self, cmd: str) -> bool:
        if not self.model:
            self.console.print(f"[{self.theme['error']}]Model secili degil[/]\n")
            return True

        arg = cmd[6:].strip() if len(cmd) > 6 else ""
        run_all = False
        prompt = self.config.benchmark_prompt

        if arg:
            parts = arg.split(maxsplit=1)
            if parts[0].lower() in ["all", "--all"]:
                run_all = True
                if len(parts) > 1:
                    prompt = parts[1]
            else:
                prompt = arg

        runs = max(1, int(self.config.benchmark_runs))
        models = [m["name"] for m in self.models] if run_all else [self.model]

        summary = []
        for name in models:
            result = self.benchmark_model(name, prompt, runs)
            if result:
                summary.append(result)

        if summary:
            table = Table(
                box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2)
            )
            table.add_column("Model", style="bold white")
            table.add_column("Run", style=self.theme["muted"], justify="right")
            table.add_column("Sure (s)", style=self.theme["accent"], justify="right")
            table.add_column("Prompt", style=self.theme["muted"], justify="right")
            table.add_column("Completion", style=self.theme["muted"], justify="right")
            table.add_column("Toplam", style=self.theme["muted"], justify="right")
            table.add_column("TPS", style=self.theme["success"], justify="right")

            for item in summary:
                table.add_row(
                    item["model"],
                    str(item["runs"]),
                    f"{item['avg_elapsed']:.2f}",
                    f"{item['avg_prompt_tokens']:.0f}",
                    f"{item['avg_completion_tokens']:.0f}",
                    f"{item['avg_total_tokens']:.0f}",
                    f"{item['avg_tps']:.2f}",
                )

            self.console.print(
                Panel(
                    table,
                    title="[bold]Benchmark Sonucu[/]",
                    border_style=self.theme["primary"],
                )
            )
            self.console.print()

        return True

    def cmd_export(self, cmd: str) -> bool:
        format_arg = cmd[7:].strip().lower() if len(cmd) > 7 else ""
        if format_arg not in ["html", "json", "txt", "md"]:
            self.console.print(
                f"[{self.theme['muted']}]Formatlar: html, json, txt, md[/]"
            )
            format_arg = (
                self.session.prompt(
                    HTML(f'<style fg="{self.theme["accent"]}">Format [html]: </style>')
                )
                or "html"
            )
        if format_arg in ["html", "json", "txt", "md"]:
            self.export_chat(format_arg)
        else:
            self.console.print(
                f"[{self.theme['error']}]Gecersiz format: {format_arg}[/]\n"
            )
        return True

    def cmd_paste(self, cmd: str) -> bool:
        if not self.supports_vision(self.model):
            self.console.print(f"[{self.theme['error']}]Vision model sec (/model)[/]\n")
            return True

        img_data, error = paste_image_from_clipboard(self.logger)
        if error:
            self.console.print(f"[{self.theme['error']}]{error}[/]\n")
            return True

        paste_question = cmd[6:].strip() if len(cmd) > 6 else ""
        if not paste_question:
            paste_question = (
                self.session.prompt(
                    HTML(
                        f'<style fg="{self.theme["accent"]}">Soru [Bu resimde ne var?]: </style>'
                    )
                )
                or "Bu resimde ne var?"
            )

        self.console.print(f"[{self.theme['success']}]✓ Panodan resim alindi[/]")
        self.send_user_message(paste_question, images=[img_data])
        return True

    def cmd_persona(self, cmd: str) -> bool:
        persona_arg = cmd[8:].strip().lower() if len(cmd) > 8 else ""

        if not persona_arg:
            self.console.print(f"\n[{self.theme['accent']}]Mevcut Personalar:[/]")
            for key, persona in PERSONAS.items():
                active = " ★" if self.current_persona == key else ""
                self.console.print(
                    f"  {persona['icon']} [bold]{key}[/] - {persona['name']}{active}"
                )
            self.console.print(
                f"\n[{self.theme['muted']}]Kullanim: /persona <isim> veya /persona off[/]\n"
            )
        elif persona_arg == "off":
            self.current_persona = None
            self.update_system_message()
            self.console.print(f"[{self.theme['success']}]✓ Persona kapatildi[/]\n")
        elif persona_arg in PERSONAS:
            self.current_persona = persona_arg
            persona = PERSONAS[persona_arg]
            self.update_system_message()
            self.console.print(
                f"[{self.theme['success']}]✓ Persona: {persona['icon']} {persona['name']}[/]\n"
            )
        else:
            self.console.print(
                f"[{self.theme['error']}]Persona bulunamadi: {persona_arg}[/]\n"
            )
        return True

    def cmd_profile(self, cmd: str) -> bool:
        parts = cmd.split()
        profiles = self.config.profiles
        if len(parts) == 1:
            if not profiles:
                self.console.print(f"[{self.theme['muted']}]Profil yok.[/]\n")
                return True
            self.console.print(f"\n[{self.theme['accent']}]Mevcut Profiller:[/]")
            for name, profile in profiles.items():
                active = " ★" if self.config.active_profile == name else ""
                model = profile.model or "-"
                temp = profile.temperature if profile.temperature is not None else "-"
                self.console.print(f"  [bold]{name}[/] ({model}, {temp}){active}")
            self.console.print(
                f"\n[{self.theme['muted']}]Kullanim: /profile <isim> veya /profile off[/]\n"
            )
            return True

        profile_name = parts[1]
        if profile_name == "off":
            self.config.active_profile = None
            self.profile_prompt = ""
            self.apply_model_profiles(self.model)
            save_config(self.config, self.paths, self.logger)
            self.update_system_message()
            self.console.print(f"[{self.theme['success']}]✓ Profil kapatildi[/]\n")
            return True

        if profile_name not in profiles:
            self.console.print(
                f"[{self.theme['error']}]Profil bulunamadi: {profile_name}[/]\n"
            )
            return True

        profile = profiles[profile_name]
        self.config.active_profile = profile_name
        save_config(self.config, self.paths, self.logger)

        if profile.model:
            available = [m["name"] for m in self.models]
            if profile.model in available:
                self.model = profile.model
                self.console.print(
                    f"[{self.theme['muted']}]Profil modeli secildi: {self.model}[/]"
                )
            else:
                self.console.print(
                    f"[{self.theme['error']}]Profil modeli bulunamadi: {profile.model}[/]"
                )

        self.apply_model_profiles(self.model)
        prompt_info = get_model_prompt(self.model, self.prompts)
        self.base_system_prompt = prompt_info.get("system_prompt", "")
        self.update_system_message()

        self.console.print(
            f"[{self.theme['success']}]✓ Profil etkin: {profile_name}[/]\n"
        )
        return True

    def cmd_security(self, cmd: str) -> bool:
        parts = cmd.split()
        if len(parts) == 1:
            key_source = "env" if os.environ.get("OLLAMA_CLI_KEY") else "config"
            key_status = (
                "var"
                if self.config.encryption_key or os.environ.get("OLLAMA_CLI_KEY")
                else "yok"
            )
            table = Table(
                box=ROUNDED, border_style=self.theme["primary"], padding=(0, 2)
            )
            table.add_column("Ayar", style=f"bold {self.theme['accent']}")
            table.add_column("Durum", style="bold white")
            table.add_row(
                "Maskeleme", "acik" if self.config.mask_sensitive else "kapali"
            )
            table.add_row(
                "Sifreleme", "acik" if self.config.encryption_enabled else "kapali"
            )
            table.add_row(
                "Export sifre", "acik" if self.config.encrypt_exports else "kapali"
            )
            table.add_row("Anahtar", f"{key_status} ({key_source})")
            self.console.print(
                Panel(
                    table,
                    title="[bold]Guvenlik Ayarlari[/]",
                    border_style=self.theme["primary"],
                )
            )
            self.console.print()
            return True

        action = parts[1]
        if action == "mask" and len(parts) > 2:
            state = parts[2].lower() in ["on", "ac", "true"]
            self.config.mask_sensitive = state
            save_config(self.config, self.paths, self.logger)
            self.session_store.update_config(self.config)
            self.console.print(
                f"[{self.theme['success']}]✓ Maskeleme {'acildi' if state else 'kapatildi'}[/]\n"
            )
            return True

        if action == "encrypt" and len(parts) > 2:
            state = parts[2].lower() in ["on", "ac", "true"]
            if state and not (
                self.config.encryption_key or os.environ.get("OLLAMA_CLI_KEY")
            ):
                self.console.print(
                    f"[{self.theme['error']}]Anahtar yok. /security keygen veya OLLAMA_CLI_KEY kullan[/]\n"
                )
                return True
            self.config.encryption_enabled = state
            save_config(self.config, self.paths, self.logger)
            self.session_store.update_config(self.config)
            self.console.print(
                f"[{self.theme['success']}]✓ Sifreleme {'acildi' if state else 'kapatildi'}[/]\n"
            )
            return True

        if action == "export" and len(parts) > 2:
            state = parts[2].lower() in ["on", "ac", "true"]
            self.config.encrypt_exports = state
            save_config(self.config, self.paths, self.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Export sifreleme {'acildi' if state else 'kapatildi'}[/]\n"
            )
            return True

        if action == "keygen":
            try:
                new_key = generate_key()
            except SecurityError as exc:
                self.console.print(
                    f"[{self.theme['error']}]Guvenlik hatasi: {exc}[/]\n"
                )
                return True
            self.config.encryption_key = new_key
            self.config.encryption_enabled = True
            save_config(self.config, self.paths, self.logger)
            self.session_store.update_config(self.config)
            self.console.print(
                f"[{self.theme['success']}]✓ Yeni anahtar uretildi ve sifreleme acildi[/]\n"
            )
            self.console.print(
                f"[{self.theme['muted']}]Anahtar (gizli tut): {new_key}[/]\n"
            )
            return True

        if action == "key" and len(parts) > 2:
            new_key = parts[2]
            self.config.encryption_key = new_key
            self.config.encryption_enabled = True
            save_config(self.config, self.paths, self.logger)
            self.session_store.update_config(self.config)
            self.console.print(f"[{self.theme['success']}]✓ Anahtar guncellendi[/]\n")
            return True

        self.console.print(
            f"[{self.theme['muted']}]Kullanim: /security mask on|off, /security encrypt on|off, /security export on|off, /security keygen, /security key <anahtar>[/]\n"
        )
        return True

    def cmd_continue(self, _: str) -> bool:
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.console.print(f"[{self.theme['accent']}]⏩ Devam ediliyor...[/]")
            self.send_user_message("Devam et, kaldigin yerden devam et.")
        else:
            self.console.print(
                f"[{self.theme['muted']}]Devam ettirilecek yanit yok[/]\n"
            )
        return True

    def cmd_temp(self, cmd: str) -> bool:
        temp_arg = cmd[5:].strip() if len(cmd) > 5 else ""

        if not temp_arg:
            current_temp = (
                self.current_temperature if self.current_temperature else "varsayilan"
            )
            self.console.print(
                f"[{self.theme['muted']}]Mevcut sicaklik: {current_temp}[/]"
            )
            self.console.print(
                f"[{self.theme['muted']}]Kullanim: /temp <0.0-2.0> veya /temp off[/]\n"
            )
        elif temp_arg == "off":
            self.current_temperature = None
            self.apply_model_profiles(self.model)
            self.console.print(
                f"[{self.theme['success']}]✓ Sicaklik varsayilana dondu[/]\n"
            )
        else:
            try:
                temp_val = float(temp_arg)
                if 0.0 <= temp_val <= 2.0:
                    self.current_temperature = temp_val
                    self.console.print(
                        f"[{self.theme['success']}]✓ Sicaklik: {temp_val}[/]\n"
                    )
                else:
                    self.console.print(
                        f"[{self.theme['error']}]0.0-2.0 arasi bir deger gir[/]\n"
                    )
            except Exception:
                self.console.print(
                    f"[{self.theme['error']}]Gecersiz deger: {temp_arg}[/]\n"
                )
        return True

    def cmd_diag(self, cmd: str) -> bool:
        arg = cmd[5:].strip().lower() if len(cmd) > 5 else ""
        if not arg:
            status = "acik" if self.config.diagnostic else "kapali"
            self.console.print(f"[{self.theme['muted']}]Diagnostik mod: {status}[/]")
            self.console.print(
                f"[{self.theme['muted']}]Log dosyasi: {self.paths.log_file}[/]\n"
            )
            return True

        if arg in ["on", "ac", "true"]:
            self.config.diagnostic = True
            set_log_level(self.logger, True)
            save_config(self.config, self.paths, self.logger)
            self.console.print(f"[{self.theme['success']}]✓ Diagnostik acildi[/]\n")
            return True
        if arg in ["off", "kapat", "false"]:
            self.config.diagnostic = False
            set_log_level(self.logger, False)
            save_config(self.config, self.paths, self.logger)
            self.console.print(f"[{self.theme['success']}]✓ Diagnostik kapatildi[/]\n")
            return True

        self.console.print(f"[{self.theme['error']}]Kullanim: /diag [on|off][/]\n")
        return True

    def cmd_markdown(self, cmd: str) -> bool:
        parts = cmd.split()
        if len(parts) == 1:
            status = "acik" if self.config.render_markdown else "kapali"
            self.console.print(f"[{self.theme['muted']}]Markdown gorunumu: {status}[/]")
            self.console.print(
                f"[{self.theme['muted']}]Kullanim: /markdown on|off[/]\n"
            )
            return True

        arg = parts[1].lower()
        if arg in ["on", "ac", "true"]:
            self.config.render_markdown = True
            save_config(self.config, self.paths, self.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Markdown gorunumu acildi[/]\n"
            )
            return True
        if arg in ["off", "kapat", "false"]:
            self.config.render_markdown = False
            save_config(self.config, self.paths, self.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Markdown gorunumu kapatildi[/]\n"
            )
            return True

        self.console.print(f"[{self.theme['error']}]Kullanim: /markdown on|off[/]\n")
        return True
