"""Command registry and handlers for Ollama CLI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from rich.box import ROUNDED
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .chat_engine import PERSONAS
from .clipboard import copy_text
from .media import encode_image, paste_image_from_clipboard
from .security import SecurityError, generate_key
from .storage import save_config, save_favorites
from .utils import estimate_message_tokens, get_model_prompt

if TYPE_CHECKING:
    from .app import ChatApp

CommandHandler = Callable[[str], bool]


@dataclass(frozen=True)
class Command:
    name: str
    aliases: Tuple[str, ...]
    description: str
    usage: Optional[str]
    handler: CommandHandler


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: Dict[str, Command] = {}

    def register(self, command: Command) -> None:
        for key in (command.name, *command.aliases):
            self._commands[key] = command

    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def list_commands(self) -> List[Command]:
        unique = {cmd.name: cmd for cmd in self._commands.values()}
        return list(unique.values())

    def command_strings(self) -> List[str]:
        return sorted(self._commands.keys())


class SmartCompleter(Completer):
    """Akilli tamamlayici - komutlar, favoriler, modeller."""

    def __init__(
        self,
        registry: CommandRegistry,
        favorites,
        models,
        profiles: Dict,
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

            if text.startswith("/prompts "):
                prefix = text[9:]
                for name in self.favorites.library_prompts.keys():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
                for option in ["add", "remove"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))

            if text.startswith("/clipboard "):
                prefix = text[11:]
                for option in ["on", "off"]:
                    if option.startswith(prefix):
                        yield Completion(option, start_position=-len(prefix))


class CommandHandlers:
    """All command handlers for the chat application."""

    def __init__(self, app: ChatApp) -> None:
        self.app = app

    @property
    def theme(self) -> Dict[str, str]:
        return self.app.theme

    @property
    def console(self):
        return self.app.console

    @property
    def config(self):
        return self.app.config

    @property
    def session(self):
        return self.app.session

    @property
    def messages(self):
        return self.app.messages

    @messages.setter
    def messages(self, value):
        self.app.messages = value

    def register_all(self, registry: CommandRegistry) -> None:
        """Register all commands to the registry."""
        registry.register(Command("/help", ("/?",), "Yardim", None, self.cmd_help))
        registry.register(
            Command("/quit", ("/q", "/exit"), "Cikis", None, self.cmd_quit)
        )
        registry.register(
            Command("/clear", ("/c",), "Sohbeti temizle", None, self.cmd_clear)
        )
        registry.register(
            Command("/model", ("/m",), "Model degistir", None, self.cmd_model)
        )
        registry.register(
            Command("/info", ("/i",), "Model bilgisi", None, self.cmd_info)
        )
        registry.register(
            Command("/prompt", ("/p",), "Sistem promptu", None, self.cmd_prompt)
        )
        registry.register(
            Command("/history", ("/h",), "Mesaj gecmisi", None, self.cmd_history)
        )
        registry.register(
            Command("/save", ("/s",), "Sohbeti kaydet", None, self.cmd_save)
        )
        registry.register(
            Command("/load", ("/l",), "Sohbet yukle", None, self.cmd_load)
        )
        registry.register(
            Command("/sessions", tuple(), "Kayitli sohbetler", None, self.cmd_sessions)
        )
        registry.register(
            Command("/session", tuple(), "Oturum yonetimi", None, self.cmd_session)
        )
        registry.register(
            Command("/theme", ("/t",), "Tema degistir", None, self.cmd_theme)
        )
        registry.register(
            Command("/default", ("/d",), "Varsayilan model", None, self.cmd_default)
        )
        registry.register(
            Command("/stats", tuple(), "Model durumlari", None, self.cmd_stats)
        )
        registry.register(Command("/fav", tuple(), "Favoriler", None, self.cmd_fav))
        registry.register(
            Command("/template", ("/tpl",), "Sablonlar", None, self.cmd_template)
        )
        registry.register(Command("/pull", tuple(), "Model indir", None, self.cmd_pull))
        registry.register(
            Command("/delete", tuple(), "Model sil", None, self.cmd_delete)
        )
        registry.register(Command("/img", tuple(), "Resim gonder", None, self.cmd_img))
        registry.register(
            Command("/retry", tuple(), "Son yaniti yenile", None, self.cmd_retry)
        )
        registry.register(
            Command("/edit", tuple(), "Son mesaji duzenle", None, self.cmd_edit)
        )
        registry.register(
            Command("/copy", tuple(), "Son yaniti kopyala", None, self.cmd_copy)
        )
        registry.register(
            Command("/search", tuple(), "Mesajlarda ara", None, self.cmd_search)
        )
        registry.register(
            Command("/tokens", tuple(), "Token kullanimi", None, self.cmd_tokens)
        )
        registry.register(
            Command("/context", tuple(), "Context durumu", None, self.cmd_context)
        )
        registry.register(
            Command("/summarize", tuple(), "Konusma ozetle", None, self.cmd_summarize)
        )
        registry.register(
            Command("/quick", tuple(), "Hizli model degistir", None, self.cmd_quick)
        )
        registry.register(
            Command("/title", tuple(), "Baslik ata", None, self.cmd_title)
        )
        registry.register(
            Command(
                "/compare", tuple(), "Modelleri karsilastir", None, self.cmd_compare
            )
        )
        registry.register(
            Command("/bench", tuple(), "Benchmark calistir", None, self.cmd_bench)
        )
        registry.register(
            Command("/export", tuple(), "Disari aktar", None, self.cmd_export)
        )
        registry.register(
            Command("/paste", tuple(), "Panodan resim", None, self.cmd_paste)
        )
        registry.register(
            Command("/persona", tuple(), "Persona degistir", None, self.cmd_persona)
        )
        registry.register(
            Command("/profile", tuple(), "Profil sec", None, self.cmd_profile)
        )
        registry.register(
            Command("/security", tuple(), "Guvenlik ayarlari", None, self.cmd_security)
        )
        registry.register(
            Command("/continue", tuple(), "Yaniti devam ettir", None, self.cmd_continue)
        )
        registry.register(
            Command("/temp", tuple(), "Sicaklik ayari", None, self.cmd_temp)
        )
        registry.register(
            Command("/diag", tuple(), "Diagnostik mod", None, self.cmd_diag)
        )
        registry.register(
            Command("/markdown", ("/md",), "Markdown gorunumu", None, self.cmd_markdown)
        )
        # Yeni özellikler
        registry.register(
            Command("/prompts", tuple(), "Prompt kutuphanesi", None, self.cmd_prompts)
        )
        registry.register(
            Command(
                "/yapistir", tuple(), "Panodan metin kullan", None, self.cmd_yapistir
            )
        )
        registry.register(
            Command("/clipboard", tuple(), "Clipboard izleme", None, self.cmd_clipboard)
        )

    # ─────────────────────────────────────────────────────────────
    # Basic Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_help(self, _: str) -> bool:
        self.app.ui_display.show_help()
        return True

    def cmd_quit(self, _: str) -> bool:
        if self.config.auto_save and len(self.messages) > 1:
            self.app.save_session(show_message=False)
        self.console.print(f"\n[{self.theme['accent']}]Gorusuruz! 👋[/]\n")
        return False

    def cmd_clear(self, _: str) -> bool:
        self.app.messages = self.app.chat_engine.init_conversation(self.app.model)
        self.console.clear()
        self.app.ui_display.print_header()
        self.app.model_manager.show_model_info(self.app.model)
        self.console.print(f"\n[{self.theme['success']}]✓ Temizlendi[/]\n")
        return True

    def cmd_model(self, _: str) -> bool:
        self.app.models = self.app.model_manager.get_models()
        self.app.model_manager.models = self.app.models
        self.session.completer = SmartCompleter(
            self.app.registry, self.app.favorites, self.app.models, self.config.profiles
        )
        self.app.model_manager.set_session(self.session)
        self.app.model = self.app.model_manager.select_model(self.app.models)
        self.app.messages = self.app.chat_engine.init_conversation(self.app.model)
        self.console.print()
        self.app.model_manager.show_model_info(self.app.model)
        if self.app.model_manager.supports_vision(self.app.model):
            self.console.print(f"[{self.theme['accent']}]👁️ Vision model[/]")
        self.console.print()
        return True

    def cmd_info(self, _: str) -> bool:
        self.app.model_manager.models = self.app.models
        self.app.model_manager.show_model_info(self.app.model)
        self.console.print()
        return True

    def cmd_prompt(self, _: str) -> bool:
        combined = self.app.chat_engine.build_system_prompt()
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
        self.app.save_session()
        return True

    def cmd_load(self, _: str) -> bool:
        self.app.load_chat()
        return True

    def cmd_sessions(self, _: str) -> bool:
        self.app.list_sessions()
        return True

    def cmd_session(self, cmd: str) -> bool:
        parts = cmd.split()
        if len(parts) == 1 or parts[1] == "list":
            self.app.list_sessions()
            return True

        action = parts[1]
        sessions = self.app.session_store.list_sessions()
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
            self.app._load_session(sessions[idx - 1])
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
            self.app.session_store.update_tags(meta.id, list(tags))
            if self.app.session_id == meta.id:
                self.app.session_tags = list(tags)
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
            self.app.session_store.update_tags(meta.id, tags)
            if self.app.session_id == meta.id:
                self.app.session_tags = tags
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
            self.app.session_store.delete_session(meta.id)
            if self.app.session_id == meta.id:
                self.app.session_id = None
                self.app.session_tags = []
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
            self.app.session_store.update_title(meta.id, new_title)
            if self.app.session_id == meta.id:
                self.app.chat_title = new_title
            self.console.print(f"[{self.theme['success']}]✓ Baslik guncellendi[/]\n")
            return True

        self.console.print(
            f"[{self.theme['muted']}]Kullanim: /session list|open|tag|untag|rename|delete[/]\n"
        )
        return True

    # ─────────────────────────────────────────────────────────────
    # Theme and Configuration Commands
    # ─────────────────────────────────────────────────────────────

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
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.clear()
            self.app.ui_display.print_header()
            self.app.model_manager.show_model_info(self.app.model)
            self.console.print(f"\n[{self.theme['success']}]✓ Tema: {new_theme}[/]\n")
        return True

    def cmd_default(self, _: str) -> bool:
        self.config.default_model = self.app.model
        save_config(self.config, self.app.paths, self.app.logger)
        self.console.print(
            f"[{self.theme['success']}]✓ Varsayilan: {self.app.model}[/]\n"
        )
        return True

    def cmd_stats(self, _: str) -> bool:
        self.app.ui_display.show_stats(self.app.models)
        return True

    # ─────────────────────────────────────────────────────────────
    # Favorites and Templates
    # ─────────────────────────────────────────────────────────────

    def cmd_fav(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        if cmd_lower == "/fav":
            self.app.ui_display.show_favorites()
            return True

        if cmd_lower.startswith("/fav "):
            parts = cmd[5:].strip().split(maxsplit=1)
            fav_name = parts[0] if parts else ""

            if fav_name == "add" and len(parts) > 1:
                add_parts = parts[1].split(maxsplit=1)
                if len(add_parts) == 2:
                    self.app.favorites.favorites[add_parts[0]] = add_parts[1]
                    save_favorites(self.app.favorites, self.app.paths, self.app.logger)
                    self.console.print(
                        f"[{self.theme['success']}]✓ Eklendi: {add_parts[0]}[/]\n"
                    )
                return True

            favs = self.app.favorites.favorites
            if fav_name in favs:
                extra = parts[1] if len(parts) > 1 else ""
                user_input = f"{favs[fav_name]} {extra}".strip()
                self.app.chat_engine.send_user_message(user_input)
                return True

            self.console.print(f"[{self.theme['error']}]Favori yok: {fav_name}[/]\n")
            return True

        return True

    def cmd_template(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        if cmd_lower in ["/template", "/tpl"]:
            self.app.ui_display.show_templates()
            return True

        if cmd_lower.startswith("/tpl ") and cmd_lower != "/tpl":
            parts = cmd[5:].strip().split(maxsplit=1)
            tpl_name = parts[0] if parts else ""
            templates = self.app.favorites.templates

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
                self.app.chat_engine.send_user_message(user_input)
                return True

            self.console.print(f"[{self.theme['error']}]Sablon yok: {tpl_name}[/]\n")
            return True

        return True

    # ─────────────────────────────────────────────────────────────
    # Model Management Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_pull(self, cmd: str) -> bool:
        if cmd.lower().startswith("/pull "):
            model_to_pull = cmd[6:].strip()
            if model_to_pull and self.app.model_manager.pull_model(model_to_pull):
                self.app.models = self.app.model_manager.get_models()
                self.session.completer = SmartCompleter(
                    self.app.registry,
                    self.app.favorites,
                    self.app.models,
                    self.config.profiles,
                )
        return True

    def cmd_delete(self, cmd: str) -> bool:
        if cmd.lower().startswith("/delete "):
            model_to_delete = cmd[8:].strip()
            if model_to_delete and self.app.model_manager.delete_model(model_to_delete):
                self.app.models = self.app.model_manager.get_models()
                self.session.completer = SmartCompleter(
                    self.app.registry,
                    self.app.favorites,
                    self.app.models,
                    self.config.profiles,
                )
        return True

    # ─────────────────────────────────────────────────────────────
    # Image Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_img(self, cmd: str) -> bool:
        if not self.app.model_manager.supports_vision(self.app.model):
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
        self.app.chat_engine.send_user_message(img_question, images=[img_data])
        return True

    def cmd_paste(self, cmd: str) -> bool:
        if not self.app.model_manager.supports_vision(self.app.model):
            self.console.print(f"[{self.theme['error']}]Vision model sec (/model)[/]\n")
            return True

        img_data, error = paste_image_from_clipboard(self.app.logger)
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
        self.app.chat_engine.send_user_message(paste_question, images=[img_data])
        return True

    # ─────────────────────────────────────────────────────────────
    # Message Editing Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_retry(self, _: str) -> bool:
        if len(self.messages) >= 2:
            if self.messages[-1]["role"] == "assistant":
                self.messages.pop()
            if self.messages and self.messages[-1]["role"] == "user":
                self.console.print(
                    f"[{self.theme['accent']}]🔄 Yeniden olusturuluyor...[/]"
                )
                self.app.chat_engine.maybe_summarize()
                response = self.app.chat_engine.chat_stream(
                    self.app.model, self.messages, self.app.current_temperature
                )
                self.app.chat_engine.handle_response(response)
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
                    self.app.messages = self.messages[: last_user_idx + 1]
                self.messages[last_user_idx]["content"] = new_msg
                self.console.print(
                    f"[{self.theme['accent']}]✏️ Duzenlendi, yeniden gonderiliyor...[/]"
                )
                self.app.chat_engine.maybe_summarize()
                response = self.app.chat_engine.chat_stream(
                    self.app.model, self.messages, self.app.current_temperature
                )
                self.app.chat_engine.handle_response(response)
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
            if copy_text(last_response, self.app.logger):
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
            self.app.ui_display.search_messages(keyword, self.messages)
        else:
            self.console.print(
                f"[{self.theme['error']}]Kullanim: /search <kelime>[/]\n"
            )
        return True

    # ─────────────────────────────────────────────────────────────
    # Token and Context Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_tokens(self, _: str) -> bool:
        self.app.ui_display.show_tokens()
        return True

    def cmd_context(self, _: str) -> bool:
        total = self.app.chat_engine.estimate_context_tokens()
        summary_tokens = estimate_message_tokens(
            {"content": self.app.chat_engine.summary}
        )
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
        if self.app.chat_engine.summarize_messages():
            self.console.print(f"[{self.theme['success']}]✓ Ozet guncellendi[/]\n")
        return True

    # ─────────────────────────────────────────────────────────────
    # Quick Model Switch and Title
    # ─────────────────────────────────────────────────────────────

    def cmd_quick(self, cmd: str) -> bool:
        new_model = cmd[7:].strip()
        available = [m["name"] for m in self.app.models]
        if new_model in available:
            self.app.model = new_model
            self.app.model_manager.apply_model_profiles(self.app.model)
            prompt_info = get_model_prompt(self.app.model, self.app.prompts)
            self.app.chat_engine.base_system_prompt = prompt_info.get(
                "system_prompt", ""
            )
            self.app.chat_engine.update_system_message()
            self.console.print(
                f"[{self.theme['success']}]⚡ Model degisti: {self.app.model}[/]"
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
            self.app.chat_title = title_arg
        else:
            self.app.chat_title = self.session.prompt(
                HTML(f'<style fg="{self.theme["accent"]}">Baslik: </style>')
            )
        if self.app.chat_title:
            if self.app.session_id:
                self.app.session_store.update_title(
                    self.app.session_id, self.app.chat_title
                )
            self.console.print(
                f"[{self.theme['success']}]✓ Baslik: {self.app.chat_title}[/]\n"
            )
        return True

    # ─────────────────────────────────────────────────────────────
    # Comparison and Benchmark Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_compare(self, _: str) -> bool:
        self.console.print(
            f"[{self.theme['muted']}]Karsilastirilacak modelleri sec (virgulle ayir):[/]"
        )
        available = [m["name"] for m in self.app.models]
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
                    self.app.ui_display.compare_models(question, selected_models)
            else:
                self.console.print(f"[{self.theme['error']}]En az 2 model sec[/]\n")
        except Exception:
            self.console.print(f"[{self.theme['error']}]Gecersiz secim[/]\n")
        return True

    def cmd_bench(self, cmd: str) -> bool:
        if not self.app.model:
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
        models = [m["name"] for m in self.app.models] if run_all else [self.app.model]

        summary = []
        for name in models:
            result = self.app.ui_display.benchmark_model(name, prompt, runs)
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
            self.app.ui_display.export_chat(
                format_arg, self.messages, self.app.model, self.app.chat_title
            )
        else:
            self.console.print(
                f"[{self.theme['error']}]Gecersiz format: {format_arg}[/]\n"
            )
        return True

    # ─────────────────────────────────────────────────────────────
    # Persona and Profile Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_persona(self, cmd: str) -> bool:
        persona_arg = cmd[8:].strip().lower() if len(cmd) > 8 else ""

        if not persona_arg:
            self.console.print(f"\n[{self.theme['accent']}]Mevcut Personalar:[/]")
            for key, persona in PERSONAS.items():
                active = " ★" if self.app.chat_engine.current_persona == key else ""
                self.console.print(
                    f"  {persona['icon']} [bold]{key}[/] - {persona['name']}{active}"
                )
            self.console.print(
                f"\n[{self.theme['muted']}]Kullanim: /persona <isim> veya /persona off[/]\n"
            )
        elif persona_arg == "off":
            self.app.chat_engine.set_persona(None)
            self.console.print(f"[{self.theme['success']}]✓ Persona kapatildi[/]\n")
        elif persona_arg in PERSONAS:
            self.app.chat_engine.set_persona(persona_arg)
            persona = PERSONAS[persona_arg]
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
            self.app.model_manager.profile_prompt = ""
            self.app.model_manager.apply_model_profiles(self.app.model)
            save_config(self.config, self.app.paths, self.app.logger)
            self.app.chat_engine.update_system_message()
            self.console.print(f"[{self.theme['success']}]✓ Profil kapatildi[/]\n")
            return True

        if profile_name not in profiles:
            self.console.print(
                f"[{self.theme['error']}]Profil bulunamadi: {profile_name}[/]\n"
            )
            return True

        profile = profiles[profile_name]
        self.config.active_profile = profile_name
        save_config(self.config, self.app.paths, self.app.logger)

        if profile.model:
            available = [m["name"] for m in self.app.models]
            if profile.model in available:
                self.app.model = profile.model
                self.console.print(
                    f"[{self.theme['muted']}]Profil modeli secildi: {self.app.model}[/]"
                )
            else:
                self.console.print(
                    f"[{self.theme['error']}]Profil modeli bulunamadi: {profile.model}[/]"
                )

        self.app.model_manager.apply_model_profiles(self.app.model)
        prompt_info = get_model_prompt(self.app.model, self.app.prompts)
        self.app.chat_engine.base_system_prompt = prompt_info.get("system_prompt", "")
        self.app.chat_engine.update_system_message()

        self.console.print(
            f"[{self.theme['success']}]✓ Profil etkin: {profile_name}[/]\n"
        )
        return True

    # ─────────────────────────────────────────────────────────────
    # Security Commands
    # ─────────────────────────────────────────────────────────────

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
            save_config(self.config, self.app.paths, self.app.logger)
            self.app.session_store.update_config(self.config)
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
            save_config(self.config, self.app.paths, self.app.logger)
            self.app.session_store.update_config(self.config)
            self.console.print(
                f"[{self.theme['success']}]✓ Sifreleme {'acildi' if state else 'kapatildi'}[/]\n"
            )
            return True

        if action == "export" and len(parts) > 2:
            state = parts[2].lower() in ["on", "ac", "true"]
            self.config.encrypt_exports = state
            save_config(self.config, self.app.paths, self.app.logger)
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
            save_config(self.config, self.app.paths, self.app.logger)
            self.app.session_store.update_config(self.config)
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
            save_config(self.config, self.app.paths, self.app.logger)
            self.app.session_store.update_config(self.config)
            self.console.print(f"[{self.theme['success']}]✓ Anahtar guncellendi[/]\n")
            return True

        self.console.print(
            f"[{self.theme['muted']}]Kullanim: /security mask on|off, /security encrypt on|off, "
            "/security export on|off, /security keygen, /security key <anahtar>[/]\n"
        )
        return True

    # ─────────────────────────────────────────────────────────────
    # Misc Commands
    # ─────────────────────────────────────────────────────────────

    def cmd_continue(self, _: str) -> bool:
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.console.print(f"[{self.theme['accent']}]⏩ Devam ediliyor...[/]")
            self.app.chat_engine.send_user_message(
                "Devam et, kaldigin yerden devam et."
            )
        else:
            self.console.print(
                f"[{self.theme['muted']}]Devam ettirilecek yanit yok[/]\n"
            )
        return True

    def cmd_temp(self, cmd: str) -> bool:
        temp_arg = cmd[5:].strip() if len(cmd) > 5 else ""

        if not temp_arg:
            current_temp = (
                self.app.current_temperature
                if self.app.current_temperature
                else "varsayilan"
            )
            self.console.print(
                f"[{self.theme['muted']}]Mevcut sicaklik: {current_temp}[/]"
            )
            self.console.print(
                f"[{self.theme['muted']}]Kullanim: /temp <0.0-2.0> veya /temp off[/]\n"
            )
        elif temp_arg == "off":
            self.app.current_temperature = None
            self.app.model_manager.apply_model_profiles(self.app.model)
            self.console.print(
                f"[{self.theme['success']}]✓ Sicaklik varsayilana dondu[/]\n"
            )
        else:
            try:
                temp_val = float(temp_arg)
                if 0.0 <= temp_val <= 2.0:
                    self.app.current_temperature = temp_val
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
        from .logging_utils import set_log_level

        arg = cmd[5:].strip().lower() if len(cmd) > 5 else ""
        if not arg:
            status = "acik" if self.config.diagnostic else "kapali"
            self.console.print(f"[{self.theme['muted']}]Diagnostik mod: {status}[/]")
            self.console.print(
                f"[{self.theme['muted']}]Log dosyasi: {self.app.paths.log_file}[/]\n"
            )
            return True

        if arg in ["on", "ac", "true"]:
            self.config.diagnostic = True
            set_log_level(self.app.logger, True)
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.print(f"[{self.theme['success']}]✓ Diagnostik acildi[/]\n")
            return True
        if arg in ["off", "kapat", "false"]:
            self.config.diagnostic = False
            set_log_level(self.app.logger, False)
            save_config(self.config, self.app.paths, self.app.logger)
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
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Markdown gorunumu acildi[/]\n"
            )
            return True
        if arg in ["off", "kapat", "false"]:
            self.config.render_markdown = False
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Markdown gorunumu kapatildi[/]\n"
            )
            return True

        self.console.print(f"[{self.theme['error']}]Kullanim: /markdown on|off[/]\n")
        return True

    # ─────────────────────────────────────────────────────────────
    # Prompt Kütüphanesi
    # ─────────────────────────────────────────────────────────────

    def cmd_prompts(self, cmd: str) -> bool:
        """Prompt kütüphanesini yönet."""
        parts = cmd.split(maxsplit=2)
        if len(parts) == 1:
            # /prompts - Listeyi göster
            self.app.ui_display.show_prompts(self.app.favorites.library_prompts)
            return True

        arg = parts[1].lower()

        if arg == "add":
            # /prompts add <isim> - Yeni prompt ekle (interaktif)
            if len(parts) < 3:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /prompts add <isim>[/]\n"
                )
                return True
            name = parts[2].lower().replace(" ", "-")
            if name in self.app.favorites.library_prompts:
                self.console.print(f"[{self.theme['error']}]'{name}' zaten mevcut[/]\n")
                return True

            # Interaktif bilgi al
            from rich.prompt import Prompt
            from .models import LibraryPrompt

            display_name = Prompt.ask(
                f"[{self.theme['muted']}]Görünen isim[/]", default=name.title()
            )
            description = Prompt.ask(f"[{self.theme['muted']}]Açıklama[/]")
            prompt_text = Prompt.ask(f"[{self.theme['muted']}]Prompt metni[/]")
            category = Prompt.ask(
                f"[{self.theme['muted']}]Kategori[/]", default="genel"
            )
            icon = Prompt.ask(f"[{self.theme['muted']}]Emoji[/]", default="📝")

            self.app.favorites.library_prompts[name] = LibraryPrompt(
                name=display_name,
                description=description,
                prompt=prompt_text,
                category=category,
                icon=icon,
            )
            save_favorites(self.app.favorites, self.app.paths, self.app.logger)
            self.console.print(f"[{self.theme['success']}]✓ '{name}' eklendi[/]\n")
            return True

        if arg == "remove":
            # /prompts remove <isim>
            if len(parts) < 3:
                self.console.print(
                    f"[{self.theme['error']}]Kullanim: /prompts remove <isim>[/]\n"
                )
                return True
            name = parts[2].lower()
            if name not in self.app.favorites.library_prompts:
                self.console.print(f"[{self.theme['error']}]'{name}' bulunamadi[/]\n")
                return True

            del self.app.favorites.library_prompts[name]
            save_favorites(self.app.favorites, self.app.paths, self.app.logger)
            self.console.print(f"[{self.theme['success']}]✓ '{name}' silindi[/]\n")
            return True

        # /prompts <isim> - Promptu kullan
        prompt_name = arg
        if prompt_name not in self.app.favorites.library_prompts:
            self.console.print(
                f"[{self.theme['error']}]Prompt bulunamadi: {prompt_name}[/]\n"
            )
            self.console.print(
                f"[{self.theme['muted']}]Mevcut promptlar: {', '.join(self.app.favorites.library_prompts.keys())}[/]\n"
            )
            return True

        prompt_entry = self.app.favorites.library_prompts[prompt_name]
        # Sonraki girişi prompt ile başlat
        self.console.print(
            f"[{self.theme['accent']}]{prompt_entry.icon} {prompt_entry.name}[/]"
        )
        self.console.print(f"[{self.theme['muted']}]{prompt_entry.description}[/]")
        self.console.print(
            f"[{self.theme['primary']}]Prompt: {prompt_entry.prompt}[/]\n"
        )
        self.console.print(
            f"[{self.theme['muted']}]Metninizi girin (ya da /iptal ile vazgecin):[/]\n"
        )

        # Kullanıcıdan metin al
        try:
            user_text = self.session.prompt(
                HTML(f"<ansigreen>{prompt_entry.icon}</ansigreen> "),
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print(f"\n[{self.theme['muted']}]İptal edildi[/]\n")
            return True

        if user_text.strip().lower() == "/iptal":
            self.console.print(f"[{self.theme['muted']}]İptal edildi[/]\n")
            return True

        # Prompt ile mesajı birleştir ve gönder
        full_message = f"{prompt_entry.prompt}\n\n{user_text}"
        self.app.chat_engine.send_user_message(full_message)
        return True

    def cmd_yapistir(self, cmd: str) -> bool:
        """Panodaki metni prompt olarak kullan."""
        try:
            import pyperclip

            text = pyperclip.paste()
        except ImportError:
            self.console.print(
                f"[{self.theme['error']}]pyperclip yuklu degil. pip install pyperclip[/]\n"
            )
            return True
        except Exception as e:
            self.console.print(f"[{self.theme['error']}]Pano okunamadi: {e}[/]\n")
            return True

        if not text or not text.strip():
            self.console.print(f"[{self.theme['error']}]Panoda metin yok[/]\n")
            return True

        # Opsiyonel prefix
        prefix = cmd[10:].strip() if len(cmd) > 10 else ""
        user_input = f"{prefix} {text}".strip() if prefix else text

        preview_len = 100
        preview = text[:preview_len] + "..." if len(text) > preview_len else text
        self.console.print(
            f"[{self.theme['muted']}]📋 Panodan ({len(text)} karakter):[/]"
        )
        self.console.print(f"[{self.theme['muted']}]{preview}[/]\n")

        self.app.chat_engine.send_user_message(user_input)
        return True

    def cmd_clipboard(self, cmd: str) -> bool:
        """Clipboard izlemeyi aç/kapat."""
        parts = cmd.split()
        if len(parts) == 1:
            status = "açık" if self.config.clipboard_monitor else "kapalı"
            self.console.print(f"[{self.theme['muted']}]Clipboard izleme: {status}[/]")
            self.console.print(
                f"[{self.theme['muted']}]Kullanım: /clipboard on|off[/]\n"
            )
            return True

        arg = parts[1].lower()
        if arg in ["on", "ac", "aç", "true"]:
            self.config.clipboard_monitor = True
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Clipboard izleme açıldı[/]\n"
            )
            return True
        if arg in ["off", "kapat", "false"]:
            self.config.clipboard_monitor = False
            save_config(self.config, self.app.paths, self.app.logger)
            self.console.print(
                f"[{self.theme['success']}]✓ Clipboard izleme kapatıldı[/]\n"
            )
            return True

        self.console.print(f"[{self.theme['error']}]Kullanım: /clipboard on|off[/]\n")
        return True
