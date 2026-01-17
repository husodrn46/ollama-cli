"""Model manager module - Model discovery, selection, and capabilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from .models import ConfigModel, ProfileModel
from .storage import read_json, write_json
from .utils import format_size, get_model_prompt, is_vision_model

if TYPE_CHECKING:
    from logging import Logger

MODEL_CACHE_TTL_SECONDS = 24 * 60 * 60


class ModelManager:
    """Handles model discovery, selection, caching and capabilities."""

    def __init__(
        self,
        config: ConfigModel,
        console: Console,
        logger: Logger,
        prompts: Dict,
        model_cache_file: Path,
        benchmarks_file: Path,
        get_theme: Callable[[], Dict[str, str]],
        session: Optional[PromptSession] = None,
    ) -> None:
        self.config = config
        self.console = console
        self.logger = logger
        self.prompts = prompts
        self.model_cache_file = model_cache_file
        self.benchmarks_file = benchmarks_file
        self._get_theme = get_theme
        self.session = session

        # State
        self.model_cache: Dict[str, Dict[str, object]] = self._load_model_cache()
        self.models: List[Dict[str, object]] = []
        self.current_model: Optional[str] = None

        # Profile state (managed externally but exposed here)
        self.profile_prompt: str = ""
        self.active_profile_name: Optional[str] = None
        self.current_temperature: Optional[float] = None

    @property
    def theme(self) -> Dict[str, str]:
        """Get current theme colors."""
        return self._get_theme()

    def set_session(self, session: PromptSession) -> None:
        """Set the prompt session for interactive selection."""
        self.session = session

    # ─────────────────────────────────────────────────────────────
    # Cache Management
    # ─────────────────────────────────────────────────────────────

    def _load_model_cache(self) -> Dict[str, Dict[str, object]]:
        """Load cached model info from disk."""
        data = read_json(self.model_cache_file, self.logger)
        if isinstance(data, dict):
            return data
        return {}

    def _save_model_cache(self) -> None:
        """Save model cache to disk."""
        write_json(self.model_cache_file, self.model_cache, self.logger)

    def _cache_is_fresh(self, entry: Dict[str, object]) -> bool:
        """Check if cache entry is still valid."""
        fetched_at = entry.get("fetched_at")
        if not isinstance(fetched_at, str):
            return False
        try:
            ts = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        return (datetime.utcnow() - ts).total_seconds() < MODEL_CACHE_TTL_SECONDS

    def _extract_context_length(self, model_info: object) -> Optional[int]:
        """Extract context window from model capabilities."""
        if not isinstance(model_info, dict):
            return None

        candidates: List[int] = []
        for key, value in model_info.items():
            if not isinstance(key, str):
                continue
            key_lower = key.lower()
            if "context_length" not in key_lower:
                continue
            if isinstance(value, (int, float)):
                candidates.append(int(value))
                continue
            if isinstance(value, str) and value.isdigit():
                candidates.append(int(value))

        if candidates:
            return max(candidates)
        return None

    # ─────────────────────────────────────────────────────────────
    # Model Discovery & Capabilities
    # ─────────────────────────────────────────────────────────────

    def get_models(self) -> List[Dict[str, object]]:
        """Fetch available models from Ollama API."""
        host = self.config.ollama_host
        with self.console.status("[bold cyan]Modeller yukleniyor...", spinner="dots"):
            try:
                response = requests.get(f"{host}/api/tags", timeout=10)
                response.raise_for_status()
                self.models = response.json().get("models", [])
                return self.models
            except requests.RequestException as exc:
                self.logger.exception("Model listesi alinmadi")
                self.console.print(f"[red]Baglanti hatasi: {exc}[/]")
                return []

    def get_model_capabilities(
        self, model_name: str, refresh: bool = False
    ) -> Optional[Dict[str, object]]:
        """Fetch model capabilities with caching."""
        cached = self.model_cache.get(model_name)
        if cached and not refresh and self._cache_is_fresh(cached):
            return cached

        host = self.config.ollama_host
        try:
            response = requests.post(
                f"{host}/api/show",
                json={"name": model_name},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            self.logger.exception("Model detaylari alinmadi: %s", model_name)
            return cached

        raw_caps = data.get("capabilities") or []
        if isinstance(raw_caps, str):
            raw_caps = [raw_caps]
        capabilities = [c for c in raw_caps if isinstance(c, str)]

        context_length = self._extract_context_length(data.get("model_info", {}))
        supports_vision = "vision" in capabilities or is_vision_model(model_name)
        supports_tools = "tools" in capabilities
        supports_embedding = "embedding" in capabilities
        supports_completion = "completion" in capabilities or not capabilities
        if supports_embedding and "completion" not in capabilities:
            supports_completion = False

        record = {
            "fetched_at": datetime.utcnow().isoformat(),
            "capabilities": capabilities,
            "context_length": context_length,
            "supports_vision": supports_vision,
            "supports_tools": supports_tools,
            "supports_embedding": supports_embedding,
            "supports_completion": supports_completion,
        }

        self.model_cache[model_name] = record
        self._save_model_cache()
        return record

    def supports_vision(self, model_name: str) -> bool:
        """Check if model supports vision."""
        caps = self.get_model_capabilities(model_name)
        if caps and isinstance(caps.get("supports_vision"), bool):
            return bool(caps["supports_vision"])
        return is_vision_model(model_name)

    # ─────────────────────────────────────────────────────────────
    # Model Selection
    # ─────────────────────────────────────────────────────────────

    def select_model(self, models: Optional[List[Dict[str, object]]] = None) -> str:
        """Interactive model selection with table display."""
        if models is None:
            models = self.models

        if not models:
            raise ValueError("No models available")

        if not self.session:
            raise ValueError("Session not set - call set_session() first")

        default_model = self.config.default_model

        table = Table(
            title="[bold]Mevcut Modeller[/]",
            box=ROUNDED,
            border_style=self.theme["primary"],
            header_style=f"bold {self.theme['accent']}",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("#", style="bold cyan", justify="center", width=4)
        table.add_column("", width=3)
        table.add_column("Model", style="bold white", min_width=20)
        table.add_column("Rol", style=self.theme["muted"], min_width=20)
        table.add_column("Boyut", style=self.theme["muted"], justify="right", width=10)

        default_idx = 1
        for i, model in enumerate(models, 1):
            name = model.get("name", "?")
            size = format_size(int(model.get("size", 0)))
            prompt_info = get_model_prompt(name, self.prompts)
            icon = prompt_info.get("icon", "\U0001f916")

            cached_caps = self.model_cache.get(name, {})
            if cached_caps.get("supports_embedding") and not cached_caps.get(
                "supports_completion", True
            ):
                icon = "\U0001f9e9"
            elif cached_caps.get("supports_vision") or is_vision_model(name):
                icon = "\U0001f441\ufe0f"

            role = prompt_info.get("description", "-")
            display_name = (
                f"{name} \u2605" if default_model and default_model in name else name
            )
            if default_model and default_model in name:
                default_idx = i

            table.add_row(str(i), icon, display_name, role, size)

        self.console.print(table)
        self.console.print()

        while True:
            try:
                choice = self.session.prompt(
                    HTML(
                        f'<style fg="{self.theme["primary"]}">Model sec [{default_idx}]: </style>'
                    ),
                ) or str(default_idx)
                idx = int(choice)
                if 1 <= idx <= len(models):
                    self.current_model = models[idx - 1]["name"]
                    return self.current_model
                self.console.print(f"[red]1-{len(models)} arasi bir sayi gir[/]")
            except ValueError:
                self.console.print("[red]Gecerli bir sayi gir[/]")
            except (KeyboardInterrupt, EOFError):
                raise SystemExit(0)

    def show_model_info(self, model_name: str) -> None:
        """Display model capabilities and details."""
        model_data = next((m for m in self.models if m["name"] == model_name), None)
        prompt_info = get_model_prompt(model_name, self.prompts)

        if model_data:
            details = model_data.get("details", {})
            info_items = []
            if details.get("parameter_size"):
                info_items.append(f"[bold]{details['parameter_size']}[/]")
            if details.get("quantization_level"):
                info_items.append(f"Q{details['quantization_level']}")
            if details.get("family"):
                info_items.append(details["family"])

            caps = self.get_model_capabilities(model_name)
            if caps:
                context_length = caps.get("context_length")
                if isinstance(context_length, int):
                    info_items.append(f"Ctx {context_length:,}")
                if caps.get("supports_vision"):
                    info_items.append("\U0001f441\ufe0f Vision")
                if caps.get("supports_tools"):
                    info_items.append("\U0001f6e0 Tools")
                if caps.get("supports_embedding"):
                    info_items.append("\U0001f9e9 Embed")
                if "thinking" in (caps.get("capabilities") or []):
                    info_items.append("\U0001f9e0 Thinking")

            icon = prompt_info.get("icon", "\U0001f916")
            prompt_name = prompt_info.get("name", "")
            prompt_desc = prompt_info.get("description", "")
            info_str = " \u2022 ".join(info_items)
            success_color = self.theme["success"]
            muted_color = self.theme["muted"]
            accent_color = self.theme["accent"]
            self.console.print(
                Panel(
                    f"[bold {success_color}]\u2713[/] {icon} [bold white]{model_name}[/]\n"
                    f"[{muted_color}]{info_str}[/]\n"
                    f"[{accent_color}]{prompt_name}[/] [dim]- {prompt_desc}[/]",
                    box=ROUNDED,
                    border_style=self.theme["success"],
                    padding=(0, 2),
                )
            )

            if caps and not caps.get("supports_completion", True):
                self.console.print(
                    f"[{self.theme['error']}]Uyari: Bu model embedding odakli, sohbet yaniti uretmeyebilir.[/]"
                )
            if caps and isinstance(caps.get("context_length"), int):
                ctx_len = int(caps["context_length"])
                if self.config.context_token_budget > ctx_len > 0:
                    self.console.print(
                        f"[{self.theme['muted']}]Not: context butcesi model limitinin uzerinde ({self.config.context_token_budget} > {ctx_len}).[/]"
                    )

    # ─────────────────────────────────────────────────────────────
    # Model Operations
    # ─────────────────────────────────────────────────────────────

    def pull_model(self, model_name: str) -> bool:
        """Download model from registry with progress display."""
        host = self.config.ollama_host

        self.console.print(
            f"[{self.theme['accent']}]\U0001f4e5 Model indiriliyor: {model_name}[/]\n"
        )

        try:
            response = requests.post(
                f"{host}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            )
            response.raise_for_status()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"[cyan]{model_name}", total=100)

                for line in response.iter_lines():
                    if line:
                        try:
                            data = line.decode("utf-8")
                            import json

                            info = json.loads(data)
                            status = info.get("status", "")
                            completed = info.get("completed", 0)
                            total = info.get("total", 0)

                            if total > 0:
                                pct = (completed / total) * 100
                                progress.update(task, completed=pct, description=status)
                            else:
                                progress.update(task, description=status)
                        except (json.JSONDecodeError, KeyError):
                            continue

            self.console.print(
                f"\n[{self.theme['success']}]\u2713 Model indirildi: {model_name}[/]\n"
            )
            # Refresh model list
            self.get_models()
            return True

        except requests.RequestException as exc:
            self.logger.exception("Model indirilemedi: %s", model_name)
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return False

    def delete_model(self, model_name: str, confirm: bool = True) -> bool:
        """Delete model with confirmation."""
        if confirm:
            if not Confirm.ask(
                f"[{self.theme['error']}]{model_name} silinsin mi?[/]",
                console=self.console,
            ):
                self.console.print(f"[{self.theme['muted']}]Iptal edildi[/]\n")
                return False

        host = self.config.ollama_host

        try:
            response = requests.delete(
                f"{host}/api/delete",
                json={"name": model_name},
                timeout=30,
            )
            response.raise_for_status()

            self.console.print(
                f"[{self.theme['success']}]\u2713 Silindi: {model_name}[/]\n"
            )

            # Remove from cache
            if model_name in self.model_cache:
                del self.model_cache[model_name]
                self._save_model_cache()

            # Refresh model list
            self.get_models()
            return True

        except requests.RequestException as exc:
            self.logger.exception("Model silinemedi: %s", model_name)
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return False

    # ─────────────────────────────────────────────────────────────
    # Profile Management
    # ─────────────────────────────────────────────────────────────

    def apply_model_profiles(self, model_name: str) -> None:
        """Apply temperature and prompt from model profiles."""
        self.profile_prompt = ""
        self.active_profile_name = None
        self.current_temperature = None

        profile, profile_name = self._find_model_profile(model_name)
        if profile:
            if profile.system_prompt:
                self.profile_prompt = profile.system_prompt
            if profile.temperature is not None:
                self.current_temperature = profile.temperature
            self.active_profile_name = profile_name

        if self.config.active_profile:
            active = self.config.profiles.get(self.config.active_profile)
            if active:
                if active.system_prompt:
                    self.profile_prompt = active.system_prompt
                if active.temperature is not None:
                    self.current_temperature = active.temperature
                self.active_profile_name = self.config.active_profile

    def _find_model_profile(
        self, model_name: str
    ) -> Tuple[Optional[ProfileModel], Optional[str]]:
        """Locate profile by model name."""
        name = model_name.lower()
        best_key = None
        best_profile = None

        for key, profile in self.config.model_profiles.items():
            key_lower = key.lower()
            if name == key_lower or name.startswith(key_lower) or key_lower in name:
                if best_key is None or len(key_lower) > len(best_key):
                    best_key = key_lower
                    best_profile = profile

        if best_profile:
            return best_profile, best_key

        for prof_name, profile in self.config.profiles.items():
            if not profile.auto_apply or not profile.model:
                continue
            key_lower = profile.model.lower()
            if name == key_lower or name.startswith(key_lower) or key_lower in name:
                return profile, prof_name

        return None, None

    # ─────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────

    def save_benchmark_result(self, result: Dict[str, object]) -> None:
        """Save benchmark timing results."""
        data = read_json(self.benchmarks_file, self.logger)
        if not isinstance(data, list):
            data = []
        data.append(result)
        write_json(self.benchmarks_file, data, self.logger)
