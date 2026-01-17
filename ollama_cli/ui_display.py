"""UI display module - Display helpers, export, and benchmark."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import requests
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import ConfigModel, FavoritesModel, TokenStats
from .security import (
    SecurityError,
    encrypt_text,
    get_encryption_key,
    mask_messages,
    mask_sensitive_text,
)
from .templates import generate_html_export as _generate_html_template
from .utils import get_model_prompt

if TYPE_CHECKING:
    from logging import Logger


class UIDisplay:
    """Handles UI display, export, and benchmark functionality."""

    def __init__(
        self,
        config: ConfigModel,
        console: Console,
        logger: Logger,
        favorites: FavoritesModel,
        prompts: Dict,
        token_stats: TokenStats,
        get_theme: Callable[[], Dict[str, str]],
    ) -> None:
        self.config = config
        self.console = console
        self.logger = logger
        self.favorites = favorites
        self.prompts = prompts
        self.token_stats = token_stats
        self._get_theme = get_theme

    @property
    def theme(self) -> Dict[str, str]:
        """Get current theme colors."""
        return self._get_theme()

    # ─────────────────────────────────────────────────────────────
    # Header & Help
    # ─────────────────────────────────────────────────────────────

    def print_header(self) -> None:
        """Display app header with version and theme info."""
        header = Text()
        header.append("  \u25c9 ", style=f"bold {self.theme['primary']}")
        header.append("OLLAMA ", style="bold white")
        header.append("CLI ", style=f"bold {self.theme['secondary']}")
        header.append("PRO ", style=f"bold {self.theme['accent']}")
        header.append("v5.1", style="dim")

        info = Text()
        info.append(f"  \u26a1 {self.config.ollama_host}  ", style=self.theme["muted"])
        info.append(
            f"\U0001f3a8 {self.config.theme}", style=f"dim {self.theme['accent']}"
        )

        self.console.print(
            Panel(
                Text.assemble(header, "\n", info),
                box=DOUBLE,
                border_style=self.theme["primary"],
                padding=(1, 2),
            )
        )
        self.console.print()

    def show_help(self) -> None:
        """Display help text with all commands."""
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
                "Prompt & Pano",
                [
                    ("/prompts", "Prompt kutuphanesi"),
                    ("/prompts <isim>", "Hazir promptu kullan"),
                    ("/yapistir", "Panodan metin kullan"),
                    ("/clipboard", "Pano izleme ac/kapa"),
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

    # ─────────────────────────────────────────────────────────────
    # Favorites & Templates
    # ─────────────────────────────────────────────────────────────

    def show_favorites(self) -> None:
        """Display saved favorites."""
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
        """Display template list."""
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

    def show_prompts(self, library_prompts: Dict) -> None:
        """Display prompt library grouped by category."""
        if not library_prompts:
            self.console.print(
                f"[{self.theme['muted']}]Prompt kütüphanesi boş. /prompts add <isim>[/]\n"
            )
            return

        # Kategorilere göre grupla
        by_category: Dict[str, List] = {}
        for key, prompt in library_prompts.items():
            cat = prompt.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((key, prompt))

        # Her kategori için tablo oluştur
        for category, items in sorted(by_category.items()):
            table = Table(
                title=f"[bold]{category.title()}[/]",
                box=ROUNDED,
                border_style=self.theme["primary"],
                padding=(0, 1),
            )
            table.add_column("", style="white", width=3)  # Icon
            table.add_column("Komut", style=f"bold {self.theme['accent']}")
            table.add_column("Açıklama", style="white", max_width=40)
            table.add_column("Prompt", style=self.theme["muted"], max_width=35)

            for key, prompt in items:
                prompt_preview = (
                    prompt.prompt[:32] + "..."
                    if len(prompt.prompt) > 35
                    else prompt.prompt
                )
                table.add_row(prompt.icon, key, prompt.description, prompt_preview)

            self.console.print(table)
            self.console.print()

        # Kullanım bilgisi
        self.console.print(
            f"[{self.theme['muted']}]Kullanım: /prompts <isim> - örn: /prompts ozetle[/]"
        )
        self.console.print(
            f"[{self.theme['muted']}]Yeni ekle: /prompts add <isim>[/]\n"
        )

    # ─────────────────────────────────────────────────────────────
    # Statistics & Tokens
    # ─────────────────────────────────────────────────────────────

    def show_stats(self, models: List[Dict]) -> None:
        """Display model statistics and VRAM usage."""
        host = self.config.ollama_host

        try:
            response = requests.get(f"{host}/api/ps", timeout=10)
            response.raise_for_status()
            running = response.json().get("models", [])
        except Exception:
            running = []

        table = Table(
            title="[bold]Model Durumlari[/]",
            box=ROUNDED,
            border_style=self.theme["primary"],
        )
        table.add_column("Model", style="bold white")
        table.add_column("Durum", style=self.theme["accent"])
        table.add_column("VRAM", style=self.theme["muted"], justify="right")

        running_names = {m.get("name") for m in running}

        for model in models:
            name = model.get("name", "?")
            if name in running_names:
                run_info = next((r for r in running if r.get("name") == name), {})
                vram = run_info.get("size_vram", 0)
                vram_str = f"{vram / 1024 / 1024 / 1024:.1f} GB" if vram else "-"
                table.add_row(name, f"[{self.theme['success']}]Yuklendi[/]", vram_str)
            else:
                table.add_row(name, f"[{self.theme['muted']}]Beklemede[/]", "-")

        self.console.print(table)
        self.console.print()

    def show_tokens(self) -> None:
        """Display token usage statistics."""
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

    # ─────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────

    def search_messages(self, keyword: str, messages: List[Dict]) -> None:
        """Search conversation history by keyword."""
        results = []
        keyword_lower = keyword.lower()

        for i, msg in enumerate(messages):
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

    # ─────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────

    def export_chat(
        self,
        format_type: str,
        messages: List[Dict],
        model: str,
        chat_title: Optional[str] = None,
    ) -> Optional[Path]:
        """Export conversation to multiple formats."""
        try:
            save_dir = Path(self.config.save_directory).expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = model.split(":")[0].replace("/", "-")
            title = chat_title or f"Chat with {model}"
            title_slug = title.replace(" ", "_")[:30]

            export_messages = messages
            if self.config.mask_sensitive:
                export_messages = mask_messages(messages, self.config.mask_patterns)
                title = mask_sensitive_text(title, self.config.mask_patterns)

            content = ""
            extension = format_type

            if format_type == "json":
                export_data = {
                    "title": title,
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "messages": export_messages,
                    "token_stats": {
                        "prompt_tokens": self.token_stats.prompt_tokens,
                        "completion_tokens": self.token_stats.completion_tokens,
                        "total_tokens": self.token_stats.total_tokens,
                    },
                }
                content = json.dumps(export_data, indent=2, ensure_ascii=False)

            elif format_type == "txt":
                lines = [
                    f"Ollama Chat - {model}",
                    f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                if title:
                    lines.append(f"Baslik: {title}")
                lines.append("=" * 50)
                lines.append("")
                for msg in export_messages:
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
                content = self.generate_html_export(export_messages, model, title)

            else:  # markdown
                extension = "md"
                lines = [
                    f"# {title}",
                    "",
                    f"**Model:** {model}  ",
                    f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "---",
                    "",
                ]
                for msg in export_messages:
                    if msg["role"] == "system":
                        continue
                    role = (
                        "\U0001f9d1 Sen"
                        if msg["role"] == "user"
                        else f"\U0001f916 {model_short.upper()}"
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
                f"[{self.theme['success']}]\u2713 Disari aktarildi: {filepath}[/]\n"
            )
            return filepath

        except Exception as exc:
            self.logger.exception("Disa aktarma hatasi")
            self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]\n")
            return None

    def generate_html_export(
        self,
        messages: List[Dict],
        model: str,
        title: str,
    ) -> str:
        """Generate styled HTML output using shared template."""
        return _generate_html_template(
            messages=messages,
            model=model,
            title=title,
            theme=self.theme,
            total_tokens=self.token_stats.total_tokens,
        )

    def _format_html_content(self, content: str) -> str:
        """Format content for HTML export with code highlighting.

        Note: This method is kept for backward compatibility but the actual
        implementation is now in templates.py.
        """
        from .templates import format_html_content

        return format_html_content(content)

    # ─────────────────────────────────────────────────────────────
    # Benchmark & Compare
    # ─────────────────────────────────────────────────────────────

    def compare_models(
        self,
        question: str,
        model_names: List[str],
        save_benchmark: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, str]:
        """Get responses from multiple models for comparison."""
        host = self.config.ollama_host
        results: Dict[str, str] = {}

        self.console.print(
            f"\n[{self.theme['accent']}]\U0001f500 {len(model_names)} model karsilastiriliyor...[/]\n"
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
            icon = prompt_info.get("icon", "\U0001f916")
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
        self,
        model_name: str,
        prompt: str,
        runs: int,
        save_benchmark: Optional[Callable[[Dict], None]] = None,
    ) -> Optional[Dict[str, object]]:
        """Time model response with multiple runs."""
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

                if save_benchmark:
                    save_benchmark(result)

            except Exception as exc:
                self.logger.exception("Benchmark hatasi: %s", model_name)
                self.console.print(f"[{self.theme['error']}]Hata: {exc}[/]")
                return None

        avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
        avg_prompt = sum(r["prompt_tokens"] for r in results) / len(results)
        avg_completion = sum(r["completion_tokens"] for r in results) / len(results)
        avg_total = sum(r["total_tokens"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)

        # Display results
        table = Table(box=ROUNDED, border_style=self.theme["primary"])
        table.add_column("Metrik", style=f"bold {self.theme['accent']}")
        table.add_column("Deger", style="bold white", justify="right")

        table.add_row("Model", model_name)
        table.add_row("Calisma Sayisi", str(runs))
        table.add_row("Ort. Sure", f"{avg_elapsed:.2f}s")
        table.add_row("Ort. Token/s", f"{avg_tps:.1f}")
        table.add_row("Ort. Toplam Token", f"{avg_total:.0f}")

        self.console.print(
            Panel(
                table,
                title="[bold]Benchmark Sonucu[/]",
                border_style=self.theme["success"],
            )
        )
        self.console.print()

        return {
            "model": model_name,
            "runs": len(results),
            "avg_elapsed": avg_elapsed,
            "avg_prompt_tokens": avg_prompt,
            "avg_completion_tokens": avg_completion,
            "avg_total_tokens": avg_total,
            "avg_tps": avg_tps,
        }
