from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Theme(BaseModel):
    primary: str = "#00d4aa"
    secondary: str = "#7c3aed"
    accent: str = "#f59e0b"
    success: str = "#10b981"
    error: str = "#ef4444"
    muted: str = "#6b7280"
    user: str = "#3b82f6"
    assistant: str = "#10b981"
    code_bg: str = "#1e1e1e"

    model_config = ConfigDict(extra="allow")


def default_themes() -> Dict[str, Theme]:
    return {
        "dark": Theme(),
        "light": Theme(
            primary="#059669",
            secondary="#7c3aed",
            accent="#d97706",
            success="#059669",
            error="#dc2626",
            muted="#6b7280",
            user="#2563eb",
            assistant="#059669",
            code_bg="#f3f4f6",
        ),
        "ocean": Theme(
            primary="#06b6d4",
            secondary="#8b5cf6",
            accent="#f472b6",
            success="#34d399",
            error="#f87171",
            muted="#94a3b8",
            user="#38bdf8",
            assistant="#34d399",
            code_bg="#0f172a",
        ),
        "forest": Theme(
            primary="#22c55e",
            secondary="#a3e635",
            accent="#fbbf24",
            success="#4ade80",
            error="#f87171",
            muted="#71717a",
            user="#86efac",
            assistant="#4ade80",
            code_bg="#14532d",
        ),
    }


class ProfileModel(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    description: str = ""
    auto_apply: bool = False

    model_config = ConfigDict(extra="allow")


def default_mask_patterns() -> List[str]:
    return [
        r"(?i)api[_-]?key\s*[:=]\s*['\"]?([A-Za-z0-9_-]{16,})",
        r"(?i)secret\s*[:=]\s*['\"]?([A-Za-z0-9_-]{16,})",
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"(?s)-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----",
    ]


class ConfigModel(BaseModel):
    ollama_host: str = "http://localhost:11434"
    default_model: Optional[str] = None
    theme: str = "dark"
    themes: Dict[str, Theme] = Field(default_factory=default_themes)
    save_directory: str = "~/ollama-chats"
    multiline_trigger: str = '"""'
    show_metrics: bool = True
    auto_save: bool = False
    diagnostic: bool = False
    context_token_budget: int = 8192
    context_keep_last: int = 6
    context_autosummarize: bool = True
    summary_model: Optional[str] = None
    summary_prompt: str = (
        "Kisa, net ve yapilandirilmis bir ozet yaz. Teknik terimleri koru, "
        "gereksiz detaylari atla. Gerektiginde madde isaretleri kullan."
    )
    profiles: Dict[str, ProfileModel] = Field(default_factory=dict)
    model_profiles: Dict[str, ProfileModel] = Field(default_factory=dict)
    active_profile: Optional[str] = None
    session_retention_count: int = 200
    session_retention_days: int = 0
    mask_sensitive: bool = False
    mask_patterns: List[str] = Field(default_factory=default_mask_patterns)
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    encrypt_exports: bool = False
    render_markdown: bool = True
    benchmark_prompt: str = "Yapay zeka nedir? Kisa bir cumleyle acikla."
    benchmark_runs: int = 1
    benchmark_timeout: int = 120
    benchmark_temperature: float = 0.2

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class TemplateEntry(BaseModel):
    name: str = ""
    prompt: str = ""

    model_config = ConfigDict(extra="allow")


class FavoritesModel(BaseModel):
    favorites: Dict[str, str] = Field(default_factory=dict)
    templates: Dict[str, TemplateEntry] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class PromptEntry(BaseModel):
    name: str = "Varsayilan"
    icon: str = "🤖"
    description: str = "Genel asistan"
    system_prompt: str = "Sen yardimci bir AI asistansin. Turkce yanit ver."

    model_config = ConfigDict(extra="allow")


DEFAULT_PROMPT = PromptEntry()


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
