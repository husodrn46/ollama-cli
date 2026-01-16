from __future__ import annotations

from typing import Dict, Any

from .models import DEFAULT_PROMPT


def format_size(size_bytes: int) -> str:
    gb = size_bytes / (1024**3)
    return f"{gb:.1f} GB" if gb >= 1 else f"{size_bytes / (1024**2):.0f} MB"


def is_vision_model(model_name: str) -> bool:
    patterns = ["vl", "vision", "llava", "minicpm-v", "cogvlm", "qwen3-vl"]
    name = model_name.lower()
    return any(p in name for p in patterns)


def get_model_prompt(model_name: str, prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    model_base = model_name.split(":")[0].lower()

    for key, value in prompts.items():
        if key.startswith("_"):
            continue
        key_lower = key.lower()
        if key_lower == model_base or key_lower in model_base or model_base in key_lower:
            return value

    return prompts.get("_default", DEFAULT_PROMPT.model_dump(mode="json"))


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, list):
        return 256
    if isinstance(content, str):
        return estimate_tokens(content)
    return 0
