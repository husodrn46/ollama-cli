from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple

from .clipboard import get_image_bytes


def encode_image(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        path = Path(image_path).expanduser()
        if not path.exists():
            return None, f"Dosya bulunamadi: {path}"
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), None
    except Exception as exc:
        return None, str(exc)


def paste_image_from_clipboard(logger) -> Tuple[Optional[str], Optional[str]]:
    image_bytes, error = get_image_bytes(logger)
    if error or not image_bytes:
        return None, error
    return base64.b64encode(image_bytes).decode("utf-8"), None
