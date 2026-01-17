from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from typing import Optional, Tuple


class ClipboardTracker:
    """Track clipboard changes for monitoring feature."""

    def __init__(self) -> None:
        self._last_hash: Optional[str] = None
        self._last_content: Optional[str] = None
        self._last_type: Optional[str] = None  # "text" | "image"

    def _hash(self, content: bytes) -> str:
        """Generate MD5 hash of content."""
        return hashlib.md5(content).hexdigest()

    def check_change(self, logger) -> Optional[Tuple[str, object]]:
        """Check if clipboard has changed. Returns (type, content) if changed."""
        try:
            # First check for image
            img_bytes, _ = get_image_bytes(logger)
            if img_bytes:
                h = self._hash(img_bytes)
                if h != self._last_hash:
                    self._last_hash = h
                    self._last_type = "image"
                    return ("image", img_bytes)

            # Then check for text
            try:
                import pyperclip

                text = pyperclip.paste()
            except ImportError:
                return None
            except Exception:
                return None

            if text:
                h = self._hash(text.encode("utf-8"))
                if h != self._last_hash:
                    self._last_hash = h
                    self._last_content = text
                    self._last_type = "text"
                    return ("text", text)
        except Exception:
            logger.debug("Clipboard kontrol hatası", exc_info=True)
        return None

    def get_last(self) -> Tuple[Optional[str], Optional[object]]:
        """Get last clipboard type and content."""
        return (self._last_type, self._last_content)

    def reset(self) -> None:
        """Reset tracker state."""
        self._last_hash = None
        self._last_content = None
        self._last_type = None


def copy_text(text: str, logger) -> bool:
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        logger.debug("pyperclip kopyalama basarisiz, fallback deneniyor", exc_info=True)

    try:
        if sys.platform.startswith("darwin"):
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            process.communicate(text.encode("utf-8"))
            return process.returncode == 0
        if sys.platform.startswith("win"):
            process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
            process.communicate(text.encode("utf-8"))
            return process.returncode == 0

        if _run_text_command(["xclip", "-selection", "clipboard"], text):
            return True
        if _run_text_command(["xsel", "--clipboard", "--input"], text):
            return True
        if _run_text_command(["wl-copy"], text):
            return True
    except Exception:
        logger.exception("Clipboard kopyalama hatasi")

    return False


def _run_text_command(cmd: list[str], text: str) -> bool:
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-8"))
        return process.returncode == 0
    except FileNotFoundError:
        return False


def get_image_bytes(logger) -> Tuple[Optional[bytes], Optional[str]]:
    image_bytes = _grab_image_via_pillow(logger)
    if image_bytes:
        return image_bytes, None

    image_bytes = _grab_image_via_linux_tools(logger)
    if image_bytes:
        return image_bytes, None

    return None, "Panoda resim bulunamadi veya desteklenmiyor"


def _grab_image_via_pillow(logger) -> Optional[bytes]:
    try:
        from PIL import ImageGrab, Image

        grabbed = ImageGrab.grabclipboard()
        if isinstance(grabbed, Image.Image):
            with io.BytesIO() as buf:
                grabbed.save(buf, format="PNG")
                return buf.getvalue()
    except Exception:
        logger.debug("Pillow ImageGrab basarisiz", exc_info=True)
    return None


def _grab_image_via_linux_tools(logger) -> Optional[bytes]:
    commands = [
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
        ["wl-paste", "--type", "image/png"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("Clipboard resim okuma basarisiz", exc_info=True)
    return None
