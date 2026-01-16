from __future__ import annotations

import io
import subprocess
import sys
from typing import Optional, Tuple


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
