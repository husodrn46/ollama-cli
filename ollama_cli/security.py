from __future__ import annotations

import os
import re
from typing import Iterable, List


class SecurityError(RuntimeError):
    pass


def mask_sensitive_text(text: str, patterns: Iterable[str]) -> str:
    masked = text
    for pattern in patterns:
        masked = re.sub(pattern, "[REDACTED]", masked)
    return masked


def mask_messages(messages: list[dict], patterns: Iterable[str]) -> list[dict]:
    sanitized = []
    for msg in messages:
        new_msg = dict(msg)
        content = msg.get("content", "")
        if isinstance(content, str):
            new_msg["content"] = mask_sensitive_text(content, patterns)
        sanitized.append(new_msg)
    return sanitized


def get_encryption_key(config) -> str | None:
    env_key = os.environ.get("OLLAMA_CLI_KEY", "").strip()
    if env_key:
        return env_key
    key = (config.encryption_key or "").strip()
    return key or None


def encrypt_text(plain_text: str, key: str) -> str:
    try:
        from cryptography.fernet import Fernet
    except Exception as exc:
        raise SecurityError("Sifreleme icin cryptography gereklidir") from exc

    try:
        fernet = Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise SecurityError("Gecersiz sifreleme anahtari") from exc

    token = fernet.encrypt(plain_text.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_text(cipher_text: str, key: str) -> str:
    try:
        from cryptography.fernet import Fernet
    except Exception as exc:
        raise SecurityError("Sifre cozumleme icin cryptography gereklidir") from exc

    try:
        fernet = Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise SecurityError("Gecersiz sifreleme anahtari") from exc

    try:
        plain = fernet.decrypt(cipher_text.encode("utf-8"))
    except Exception as exc:
        raise SecurityError("Sifre cozumleme basarisiz") from exc

    return plain.decode("utf-8")


def generate_key() -> str:
    try:
        from cryptography.fernet import Fernet
    except Exception as exc:
        raise SecurityError("Sifreleme icin cryptography gereklidir") from exc

    return Fernet.generate_key().decode("utf-8")
