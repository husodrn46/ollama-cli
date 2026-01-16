from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "ollama_cli"


def setup_logging(log_path: Path, diagnostic: bool) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(logging.DEBUG if diagnostic else logging.INFO)
        return logger

    logger.setLevel(logging.DEBUG if diagnostic else logging.INFO)
    logger.propagate = False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_log_level(logger: logging.Logger, diagnostic: bool) -> None:
    logger.setLevel(logging.DEBUG if diagnostic else logging.INFO)
