from __future__ import annotations

import argparse

from .app import ChatApp
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ollama CLI Pro - terminalden Ollama modelleriyle sohbet",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Diagnostik modu ac (detayli loglama)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = ChatApp(diagnostic_override=args.diag)
    return app.run()
