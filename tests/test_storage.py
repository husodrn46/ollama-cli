import os

from ollama_cli.logging_utils import setup_logging
from ollama_cli.storage import load_config, resolve_paths


def test_load_config_creates_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLI_HOME", str(tmp_path))
    paths = resolve_paths()
    logger = setup_logging(paths.log_file, diagnostic=False)

    config = load_config(paths, logger)

    assert config.ollama_host
    assert paths.config_file.exists()


