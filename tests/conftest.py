"""Shared pytest fixtures for ollama-cli tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ollama_cli.models import ConfigModel, TokenStats, FavoritesModel
from ollama_cli.storage import resolve_paths
from ollama_cli.logging_utils import setup_logging


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Set up temporary home directory for tests."""
    monkeypatch.setenv("OLLAMA_CLI_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def paths(temp_home):
    """Resolve paths with temporary home."""
    return resolve_paths()


@pytest.fixture
def logger(paths):
    """Set up logger for tests."""
    return setup_logging(paths.log_file, diagnostic=False)


@pytest.fixture
def mock_config():
    """Default test configuration."""
    config = ConfigModel()
    config.ollama_host = "http://localhost:11434"
    config.default_model = "llama3:latest"
    config.theme = "dark"
    config.context_token_budget = 4096
    config.context_keep_last = 6
    config.context_autosummarize = True
    config.render_markdown = True
    config.show_metrics = True
    config.auto_save = False
    config.mask_sensitive = False
    config.encryption_enabled = False
    return config


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    console = MagicMock()
    console.print = MagicMock()
    console.status = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )
    console.clear = MagicMock()
    return console


@pytest.fixture
def mock_token_stats():
    """Default token stats for testing."""
    stats = TokenStats()
    stats.prompt_tokens = 100
    stats.completion_tokens = 200
    stats.total_tokens = 300
    return stats


@pytest.fixture
def mock_theme():
    """Default theme colors for testing."""
    return {
        "primary": "#3b82f6",
        "secondary": "#8b5cf6",
        "accent": "#06b6d4",
        "success": "#22c55e",
        "error": "#ef4444",
        "muted": "#64748b",
        "user": "#3b82f6",
        "assistant": "#8b5cf6",
    }


@pytest.fixture
def mock_prompts():
    """Default prompts configuration."""
    return {
        "_default": {
            "name": "Default",
            "description": "Default assistant",
            "icon": "\U0001f916",
            "system_prompt": "You are a helpful assistant.",
        },
        "llama3": {
            "name": "Llama 3",
            "description": "Meta's Llama 3",
            "icon": "\U0001f999",
            "system_prompt": "You are Llama, a helpful AI assistant.",
        },
    }


@pytest.fixture
def mock_favorites():
    """Default favorites for testing."""
    return FavoritesModel(
        favorites={
            "explain": "Bunu acikla:",
            "code": "Bu kod ne yapiyor:",
        },
        templates={},
    )


@pytest.fixture
def sample_conversation():
    """Sample conversation messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Merhaba!"},
        {"role": "assistant", "content": "Merhaba! Size nasil yardimci olabilirim?"},
        {"role": "user", "content": "Python nedir?"},
        {
            "role": "assistant",
            "content": "Python, yuksek seviyeli bir programlama dilidir.",
        },
    ]


@pytest.fixture
def sample_models():
    """Sample model list for testing."""
    return [
        {
            "name": "llama3:latest",
            "size": 4_000_000_000,
            "details": {
                "family": "llama",
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
            },
        },
        {
            "name": "mistral:latest",
            "size": 4_500_000_000,
            "details": {
                "family": "mistral",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "qwen3-vl:latest",
            "size": 8_000_000_000,
            "details": {
                "family": "qwen",
                "parameter_size": "14B",
                "quantization_level": "Q4_0",
            },
        },
    ]


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API chat response."""
    return {
        "message": {"content": "Bu bir test yaniti."},
        "done": True,
        "prompt_eval_count": 50,
        "eval_count": 100,
    }


@pytest.fixture
def mock_ollama_stream_response():
    """Mock Ollama streaming response lines."""
    return [
        b'{"message": {"content": "Bu "}, "done": false}',
        b'{"message": {"content": "bir "}, "done": false}',
        b'{"message": {"content": "test "}, "done": false}',
        b'{"message": {"content": "yaniti."}, "done": true, "prompt_eval_count": 50, "eval_count": 100}',
    ]
