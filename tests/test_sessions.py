from ollama_cli.logging_utils import setup_logging
from ollama_cli.models import ConfigModel
from ollama_cli.session_store import SessionStore
from ollama_cli.storage import resolve_paths


def test_session_store_save_load(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLI_HOME", str(tmp_path))
    paths = resolve_paths()
    logger = setup_logging(paths.log_file, diagnostic=False)
    config = ConfigModel()
    config.encryption_enabled = False

    store = SessionStore(paths, logger, config)

    messages = [{"role": "user", "content": "Merhaba"}]
    token_stats = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    meta = store.save_session(
        session_id=None,
        title="Test",
        model="demo",
        messages=messages,
        token_stats=token_stats,
        tags=["tag"],
        summary="",
        show_log=False,
    )

    data = store.load_session(meta.id)
    assert data is not None
    assert data.meta.title == "Test"
    assert data.messages[0]["content"] == "Merhaba"
