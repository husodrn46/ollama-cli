"""Tests for chat_engine module."""

import pytest
from unittest.mock import MagicMock, patch

from ollama_cli.chat_engine import ChatEngine, PERSONAS, SUMMARY_PREFIX


class TestChatEngineInit:
    """Tests for ChatEngine initialization."""

    def test_init_creates_empty_messages(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        assert engine.messages == []
        assert engine.summary == ""
        assert engine.current_persona is None

    def test_init_conversation_creates_system_message(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        messages = engine.init_conversation("llama3:latest")

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Llama" in messages[0]["content"]

    def test_init_conversation_no_system_when_no_prompt(
        self, mock_config, mock_console, logger, mock_token_stats, mock_theme
    ):
        prompts = {"_default": {"system_prompt": ""}}
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        messages = engine.init_conversation("unknown:latest")

        assert len(messages) == 0


class TestSystemPromptBuilding:
    """Tests for system prompt construction."""

    def test_build_system_prompt_base_only(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.base_system_prompt = "Base prompt"

        result = engine.build_system_prompt()

        assert result == "Base prompt"

    def test_build_system_prompt_with_persona(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.base_system_prompt = "Base"
        engine.current_persona = "developer"

        result = engine.build_system_prompt()

        assert "Base" in result
        assert "yazilim gelistirici" in result.lower()

    def test_build_system_prompt_with_profile(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.base_system_prompt = "Base"
        engine.profile_prompt = "Profile specific instructions"

        result = engine.build_system_prompt()

        assert "Base" in result
        assert "Profile specific instructions" in result


class TestConversationManagement:
    """Tests for conversation message management."""

    def test_update_system_message_inserts_when_missing(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [{"role": "user", "content": "Hello"}]
        engine.base_system_prompt = "System prompt"

        engine.update_system_message()

        assert len(engine.messages) == 2
        assert engine.messages[0]["role"] == "system"

    def test_update_system_message_updates_existing(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [
            {"role": "system", "content": "Old prompt"},
            {"role": "user", "content": "Hello"},
        ]
        engine.base_system_prompt = "New prompt"

        engine.update_system_message()

        assert engine.messages[0]["content"] == "New prompt"

    def test_find_summary_index_returns_none_when_missing(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [{"role": "user", "content": "Hello"}]

        result = engine._find_summary_index()

        assert result is None

    def test_find_summary_index_returns_index(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [
            {"role": "system", "content": "Base prompt"},
            {"role": "system", "content": f"{SUMMARY_PREFIX}\nSummary content"},
        ]

        result = engine._find_summary_index()

        assert result == 1

    def test_is_summary_message_true(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        msg = {"role": "system", "content": f"{SUMMARY_PREFIX}\nSummary"}

        assert engine._is_summary_message(msg) is True

    def test_is_summary_message_false(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        msg = {"role": "system", "content": "Regular system prompt"}

        assert engine._is_summary_message(msg) is False


class TestTokenEstimation:
    """Tests for token counting."""

    def test_estimate_context_tokens_empty(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = []

        result = engine.estimate_context_tokens()

        assert result == 0

    def test_estimate_context_tokens_with_messages(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
        sample_conversation,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = sample_conversation

        result = engine.estimate_context_tokens()

        assert result > 0


class TestSummarization:
    """Tests for conversation summarization."""

    def test_maybe_summarize_disabled(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_config.context_autosummarize = False
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = engine.maybe_summarize()

        assert result is False

    def test_maybe_summarize_under_budget(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_config.context_autosummarize = True
        mock_config.context_token_budget = 10000
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [{"role": "user", "content": "Hello"}]

        result = engine.maybe_summarize()

        assert result is False

    def test_split_messages_for_summary_keeps_recent(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_config.context_keep_last = 2
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Reply 3"},
        ]

        to_summarize, keep = engine._split_messages_for_summary()

        assert len(keep) == 2
        assert keep[0]["content"] == "Message 3"
        assert len(to_summarize) == 4

    def test_split_messages_empty_when_too_few(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_config.context_keep_last = 6
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
        ]

        to_summarize, keep = engine._split_messages_for_summary()

        assert len(to_summarize) == 0
        assert len(keep) == 2

    def test_build_summary_input_format(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = engine._build_summary_input(messages)

        assert "Kullanici: Hello" in result
        assert "Asistan: Hi there!" in result
        assert "Yeni, guncel bir ozet yaz" in result


class TestPersonaManagement:
    """Tests for persona handling."""

    def test_set_persona_valid(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = engine.set_persona("developer")

        assert result is True
        assert engine.current_persona == "developer"

    def test_set_persona_invalid(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = engine.set_persona("invalid_persona")

        assert result is False
        assert engine.current_persona is None

    def test_set_persona_none_clears(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.current_persona = "developer"

        result = engine.set_persona(None)

        assert result is True
        assert engine.current_persona is None

    def test_list_personas_returns_all(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        personas = engine.list_personas()

        assert "developer" in personas
        assert "teacher" in personas
        assert "assistant" in personas

    def test_get_persona_info(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        info = engine.get_persona_info("developer")

        assert info is not None
        assert "name" in info
        assert "icon" in info
        assert "prompt" in info


class TestResponseHandling:
    """Tests for response processing."""

    def test_handle_response_adds_to_messages(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = []

        engine.handle_response("Test response")

        assert len(engine.messages) == 1
        assert engine.messages[0]["role"] == "assistant"
        assert engine.messages[0]["content"] == "Test response"

    def test_handle_response_ignores_none(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        engine.messages = []

        engine.handle_response(None)

        assert len(engine.messages) == 0

    def test_handle_response_calls_autosave(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        autosave_called = []

        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
            on_autosave=lambda: autosave_called.append(True),
        )
        engine.messages = []

        engine.handle_response("Test response")

        assert len(autosave_called) == 1


class TestExtractSummary:
    """Tests for summary extraction."""

    def test_extract_summary_found(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {
                "role": "system",
                "content": f"{SUMMARY_PREFIX}\nThis is the summary content.",
            },
        ]

        result = engine.extract_summary(messages)

        assert result == "This is the summary content."

    def test_extract_summary_not_found(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        engine = ChatEngine(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "system", "content": "Regular system prompt"},
        ]

        result = engine.extract_summary(messages)

        assert result == ""
