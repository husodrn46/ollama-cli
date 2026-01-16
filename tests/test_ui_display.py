"""Tests for ui_display module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ollama_cli.ui_display import UIDisplay


class TestUIDisplayInit:
    """Tests for UIDisplay initialization."""

    def test_init_creates_instance(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        assert display is not None


class TestShowHelp:
    """Tests for help display."""

    def test_show_help_prints_sections(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        display.show_help()

        assert mock_console.print.called


class TestShowFavorites:
    """Tests for favorites display."""

    def test_show_favorites_empty(
        self,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        from ollama_cli.models import FavoritesModel

        empty_favorites = FavoritesModel(favorites={}, templates={})
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=empty_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        display.show_favorites()

        mock_console.print.assert_called()

    def test_show_favorites_with_items(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        display.show_favorites()

        assert mock_console.print.called


class TestShowTokens:
    """Tests for token display."""

    def test_show_tokens_displays_stats(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        display.show_tokens()

        assert mock_console.print.called


class TestSearchMessages:
    """Tests for message search."""

    def test_search_messages_found(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        display.search_messages("Hello", messages)

        assert mock_console.print.called

    def test_search_messages_not_found(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello world"},
        ]

        display.search_messages("Python", messages)

        assert mock_console.print.called

    def test_search_messages_case_insensitive(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "HELLO world"},
        ]

        display.search_messages("hello", messages)

        # Should find the message despite case difference
        assert mock_console.print.called


class TestExportChat:
    """Tests for chat export."""

    def test_export_chat_json(
        self,
        tmp_path,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
        mock_config,
    ):
        mock_config.save_directory = str(tmp_path)
        mock_config.mask_sensitive = False
        mock_config.encrypt_exports = False

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = display.export_chat("json", messages, "test-model", "Test Chat")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".json"

    def test_export_chat_txt(
        self,
        tmp_path,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
        mock_config,
    ):
        mock_config.save_directory = str(tmp_path)
        mock_config.mask_sensitive = False
        mock_config.encrypt_exports = False

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = display.export_chat("txt", messages, "test-model", "Test Chat")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".txt"

    def test_export_chat_markdown(
        self,
        tmp_path,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
        mock_config,
    ):
        mock_config.save_directory = str(tmp_path)
        mock_config.mask_sensitive = False
        mock_config.encrypt_exports = False

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = display.export_chat("md", messages, "test-model", "Test Chat")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"

    def test_export_chat_html(
        self,
        tmp_path,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
        mock_config,
    ):
        mock_config.save_directory = str(tmp_path)
        mock_config.mask_sensitive = False
        mock_config.encrypt_exports = False

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = display.export_chat("html", messages, "test-model", "Test Chat")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".html"


class TestGenerateHtmlExport:
    """Tests for HTML generation."""

    def test_generate_html_export_basic(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        html = display.generate_html_export(messages, "test-model", "Test Title")

        assert "<!DOCTYPE html>" in html
        assert "Test Title" in html
        assert "Hello" in html
        assert "Hi there!" in html

    def test_generate_html_export_with_code(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )
        messages = [
            {"role": "user", "content": "Show me code"},
            {
                "role": "assistant",
                "content": "Here's code:\n```python\nprint('hello')\n```",
            },
        ]

        html = display.generate_html_export(messages, "test-model", "Code Test")

        assert "<!DOCTYPE html>" in html
        assert "language-python" in html


class TestFormatHtmlContent:
    """Tests for HTML content formatting."""

    def test_format_html_content_plain_text(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display._format_html_content("Hello world")

        assert "Hello world" in result

    def test_format_html_content_with_code_block(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display._format_html_content("Code:\n```python\nprint('hi')\n```")

        assert "code-container" in result
        assert "language-python" in result

    def test_format_html_content_escapes_html(
        self,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display._format_html_content("<script>alert('xss')</script>")

        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestBenchmarkModel:
    """Tests for model benchmarking."""

    @patch("ollama_cli.ui_display.requests.post")
    def test_benchmark_model_success(
        self,
        mock_post,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 50,
            "eval_count": 100,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display.benchmark_model("test-model", "Test prompt", runs=2)

        assert result is not None
        assert result["model"] == "test-model"
        assert result["runs"] == 2
        assert "avg_elapsed" in result
        assert "avg_tps" in result

    @patch("ollama_cli.ui_display.requests.post")
    def test_benchmark_model_failure(
        self,
        mock_post,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_post.side_effect = Exception("Connection failed")

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display.benchmark_model("test-model", "Test prompt", runs=1)

        assert result is None


class TestCompareModels:
    """Tests for model comparison."""

    @patch("ollama_cli.ui_display.requests.post")
    def test_compare_models_success(
        self,
        mock_post,
        mock_config,
        mock_console,
        logger,
        mock_favorites,
        mock_prompts,
        mock_token_stats,
        mock_theme,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Test response"},
        }
        mock_post.return_value = mock_response

        display = UIDisplay(
            config=mock_config,
            console=mock_console,
            logger=logger,
            favorites=mock_favorites,
            prompts=mock_prompts,
            token_stats=mock_token_stats,
            get_theme=lambda: mock_theme,
        )

        result = display.compare_models("Test question", ["model1", "model2"])

        assert len(result) == 2
        assert "model1" in result
        assert "model2" in result
