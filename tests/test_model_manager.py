"""Tests for model_manager module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock

from ollama_cli.model_manager import ModelManager, MODEL_CACHE_TTL_SECONDS


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_init_creates_empty_cache(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )
        assert manager.model_cache == {}
        assert manager.models == []
        assert manager.current_model is None


class TestCacheManagement:
    """Tests for model cache operations."""

    def test_cache_is_fresh_valid(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        entry = {"fetched_at": datetime.utcnow().isoformat()}
        assert manager._cache_is_fresh(entry) is True

    def test_cache_is_fresh_expired(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        old_time = datetime.utcnow() - timedelta(seconds=MODEL_CACHE_TTL_SECONDS + 1)
        entry = {"fetched_at": old_time.isoformat()}
        assert manager._cache_is_fresh(entry) is False

    def test_cache_is_fresh_invalid_format(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        entry = {"fetched_at": "invalid"}
        assert manager._cache_is_fresh(entry) is False

    def test_cache_is_fresh_missing_key(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        entry = {}
        assert manager._cache_is_fresh(entry) is False


class TestContextLengthExtraction:
    """Tests for context length parsing."""

    def test_extract_context_length_found(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        model_info = {"context_length": 4096}
        result = manager._extract_context_length(model_info)
        assert result == 4096

    def test_extract_context_length_string(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        model_info = {"context_length": "8192"}
        result = manager._extract_context_length(model_info)
        assert result == 8192

    def test_extract_context_length_not_found(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        model_info = {"other_field": "value"}
        result = manager._extract_context_length(model_info)
        assert result is None

    def test_extract_context_length_not_dict(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        result = manager._extract_context_length("not a dict")
        assert result is None


class TestVisionSupport:
    """Tests for vision model detection."""

    def test_supports_vision_from_cache(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )
        manager.model_cache["test-model"] = {
            "fetched_at": datetime.utcnow().isoformat(),
            "supports_vision": True,
        }

        result = manager.supports_vision("test-model")
        assert result is True

    def test_supports_vision_from_name(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        # Model names containing vision indicators should return True
        assert manager.supports_vision("llava:latest") is True
        assert manager.supports_vision("qwen3-vl:latest") is True


class TestProfileManagement:
    """Tests for model profile handling."""

    def test_apply_model_profiles_empty(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        manager.apply_model_profiles("unknown-model")

        assert manager.profile_prompt == ""
        assert manager.active_profile_name is None
        assert manager.current_temperature is None

    def test_find_model_profile_not_found(
        self,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        profile, name = manager._find_model_profile("unknown-model")

        assert profile is None
        assert name is None


class TestGetModels:
    """Tests for model fetching."""

    @patch("ollama_cli.model_manager.requests.get")
    def test_get_models_success(
        self,
        mock_get,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
        sample_models,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": sample_models}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        models = manager.get_models()

        assert len(models) == 3
        assert models[0]["name"] == "llama3:latest"

    @patch("ollama_cli.model_manager.requests.get")
    def test_get_models_connection_error(
        self,
        mock_get,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_get.side_effect = Exception("Connection refused")

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        models = manager.get_models()

        assert models == []


class TestPullModel:
    """Tests for model downloading."""

    @patch("ollama_cli.model_manager.requests.post")
    def test_pull_model_success(
        self,
        mock_post,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"status": "pulling", "completed": 50, "total": 100}',
            b'{"status": "done", "completed": 100, "total": 100}',
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        # Mock get_models to avoid second API call
        manager.get_models = MagicMock(return_value=[])

        result = manager.pull_model("test-model")

        assert result is True

    @patch("ollama_cli.model_manager.requests.post")
    def test_pull_model_failure(
        self,
        mock_post,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_post.side_effect = Exception("Download failed")

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        result = manager.pull_model("test-model")

        assert result is False


class TestDeleteModel:
    """Tests for model deletion."""

    @patch("ollama_cli.model_manager.Confirm.ask")
    @patch("ollama_cli.model_manager.requests.delete")
    def test_delete_model_confirmed(
        self,
        mock_delete,
        mock_confirm,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_confirm.return_value = True
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )
        manager.get_models = MagicMock(return_value=[])

        result = manager.delete_model("test-model")

        assert result is True

    @patch("ollama_cli.model_manager.Confirm.ask")
    def test_delete_model_cancelled(
        self,
        mock_confirm,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_confirm.return_value = False

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )

        result = manager.delete_model("test-model")

        assert result is False

    @patch("ollama_cli.model_manager.requests.delete")
    def test_delete_model_no_confirm(
        self,
        mock_delete,
        temp_home,
        mock_config,
        mock_console,
        logger,
        mock_prompts,
        mock_theme,
        paths,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response

        manager = ModelManager(
            config=mock_config,
            console=mock_console,
            logger=logger,
            prompts=mock_prompts,
            model_cache_file=paths.model_cache_file,
            benchmarks_file=paths.benchmarks_file,
            get_theme=lambda: mock_theme,
        )
        manager.get_models = MagicMock(return_value=[])

        result = manager.delete_model("test-model", confirm=False)

        assert result is True
