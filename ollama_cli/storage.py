from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from platformdirs import user_config_dir, user_data_dir
from pydantic import ValidationError

from .models import ConfigModel, FavoritesModel, PromptEntry, DEFAULT_PROMPT

APP_NAME = "ollama-cli-pro"


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_dir: Path
    config_file: Path
    prompts_file: Path
    favorites_file: Path
    history_file: Path
    log_file: Path
    sessions_dir: Path
    sessions_index_file: Path
    model_cache_file: Path
    benchmarks_file: Path
    legacy_config_file: Path
    legacy_prompts_file: Path
    legacy_favorites_file: Path
    legacy_history_file: Path


def resolve_paths() -> AppPaths:
    override_home = os.environ.get("OLLAMA_CLI_HOME", "").strip()
    if override_home:
        base_dir = Path(override_home).expanduser()
        config_dir = base_dir
        data_dir = base_dir
    else:
        config_dir = Path(user_config_dir(APP_NAME))
        data_dir = Path(user_data_dir(APP_NAME))

    config_file = config_dir / "config.json"
    prompts_file = config_dir / "prompts.json"
    favorites_file = data_dir / "favorites.json"
    history_file = data_dir / "history.txt"
    log_file = data_dir / "ollama-cli.log"
    sessions_dir = data_dir / "sessions"
    sessions_index_file = sessions_dir / "index.json"
    model_cache_file = data_dir / "model_cache.json"
    benchmarks_file = data_dir / "benchmarks.json"

    package_root = Path(__file__).resolve().parents[1]
    legacy_config_file = package_root / "config.json"
    legacy_prompts_file = package_root / "prompts.json"
    legacy_favorites_file = package_root / "favorites.json"
    legacy_history_file = Path.home() / ".ollama_chat_history"

    return AppPaths(
        config_dir=config_dir,
        data_dir=data_dir,
        config_file=config_file,
        prompts_file=prompts_file,
        favorites_file=favorites_file,
        history_file=history_file,
        log_file=log_file,
        sessions_dir=sessions_dir,
        sessions_index_file=sessions_index_file,
        model_cache_file=model_cache_file,
        benchmarks_file=benchmarks_file,
        legacy_config_file=legacy_config_file,
        legacy_prompts_file=legacy_prompts_file,
        legacy_favorites_file=legacy_favorites_file,
        legacy_history_file=legacy_history_file,
    )


def ensure_dirs(paths: AppPaths) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.sessions_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, logger) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Json okunamadi: %s", path)
        return None


def write_json(path: Path, data: Dict[str, Any], logger) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Json yazilamadi: %s", path)


def migrate_legacy_file(target: Path, legacy: Path, logger) -> None:
    if target.exists() or not legacy.exists():
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
        logger.info("Legacy dosya tasindi: %s -> %s", legacy, target)
    except Exception:
        logger.exception("Legacy dosya tasinamadi: %s", legacy)


def ensure_default_prompts(paths: AppPaths, logger) -> None:
    if paths.prompts_file.exists():
        return
    write_json(
        paths.prompts_file,
        {"_default": DEFAULT_PROMPT.model_dump(mode="json")},
        logger,
    )


def ensure_default_favorites(paths: AppPaths, logger) -> None:
    if paths.favorites_file.exists():
        return
    write_json(paths.favorites_file, {"favorites": {}, "templates": {}}, logger)


def ensure_default_config(paths: AppPaths, logger) -> None:
    if paths.config_file.exists():
        return
    write_json(paths.config_file, ConfigModel().model_dump(mode="json"), logger)


def load_config(paths: AppPaths, logger) -> ConfigModel:
    ensure_dirs(paths)
    migrate_legacy_file(paths.config_file, paths.legacy_config_file, logger)
    ensure_default_config(paths, logger)

    data = read_json(paths.config_file, logger) or {}
    try:
        config = ConfigModel.model_validate(data)
    except ValidationError:
        logger.exception("Config dogrulanamadi, varsayilanlar kullaniliyor")
        config = ConfigModel()
    return config


def save_config(config: ConfigModel, paths: AppPaths, logger) -> None:
    write_json(paths.config_file, config.model_dump(mode="json"), logger)


def load_favorites(paths: AppPaths, logger) -> FavoritesModel:
    ensure_dirs(paths)
    migrate_legacy_file(paths.favorites_file, paths.legacy_favorites_file, logger)
    ensure_default_favorites(paths, logger)

    data = read_json(paths.favorites_file, logger) or {}
    try:
        favorites = FavoritesModel.model_validate(data)
    except ValidationError:
        logger.exception("Favoriler dogrulanamadi, varsayilanlar kullaniliyor")
        favorites = FavoritesModel()
    return favorites


def save_favorites(favorites: FavoritesModel, paths: AppPaths, logger) -> None:
    write_json(paths.favorites_file, favorites.model_dump(mode="json"), logger)


def load_prompts(paths: AppPaths, logger) -> Dict[str, Dict[str, Any]]:
    ensure_dirs(paths)
    migrate_legacy_file(paths.prompts_file, paths.legacy_prompts_file, logger)
    ensure_default_prompts(paths, logger)

    data = read_json(paths.prompts_file, logger) or {}
    prompts: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if key.startswith("_"):
            prompts[key] = value
            continue
        if isinstance(value, dict):
            try:
                entry = PromptEntry.model_validate(value)
                prompts[key] = entry.model_dump(mode="json")
            except ValidationError:
                logger.warning("Prompt dogrulanamadi: %s", key)
    if "_default" not in prompts:
        prompts["_default"] = DEFAULT_PROMPT.model_dump(mode="json")
    return prompts


def migrate_history(paths: AppPaths, logger) -> None:
    if paths.history_file.exists() or not paths.legacy_history_file.exists():
        return
    try:
        paths.history_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.legacy_history_file, paths.history_file)
        logger.info("Legacy gecmis tasindi: %s", paths.history_file)
    except Exception:
        logger.exception("Legacy gecmis tasinamadi")
