from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .security import (
    SecurityError,
    decrypt_text,
    encrypt_text,
    get_encryption_key,
    mask_messages,
    mask_sensitive_text,
)


class SessionMeta(BaseModel):
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int
    token_total: int
    tags: List[str] = Field(default_factory=list)
    encrypted: bool = False
    path: str
    summary_excerpt: str = ""

    model_config = ConfigDict(extra="allow")


class SessionData(BaseModel):
    meta: SessionMeta
    messages: List[Dict[str, Any]]
    token_stats: Dict[str, int]
    summary: str = ""

    model_config = ConfigDict(extra="allow")


class SessionStore:
    def __init__(self, paths, logger, config) -> None:
        self.paths = paths
        self.logger = logger
        self.config = config
        self.paths.sessions_dir.mkdir(parents=True, exist_ok=True)

    def update_config(self, config) -> None:
        self.config = config

    def list_sessions(self) -> List[SessionMeta]:
        index = self._load_index()
        sessions = []
        for item in index.get("sessions", []):
            try:
                sessions.append(SessionMeta.model_validate(item))
            except ValidationError:
                self.logger.warning("Session meta dogrulanamadi")
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def save_session(
        self,
        session_id: Optional[str],
        title: str,
        model: str,
        messages: List[Dict[str, Any]],
        token_stats: Dict[str, int],
        tags: List[str],
        summary: str,
        show_log: bool = True,
    ) -> SessionMeta:
        now = datetime.utcnow().isoformat()
        session_id = session_id or self._generate_session_id()

        if self.config.mask_sensitive:
            messages = mask_messages(messages, self.config.mask_patterns)
            summary = mask_sensitive_text(summary, self.config.mask_patterns)
            title = mask_sensitive_text(title, self.config.mask_patterns)

        meta = SessionMeta(
            id=session_id,
            title=title,
            model=model,
            created_at=now,
            updated_at=now,
            message_count=len([m for m in messages if m.get("role") != "system"]),
            token_total=token_stats.get("total_tokens", 0),
            tags=sorted(set(tags)),
            encrypted=self.config.encryption_enabled,
            path="",
            summary_excerpt=summary[:120],
        )

        index = self._load_index()
        existing = self._find_meta(index, session_id)
        if existing:
            meta.created_at = existing.get("created_at", now)

        data = SessionData(
            meta=meta,
            messages=messages,
            token_stats=token_stats,
            summary=summary,
        )

        payload = json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2)
        file_path = self._session_file_path(session_id, self.config.encryption_enabled)

        if self.config.encryption_enabled:
            key = get_encryption_key(self.config)
            if not key:
                raise SecurityError("Sifreleme acik ama anahtar yok")
            payload = encrypt_text(payload, key)

        try:
            file_path.write_text(payload, encoding="utf-8")
        except Exception:
            self.logger.exception("Session dosyasi yazilamadi: %s", file_path)
            raise

        meta.path = file_path.name
        self._upsert_index(index, meta)
        self._save_index(index)

        if show_log:
            self.logger.info("Session kaydedildi: %s", meta.id)

        return meta

    def load_session(self, session_id: str) -> Optional[SessionData]:
        index = self._load_index()
        meta = self._find_meta(index, session_id)
        if not meta:
            return None

        path = self.paths.sessions_dir / meta.get("path", "")
        if not path.exists():
            return None

        raw = path.read_text(encoding="utf-8")
        if meta.get("encrypted") or path.suffix == ".enc":
            key = get_encryption_key(self.config)
            if not key:
                raise SecurityError("Sifreli session icin anahtar gerekli")
            raw = decrypt_text(raw, key)

        data = json.loads(raw)
        return SessionData.model_validate(data)

    def delete_session(self, session_id: str) -> bool:
        index = self._load_index()
        meta = self._find_meta(index, session_id)
        if not meta:
            return False

        path = self.paths.sessions_dir / meta.get("path", "")
        if path.exists():
            try:
                path.unlink()
            except Exception:
                self.logger.exception("Session dosyasi silinemedi: %s", path)
        self._remove_meta(index, session_id)
        self._save_index(index)
        return True

    def update_tags(self, session_id: str, tags: List[str]) -> bool:
        index = self._load_index()
        meta = self._find_meta(index, session_id)
        if not meta:
            return False

        meta["tags"] = sorted(set(tags))
        meta["updated_at"] = datetime.utcnow().isoformat()
        self._save_index(index)
        return True

    def update_title(self, session_id: str, title: str) -> bool:
        index = self._load_index()
        meta = self._find_meta(index, session_id)
        if not meta:
            return False

        meta["title"] = title
        meta["updated_at"] = datetime.utcnow().isoformat()
        self._save_index(index)
        return True

    def prune_sessions(self, keep_ids: List[str]) -> None:
        index = self._load_index()
        sessions: List[SessionMeta] = []
        for item in index.get("sessions", []):
            try:
                sessions.append(SessionMeta.model_validate(item))
            except ValidationError:
                self.logger.warning("Session meta dogrulanamadi, atlandi")
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        cutoff = None
        if (
            self.config.session_retention_days
            and self.config.session_retention_days > 0
        ):
            cutoff = datetime.utcnow() - timedelta(
                days=self.config.session_retention_days
            )

        kept = []
        removed = []
        for session in sessions:
            if session.id in keep_ids:
                kept.append(session)
                continue
            if cutoff and datetime.fromisoformat(session.updated_at) < cutoff:
                removed.append(session)
                continue
            kept.append(session)

        if (
            self.config.session_retention_count
            and self.config.session_retention_count > 0
        ):
            if len(kept) > self.config.session_retention_count:
                removed.extend(kept[self.config.session_retention_count :])
                kept = kept[: self.config.session_retention_count]

        for session in removed:
            self.delete_session(session.id)

        index["sessions"] = [session.model_dump(mode="json") for session in kept]
        self._save_index(index)

    def _load_index(self) -> Dict[str, Any]:
        path = self.paths.sessions_index_file
        if not path.exists():
            return {"sessions": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.logger.exception("Session index okunamadi")
            return {"sessions": []}

    def _save_index(self, data: Dict[str, Any]) -> None:
        try:
            self.paths.sessions_dir.mkdir(parents=True, exist_ok=True)
            self.paths.sessions_index_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            self.logger.exception("Session index yazilamadi")

    def _find_meta(
        self, index: Dict[str, Any], session_id: str
    ) -> Optional[Dict[str, Any]]:
        for item in index.get("sessions", []):
            if item.get("id") == session_id:
                return item
        return None

    def _remove_meta(self, index: Dict[str, Any], session_id: str) -> None:
        index["sessions"] = [
            item for item in index.get("sessions", []) if item.get("id") != session_id
        ]

    def _upsert_index(self, index: Dict[str, Any], meta: SessionMeta) -> None:
        updated = False
        for idx, item in enumerate(index.get("sessions", [])):
            if item.get("id") == meta.id:
                index["sessions"][idx] = meta.model_dump(mode="json")
                updated = True
                break
        if not updated:
            index.setdefault("sessions", []).append(meta.model_dump(mode="json"))

    def _generate_session_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{secrets.token_hex(3)}"

    def _session_file_path(self, session_id: str, encrypted: bool) -> Path:
        suffix = ".json.enc" if encrypted else ".json"
        return self.paths.sessions_dir / f"{session_id}{suffix}"
