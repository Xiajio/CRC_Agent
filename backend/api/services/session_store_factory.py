from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.session_store_base import SessionStore
from backend.api.services.sqlite_session_store import SqliteSessionStore


def build_session_store(settings: Any, runtime_root: Path) -> SessionStore:
    backend = getattr(settings, "session_store_backend", "memory")
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        configured_path = getattr(settings, "session_store_sqlite_path", None)
        db_path = Path(configured_path) if configured_path else runtime_root / "sessions.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ttl_days = getattr(settings, "session_store_ttl_days", 7)
        return SqliteSessionStore(db_path, ttl_days=ttl_days)
    raise ValueError(f"unknown session_store_backend: {backend!r}")
