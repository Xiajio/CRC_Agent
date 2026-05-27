from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.session_store_factory import build_session_store
from backend.api.services.settings import RuntimeSettings
from backend.api.services.sqlite_session_store import SqliteSessionStore


def test_session_store_factory_defaults_to_memory(tmp_path) -> None:
    store = build_session_store(RuntimeSettings(session_store_backend="memory"), tmp_path)

    assert isinstance(store, InMemorySessionStore)


def test_session_store_factory_uses_runtime_default_sqlite_path(tmp_path) -> None:
    store = build_session_store(RuntimeSettings(session_store_backend="sqlite"), tmp_path)

    assert isinstance(store, SqliteSessionStore)
    assert (tmp_path / "sessions.db").exists()
    store.close()


def test_session_store_factory_uses_custom_sqlite_path(tmp_path) -> None:
    custom_path = tmp_path / "custom" / "sessions.sqlite3"
    store = build_session_store(
        RuntimeSettings(
            session_store_backend="sqlite",
            session_store_sqlite_path=str(custom_path),
        ),
        tmp_path,
    )

    assert isinstance(store, SqliteSessionStore)
    assert custom_path.exists()
    store.close()


def test_session_store_factory_rejects_unknown_backend(tmp_path) -> None:
    settings = SimpleNamespace(
        session_store_backend="bogus",
        session_store_sqlite_path=None,
        session_store_ttl_days=7,
    )

    with pytest.raises(ValueError, match="unknown session_store_backend"):
        build_session_store(settings, tmp_path)
