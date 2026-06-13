from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage

from backend.api.services.sqlite_session_store import SqliteSessionStore


def test_sqlite_session_store_rebuilds_session_metadata(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)
    meta = store.create_session(scene="patient", patient_id=42)
    meta.uploaded_assets["asset-1"] = {
        "asset_id": "asset-1",
        "filename": "report.pdf",
        "patient_id": 42,
    }
    meta.processed_files["42:abc"] = {"asset_id": "asset-1"}
    store.touch(meta.session_id)
    store.enqueue_context_message(meta.session_id, HumanMessage(content="queued context"))
    store.merge_context_state(meta.session_id, {"summary_memory": "short summary"})
    store.set_context_maintenance(meta.session_id, {"status": "running"})
    snapshot_version = store.bump_snapshot_version(meta.session_id)
    session_id = meta.session_id
    thread_id = meta.thread_id
    store.close()

    reopened = SqliteSessionStore(db_path, ttl_days=None)
    restored = reopened.get_session(session_id)

    assert restored is not None
    assert restored.thread_id == thread_id
    assert restored.scene == "patient"
    assert restored.patient_id == 42
    assert restored.snapshot_version == snapshot_version
    assert restored.uploaded_assets["asset-1"]["filename"] == "report.pdf"
    assert restored.processed_files["42:abc"]["asset_id"] == "asset-1"
    assert restored.pending_context_messages == [HumanMessage(content="queued context")]
    assert restored.context_maintenance == {"status": "running"}
    assert restored.context_state == {"summary_memory": "short summary"}
    reopened.close()


def test_sqlite_session_store_does_not_persist_runtime_run_lock(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)
    meta = store.create_session()
    assert store.try_acquire_run_lock(meta.session_id, "run-old")
    assert store.get_session(meta.session_id).active_run_id == "run-old"
    store.close()

    reopened = SqliteSessionStore(db_path, ttl_days=None)
    restored = reopened.get_session(meta.session_id)

    assert restored is not None
    assert restored.active_run_id is None
    assert reopened.try_acquire_run_lock(meta.session_id, "run-new")
    reopened.release_run_lock(meta.session_id, "run-new")
    reopened.close()


def test_sqlite_session_store_purges_expired_rows_on_startup(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)
    expired = store.create_session()
    current = store.create_session()
    store.close()

    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET last_used_at = ? WHERE session_id = ?",
            (old_timestamp, expired.session_id),
        )

    reopened = SqliteSessionStore(db_path, ttl_days=7)

    assert reopened.get_session(expired.session_id) is None
    assert reopened.get_session(current.session_id) is not None
    reopened.close()


def test_sqlite_session_store_skips_corrupt_pending_context_rows(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)
    good = store.create_session()
    store.close()

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, thread_id, scene, patient_id, snapshot_version,
                uploaded_assets, processed_files, pending_context,
                pending_context_version, context_maintenance, context_state,
                created_at, last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sess_corrupt",
                "thread_corrupt",
                "doctor",
                None,
                0,
                "{}",
                "{}",
                b"not-a-pickle",
                1,
                None,
                "{}",
                now,
                now,
            ),
        )

    reopened = SqliteSessionStore(db_path, ttl_days=None)

    assert reopened.get_session(good.session_id) is not None
    assert reopened.get_session("sess_corrupt") is None
    reopened.close()


def test_sqlite_session_store_missing_session_cleanup_paths_do_not_raise(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)

    assert store.drain_context_messages("sess_missing") == []
    store.restore_context_messages("sess_missing", [HumanMessage(content="queued")])
    store.touch("sess_missing")

    store.close()
