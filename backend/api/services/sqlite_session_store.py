from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from backend.api.services.session_store import InMemorySessionStore, SessionMeta

logger = logging.getLogger(__name__)

PENDING_CONTEXT_VERSION = 1


class SqliteSessionStore(InMemorySessionStore):
    def __init__(self, db_path: Path, *, ttl_days: int | None = 7) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._ttl_days = ttl_days
        self._persist_lock = Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._purge_expired()
        self._rebuild_from_disk()

    def close(self) -> None:
        with self._persist_lock:
            self._conn.close()

    def reset(self) -> None:
        with self._store_lock:
            super().reset()
            with self._persist_lock:
                self._conn.execute("DELETE FROM sessions")

    def create_session(self, *, scene: str = "doctor", patient_id: int | None = None) -> SessionMeta:
        with self._store_lock:
            meta = super().create_session(scene=scene, patient_id=patient_id)
            self._persist(meta)
            return meta

    def rotate_thread(self, session_id: str, *, clear_patient_id: bool = False) -> SessionMeta:
        with self._store_lock:
            meta = super().rotate_thread(session_id, clear_patient_id=clear_patient_id)
            self._persist(meta)
            return meta

    def set_patient_id(
        self,
        session_id: str,
        patient_id: int | None,
        *,
        allow_replace: bool = False,
    ) -> SessionMeta:
        with self._store_lock:
            meta = super().set_patient_id(session_id, patient_id, allow_replace=allow_replace)
            self._persist(meta)
            return meta

    def bind_patient(self, session_id: str, patient_id: int) -> SessionMeta:
        with self._store_lock:
            meta = super().bind_patient(session_id, patient_id)
            self._persist(meta)
            return meta

    def enqueue_context_message(self, session_id: str, message: Any) -> None:
        with self._store_lock:
            super().enqueue_context_message(session_id, message)
            self._persist(self._sessions[session_id])

    def drain_context_messages(self, session_id: str) -> list[Any]:
        with self._store_lock:
            drained = super().drain_context_messages(session_id)
            self._persist(self._sessions[session_id])
            return drained

    def restore_context_messages(self, session_id: str, messages: list[Any]) -> None:
        with self._store_lock:
            super().restore_context_messages(session_id, messages)
            self._persist(self._sessions[session_id])

    def touch(self, session_id: str) -> None:
        with self._store_lock:
            self._persist(self._sessions[session_id])

    def bump_snapshot_version(self, session_id: str) -> int:
        with self._store_lock:
            version = super().bump_snapshot_version(session_id)
            self._persist(self._sessions[session_id])
            return version

    def set_context_maintenance(self, session_id: str, payload: dict[str, Any] | None) -> None:
        with self._store_lock:
            super().set_context_maintenance(session_id, payload)
            self._persist(self._sessions[session_id])

    def merge_context_state(self, session_id: str, updates: dict[str, Any]) -> None:
        if not updates:
            return
        with self._store_lock:
            super().merge_context_state(session_id, updates)
            self._persist(self._sessions[session_id])

    def _init_schema(self) -> None:
        with self._persist_lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    patient_id INTEGER,
                    snapshot_version INTEGER NOT NULL DEFAULT 0,
                    uploaded_assets TEXT NOT NULL DEFAULT '{}',
                    processed_files TEXT NOT NULL DEFAULT '{}',
                    pending_context BLOB NOT NULL,
                    pending_context_version INTEGER NOT NULL DEFAULT 1,
                    context_maintenance TEXT,
                    context_state TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS sessions_last_used_idx ON sessions(last_used_at)"
            )

    def _persist(self, meta: SessionMeta) -> None:
        now = datetime.now(timezone.utc).isoformat()
        pending_context = pickle.dumps(list(meta.pending_context_messages))
        with self._persist_lock:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, thread_id, scene, patient_id, snapshot_version,
                    uploaded_assets, processed_files, pending_context,
                    pending_context_version, context_maintenance, context_state,
                    created_at, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    scene = excluded.scene,
                    patient_id = excluded.patient_id,
                    snapshot_version = excluded.snapshot_version,
                    uploaded_assets = excluded.uploaded_assets,
                    processed_files = excluded.processed_files,
                    pending_context = excluded.pending_context,
                    pending_context_version = excluded.pending_context_version,
                    context_maintenance = excluded.context_maintenance,
                    context_state = excluded.context_state,
                    last_used_at = excluded.last_used_at
                """,
                (
                    meta.session_id,
                    meta.thread_id,
                    meta.scene,
                    meta.patient_id,
                    meta.snapshot_version,
                    json.dumps(meta.uploaded_assets, ensure_ascii=False),
                    json.dumps(meta.processed_files, ensure_ascii=False),
                    pending_context,
                    PENDING_CONTEXT_VERSION,
                    json.dumps(meta.context_maintenance, ensure_ascii=False)
                    if meta.context_maintenance is not None
                    else None,
                    json.dumps(meta.context_state, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def _purge_expired(self) -> None:
        if self._ttl_days is None:
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._ttl_days)).isoformat()
        with self._persist_lock:
            expired_ids = [
                row["session_id"]
                for row in self._conn.execute(
                    "SELECT session_id FROM sessions WHERE last_used_at < ?",
                    (cutoff,),
                )
            ]
            if not expired_ids:
                return
            self._conn.executemany(
                "DELETE FROM sessions WHERE session_id = ?",
                [(session_id,) for session_id in expired_ids],
            )
        with self._store_lock:
            for session_id in expired_ids:
                self._sessions.pop(session_id, None)
                self._run_locks.pop(session_id, None)
        logger.info(
            "purged %d expired sessions (older than %d days)",
            len(expired_ids),
            self._ttl_days,
        )

    def _rebuild_from_disk(self) -> None:
        with self._persist_lock:
            rows = list(self._conn.execute("SELECT * FROM sessions"))
        corrupt_session_ids: list[str] = []
        rebuilt: dict[str, SessionMeta] = {}
        for row in rows:
            try:
                meta = self._row_to_session_meta(row)
            except Exception as exc:
                session_id = str(row["session_id"])
                corrupt_session_ids.append(session_id)
                logger.warning("dropping corrupt persisted session %s: %s", session_id, exc)
                continue
            rebuilt[meta.session_id] = meta

        with self._store_lock:
            self._sessions.update(rebuilt)
            for session_id in rebuilt:
                self._run_locks[session_id] = Lock()

        if corrupt_session_ids:
            with self._persist_lock:
                self._conn.executemany(
                    "DELETE FROM sessions WHERE session_id = ?",
                    [(session_id,) for session_id in corrupt_session_ids],
                )

    def _row_to_session_meta(self, row: sqlite3.Row) -> SessionMeta:
        uploaded_assets = self._load_json_mapping(row["uploaded_assets"])
        processed_files = self._load_json_mapping(row["processed_files"])
        context_state = self._load_json_mapping(row["context_state"])
        context_maintenance = (
            self._load_json_mapping(row["context_maintenance"])
            if row["context_maintenance"] is not None
            else None
        )
        pending_context = pickle.loads(row["pending_context"])
        if not isinstance(pending_context, list):
            pending_context = []
        return SessionMeta(
            session_id=str(row["session_id"]),
            thread_id=str(row["thread_id"]),
            scene=str(row["scene"]),
            patient_id=row["patient_id"],
            snapshot_version=int(row["snapshot_version"]),
            uploaded_assets=uploaded_assets,
            processed_files=processed_files,
            pending_context_messages=pending_context,
            active_run_id=None,
            context_maintenance=context_maintenance,
            context_state=context_state,
        )

    @staticmethod
    def _load_json_mapping(value: Any) -> dict[str, Any]:
        loaded = json.loads(value or "{}")
        if not isinstance(loaded, dict):
            return {}
        return loaded
