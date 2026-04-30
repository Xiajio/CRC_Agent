from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class SessionMeta:
    session_id: str
    thread_id: str
    scene: str = "doctor"
    patient_id: int | None = None
    snapshot_version: int = 0
    uploaded_assets: dict[str, Any] = field(default_factory=dict)
    processed_files: dict[str, Any] = field(default_factory=dict)
    pending_context_messages: list[Any] = field(default_factory=list)
    active_run_id: str | None = None
    context_maintenance: dict[str, Any] | None = None
    context_state: dict[str, Any] = field(default_factory=dict)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionMeta] = {}
        self._run_locks: dict[str, Lock] = {}
        self._store_lock = Lock()

    def reset(self) -> None:
        with self._store_lock:
            self._sessions.clear()
            self._run_locks.clear()

    def create_session(self, *, scene: str = "doctor", patient_id: int | None = None) -> SessionMeta:
        session_id = f"sess_{uuid4().hex}"
        thread_id = f"thread_{uuid4().hex}"
        meta = SessionMeta(
            session_id=session_id,
            thread_id=thread_id,
            scene=scene,
            patient_id=patient_id,
        )
        with self._store_lock:
            self._sessions[session_id] = meta
            self._run_locks[session_id] = Lock()
        return meta

    def get_session(self, session_id: str) -> SessionMeta | None:
        with self._store_lock:
            return self._sessions.get(session_id)

    def find_uploaded_asset(self, asset_id: str) -> tuple[SessionMeta, dict[str, Any]] | None:
        with self._store_lock:
            for meta in self._sessions.values():
                asset = meta.uploaded_assets.get(asset_id)
                if isinstance(asset, dict):
                    return meta, asset
        return None

    def rotate_thread(self, session_id: str, *, clear_patient_id: bool = False) -> SessionMeta:
        with self._store_lock:
            meta = self._sessions[session_id]
            meta.thread_id = f"thread_{uuid4().hex}"
            if clear_patient_id:
                meta.patient_id = None
            meta.context_maintenance = None
            meta.context_state.clear()
            meta.pending_context_messages.clear()
            return meta

    def set_patient_id(
        self,
        session_id: str,
        patient_id: int | None,
        *,
        allow_replace: bool = False,
    ) -> SessionMeta:
        with self._store_lock:
            meta = self._sessions[session_id]
            if not allow_replace and meta.patient_id not in {None, patient_id}:
                raise ValueError("Session already bound to a different patient")
            meta.patient_id = patient_id
            return meta

    def bind_patient(self, session_id: str, patient_id: int) -> SessionMeta:
        with self._store_lock:
            meta = self._sessions[session_id]
            if meta.scene != "doctor":
                raise ValueError("Only doctor sessions can bind patients")
            if meta.patient_id not in {None, patient_id}:
                raise ValueError("Session already bound to a different patient")
            meta.patient_id = patient_id
            return meta

    def try_acquire_run_lock(self, session_id: str, run_id: str) -> bool:
        meta = self._sessions[session_id]
        lock = self._run_locks[session_id]
        if not lock.acquire(blocking=False):
            return False
        meta.active_run_id = run_id
        return True

    def release_run_lock(self, session_id: str, run_id: str) -> bool:
        meta = self._sessions[session_id]
        lock = self._run_locks[session_id]
        if lock.locked() and meta.active_run_id == run_id:
            meta.active_run_id = None
            lock.release()
            return True
        return False

    def enqueue_context_message(self, session_id: str, message: Any) -> None:
        self._sessions[session_id].pending_context_messages.append(message)

    def drain_context_messages(self, session_id: str) -> list[Any]:
        meta = self._sessions[session_id]
        drained = list(meta.pending_context_messages)
        meta.pending_context_messages.clear()
        return drained

    def restore_context_messages(self, session_id: str, messages: list[Any]) -> None:
        self._sessions[session_id].pending_context_messages.extend(messages)

    def bump_snapshot_version(self, session_id: str) -> int:
        with self._store_lock:
            meta = self._sessions[session_id]
            meta.snapshot_version += 1
            return meta.snapshot_version

    def set_context_maintenance(self, session_id: str, payload: dict[str, Any] | None) -> None:
        with self._store_lock:
            meta = self._sessions[session_id]
            meta.context_maintenance = deepcopy(payload) if isinstance(payload, dict) else None

    def set_patient_context_cache(self, session_id: str, cache: dict[str, Any]) -> None:
        self.merge_context_state(
            session_id,
            {"patient_context_cache": cache, "medical_card": None},
        )

    def clear_legacy_medical_card(self, session_id: str) -> None:
        self.merge_context_state(session_id, {"medical_card": None})

    def merge_context_state(self, session_id: str, updates: dict[str, Any]) -> None:
        if not updates:
            return
        with self._store_lock:
            meta = self._sessions[session_id]
            next_state = dict(meta.context_state)
            for key, value in updates.items():
                if value is None:
                    next_state.pop(key, None)
                else:
                    next_state[key] = deepcopy(value)
            meta.context_state = next_state
