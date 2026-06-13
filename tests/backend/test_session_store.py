from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.api.services.session_store import InMemorySessionStore


def test_run_lock_methods_return_false_for_missing_session() -> None:
    store = InMemorySessionStore()

    assert store.try_acquire_run_lock("sess_missing", "run-1") is False
    assert store.release_run_lock("sess_missing", "run-1") is False


def test_run_lock_methods_return_false_when_lock_entry_is_missing() -> None:
    store = InMemorySessionStore()
    meta = store.create_session()

    store._run_locks.pop(meta.session_id)

    assert store.try_acquire_run_lock(meta.session_id, "run-1") is False
    assert store.release_run_lock(meta.session_id, "run-1") is False
    assert store.get_session(meta.session_id).active_run_id is None


def test_pending_context_cleanup_methods_tolerate_missing_session() -> None:
    store = InMemorySessionStore()

    assert store.drain_context_messages("sess_missing") == []
    store.restore_context_messages("sess_missing", [HumanMessage(content="queued")])
    assert store.get_session("sess_missing") is None


def test_pending_context_methods_preserve_existing_session_behavior() -> None:
    store = InMemorySessionStore()
    meta = store.create_session()
    message = HumanMessage(content="queued")

    store.enqueue_context_message(meta.session_id, message)
    assert store.drain_context_messages(meta.session_id) == [message]
    assert store.drain_context_messages(meta.session_id) == []

    store.restore_context_messages(meta.session_id, [message])
    assert store.get_session(meta.session_id).pending_context_messages == [message]
