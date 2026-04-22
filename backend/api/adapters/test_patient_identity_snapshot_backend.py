from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from backend.api.adapters.state_snapshot import build_message_history, build_recovery_snapshot
from backend.api.schemas.responses import RecoverySnapshot
from backend.api.services.session_store import SessionMeta


def test_recovery_snapshot_allows_patient_identity_field_shape() -> None:
    snapshot = RecoverySnapshot(
        snapshot_version=3,
        patient_identity={
            "patient_name": None,
            "patient_number": None,
            "identity_locked": False,
        },
    )

    assert snapshot.patient_identity == {
        "patient_name": None,
        "patient_number": None,
        "identity_locked": False,
    }


def test_build_recovery_snapshot_defaults_patient_identity_to_none_when_unset() -> None:
    session_meta = SessionMeta(session_id="sess_1", thread_id="thread_1", scene="patient")

    snapshot = build_recovery_snapshot(session_meta, {"messages": []})

    assert snapshot.patient_identity is None


def test_build_message_history_filters_internal_ai_control_messages() -> None:
    history = build_message_history(
        {
            "messages": [
                HumanMessage(content="user question"),
                AIMessage(content="[Router] internal route decision"),
                AIMessage(content="visible answer"),
            ]
        }
    )

    assert history.messages_total == 2
    assert [message.content for message in history.messages] == [
        "user question",
        "visible answer",
    ]
