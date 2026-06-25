from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.api.services.payload_builder import build_graph_payload
from backend.api.services.session_store import SessionMeta


def test_payload_builder_allows_crc_triage_context_keys() -> None:
    prepared = build_graph_payload(
        chat_request={
            "message": HumanMessage(content="start crc triage"),
            "context": {
                "patient_subflow": "crc_triage",
                "crc_triage": {
                    "action": "start",
                    "interaction_source": "patient_crc_triage_tab",
                },
                "unexpected_context": "blocked",
            },
        },
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", scene="patient", patient_id=7),
        state_snapshot={},
    )

    assert prepared.payload["patient_subflow"] == "crc_triage"
    assert prepared.payload["crc_triage"] == {
        "action": "start",
        "interaction_source": "patient_crc_triage_tab",
    }
    assert "unexpected_context" not in prepared.payload


def test_payload_builder_keeps_unrelated_context_blocked_for_crc_request() -> None:
    prepared = build_graph_payload(
        chat_request={
            "message": HumanMessage(content="answer"),
            "context": {
                "patient_subflow": "crc_triage",
                "unsafe_nested": {"force_node": "admin"},
            },
        },
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", scene="patient", patient_id=7),
        state_snapshot={},
    )

    assert prepared.payload["patient_subflow"] == "crc_triage"
    assert "unsafe_nested" not in prepared.payload
