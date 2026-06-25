from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.api.services.payload_builder import build_graph_payload
from backend.api.services.session_store import SessionMeta
from src.graph_builder import route_after_patient_intent
from src.nodes.triage_nodes import node_clinical_entry_resolver, node_outpatient_triage
from src.state import CRCAgentState


def test_stale_crc_findings_do_not_force_future_patient_intent_route() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="hello")],
        findings={
            "source_subflow": "crc_triage",
            "patient_subflow": "crc_triage",
            "user_intent": "general_chat",
        },
    )

    assert route_after_patient_intent(state) == "general_chat"


def test_payload_builder_clears_crc_subflow_markers_without_current_context() -> None:
    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="normal patient question")},
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", scene="patient", patient_id=7),
        state_snapshot={
            "findings": {
                "patient_subflow": "crc_triage",
                "source_subflow": "crc_triage",
                "crc_triage": {"action": "start"},
                "triage_summary": "previous crc summary",
            },
        },
    )

    assert prepared.payload["patient_subflow"] is None
    assert prepared.payload["source_subflow"] is None
    assert prepared.payload["crc_triage"] == {}
    assert prepared.payload["findings"]["patient_subflow"] is None
    assert prepared.payload["findings"]["source_subflow"] is None
    assert prepared.payload["findings"]["crc_triage"] == {}


def test_clinical_entry_resolver_clears_stale_crc_findings_for_normal_turn() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="normal patient question")],
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {"action": "start"},
        },
    )

    result = node_clinical_entry_resolver(show_thinking=False)(state)

    assert result["findings"]["patient_subflow"] is None
    assert result["findings"]["source_subflow"] is None
    assert result["findings"]["crc_triage"] == {}


def test_clinical_entry_resolver_does_not_continue_stale_crc_active_inquiry_for_normal_turn() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="normal patient question")],
        findings={
            "encounter_track": "outpatient_triage",
            "active_inquiry": True,
            "inquiry_type": "outpatient_triage",
            "inquiry_message": "old crc question",
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {"action": "start"},
        },
    )

    result = node_clinical_entry_resolver(show_thinking=False)(state)

    assert result["encounter_track"] == "crc_clinical"
    assert result["findings"]["active_inquiry"] is False
    assert result["findings"]["patient_subflow"] is None
    assert result["findings"]["source_subflow"] is None


def test_outpatient_triage_clears_stale_crc_findings_for_normal_turn() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="I have a headache today")],
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {"action": "start"},
        },
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    assert result["findings"]["patient_subflow"] is None
    assert result["findings"]["source_subflow"] is None
    assert result["findings"]["crc_triage"] == {}
