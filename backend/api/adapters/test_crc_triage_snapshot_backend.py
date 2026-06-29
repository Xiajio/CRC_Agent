from __future__ import annotations

from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.services.session_store import SessionMeta


def test_recovery_snapshot_preserves_crc_triage_protocol_state_from_top_level_fields() -> None:
    session_meta = SessionMeta(session_id="sess_crc", thread_id="thread_crc", scene="patient")

    snapshot = build_recovery_snapshot(
        session_meta,
        {
            "messages": [],
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {"action": "answer", "question_id": "vitals_shock_or_consciousness"},
            "crc_triage_state": {
                "stage": "vitals",
                "active_inquiry": True,
                "current_question": {
                    "id": "vitals_shock_or_consciousness",
                    "stage": "vitals",
                    "text": "current CRC question",
                    "options": ["no", "yes", "unknown"],
                },
            },
            "active_inquiry": True,
            "inquiry_type": "crc_triage",
            "inquiry_message": "current CRC question",
            "triage_current_field": "vitals_shock_or_consciousness",
            "triage_pending_fields": ["vitals_shock_or_consciousness"],
            "missing_critical_data": ["test result"],
            "findings": {},
        },
    )

    findings = snapshot.findings
    assert findings["patient_subflow"] == "crc_triage"
    assert findings["source_subflow"] == "crc_triage"
    assert findings["crc_triage"]["question_id"] == "vitals_shock_or_consciousness"
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert findings["active_inquiry"] is True
    assert findings["inquiry_type"] == "crc_triage"
    assert findings["triage_pending_fields"] == ["vitals_shock_or_consciousness"]
    assert findings["missing_critical_data"] == ["test result"]
