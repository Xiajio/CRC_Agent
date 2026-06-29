from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import patient_registry as patient_registry_routes
from backend.api.services.patient_care_cards import build_patient_care_cards
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService


def test_crc_triage_assessment_persists_traceability_fields_and_care_cards(
    tmp_path: Path,
) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    assessment = {
        "assessment_id": "crc_assessment_test_001",
        "record_type": "crc_triage_assessment",
        "chief_complaint": "反复便血",
        "symptom_group": "便血与排便习惯改变",
        "risk_level": "high",
        "disposition": "urgent_gi_clinic",
        "red_flags": ["rectal_bleeding"],
        "known_crc_signals": {"rectal_bleeding": True},
        "suggested_tests": ["肠镜", "血常规"],
        "missing_information": ["家族史"],
        "qa_summary": [],
        "node_results": [],
        "protocol_state": {"stage": "final", "active_inquiry": False},
        "patient_summary": "患者近两周反复便血，建议尽快门诊评估。",
        "next_step": "urgent_gi_clinic",
        "source_session_id": "sess_patient_1",
        "source_subflow": "crc_triage",
        "safety_policy_version": "crc_safety_policy_v0",
        "matched_rules": ["rectal_bleeding_age_escalation"],
        "hard_fail_flags": ["rectal_bleeding_age_escalation"],
        "patient_message_key": "urgent_clinical_review",
    }

    result = commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=assessment,
        source_session_id="sess_patient_1",
    )

    records = registry.list_patient_records(patient.patient_id)
    assert len(records) == 1
    record = records[0]
    payload = json.loads(record["normalized_payload_json"])
    care_cards = build_patient_care_cards(records)

    assert result.record_id == record["record_id"]
    assert payload["assessment_id"] == "crc_assessment_test_001"
    assert payload["safety_policy_version"] == "crc_safety_policy_v0"
    assert payload["matched_rules"] == ["rectal_bleeding_age_escalation"]
    assert "留意便血或黑便是否加重" in care_cards["focusMetrics"]
    assert "尽快预约消化专科门诊" in care_cards["periodicChecks"]

    with registry._connect() as connection:
        event = connection.execute(
            """
            SELECT event_payload_json
            FROM patient_events
            WHERE event_id = ?
            """,
            (result.event_ids[0],),
        ).fetchone()
        snapshot = connection.execute(
            """
            SELECT record_refs_json, source_event_ids_json
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (patient.patient_id,),
        ).fetchone()

    assert event is not None
    event_payload = json.loads(event["event_payload_json"])
    assert event_payload["assessment"]["assessment_id"] == "crc_assessment_test_001"

    assert snapshot is not None
    assert {
        "record_id": result.record_id,
        "document_type": "crc_triage_assessment",
    } in json.loads(snapshot["record_refs_json"])
    assert result.event_ids[0] in json.loads(snapshot["source_event_ids_json"])


def test_patient_registry_records_endpoint_returns_derived_clinical_assertions(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    app.state.runtime = SimpleNamespace(patient_registry_service=registry)
    app.include_router(patient_registry_routes.router)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment={
            "record_type": "crc_triage_assessment",
            "known_crc_signals": {"rectal_bleeding": True},
            "source_session_id": "sess_patient_1",
        },
        source_session_id="sess_patient_1",
    )
    client = TestClient(app)

    response = client.get(f"/api/patient-registry/patients/{patient.patient_id}/records")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["clinical_assertions"][0]["normalized_fact"] == {
        "type": "condition_signal",
        "name": "rectal_bleeding",
        "value": True,
    }
    assert item["clinical_assertion_refs"] == [
        assertion["assertion_id"] for assertion in item["clinical_assertions"]
    ]
