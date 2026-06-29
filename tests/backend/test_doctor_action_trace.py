from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import doctor_review
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def _assessment(
    patient_session_id: str,
    *,
    assessment_id: str = "crc_assessment_abc123",
) -> dict[str, object]:
    return {
        "assessment_id": assessment_id,
        "record_type": "crc_triage_assessment",
        "chief_complaint": "rectal bleeding",
        "symptom_group": "bowel habit change",
        "risk_level": "high",
        "disposition": "urgent_gi_clinic",
        "red_flags": ["weight_loss"],
        "known_crc_signals": {"rectal_bleeding": True},
        "suggested_tests": ["colonoscopy"],
        "missing_information": ["family_history"],
        "qa_summary": [],
        "node_results": [],
        "protocol_state": {"stage": "final", "active_inquiry": False},
        "patient_summary": "Rectal bleeding with weight loss needs urgent GI review.",
        "next_step": "urgent_gi_clinic",
        "source_session_id": patient_session_id,
        "source_subflow": "crc_triage",
        "safety_policy_version": "crc_safety_policy_v0",
        "matched_rules": ["rectal_bleeding_age_escalation"],
        "hard_fail_flags": [],
        "patient_message_key": "urgent_clinical_review",
    }


def _client(tmp_path):
    app = FastAPI()
    session_store = InMemorySessionStore()
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)

    patient_session = session_store.create_session(scene="patient")
    patient = commands.create_patient(created_by_session_id=patient_session.session_id)
    session_store.set_patient_id(
        patient_session.session_id,
        patient.patient_id,
        allow_replace=True,
    )
    doctor_session = session_store.create_session(
        scene="doctor",
        patient_id=patient.patient_id,
    )

    app.state.runtime = SimpleNamespace(
        session_store=session_store,
        patient_registry_service=registry,
        patient_command_service=commands,
    )
    app.include_router(doctor_review.router)
    return TestClient(app), patient.patient_id, doctor_session.session_id, registry


def _valid_edit_payload() -> dict[str, object]:
    return {
        "action_type": "edit",
        "target_object": "risk_summary",
        "target_refs": {
            "draft_id": "draft_crc_review_1_latest",
            "assertion_id": "assertion_rectal_bleeding",
        },
        "before_after": {
            "before": "Urgent clinic review is suggested.",
            "after": "Urgent GI clinic review is required.",
        },
        "reason_code": "workflow_mismatch",
    }


def _valid_accept_payload() -> dict[str, object]:
    return {
        "action_type": "accept",
        "target_object": None,
        "target_refs": {
            "draft_id": "draft_crc_review_1_latest",
        },
        "reason_code": "workflow_mismatch",
    }


def _latest_event(registry: PatientRegistryService, patient_id: int) -> dict[str, object]:
    with registry.transaction() as connection:
        row = connection.execute(
            """
            SELECT event_type, event_payload_json, actor_type, idempotency_key
            FROM patient_events
            WHERE patient_id = ?
            ORDER BY patient_version DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _snapshot_row(registry: PatientRegistryService, patient_id: int) -> dict[str, object]:
    with registry.transaction() as connection:
        row = connection.execute(
            """
            SELECT patient_version, projection_version, updated_at, source_event_ids_json
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _doctor_action_event_count(
    registry: PatientRegistryService,
    patient_id: int,
) -> int:
    with registry.transaction() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM patient_events
            WHERE patient_id = ? AND event_type = 'doctor.action_trace_recorded'
            """,
            (patient_id,),
        ).fetchone()
    assert row is not None
    return int(row["count"])


def test_record_doctor_action_trace_stores_append_only_event(tmp_path) -> None:
    client, patient_id, doctor_session_id, registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=_valid_edit_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == patient_id
    assert body["trace"]["action_type"] == "edit"
    assert body["trace"]["trace_id"].startswith("doctor_trace_")
    assert body["trace"]["patient_id"] == patient_id
    assert body["trace"]["session_id"] == doctor_session_id
    assert "timestamp" in body["trace"]
    assert "created_at" not in body["trace"]
    assert body["event_ids"]
    assert body["snapshot_changed"] is False

    latest_event = _latest_event(registry, patient_id)
    assert latest_event["event_type"] == "doctor.action_trace_recorded"
    assert latest_event["actor_type"] == "physician_reviewer"
    assert latest_event["idempotency_key"] is None
    payload = json.loads(str(latest_event["event_payload_json"]))
    assert payload["trace"]["deidentified"] is True
    assert payload["trace"]["reason_code"] == "workflow_mismatch"

    second = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=_valid_edit_payload(),
    )
    assert second.status_code == 200
    assert second.json()["event_ids"] != body["event_ids"]


def test_record_doctor_action_trace_returns_stable_null_optional_fields(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=_valid_accept_payload(),
    )

    assert response.status_code == 200
    trace = response.json()["trace"]
    assert trace["target_object"] is None
    assert trace["before_after"] is None


def test_record_doctor_action_trace_accepts_mark_unsafe_assertion_target(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json={
            "action_type": "mark_unsafe",
            "target_object": "assertion",
            "target_refs": {"assertion_id": "assertion-1"},
            "reason_code": "unsafe_disposition",
        },
    )

    assert response.status_code == 200
    trace = response.json()["trace"]
    assert trace["action_type"] == "mark_unsafe"
    assert trace["target_object"] == "assertion"
    assert trace["target_refs"]["assertion_id"] == "assertion-1"


def test_record_doctor_action_trace_accepts_mark_unsafe_draft_risk_summary(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json={
            "action_type": "mark_unsafe",
            "target_object": "draft.risk_summary",
            "target_refs": {"draft_id": "draft_crc_review_1_latest"},
            "reason_code": "missing_red_flag",
        },
    )

    assert response.status_code == 200
    assert response.json()["trace"]["target_object"] == "draft.risk_summary"


def test_record_doctor_action_trace_rejects_mark_unsafe_nonclinical_target(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json={
            "action_type": "mark_unsafe",
            "target_object": "draft.tone",
            "target_refs": {"draft_id": "draft_crc_review_1_latest"},
            "reason_code": "unsafe_disposition",
        },
    )

    assert response.status_code == 422


def test_record_doctor_action_trace_rejects_cross_patient_record_ref(tmp_path) -> None:
    client, patient_a_id, doctor_session_id, registry = _client(tmp_path)
    commands = PatientCommandService(registry)
    patient_b = commands.create_patient(created_by_session_id="patient_b_session")
    patient_b_record = commands.record_crc_triage_assessment(
        patient_id=patient_b.patient_id,
        assessment=_assessment("patient_b_session", assessment_id="crc_assessment_b"),
        source_session_id="patient_b_session",
    )
    before_count = _doctor_action_event_count(registry, patient_a_id)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json={
            "action_type": "request_evidence",
            "target_object": "record",
            "target_refs": {"record_id": str(patient_b_record.record_id)},
            "reason_code": "unsupported_claim",
        },
    )

    assert response.status_code == 422
    assert _doctor_action_event_count(registry, patient_a_id) == before_count


def test_record_doctor_action_trace_rejects_invalid_assessment_ref(tmp_path) -> None:
    client, patient_id, doctor_session_id, registry = _client(tmp_path)
    before_count = _doctor_action_event_count(registry, patient_id)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json={
            "action_type": "request_evidence",
            "target_object": "assessment",
            "target_refs": {"assessment_id": "crc_assessment_missing"},
            "reason_code": "unsupported_claim",
        },
    )

    assert response.status_code == 422
    assert _doctor_action_event_count(registry, patient_id) == before_count


def test_patient_command_service_records_plain_dict_trace_payload(tmp_path) -> None:
    _client_instance, patient_id, doctor_session_id, registry = _client(tmp_path)
    commands = PatientCommandService(registry)
    trace_payload = {
        "trace_id": "doctor_trace_plain_dict",
        "patient_id": patient_id,
        "session_id": doctor_session_id,
        "action_type": "accept",
        "target_object": None,
        "target_refs": {"draft_id": "draft_crc_review_1_latest"},
        "before_after": None,
        "reason_code": "workflow_mismatch",
        "reviewer_role": "physician_reviewer",
        "deidentified": True,
        "timestamp": "2026-06-29T00:00:00Z",
    }

    result = commands.record_doctor_action_trace(
        patient_id=patient_id,
        trace=trace_payload,
        source_session_id=doctor_session_id,
    )

    assert result.event_ids
    latest_event = _latest_event(registry, patient_id)
    payload = json.loads(str(latest_event["event_payload_json"]))
    assert payload["trace"] == trace_payload


def test_patient_command_service_does_not_import_api_schema() -> None:
    service_source = Path("backend/api/services/patient_commands.py").read_text(
        encoding="utf-8"
    )

    assert "backend.api.schemas.doctor_action_trace" not in service_source


def test_record_doctor_action_trace_rejects_unknown_reason_code(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)
    payload = {**_valid_edit_payload(), "reason_code": "unknown_reason"}

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=payload,
    )

    assert response.status_code == 422


def test_record_doctor_action_trace_rejects_edit_without_before_after(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)
    payload = _valid_edit_payload()
    payload.pop("before_after")

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=payload,
    )

    assert response.status_code == 422


def test_record_doctor_action_trace_rejects_missing_target(tmp_path) -> None:
    client, _patient_id, doctor_session_id, _registry = _client(tmp_path)
    payload = _valid_edit_payload()
    payload["target_object"] = None
    payload["target_refs"] = {}

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=payload,
    )

    assert response.status_code == 422


def test_record_doctor_action_trace_does_not_update_snapshot_projection(tmp_path) -> None:
    client, patient_id, doctor_session_id, registry = _client(tmp_path)
    before = _snapshot_row(registry, patient_id)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/doctor-review/action-traces",
        json=_valid_edit_payload(),
    )

    assert response.status_code == 200
    after = _snapshot_row(registry, patient_id)
    assert after == before
    assert response.json()["projection_version"] == before["projection_version"]
