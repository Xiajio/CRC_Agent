from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import doctor_review
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


AVAILABLE_ACTIONS = [
    "accept",
    "edit",
    "reject",
    "escalate",
    "request_evidence",
    "mark_unsafe",
]


def _assessment(patient_session_id: str) -> dict[str, object]:
    return {
        "assessment_id": "crc_assessment_abc123",
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
    unbound_doctor_session = session_store.create_session(scene="doctor")

    commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=_assessment(patient_session.session_id),
        source_session_id=patient_session.session_id,
    )

    app.state.runtime = SimpleNamespace(
        session_store=session_store,
        patient_registry_service=registry,
    )
    app.include_router(doctor_review.router)
    return (
        TestClient(app),
        patient.patient_id,
        patient_session.session_id,
        doctor_session.session_id,
        unbound_doctor_session.session_id,
        registry,
    )


def test_doctor_review_rejects_patient_session(tmp_path) -> None:
    client, _patient_id, patient_session_id, _doctor_session_id, _unbound_id, _registry = (
        _client(tmp_path)
    )

    response = client.get(f"/api/sessions/{patient_session_id}/doctor-review")

    assert response.status_code == 409
    assert response.json()["detail"] == "NOT_DOCTOR_SESSION"


def test_doctor_review_rejects_unbound_doctor_session(tmp_path) -> None:
    client, _patient_id, _patient_session_id, _doctor_session_id, unbound_id, _registry = (
        _client(tmp_path)
    )

    response = client.get(f"/api/sessions/{unbound_id}/doctor-review")

    assert response.status_code == 409
    assert response.json()["detail"] == "PATIENT_BINDING_REQUIRED"


def test_doctor_review_returns_timeline_assertions_draft_and_actions(tmp_path) -> None:
    client, patient_id, _patient_session_id, doctor_session_id, _unbound_id, _registry = (
        _client(tmp_path)
    )

    response = client.get(f"/api/sessions/{doctor_session_id}/doctor-review")

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == patient_id
    assert body["session_id"] == doctor_session_id
    assert body["feature_flag"] == "doctor_review_cockpit_v0"
    assert body["available_actions"] == AVAILABLE_ACTIONS

    assert len(body["timeline"]) == 1
    timeline_item = body["timeline"][0]
    assert timeline_item["kind"] == "crc_triage_assessment"
    assert timeline_item["title"] == "Rectal bleeding with weight loss needs urgent GI review."
    assert timeline_item["assertion_refs"]

    facts = [assertion["normalized_fact"] for assertion in body["assertions"]]
    assert {
        "type": "condition_signal",
        "name": "rectal_bleeding",
        "value": True,
    } in facts
    assert {
        "type": "symptom",
        "name": "weight_loss",
        "value": True,
    } in facts
    assert {
        "type": "risk_disposition",
        "name": "disposition",
        "value": "urgent_gi_clinic",
    } in facts
    assert {
        "type": "missing_information",
        "name": "family_history",
        "value": True,
    } in facts
    assert {
        "type": "safety_rule_match",
        "name": "rectal_bleeding_age_escalation",
        "value": True,
    } in facts

    assert body["draft"]["draft_id"] == f"draft_crc_review_{patient_id}_latest"
    draft_sections = body["draft"]["sections"]
    assert all("text" in section for section in draft_sections)
    assert all("provenance" in section for section in draft_sections)
    assert all("verification_status" in section for section in draft_sections)
    assert all("body" not in section for section in draft_sections)
    assert all("provenance_refs" not in section for section in draft_sections)
    traceable_section = next(
        section
        for section in draft_sections
        if section["verification_status"] == "traceable"
    )
    unverified_section = next(
        section
        for section in draft_sections
        if section["verification_status"] == "model_generated_unverified"
    )
    assert traceable_section["text"]
    assert traceable_section["provenance"]
    assert {
        "kind": "clinical_assertion",
        "assertion_id": timeline_item["assertion_refs"][0],
        "record_id": timeline_item["item_id"],
        "safety_policy_version": "crc_safety_policy_v0",
    } in traceable_section["provenance"]
    assert unverified_section["text"]
    assert unverified_section["provenance"] == []


def test_doctor_review_rejects_missing_session(tmp_path) -> None:
    client, _patient_id, _patient_session_id, _doctor_session_id, _unbound_id, _registry = (
        _client(tmp_path)
    )

    response = client.get("/api/sessions/sess_missing/doctor-review")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_doctor_review_returns_404_when_bound_patient_was_deleted(tmp_path) -> None:
    client, patient_id, _patient_session_id, doctor_session_id, _unbound_id, registry = (
        _client(tmp_path)
    )
    registry.delete_patient(patient_id)

    response = client.get(f"/api/sessions/{doctor_session_id}/doctor-review")

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found"
