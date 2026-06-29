from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import sessions
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def _assessment() -> dict[str, object]:
    return {
        "record_type": "crc_triage_assessment",
        "chief_complaint": "反复便血",
        "symptom_group": "CRC相关门诊分诊",
        "risk_level": "medium",
        "disposition": "urgent_gi_clinic",
        "red_flags": ["rectal_bleeding"],
        "known_crc_signals": {"rectal_bleeding": True},
        "suggested_tests": ["血常规", "肠镜"],
        "missing_information": [],
        "qa_summary": [],
        "patient_summary": "建议尽快消化专科评估。",
        "next_step": "urgent_gi_clinic",
        "source_session_id": "sess_patient",
        "source_subflow": "crc_triage",
    }


def _client(tmp_path, monkeypatch):
    app = FastAPI()
    store = InMemorySessionStore()
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient")
    patient_session = store.create_session(scene="patient", patient_id=created.patient_id)
    doctor_session = store.create_session(scene="doctor")
    no_patient_session = store.create_session(scene="patient")
    commands.record_crc_triage_assessment(
        patient_id=created.patient_id,
        assessment=_assessment(),
        source_session_id=patient_session.session_id,
    )

    monkeypatch.setattr(sessions, "session_store", store)
    monkeypatch.setattr(sessions, "patient_registry_service", registry)
    monkeypatch.setattr(sessions, "patient_command_service", commands)
    app.include_router(sessions.router)
    return TestClient(app), patient_session.session_id, doctor_session.session_id, no_patient_session.session_id


def test_get_session_patient_records_returns_current_patient_records(tmp_path, monkeypatch) -> None:
    client, session_id, _doctor_session_id, _no_patient_session_id = _client(tmp_path, monkeypatch)

    response = client.get(f"/api/sessions/{session_id}/patient-records")

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["record_type"] == "crc_triage_assessment"
    assert body["items"][0]["patient_id"] > 0


def test_get_session_care_cards_returns_current_patient_guidance(tmp_path, monkeypatch) -> None:
    client, session_id, _doctor_session_id, _no_patient_session_id = _client(tmp_path, monkeypatch)

    response = client.get(f"/api/sessions/{session_id}/care-cards")

    assert response.status_code == 200
    body = response.json()
    assert "留意便血或黑便是否加重" in body["focusMetrics"]
    assert "尽快预约消化专科门诊" in body["periodicChecks"]
    assert body["dailyActions"]


def test_session_patient_records_rejects_doctor_session(tmp_path, monkeypatch) -> None:
    client, _session_id, doctor_session_id, _no_patient_session_id = _client(tmp_path, monkeypatch)

    response = client.get(f"/api/sessions/{doctor_session_id}/patient-records")

    assert response.status_code == 409
    assert response.json()["detail"] == "NOT_PATIENT_SESSION"


def test_session_care_cards_requires_patient_identity(tmp_path, monkeypatch) -> None:
    client, _session_id, _doctor_session_id, no_patient_session_id = _client(tmp_path, monkeypatch)

    response = client.get(f"/api/sessions/{no_patient_session_id}/care-cards")

    assert response.status_code == 409
    assert response.json()["detail"] == "PATIENT_IDENTITY_NOT_FOUND"
