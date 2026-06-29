from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import crc_triage
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def _assessment() -> dict[str, object]:
    return {
        "record_type": "crc_triage_assessment",
        "chief_complaint": "反复便血",
        "symptom_group": "便血与排便习惯改变",
        "risk_level": "medium",
        "disposition": "urgent_gi_clinic",
        "red_flags": ["rectal_bleeding"],
        "known_crc_signals": {"rectal_bleeding": True},
        "suggested_tests": ["肠镜", "血常规"],
        "missing_information": ["家族史"],
        "qa_summary": [],
        "patient_summary": "患者近两周反复便血，建议尽快门诊评估。",
        "next_step": "urgent_gi_clinic",
        "source_session_id": "sess_patient",
        "source_subflow": "crc_triage",
    }


def _client(tmp_path):
    app = FastAPI()
    store = InMemorySessionStore()
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient")
    patient_session = store.create_session(scene="patient", patient_id=created.patient_id)
    doctor_session = store.create_session(scene="doctor")
    no_patient_session = store.create_session(scene="patient")
    app.state.runtime = SimpleNamespace(
        session_store=store,
        patient_command_service=commands,
    )
    app.include_router(crc_triage.router)
    return (
        TestClient(app),
        patient_session.session_id,
        doctor_session.session_id,
        no_patient_session.session_id,
        registry,
    )


def test_save_crc_triage_assessment_returns_record_result(tmp_path) -> None:
    client, session_id, _doctor_session_id, _no_patient_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{session_id}/crc-triage/assessments",
        json={"assessment": _assessment()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] > 0
    assert body["record_id"] > 0
    assert body["event_ids"]
    assert body["reused"] is False


def test_save_crc_triage_assessment_persists_rich_payload_fields(tmp_path) -> None:
    client, session_id, _doctor_session_id, _no_patient_session_id, registry = _client(tmp_path)
    payload = _assessment()
    payload["qa_summary"] = [
        {
            "stage": "vitals",
            "question_id": "vitals_shock_or_consciousness",
            "question": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
            "answer": "没有",
        }
    ]
    payload["node_results"] = [
        {
            "stage": "vitals",
            "title": "节点1：生命体征评估",
            "risk_level": "生命体征平稳",
            "summary": "未识别到意识异常、休克表现、明显心率或呼吸异常。",
            "next_step": "进入节点2：全系统危险信号筛查。",
        }
    ]
    payload["protocol_state"] = {"stage": "final", "active_inquiry": False}

    response = client.post(
        f"/api/sessions/{session_id}/crc-triage/assessments",
        json={"assessment": payload},
    )

    assert response.status_code == 200
    record = registry.list_patient_records(response.json()["patient_id"])[0]
    normalized_payload = json.loads(record["normalized_payload_json"])
    assert normalized_payload["qa_summary"][0]["question_id"] == "vitals_shock_or_consciousness"
    assert normalized_payload["node_results"][0]["stage"] == "vitals"
    assert normalized_payload["protocol_state"]["stage"] == "final"


def test_save_crc_triage_assessment_rejects_doctor_session(tmp_path) -> None:
    client, _session_id, doctor_session_id, _no_patient_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{doctor_session_id}/crc-triage/assessments",
        json={"assessment": _assessment()},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "NOT_PATIENT_SESSION"


def test_save_crc_triage_assessment_requires_patient_identity(tmp_path) -> None:
    client, _session_id, _doctor_session_id, no_patient_session_id, _registry = _client(tmp_path)

    response = client.post(
        f"/api/sessions/{no_patient_session_id}/crc-triage/assessments",
        json={"assessment": _assessment()},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "PATIENT_IDENTITY_NOT_FOUND"


def test_save_crc_triage_assessment_requires_crc_subflow(tmp_path) -> None:
    client, session_id, _doctor_session_id, _no_patient_session_id, _registry = _client(tmp_path)
    payload = _assessment()
    payload["source_subflow"] = "other"

    response = client.post(
        f"/api/sessions/{session_id}/crc-triage/assessments",
        json={"assessment": payload},
    )

    assert response.status_code == 422
