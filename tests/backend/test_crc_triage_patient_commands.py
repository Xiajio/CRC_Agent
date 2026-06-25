from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService


def _crc_triage_assessment() -> dict[str, object]:
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
        "source_session_id": "sess_patient_1",
        "source_subflow": "crc_triage",
    }


def test_record_crc_triage_assessment_writes_event_record_and_snapshot_ref(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")

    result = commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=_crc_triage_assessment(),
        source_session_id="sess_patient_1",
    )

    assert result.record_id is not None
    assert result.asset_id is not None
    assert result.patient_version == 2
    assert result.projection_version == 2
    assert result.snapshot_changed is True

    with registry._connect() as connection:
        event = connection.execute(
            """
            SELECT event_type, event_payload_json, actor_type
            FROM patient_events
            WHERE patient_id = ? AND patient_version = ?
            """,
            (patient.patient_id, result.patient_version),
        ).fetchone()
        record = connection.execute(
            """
            SELECT record_type, document_type, normalized_payload_json, summary_text, source
            FROM patient_records
            WHERE record_id = ?
            """,
            (result.record_id,),
        ).fetchone()
        snapshot = connection.execute(
            """
            SELECT record_refs_json, asset_refs_json, source_event_ids_json
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (patient.patient_id,),
        ).fetchone()

    assert event["event_type"] == "patient.crc_triage_assessed"
    assert event["actor_type"] == "patient"
    assert json.loads(event["event_payload_json"])["assessment"]["source_subflow"] == "crc_triage"
    assert record["record_type"] == "crc_triage_assessment"
    assert record["document_type"] == "crc_triage_assessment"
    assert record["summary_text"] == "患者近两周反复便血，建议尽快门诊评估。"
    assert record["source"] == "patient_generated"
    assert json.loads(record["normalized_payload_json"])["chief_complaint"] == "反复便血"
    assert {"record_id": result.record_id, "document_type": "crc_triage_assessment"} in json.loads(
        snapshot["record_refs_json"]
    )
    assert result.asset_id in [ref["asset_id"] for ref in json.loads(snapshot["asset_refs_json"])]
    assert result.event_ids[0] in json.loads(snapshot["source_event_ids_json"])


def test_record_crc_triage_assessment_is_idempotent_by_session_and_payload(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    assessment = _crc_triage_assessment()

    first = commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=assessment,
        source_session_id="sess_patient_1",
    )
    second = commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=assessment,
        source_session_id="sess_patient_1",
    )

    assert second.reused is True
    assert second.patient_version == first.patient_version
    assert second.projection_version == first.projection_version
    assert second.asset_id == first.asset_id
    assert second.record_id == first.record_id
    assert second.event_ids == first.event_ids

    with registry._connect() as connection:
        event_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM patient_events
            WHERE patient_id = ? AND event_type = 'patient.crc_triage_assessed'
            """,
            (patient.patient_id,),
        ).fetchone()
        record_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM patient_records
            WHERE patient_id = ? AND record_type = 'crc_triage_assessment'
            """,
            (patient.patient_id,),
        ).fetchone()

    assert int(event_count["count"]) == 1
    assert int(record_count["count"]) == 1


def test_record_crc_triage_assessment_does_not_create_snapshot_alert(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")

    commands.record_crc_triage_assessment(
        patient_id=patient.patient_id,
        assessment=_crc_triage_assessment(),
        source_session_id="sess_patient_1",
    )

    assert registry.list_patient_alerts(patient.patient_id) == []
