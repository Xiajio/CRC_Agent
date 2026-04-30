from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table_name})")}


def test_patient_event_schema_is_initialized(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")

    with service._connect() as connection:
        event_columns = _table_columns(connection, "patient_events")
        asset_columns = _table_columns(connection, "patient_assets")
        record_columns = _table_columns(connection, "patient_records")
        snapshot_columns = _table_columns(connection, "patient_snapshots")
        state_columns = _table_columns(connection, "patient_projection_state")

    assert {
        "event_id",
        "patient_id",
        "patient_version",
        "event_type",
        "event_payload_json",
        "actor_type",
        "actor_id",
        "source_session_id",
        "idempotency_key",
        "causation_id",
        "correlation_id",
        "created_at",
    }.issubset(event_columns)
    assert {"upload_event_id", "storage_status", "parse_status", "patient_version"}.issubset(asset_columns)
    assert {"source_event_id", "patient_version"}.issubset(record_columns)
    assert {
        "patient_id",
        "patient_version",
        "projection_version",
        "medical_card_snapshot_json",
        "summary_json",
        "active_alerts_json",
        "record_refs_json",
        "asset_refs_json",
        "source_event_ids_json",
        "updated_at",
    }.issubset(snapshot_columns)
    assert {
        "patient_id",
        "projector_name",
        "projector_schema_version",
        "last_projected_patient_version",
        "projection_version",
        "updated_at",
    }.issubset(state_columns)


def test_create_patient_appends_created_event_and_snapshot(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)

    result = commands.create_patient(created_by_session_id="sess_patient_1")

    assert result.patient_id > 0
    assert result.patient_version == 1
    assert result.event_ids
    with registry._connect() as connection:
        event = connection.execute(
            "SELECT event_type, patient_version FROM patient_events WHERE patient_id = ?",
            (result.patient_id,),
        ).fetchone()
        snapshot = connection.execute(
            "SELECT patient_version, projection_version FROM patient_snapshots WHERE patient_id = ?",
            (result.patient_id,),
        ).fetchone()
    assert event["event_type"] == "patient.created"
    assert event["patient_version"] == 1
    assert snapshot["patient_version"] == 1
    assert snapshot["projection_version"] == 1


def test_identity_set_appends_second_patient_version(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient_1")

    result = commands.set_identity(
        patient_id=created.patient_id,
        patient_name="Alice",
        patient_number="p-001",
        source_session_id="sess_patient_1",
    )

    assert result.patient_version == 2
    assert registry.get_patient_identity(created.patient_id) == {
        "patient_name": "Alice",
        "patient_number": "p-001",
        "identity_locked": True,
    }
    with registry._connect() as connection:
        versions = [
            int(row["patient_version"])
            for row in connection.execute(
                "SELECT patient_version FROM patient_events WHERE patient_id = ? ORDER BY patient_version",
                (created.patient_id,),
            )
        ]
    assert versions == [1, 2]
