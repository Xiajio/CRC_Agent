from __future__ import annotations

import sqlite3
from pathlib import Path

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
