from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import (
    PatientNumberConflictError,
    PatientRegistryService,
)


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


def test_identity_set_maps_unique_index_race_to_patient_number_conflict(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient_1")

    class RacingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> sqlite3.Cursor:
            if "UPDATE patients" in sql and "patient_number_normalized" in sql:
                raise sqlite3.IntegrityError("UNIQUE constraint failed: patients.patient_number_normalized")
            return self._connection.execute(sql, parameters)

    @contextmanager
    def racing_transaction() -> Iterator[RacingConnection]:
        with registry._connect() as connection:
            yield RacingConnection(connection)

    registry.transaction = racing_transaction  # type: ignore[method-assign]

    with pytest.raises(PatientNumberConflictError):
        commands.set_identity(
            patient_id=created.patient_id,
            patient_name="Alice",
            patient_number="p-001",
            source_session_id="sess_patient_1",
        )
    with registry._connect() as connection:
        identity_event = connection.execute(
            """
            SELECT 1 FROM patient_events
            WHERE patient_id = ? AND event_type = 'patient.identity_set'
            """,
            (created.patient_id,),
        ).fetchone()
    assert identity_event is None


def test_created_by_session_id_is_unique_for_direct_patient_inserts(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")

    with registry._connect() as connection:
        connection.execute(
            """
            INSERT INTO patients (
                status, created_by_session_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("draft", "sess_patient_1", "2026-04-30T00:00:00+00:00", "2026-04-30T00:00:00+00:00"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO patients (
                    status, created_by_session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    "draft",
                    "sess_patient_1",
                    "2026-04-30T00:00:01+00:00",
                    "2026-04-30T00:00:01+00:00",
                ),
            )


def test_create_patient_reuses_existing_event_sourced_patient_for_session(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)

    first = commands.create_patient(created_by_session_id="sess_patient_1")
    second = commands.create_patient(created_by_session_id="sess_patient_1")

    assert second.reused is True
    assert second.patient_id == first.patient_id
    assert second.patient_version == first.patient_version
    assert second.event_ids == first.event_ids
    with registry._connect() as connection:
        patient_count = connection.execute("SELECT COUNT(*) AS count FROM patients").fetchone()
        event_count = connection.execute("SELECT COUNT(*) AS count FROM patient_events").fetchone()
    assert int(patient_count["count"]) == 1
    assert int(event_count["count"]) == 1


def test_create_patient_unique_index_race_reuses_existing_patient(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    first = commands.create_patient(created_by_session_id="sess_patient_1")
    original_lookup = commands._get_created_patient_for_session
    calls = 0

    def race_lookup(connection: sqlite3.Connection, created_by_session_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return original_lookup(connection, created_by_session_id)

    commands._get_created_patient_for_session = race_lookup  # type: ignore[method-assign]

    second = commands.create_patient(created_by_session_id="sess_patient_1")

    assert second.reused is True
    assert second.patient_id == first.patient_id
    assert second.patient_version == first.patient_version
    assert second.event_ids == first.event_ids
    with registry._connect() as connection:
        patient_count = connection.execute("SELECT COUNT(*) AS count FROM patients").fetchone()
        event_count = connection.execute("SELECT COUNT(*) AS count FROM patient_events").fetchone()
    assert int(patient_count["count"]) == 1
    assert int(event_count["count"]) == 1


def test_identity_set_preserves_snapshot_summary_and_source_events(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient_1")
    identity = commands.set_identity(
        patient_id=created.patient_id,
        patient_name="Alice",
        patient_number="p-001",
        source_session_id="sess_patient_1",
    )

    with registry._connect() as connection:
        snapshot = connection.execute(
            """
            SELECT summary_json, source_event_ids_json
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (created.patient_id,),
        ).fetchone()

    assert json.loads(snapshot["summary_json"]) == {
        "status": "draft",
        "patient_name": "Alice",
        "patient_number": "p-001",
    }
    assert json.loads(snapshot["source_event_ids_json"]) == [
        created.event_ids[0],
        identity.event_ids[0],
    ]


def test_upload_received_creates_available_asset_projection(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")

    result = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=12,
        sha256="abc123",
        storage_path=str(tmp_path / "assets" / "report.pdf"),
        source_session_id="sess_patient_1",
    )

    assert result.patient_version == 2
    assert result.asset_id is not None
    with registry._connect() as connection:
        asset = connection.execute(
            "SELECT sha256, storage_status, parse_status, patient_version FROM patient_assets WHERE asset_id = ?",
            (result.asset_id,),
        ).fetchone()
    assert asset["sha256"] == "abc123"
    assert asset["storage_status"] == "available"
    assert asset["parse_status"] == "pending"
    assert asset["patient_version"] == 2


def test_upload_received_reuses_existing_asset_by_sha256(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    first = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=12,
        sha256="abc123",
        storage_path=str(tmp_path / "assets" / "report.pdf"),
        source_session_id="sess_patient_1",
    )

    second = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="report-copy.pdf",
        content_type="application/pdf",
        size_bytes=12,
        sha256="abc123",
        storage_path=str(tmp_path / "assets" / "report-copy.pdf"),
        source_session_id="sess_patient_1",
    )

    assert second.reused is True
    assert second.asset_id == first.asset_id
    assert second.patient_version == first.patient_version
    assert second.projection_version == first.projection_version
    assert second.event_ids == []
    assert second.snapshot_changed is False
    with registry._connect() as connection:
        event_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM patient_events
            WHERE patient_id = ? AND event_type = 'patient.upload_received'
            """,
            (patient.patient_id,),
        ).fetchone()
    assert int(event_count["count"]) == 1


def test_upload_parse_failed_updates_asset_without_snapshot(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    upload = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="broken.pdf",
        content_type="application/pdf",
        size_bytes=10,
        sha256="broken-sha",
        storage_path=str(tmp_path / "assets" / "broken.pdf"),
        source_session_id="sess_patient_1",
    )
    assert upload.asset_id is not None

    failed = commands.record_upload_parse_failed(
        patient_id=patient.patient_id,
        asset_id=upload.asset_id,
        error_code="CONVERTER_ERROR",
        error_message="parse failed",
        source_session_id="sess_patient_1",
    )

    assert failed.patient_version == 3
    with registry._connect() as connection:
        asset = connection.execute(
            "SELECT parse_status, parse_error_code, parse_error_message FROM patient_assets WHERE asset_id = ?",
            (upload.asset_id,),
        ).fetchone()
        snapshot = connection.execute(
            "SELECT patient_version, medical_card_snapshot_json FROM patient_snapshots WHERE patient_id = ?",
            (patient.patient_id,),
        ).fetchone()
    assert asset["parse_status"] == "failed"
    assert asset["parse_error_code"] == "CONVERTER_ERROR"
    assert "parse failed" in asset["parse_error_message"]
    assert snapshot["patient_version"] == 3
    assert snapshot["medical_card_snapshot_json"] == "{}"


def test_upload_parse_failed_reuses_existing_failure_for_same_error_code(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    upload = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="broken.pdf",
        content_type="application/pdf",
        size_bytes=10,
        sha256="broken-sha",
        storage_path=str(tmp_path / "assets" / "broken.pdf"),
        source_session_id="sess_patient_1",
    )
    assert upload.asset_id is not None
    first = commands.record_upload_parse_failed(
        patient_id=patient.patient_id,
        asset_id=upload.asset_id,
        error_code="CONVERTER_ERROR",
        error_message="parse failed",
        source_session_id="sess_patient_1",
    )

    second = commands.record_upload_parse_failed(
        patient_id=patient.patient_id,
        asset_id=upload.asset_id,
        error_code="CONVERTER_ERROR",
        error_message="parse failed again",
        source_session_id="sess_patient_1",
    )

    assert second.reused is True
    assert second.asset_id == upload.asset_id
    assert second.patient_version == first.patient_version
    assert second.projection_version == first.projection_version
    assert second.event_ids == []
    assert second.snapshot_changed is False
    with registry._connect() as connection:
        event_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM patient_events
            WHERE patient_id = ? AND event_type = 'patient.upload_parse_failed'
            """,
            (patient.patient_id,),
        ).fetchone()
        asset = connection.execute(
            """
            SELECT patient_version, parse_error_message
            FROM patient_assets
            WHERE asset_id = ?
            """,
            (upload.asset_id,),
        ).fetchone()
    assert int(event_count["count"]) == 1
    assert asset["patient_version"] == first.patient_version
    assert asset["parse_error_message"] == "parse failed"


def test_medical_card_extracted_creates_record_and_snapshot(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    upload = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="patient-report.pdf",
        content_type="application/pdf",
        size_bytes=11,
        sha256="report-sha",
        storage_path=str(tmp_path / "assets" / "patient-report.pdf"),
        source_session_id="sess_patient_1",
    )

    result = commands.record_medical_card_extracted(
        patient_id=patient.patient_id,
        asset_id=upload.asset_id,
        patient_snapshot={"clinical_stage": "cT3N1M0", "tumor_location": "rectum"},
        record_payload={"document_type": "patient_report", "data": {"staging_block": {"clinical_stage": "cT3N1M0"}}},
        summary_text="cT3N1M0 rectal cancer",
        document_type="patient_report",
        ingest_decision="record_and_snapshot",
        source_session_id="sess_patient_1",
    )

    assert result.patient_version == 3
    assert result.record_id is not None
    detail = registry.get_patient_detail(patient.patient_id)
    assert detail["clinical_stage"] == "cT3N1M0"
    assert detail["tumor_location"] == "rectum"
    with registry._connect() as connection:
        record = connection.execute(
            "SELECT source_event_id, patient_version FROM patient_records WHERE record_id = ?",
            (result.record_id,),
        ).fetchone()
        asset = connection.execute(
            "SELECT parse_status, record_ids_json FROM patient_assets WHERE asset_id = ?",
            (upload.asset_id,),
        ).fetchone()
    assert record["source_event_id"] in result.event_ids
    assert record["patient_version"] == 3
    assert asset["parse_status"] == "parsed"
    assert str(result.record_id) in asset["record_ids_json"]


def test_medical_card_extracted_record_only_does_not_update_patient_facts(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    upload = commands.record_upload_received(
        patient_id=patient.patient_id,
        filename="patient-report.pdf",
        content_type="application/pdf",
        size_bytes=11,
        sha256="record-only-report-sha",
        storage_path=str(tmp_path / "assets" / "patient-report.pdf"),
        source_session_id="sess_patient_1",
    )

    result = commands.record_medical_card_extracted(
        patient_id=patient.patient_id,
        asset_id=upload.asset_id,
        patient_snapshot={"clinical_stage": "cT3N1M0", "tumor_location": "rectum"},
        record_payload={"document_type": "unknown", "data": {"staging_block": {"clinical_stage": "cT3N1M0"}}},
        summary_text="cT3N1M0 rectal cancer",
        document_type="unknown",
        ingest_decision="record_only",
        source_session_id="sess_patient_1",
    )

    assert result.patient_version == 3
    assert result.record_id is not None
    detail = registry.get_patient_detail(patient.patient_id)
    assert detail["clinical_stage"] is None
    assert detail["tumor_location"] is None
    with registry._connect() as connection:
        asset = connection.execute(
            "SELECT parse_status, record_ids_json FROM patient_assets WHERE asset_id = ?",
            (upload.asset_id,),
        ).fetchone()
        snapshot = connection.execute(
            "SELECT medical_card_snapshot_json FROM patient_snapshots WHERE patient_id = ?",
            (patient.patient_id,),
        ).fetchone()
    assert asset["parse_status"] == "parsed"
    assert str(result.record_id) in asset["record_ids_json"]
    assert snapshot["medical_card_snapshot_json"] == "{}"
