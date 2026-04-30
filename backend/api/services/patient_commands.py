from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from backend.api.services.patient_registry_service import (
    PatientIdentityLockedError,
    PatientIdentityNotFoundError,
    PatientNumberConflictError,
    PatientRegistryService,
    _utc_now,
    normalize_patient_number,
)


class PatientEventConflictError(RuntimeError):
    """Raised when patient event append violates idempotency or version constraints."""


@dataclass(frozen=True)
class PatientCommandResult:
    patient_id: int
    patient_version: int
    projection_version: int
    event_ids: list[str]
    asset_id: int | None = None
    record_id: int | None = None
    alerts_changed: bool = False
    snapshot_changed: bool = False
    reused: bool = False


class PatientCommandService:
    def __init__(self, registry: PatientRegistryService) -> None:
        self._registry = registry

    def _next_patient_version(self, connection: sqlite3.Connection, patient_id: int) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(patient_version), 0) AS version FROM patient_events WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return int(row["version"]) + 1

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        patient_id: int,
        event_type: str,
        payload: dict[str, Any],
        source_session_id: str | None,
        idempotency_key: str | None = None,
        actor_type: str | None = "system",
        actor_id: str | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> tuple[str, int]:
        event_id = f"evt_{uuid4().hex}"
        version = self._next_patient_version(connection, patient_id)
        try:
            connection.execute(
                """
                INSERT INTO patient_events (
                    event_id, patient_id, patient_version, event_type,
                    event_payload_json, actor_type, actor_id, source_session_id,
                    idempotency_key, causation_id, correlation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    patient_id,
                    version,
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                    actor_type,
                    actor_id,
                    source_session_id,
                    idempotency_key,
                    causation_id,
                    correlation_id,
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PatientEventConflictError(str(exc)) from exc
        return event_id, version

    def _upsert_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        patient_id: int,
        patient_version: int,
        medical_card_snapshot: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        active_alerts: list[dict[str, Any]] | None = None,
        record_refs: list[dict[str, Any]] | None = None,
        asset_refs: list[dict[str, Any]] | None = None,
        source_event_ids: list[str] | None = None,
    ) -> None:
        now = _utc_now()
        projection_version = patient_version
        connection.execute(
            """
            INSERT INTO patient_snapshots (
                patient_id,
                patient_version,
                projection_version,
                medical_card_snapshot_json,
                summary_json,
                active_alerts_json,
                record_refs_json,
                asset_refs_json,
                source_event_ids_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                patient_version = excluded.patient_version,
                projection_version = excluded.projection_version,
                medical_card_snapshot_json = excluded.medical_card_snapshot_json,
                summary_json = excluded.summary_json,
                active_alerts_json = excluded.active_alerts_json,
                record_refs_json = excluded.record_refs_json,
                asset_refs_json = excluded.asset_refs_json,
                source_event_ids_json = excluded.source_event_ids_json,
                updated_at = excluded.updated_at
            """,
            (
                patient_id,
                patient_version,
                projection_version,
                json.dumps(medical_card_snapshot or {}, ensure_ascii=False),
                json.dumps(summary or {}, ensure_ascii=False),
                json.dumps(active_alerts or [], ensure_ascii=False),
                json.dumps(record_refs or [], ensure_ascii=False),
                json.dumps(asset_refs or [], ensure_ascii=False),
                json.dumps(source_event_ids or [], ensure_ascii=False),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO patient_projection_state (
                patient_id,
                projector_name,
                projector_schema_version,
                last_projected_patient_version,
                projection_version,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id, projector_name) DO UPDATE SET
                projector_schema_version = excluded.projector_schema_version,
                last_projected_patient_version = excluded.last_projected_patient_version,
                projection_version = excluded.projection_version,
                updated_at = excluded.updated_at
            """,
            (
                patient_id,
                "patient_registry_snapshot",
                1,
                patient_version,
                projection_version,
                now,
            ),
        )

    def create_patient(self, *, created_by_session_id: str) -> PatientCommandResult:
        now = _utc_now()
        with self._registry.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO patients (
                    status, created_by_session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("draft", created_by_session_id, now, now),
            )
            patient_id = int(cursor.lastrowid)
            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.created",
                payload={"status": "draft"},
                source_session_id=created_by_session_id,
                idempotency_key=f"patient.created:{created_by_session_id}",
            )
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                summary={"status": "draft"},
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            snapshot_changed=True,
        )

    def set_identity(
        self,
        *,
        patient_id: int,
        patient_name: str,
        patient_number: str,
        source_session_id: str | None,
    ) -> PatientCommandResult:
        normalized_number = normalize_patient_number(patient_number)
        now = _utc_now()
        with self._registry.transaction() as connection:
            row = connection.execute(
                "SELECT id, identity_locked FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if row is None:
                raise PatientIdentityNotFoundError(f"Patient not found: {patient_id}")
            if bool(row["identity_locked"]):
                raise PatientIdentityLockedError(f"Patient identity is locked: {patient_id}")
            duplicate = connection.execute(
                """
                SELECT 1 FROM patients
                WHERE patient_number_normalized = ? AND id != ?
                LIMIT 1
                """,
                (normalized_number, patient_id),
            ).fetchone()
            if duplicate is not None:
                raise PatientNumberConflictError(f"Patient number already exists: {normalized_number}")
            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.identity_set",
                payload={
                    "patient_name": patient_name,
                    "patient_number": patient_number,
                    "patient_number_normalized": normalized_number,
                },
                source_session_id=source_session_id,
                idempotency_key=f"patient.identity_set:{patient_id}:{normalized_number}",
            )
            connection.execute(
                """
                UPDATE patients
                SET patient_name = ?, patient_number = ?, patient_number_normalized = ?,
                    identity_locked = 1, updated_at = ?
                WHERE id = ?
                """,
                (patient_name, patient_number, normalized_number, now, patient_id),
            )
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                summary={"patient_name": patient_name, "patient_number": patient_number},
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            snapshot_changed=True,
        )
