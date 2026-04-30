from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
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

    def _load_json_mapping(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return dict(parsed)
        return {}

    def _load_json_list(self, value: Any) -> list[dict[str, Any]] | list[str]:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return list(parsed)
        return []

    def _merge_source_event_ids(
        self,
        existing_event_ids: list[dict[str, Any]] | list[str],
        incoming_event_ids: list[str] | None,
    ) -> list[str]:
        merged: list[str] = []
        for event_id in [*existing_event_ids, *(incoming_event_ids or [])]:
            if not isinstance(event_id, str):
                continue
            if event_id not in merged:
                merged.append(event_id)
        return merged

    def _merge_asset_ref(
        self,
        existing_asset_refs: list[dict[str, Any]] | list[str],
        incoming_asset_ref: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._merge_refs_by_key(existing_asset_refs, [incoming_asset_ref], "asset_id")

    def _merge_refs_by_key(
        self,
        existing_refs: list[dict[str, Any]] | list[str],
        incoming_refs: list[dict[str, Any]] | None,
        key: str,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        indexes: dict[Any, int] = {}
        for ref in existing_refs:
            if not isinstance(ref, dict):
                continue
            ref_key = ref.get(key)
            if ref_key is None:
                continue
            indexes[ref_key] = len(merged)
            merged.append(dict(ref))
        for ref in incoming_refs or []:
            if not isinstance(ref, dict):
                continue
            ref_key = ref.get(key)
            if ref_key is None:
                continue
            if ref_key in indexes:
                index = indexes[ref_key]
                merged[index] = {**merged[index], **ref}
            else:
                indexes[ref_key] = len(merged)
                merged.append(dict(ref))
        return merged

    def _deep_merge_mapping(
        self,
        existing_snapshot: Mapping[str, Any],
        incoming_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(existing_snapshot)
        for key, value in (incoming_snapshot or {}).items():
            existing_value = merged.get(key)
            if isinstance(existing_value, Mapping) and isinstance(value, Mapping):
                merged[key] = self._deep_merge_mapping(existing_value, value)
            else:
                merged[key] = value
        return merged

    def _snapshot_asset_refs_with(
        self,
        connection: sqlite3.Connection,
        *,
        patient_id: int,
        asset_ref: dict[str, Any],
    ) -> list[dict[str, Any]]:
        row = connection.execute(
            "SELECT asset_refs_json FROM patient_snapshots WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        existing_asset_refs = [] if row is None else self._load_json_list(row["asset_refs_json"])
        return self._merge_asset_ref(existing_asset_refs, asset_ref)

    def _next_patient_version(self, connection: sqlite3.Connection, patient_id: int) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(patient_version), 0) AS version FROM patient_events WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return int(row["version"]) + 1

    def _current_event_sourced_patient_result(
        self,
        connection: sqlite3.Connection,
        patient_id: int,
    ) -> PatientCommandResult | None:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS event_count,
                COALESCE(MAX(patient_version), 0) AS patient_version
            FROM patient_events
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
        if row is None or int(row["event_count"]) == 0:
            return None
        patient_version = int(row["patient_version"])
        snapshot = connection.execute(
            """
            SELECT projection_version
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
        projection_version = (
            patient_version if snapshot is None else int(snapshot["projection_version"])
        )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=patient_version,
            projection_version=projection_version,
            event_ids=[],
            reused=True,
        )

    def _missing_snapshot_fields(
        self,
        existing_snapshot: Mapping[str, Any],
        incoming_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in incoming_snapshot.items()
            if isinstance(key, str) and key not in existing_snapshot
        }

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
        existing = connection.execute(
            """
            SELECT
                medical_card_snapshot_json,
                summary_json,
                active_alerts_json,
                record_refs_json,
                asset_refs_json,
                source_event_ids_json
            FROM patient_snapshots
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
        if existing is None:
            merged_medical_card_snapshot = self._deep_merge_mapping(
                {},
                medical_card_snapshot,
            )
            merged_summary = summary or {}
            merged_active_alerts = active_alerts or []
            merged_record_refs = self._merge_refs_by_key([], record_refs, "record_id")
            merged_asset_refs = self._merge_refs_by_key([], asset_refs, "asset_id")
            merged_source_event_ids = source_event_ids or []
        else:
            merged_medical_card_snapshot = self._load_json_mapping(
                existing["medical_card_snapshot_json"]
            )
            if medical_card_snapshot is not None:
                merged_medical_card_snapshot = self._deep_merge_mapping(
                    merged_medical_card_snapshot,
                    medical_card_snapshot,
                )

            merged_summary = self._load_json_mapping(existing["summary_json"])
            if summary is not None:
                merged_summary.update(summary)

            merged_active_alerts = (
                active_alerts
                if active_alerts is not None
                else self._load_json_list(existing["active_alerts_json"])
            )
            merged_record_refs = (
                self._merge_refs_by_key(
                    self._load_json_list(existing["record_refs_json"]),
                    record_refs,
                    "record_id",
                )
                if record_refs is not None
                else self._load_json_list(existing["record_refs_json"])
            )
            merged_asset_refs = (
                self._merge_refs_by_key(
                    self._load_json_list(existing["asset_refs_json"]),
                    asset_refs,
                    "asset_id",
                )
                if asset_refs is not None
                else self._load_json_list(existing["asset_refs_json"])
            )
            merged_source_event_ids = self._merge_source_event_ids(
                self._load_json_list(existing["source_event_ids_json"]),
                source_event_ids,
            )
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
                json.dumps(merged_medical_card_snapshot, ensure_ascii=False),
                json.dumps(merged_summary, ensure_ascii=False),
                json.dumps(merged_active_alerts, ensure_ascii=False),
                json.dumps(merged_record_refs, ensure_ascii=False),
                json.dumps(merged_asset_refs, ensure_ascii=False),
                json.dumps(merged_source_event_ids, ensure_ascii=False),
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

    def _get_created_patient_for_session(
        self,
        connection: sqlite3.Connection,
        created_by_session_id: str,
    ) -> PatientCommandResult | None:
        row = connection.execute(
            """
            SELECT
                p.id AS patient_id,
                e.patient_version,
                e.event_id,
                COALESCE(s.projection_version, e.patient_version) AS projection_version
            FROM patients p
            JOIN patient_events e ON e.patient_id = p.id
            LEFT JOIN patient_snapshots s ON s.patient_id = p.id
            WHERE p.created_by_session_id = ?
              AND e.event_type = 'patient.created'
              AND e.source_session_id = ?
              AND e.idempotency_key = ?
            ORDER BY p.id ASC
            LIMIT 1
            """,
            (
                created_by_session_id,
                created_by_session_id,
                f"patient.created:{created_by_session_id}",
            ),
        ).fetchone()
        if row is None:
            return None
        return PatientCommandResult(
            patient_id=int(row["patient_id"]),
            patient_version=int(row["patient_version"]),
            projection_version=int(row["projection_version"]),
            event_ids=[str(row["event_id"])],
            reused=True,
        )

    def create_patient(self, *, created_by_session_id: str) -> PatientCommandResult:
        now = _utc_now()
        with self._registry.transaction() as connection:
            existing = self._get_created_patient_for_session(connection, created_by_session_id)
            if existing is not None:
                return existing
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO patients (
                        status, created_by_session_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    ("draft", created_by_session_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                existing = self._get_created_patient_for_session(connection, created_by_session_id)
                if existing is not None:
                    return existing
                raise PatientEventConflictError(str(exc)) from exc
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

    def bootstrap_legacy_patient(
        self,
        patient_id: int,
        *,
        legacy_medical_card: Mapping[str, Any] | None = None,
        source_session_id: str | None = None,
    ) -> PatientCommandResult:
        try:
            with self._registry.transaction() as connection:
                connection.execute("BEGIN IMMEDIATE")
                patient = connection.execute(
                    """
                    SELECT id, status, created_by_session_id
                    FROM patients
                    WHERE id = ?
                    """,
                    (patient_id,),
                ).fetchone()
                if patient is None:
                    raise PatientIdentityNotFoundError(f"Patient not found: {patient_id}")

                existing = self._current_event_sourced_patient_result(connection, patient_id)
                if existing is not None and legacy_medical_card is None:
                    return existing
                card_idempotency_key = (
                    f"patient.legacy_medical_card_imported:{patient_id}:"
                    f"{source_session_id or 'unknown'}"
                )
                if existing is not None and legacy_medical_card is not None:
                    imported = connection.execute(
                        """
                        SELECT event_id
                        FROM patient_events
                        WHERE patient_id = ? AND idempotency_key = ?
                        LIMIT 1
                        """,
                        (patient_id, card_idempotency_key),
                    ).fetchone()
                    if imported is not None:
                        return existing

                    snapshot = connection.execute(
                        """
                        SELECT medical_card_snapshot_json
                        FROM patient_snapshots
                        WHERE patient_id = ?
                        """,
                        (patient_id,),
                    ).fetchone()
                    existing_medical_card_snapshot = (
                        {}
                        if snapshot is None
                        else self._load_json_mapping(snapshot["medical_card_snapshot_json"])
                    )
                    session_snapshot = self._missing_snapshot_fields(
                        existing_medical_card_snapshot,
                        legacy_medical_card,
                    )
                    event_id, version = self._append_event(
                        connection,
                        patient_id=patient_id,
                        event_type="patient.legacy_medical_card_imported",
                        payload={
                            "legacy": True,
                            "medical_card": dict(legacy_medical_card),
                        },
                        source_session_id=source_session_id,
                        idempotency_key=card_idempotency_key,
                    )
                    self._upsert_snapshot(
                        connection,
                        patient_id=patient_id,
                        patient_version=version,
                        medical_card_snapshot=session_snapshot,
                        summary={"legacy_session_medical_card": True},
                        source_event_ids=[event_id],
                    )
                    return PatientCommandResult(
                        patient_id=patient_id,
                        patient_version=version,
                        projection_version=version,
                        event_ids=[event_id],
                        snapshot_changed=True,
                    )

                detail = self._registry.get_patient_detail_in_transaction(connection, patient_id)
                record_rows = connection.execute(
                    """
                    SELECT
                        record_id,
                        asset_id,
                        document_type,
                        normalized_payload_json,
                        summary_text,
                        snapshot_contributed
                    FROM patient_records
                    WHERE patient_id = ?
                    ORDER BY record_id ASC
                    """,
                    (patient_id,),
                ).fetchall()
                asset_rows = connection.execute(
                    """
                    SELECT asset_id, sha256, parse_status
                    FROM patient_assets
                    WHERE patient_id = ?
                    ORDER BY asset_id ASC
                    """,
                    (patient_id,),
                ).fetchall()

                medical_card_snapshot: dict[str, Any] = {}
                record_refs: list[dict[str, Any]] = []
                summary_text: str | None = None
                for record in record_rows:
                    document_type = str(record["document_type"] or "unknown")
                    record_refs.append(
                        {
                            "record_id": int(record["record_id"]),
                            "document_type": document_type,
                        }
                    )
                    if record["summary_text"]:
                        summary_text = str(record["summary_text"])
                    if bool(record["snapshot_contributed"]):
                        medical_card_snapshot = self._deep_merge_mapping(
                            medical_card_snapshot,
                            self._load_json_mapping(record["normalized_payload_json"])
                        )

                asset_refs = [
                    {
                        "asset_id": int(asset["asset_id"]),
                        "sha256": str(asset["sha256"]),
                        "parse_status": str(asset["parse_status"] or "unknown"),
                    }
                    for asset in asset_rows
                ]

                created_source_session_id = patient["created_by_session_id"]
                event_id, version = self._append_event(
                    connection,
                    patient_id=patient_id,
                    event_type="patient.created",
                    payload={
                        "status": patient["status"],
                        "legacy": True,
                        "record_refs": record_refs,
                        "asset_refs": asset_refs,
                    },
                    source_session_id=created_source_session_id,
                    idempotency_key=f"patient.created.legacy:{patient_id}",
                )
                event_ids = [event_id]
                snapshot_event_ids = [event_id]
                final_medical_card_snapshot = dict(medical_card_snapshot)
                if legacy_medical_card is not None:
                    session_snapshot = self._missing_snapshot_fields(
                        final_medical_card_snapshot,
                        legacy_medical_card,
                    )
                    card_event_id, version = self._append_event(
                        connection,
                        patient_id=patient_id,
                        event_type="patient.legacy_medical_card_imported",
                        payload={
                            "legacy": True,
                            "medical_card": dict(legacy_medical_card),
                        },
                        source_session_id=source_session_id,
                        idempotency_key=card_idempotency_key,
                    )
                    final_medical_card_snapshot = self._deep_merge_mapping(
                        final_medical_card_snapshot,
                        session_snapshot,
                    )
                    event_ids.append(card_event_id)
                    snapshot_event_ids.append(card_event_id)
                summary: dict[str, Any] = {
                    "status": patient["status"],
                    "legacy": True,
                    "detail": detail,
                }
                if summary_text is not None:
                    summary["summary_text"] = summary_text
                self._upsert_snapshot(
                    connection,
                    patient_id=patient_id,
                    patient_version=version,
                    medical_card_snapshot=final_medical_card_snapshot,
                    summary=summary,
                    record_refs=record_refs,
                    asset_refs=asset_refs,
                    source_event_ids=snapshot_event_ids,
                )
        except PatientEventConflictError:
            with self._registry.transaction() as connection:
                existing = self._current_event_sourced_patient_result(connection, patient_id)
                if existing is not None:
                    return existing
            raise

        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=event_ids,
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
            try:
                connection.execute(
                    """
                    UPDATE patients
                    SET patient_name = ?, patient_number = ?, patient_number_normalized = ?,
                        identity_locked = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (patient_name, patient_number, normalized_number, now, patient_id),
                )
            except sqlite3.IntegrityError as exc:
                raise PatientNumberConflictError(
                    f"Patient number already exists: {normalized_number}"
                ) from exc
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

    def record_upload_received(
        self,
        *,
        patient_id: int,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        storage_path: str,
        source_session_id: str | None,
    ) -> PatientCommandResult:
        now = _utc_now()
        with self._registry.transaction() as connection:
            patient_row = connection.execute(
                "SELECT id FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if patient_row is None:
                raise KeyError(f"Patient not found: {patient_id}")

            existing_asset = connection.execute(
                """
                SELECT asset_id, patient_version
                FROM patient_assets
                WHERE patient_id = ? AND sha256 = ?
                """,
                (patient_id, sha256),
            ).fetchone()
            if existing_asset is not None:
                patient_version = int(existing_asset["patient_version"])
                return PatientCommandResult(
                    patient_id=patient_id,
                    patient_version=patient_version,
                    projection_version=patient_version,
                    event_ids=[],
                    asset_id=int(existing_asset["asset_id"]),
                    reused=True,
                )

            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.upload_received",
                payload={
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "storage_path": storage_path,
                    "source": "patient_generated",
                    "storage_status": "available",
                    "parse_status": "pending",
                },
                source_session_id=source_session_id,
                idempotency_key=f"patient.upload_received:{patient_id}:{sha256}",
                actor_type="patient",
            )
            cursor = connection.execute(
                """
                INSERT INTO patient_assets (
                    patient_id,
                    filename,
                    content_type,
                    sha256,
                    storage_path,
                    source,
                    created_at,
                    upload_event_id,
                    storage_status,
                    parse_status,
                    record_ids_json,
                    patient_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    filename,
                    content_type,
                    sha256,
                    storage_path,
                    "patient_generated",
                    now,
                    event_id,
                    "available",
                    "pending",
                    "[]",
                    version,
                ),
            )
            asset_id = int(cursor.lastrowid)
            asset_ref = {
                "asset_id": asset_id,
                "sha256": sha256,
                "parse_status": "pending",
            }
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                asset_refs=self._snapshot_asset_refs_with(
                    connection,
                    patient_id=patient_id,
                    asset_ref=asset_ref,
                ),
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            asset_id=asset_id,
            snapshot_changed=True,
        )

    def record_medical_card_extracted(
        self,
        *,
        patient_id: int,
        asset_id: int | None,
        patient_snapshot: dict[str, Any],
        record_payload: dict[str, Any],
        summary_text: str,
        document_type: str,
        ingest_decision: str,
        source_session_id: str | None,
    ) -> PatientCommandResult:
        with self._registry.transaction() as connection:
            asset = connection.execute(
                """
                SELECT *
                FROM patient_assets
                WHERE patient_id = ? AND asset_id = ?
                """,
                (patient_id, asset_id),
            ).fetchone()
            if asset is None:
                raise KeyError(f"Patient asset not found: {patient_id}/{asset_id}")
            existing_record_ids = self._load_json_list(asset["record_ids_json"])
            existing_record_id = next(
                (
                    int(record_id)
                    for record_id in existing_record_ids
                    if isinstance(record_id, (int, str)) and str(record_id).isdigit()
                ),
                None,
            )
            if asset["parse_status"] == "parsed" and existing_record_id is not None:
                patient_version = int(asset["patient_version"])
                snapshot = connection.execute(
                    """
                    SELECT projection_version
                    FROM patient_snapshots
                    WHERE patient_id = ?
                    """,
                    (patient_id,),
                ).fetchone()
                projection_version = (
                    patient_version
                    if snapshot is None
                    else int(snapshot["projection_version"])
                )
                return PatientCommandResult(
                    patient_id=patient_id,
                    patient_version=patient_version,
                    projection_version=projection_version,
                    event_ids=[],
                    asset_id=asset_id,
                    record_id=existing_record_id,
                    reused=True,
                    snapshot_changed=False,
                )

            enriched_record_payload = {**record_payload, "document_type": document_type}

            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.medical_card_extracted",
                payload={
                    "asset_id": asset_id,
                    "document_type": document_type,
                    "ingest_decision": ingest_decision,
                    "summary_text": summary_text,
                    "record_payload": enriched_record_payload,
                    "patient_snapshot": patient_snapshot,
                },
                source_session_id=source_session_id,
                idempotency_key=f"patient.medical_card_extracted:{patient_id}:{asset_id}",
                actor_type="system",
            )
            merge_snapshot = (
                patient_snapshot if ingest_decision == "record_and_snapshot" else {}
            )
            record_result = self._registry.write_medical_card_record_in_transaction(
                connection,
                patient_id=patient_id,
                asset_id=asset_id,
                source_event_id=event_id,
                patient_version=version,
                asset_row=dict(asset),
                patient_snapshot=merge_snapshot,
                record_payload=enriched_record_payload,
                summary_text=summary_text,
                record_type="medical_card",
            )
            record_id = int(record_result["record_id"])
            connection.execute(
                """
                UPDATE patient_assets
                SET parse_status = ?,
                    record_ids_json = ?,
                    patient_version = ?
                WHERE patient_id = ? AND asset_id = ?
                """,
                (
                    "parsed",
                    json.dumps([record_id], ensure_ascii=False),
                    version,
                    patient_id,
                    asset_id,
                ),
            )
            detail = self._registry.get_patient_detail_in_transaction(connection, patient_id)
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                medical_card_snapshot=record_result["projection_medical_card_snapshot"],
                summary={"summary_text": summary_text, "detail": detail},
                record_refs=[{"record_id": record_id, "document_type": document_type}],
                asset_refs=[
                    {
                        "asset_id": asset_id,
                        "sha256": str(asset["sha256"]),
                        "parse_status": "parsed",
                    }
                ],
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            asset_id=asset_id,
            record_id=record_id,
            snapshot_changed=True,
        )

    def record_upload_parse_failed(
        self,
        *,
        patient_id: int,
        asset_id: int,
        error_code: str,
        error_message: str,
        source_session_id: str | None,
    ) -> PatientCommandResult:
        with self._registry.transaction() as connection:
            asset = connection.execute(
                """
                SELECT
                    asset_id,
                    sha256,
                    parse_status,
                    parse_error_code,
                    record_ids_json,
                    patient_version
                FROM patient_assets
                WHERE patient_id = ? AND asset_id = ?
                """,
                (patient_id, asset_id),
            ).fetchone()
            if asset is None:
                raise KeyError(f"Patient asset not found: {patient_id}/{asset_id}")
            if asset["parse_status"] == "parsed":
                patient_version = int(asset["patient_version"])
                snapshot = connection.execute(
                    """
                    SELECT projection_version
                    FROM patient_snapshots
                    WHERE patient_id = ?
                    """,
                    (patient_id,),
                ).fetchone()
                projection_version = (
                    patient_version
                    if snapshot is None
                    else int(snapshot["projection_version"])
                )
                existing_record_ids = self._load_json_list(asset["record_ids_json"])
                existing_record_id = next(
                    (
                        int(record_id)
                        for record_id in existing_record_ids
                        if isinstance(record_id, (int, str)) and str(record_id).isdigit()
                    ),
                    None,
                )
                return PatientCommandResult(
                    patient_id=patient_id,
                    patient_version=patient_version,
                    projection_version=projection_version,
                    event_ids=[],
                    asset_id=asset_id,
                    record_id=existing_record_id,
                    reused=True,
                    snapshot_changed=False,
                )
            if asset["parse_status"] == "failed" and asset["parse_error_code"] == error_code:
                patient_version = int(asset["patient_version"])
                return PatientCommandResult(
                    patient_id=patient_id,
                    patient_version=patient_version,
                    projection_version=patient_version,
                    event_ids=[],
                    asset_id=asset_id,
                    reused=True,
                    snapshot_changed=False,
                )

            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.upload_parse_failed",
                payload={
                    "asset_id": asset_id,
                    "error_code": error_code,
                    "error_message": error_message,
                },
                source_session_id=source_session_id,
                idempotency_key=f"patient.upload_parse_failed:{patient_id}:{asset_id}:{error_code}",
                actor_type="system",
            )
            connection.execute(
                """
                UPDATE patient_assets
                SET parse_status = ?,
                    parse_error_code = ?,
                    parse_error_message = ?,
                    patient_version = ?
                WHERE patient_id = ? AND asset_id = ?
                """,
                (
                    "failed",
                    error_code,
                    error_message,
                    version,
                    patient_id,
                    asset_id,
                ),
            )
            asset_ref = {
                "asset_id": int(asset["asset_id"]),
                "sha256": str(asset["sha256"]),
                "parse_status": "failed",
            }
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                asset_refs=self._snapshot_asset_refs_with(
                    connection,
                    patient_id=patient_id,
                    asset_ref=asset_ref,
                ),
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            asset_id=asset_id,
            snapshot_changed=True,
        )

    def record_upload_command_failed(
        self,
        *,
        patient_id: int,
        asset_id: int,
        error_message: str,
        source_session_id: str | None,
    ) -> PatientCommandResult:
        error_code = "UPLOAD_COMMAND_ERROR"
        with self._registry.transaction() as connection:
            asset = connection.execute(
                """
                SELECT asset_id, sha256, parse_status, parse_error_code, patient_version
                FROM patient_assets
                WHERE patient_id = ? AND asset_id = ?
                """,
                (patient_id, asset_id),
            ).fetchone()
            if asset is None:
                raise KeyError(f"Patient asset not found: {patient_id}/{asset_id}")
            if asset["parse_status"] == "failed" and asset["parse_error_code"] == error_code:
                patient_version = int(asset["patient_version"])
                return PatientCommandResult(
                    patient_id=patient_id,
                    patient_version=patient_version,
                    projection_version=patient_version,
                    event_ids=[],
                    asset_id=asset_id,
                    reused=True,
                    snapshot_changed=False,
                )

            event_id, version = self._append_event(
                connection,
                patient_id=patient_id,
                event_type="patient.upload_parse_failed",
                payload={
                    "asset_id": asset_id,
                    "error_code": error_code,
                    "error_message": error_message,
                },
                source_session_id=source_session_id,
                idempotency_key=f"patient.upload_command_failed:{patient_id}:{asset_id}:{error_code}",
                actor_type="system",
            )
            connection.execute(
                """
                UPDATE patient_assets
                SET parse_status = ?,
                    parse_error_code = ?,
                    parse_error_message = ?,
                    patient_version = ?
                WHERE patient_id = ? AND asset_id = ?
                """,
                (
                    "failed",
                    error_code,
                    error_message,
                    version,
                    patient_id,
                    asset_id,
                ),
            )
            asset_ref = {
                "asset_id": int(asset["asset_id"]),
                "sha256": str(asset["sha256"]),
                "parse_status": "failed",
            }
            self._upsert_snapshot(
                connection,
                patient_id=patient_id,
                patient_version=version,
                asset_refs=self._snapshot_asset_refs_with(
                    connection,
                    patient_id=patient_id,
                    asset_ref=asset_ref,
                ),
                source_event_ids=[event_id],
            )
        return PatientCommandResult(
            patient_id=patient_id,
            patient_version=version,
            projection_version=version,
            event_ids=[event_id],
            asset_id=asset_id,
            snapshot_changed=True,
        )
