from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

EMPTY_VALUES = {"not_provided", "Unknown", "pending_evaluation", None, ""}
SNAPSHOT_COLUMNS = (
    "chief_complaint",
    "age",
    "gender",
    "tumor_location",
    "mmr_status",
    "clinical_stage",
    "t_stage",
    "n_stage",
    "m_stage",
)
PLACEHOLDER_VALUES = {
    "not_provided",
    "unknown",
    "pending_assessment",
    "pending_evaluation",
    "parse_failed_text",
    "",
}
SOURCE_PRIORITY = {
    "doctor_curated": 100,
    "pathology_report": 80,
    "imaging_report": 70,
    "patient_summary": 60,
    "patient_report": 50,
    "generic_patient_report": 50,
    "medical_card": 50,
    "report": 50,
    "ct_report": 40,
    "unknown": 10,
    "guideline_or_education": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.isdigit():
            return int(candidate)
    return None


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.lower() in PLACEHOLDER_VALUES:
            return None
        return candidate or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _load_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _load_projection_json_mapping(value: Any, column_name: str) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise TypeError(f"{column_name} must contain a JSON object")
    return dict(parsed)


def _load_projection_json_list(value: Any, column_name: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise TypeError(f"{column_name} must contain a JSON array")
    return list(parsed)


def _source_priority(document_type: str | None) -> int:
    if document_type is None:
        return SOURCE_PRIORITY["unknown"]
    return SOURCE_PRIORITY.get(document_type, SOURCE_PRIORITY["unknown"])


def _extract_document_type(record_payload: dict[str, Any], asset_row: dict[str, Any], record_type: str) -> str:
    candidate = record_payload.get("document_type")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    candidate = asset_row.get("document_type")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if isinstance(record_type, str) and record_type.strip():
        return record_type.strip()
    return "unknown"


def _is_valid_snapshot_value(value: Any) -> bool:
    if value in EMPTY_VALUES:
        return False
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
        return False
    return True


def _normalize_snapshot_value(field: str, value: Any) -> Any:
    if field == "age":
        return _normalize_optional_int(value)
    return _normalize_optional_text(value)


def _row_to_alert(
    *,
    patient_id: int,
    record_row: dict[str, Any],
    field_names: list[str],
    kind: str,
    message: str,
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "record_id": int(record_row["record_id"]),
        "kind": kind,
        "message": message,
        "field_name": field_names[0] if field_names else None,
        "field_names": field_names,
        "document_type": record_row.get("document_type"),
        "created_at": record_row.get("created_at"),
    }


class PatientNumberConflictError(ValueError):
    pass


class PatientIdentityLockedError(ValueError):
    pass


class PatientIdentityNotFoundError(KeyError):
    pass


def normalize_patient_number(value: str) -> str:
    candidate = value.strip()
    normalized_chars: list[str] = []
    for char in candidate:
        if "a" <= char <= "z":
            normalized_chars.append(char.upper())
        else:
            normalized_chars.append(char)
    return "".join(normalized_chars)


class PatientRegistryService:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    created_by_session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    chief_complaint TEXT,
                    age INTEGER,
                    gender TEXT,
                    tumor_location TEXT,
                    mmr_status TEXT,
                    clinical_stage TEXT,
                    t_stage TEXT,
                    n_stage TEXT,
                    m_stage TEXT
                );

                CREATE TABLE IF NOT EXISTS patient_assets (
                    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(patient_id, sha256)
                );

                CREATE TABLE IF NOT EXISTS patient_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    asset_id INTEGER NOT NULL,
                    record_type TEXT NOT NULL,
                    normalized_payload_json TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patient_events (
                    event_id TEXT PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    patient_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    event_payload_json TEXT NOT NULL,
                    actor_type TEXT,
                    actor_id TEXT,
                    source_session_id TEXT,
                    idempotency_key TEXT,
                    causation_id TEXT,
                    correlation_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(patient_id, patient_version)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_events_idempotency
                ON patient_events(patient_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS patient_snapshots (
                    patient_id INTEGER PRIMARY KEY,
                    patient_version INTEGER NOT NULL,
                    projection_version INTEGER NOT NULL,
                    medical_card_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    active_alerts_json TEXT NOT NULL DEFAULT '[]',
                    record_refs_json TEXT NOT NULL DEFAULT '[]',
                    asset_refs_json TEXT NOT NULL DEFAULT '[]',
                    source_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS patient_projection_state (
                    patient_id INTEGER NOT NULL,
                    projector_name TEXT NOT NULL,
                    projector_schema_version INTEGER NOT NULL,
                    last_projected_patient_version INTEGER NOT NULL,
                    projection_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (patient_id, projector_name)
                );
                """
            )
            self._ensure_columns(
                connection,
                "patients",
                {
                    "patient_name": "TEXT",
                    "patient_number": "TEXT",
                    "patient_number_normalized": "TEXT",
                    "identity_locked": "INTEGER NOT NULL DEFAULT 0",
                    "snapshot_provenance_json": "TEXT NOT NULL DEFAULT '{}'",
                },
            )
            self._ensure_columns(
                connection,
                "patient_assets",
                {
                    "upload_event_id": "TEXT",
                    "storage_status": "TEXT NOT NULL DEFAULT 'available'",
                    "parse_status": "TEXT NOT NULL DEFAULT 'unknown'",
                    "parse_error_code": "TEXT",
                    "parse_error_message": "TEXT",
                    "record_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                    "patient_version": "INTEGER NOT NULL DEFAULT 0",
                },
            )
            self._ensure_columns(
                connection,
                "patient_records",
                {
                    "document_type": "TEXT NOT NULL DEFAULT 'unknown'",
                    "ingest_decision": "TEXT NOT NULL DEFAULT 'record_only'",
                    "snapshot_contributed": "INTEGER NOT NULL DEFAULT 0",
                    "conflict_detected": "INTEGER NOT NULL DEFAULT 0",
                    "snapshot_meta_json": "TEXT NOT NULL DEFAULT '{}'",
                    "source_event_id": "TEXT",
                    "patient_version": "INTEGER NOT NULL DEFAULT 0",
                },
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_patient_number_normalized_unique
                ON patients(patient_number_normalized)
                WHERE patient_number_normalized IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_created_by_session_id_unique
                ON patients(created_by_session_id)
                WHERE created_by_session_id IS NOT NULL
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            yield connection

    def _ensure_columns(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing_columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_definition in columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def _runtime_root(self) -> Path:
        return self._db_path.parent.resolve()

    def _cleanup_asset_file(self, storage_path: str | None) -> None:
        if not storage_path:
            return
        runtime_root = self._runtime_root()
        candidate = Path(storage_path)
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(runtime_root)
        except (OSError, ValueError):
            return

        if resolved.is_file():
            resolved.unlink(missing_ok=True)

        current = resolved.parent
        while current != runtime_root and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def create_draft_patient(self, *, created_by_session_id: str) -> int:
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO patients (
                    status,
                    created_by_session_id,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("draft", created_by_session_id, now, now),
            )
            return int(cursor.lastrowid)

    def get_patient_identity(self, patient_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    patient_name,
                    patient_number,
                    identity_locked
                FROM patients
                WHERE id = ?
                """,
                (patient_id,),
            ).fetchone()
        if row is None:
            raise PatientIdentityNotFoundError(f"Patient not found: {patient_id}")
        return {
            "patient_name": row["patient_name"],
            "patient_number": row["patient_number"],
            "identity_locked": bool(row["identity_locked"]),
        }

    def patient_number_exists(
        self,
        normalized_number: str,
        *,
        exclude_patient_id: int | None = None,
    ) -> bool:
        with self._connect() as connection:
            if exclude_patient_id is None:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM patients
                    WHERE patient_number_normalized = ?
                    LIMIT 1
                    """,
                    (normalized_number,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM patients
                    WHERE patient_number_normalized = ?
                      AND id != ?
                    LIMIT 1
                    """,
                    (normalized_number, exclude_patient_id),
                ).fetchone()
        return row is not None

    def set_patient_identity(
        self,
        patient_id: int,
        patient_name: str,
        patient_number: str,
    ) -> dict[str, Any]:
        normalized_number = normalize_patient_number(patient_number)
        now = _utc_now()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, identity_locked
                FROM patients
                WHERE id = ?
                """,
                (patient_id,),
            ).fetchone()
            if row is None:
                raise PatientIdentityNotFoundError(f"Patient not found: {patient_id}")
            if bool(row["identity_locked"]):
                raise PatientIdentityLockedError(f"Patient identity is locked: {patient_id}")
            if self.patient_number_exists(normalized_number, exclude_patient_id=patient_id):
                raise PatientNumberConflictError(
                    f"Patient number already exists: {normalized_number}"
                )
            try:
                connection.execute(
                    """
                    UPDATE patients
                    SET
                        patient_name = ?,
                        patient_number = ?,
                        patient_number_normalized = ?,
                        identity_locked = 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        patient_name,
                        patient_number,
                        normalized_number,
                        now,
                        patient_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise PatientNumberConflictError(
                    f"Patient number already exists: {normalized_number}"
                ) from exc

        return self.get_patient_identity(patient_id)

    def get_patient_detail(self, patient_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            return self.get_patient_detail_in_transaction(connection, patient_id)

    def get_patient_context_projection(self, patient_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
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
                FROM patient_snapshots
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Patient projection not found: {patient_id}")
        return {
            "patient_id": int(row["patient_id"]),
            "patient_version": int(row["patient_version"]),
            "projection_version": int(row["projection_version"]),
            "medical_card_snapshot": _load_projection_json_mapping(
                row["medical_card_snapshot_json"],
                "medical_card_snapshot_json",
            ),
            "summary": _load_projection_json_mapping(row["summary_json"], "summary_json"),
            "alerts": _load_projection_json_list(
                row["active_alerts_json"],
                "active_alerts_json",
            ),
            "record_refs": _load_projection_json_list(
                row["record_refs_json"],
                "record_refs_json",
            ),
            "asset_refs": _load_projection_json_list(row["asset_refs_json"], "asset_refs_json"),
            "source_event_ids": _load_projection_json_list(
                row["source_event_ids_json"],
                "source_event_ids_json",
            ),
            "cached_at": row["updated_at"],
        }

    def get_patient_detail_in_transaction(
        self,
        connection: sqlite3.Connection,
        patient_id: int,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT
                id AS patient_id,
                status,
                created_by_session_id,
                created_at,
                updated_at,
                chief_complaint,
                age,
                gender,
                tumor_location,
                mmr_status,
                clinical_stage,
                t_stage,
                n_stage,
                m_stage
            FROM patients
            WHERE id = ?
            """,
            (patient_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Patient not found: {patient_id}")
        payload = dict(row)
        payload["age"] = _normalize_optional_int(payload.get("age"))
        for field in SNAPSHOT_COLUMNS:
            if field == "age":
                continue
            if field in payload:
                payload[field] = _normalize_optional_text(payload[field])
        return payload

    def delete_patient(self, patient_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            patient_row = connection.execute(
                "SELECT id FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if patient_row is None:
                raise KeyError(f"Patient not found: {patient_id}")

            asset_rows = connection.execute(
                """
                SELECT asset_id, storage_path
                FROM patient_assets
                WHERE patient_id = ?
                ORDER BY asset_id ASC
                """,
                (patient_id,),
            ).fetchall()
            record_rows = connection.execute(
                """
                SELECT record_id
                FROM patient_records
                WHERE patient_id = ?
                ORDER BY record_id ASC
                """,
                (patient_id,),
            ).fetchall()

            deleted_asset_paths = [
                str(row["storage_path"])
                for row in asset_rows
                if row["storage_path"]
            ]
            deleted_assets = len(asset_rows)
            deleted_records = len(record_rows)
            record_ids = [int(row["record_id"]) for row in record_rows]

            connection.execute("DELETE FROM patient_records WHERE patient_id = ?", (patient_id,))
            connection.execute("DELETE FROM patient_assets WHERE patient_id = ?", (patient_id,))
            connection.execute("DELETE FROM patients WHERE id = ?", (patient_id,))

        for asset_path in deleted_asset_paths:
            self._cleanup_asset_file(asset_path)

        return {
            "patient_id": patient_id,
            "deleted_records": deleted_records,
            "deleted_assets": deleted_assets,
            "deleted_asset_paths": deleted_asset_paths,
            "record_ids": record_ids,
        }

    def clear_registry(self) -> dict[str, Any]:
        with self._connect() as connection:
            patient_rows = connection.execute(
                "SELECT id FROM patients ORDER BY id ASC"
            ).fetchall()
            asset_rows = connection.execute(
                """
                SELECT asset_id, patient_id, storage_path
                FROM patient_assets
                ORDER BY asset_id ASC
                """
            ).fetchall()
            record_rows = connection.execute(
                """
                SELECT record_id
                FROM patient_records
                ORDER BY record_id ASC
                """
            ).fetchall()

            patient_ids = [int(row["id"]) for row in patient_rows]
            deleted_asset_paths = [
                str(row["storage_path"])
                for row in asset_rows
                if row["storage_path"]
            ]
            deleted_assets = len(asset_rows)
            deleted_records = len(record_rows)

            connection.execute("DELETE FROM patient_records")
            connection.execute("DELETE FROM patient_assets")
            connection.execute("DELETE FROM patients")

        for asset_path in deleted_asset_paths:
            self._cleanup_asset_file(asset_path)

        return {
            "deleted_patients": len(patient_ids),
            "deleted_records": deleted_records,
            "deleted_assets": deleted_assets,
            "patient_ids": patient_ids,
            "deleted_asset_paths": deleted_asset_paths,
        }

    def write_medical_card_record(
        self,
        *,
        patient_id: int,
        asset_row: dict[str, Any],
        patient_snapshot: dict[str, Any],
        record_payload: dict[str, Any],
        summary_text: str,
        record_type: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            return self.write_medical_card_record_in_transaction(
                connection,
                patient_id=patient_id,
                asset_id=None,
                source_event_id=None,
                patient_version=0,
                asset_row=asset_row,
                patient_snapshot=patient_snapshot,
                record_payload=record_payload,
                summary_text=summary_text,
                record_type=record_type,
            )

    def write_medical_card_record_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        patient_id: int,
        asset_id: int | None,
        source_event_id: str | None,
        patient_version: int,
        asset_row: dict[str, Any],
        patient_snapshot: dict[str, Any],
        record_payload: dict[str, Any],
        summary_text: str,
        record_type: str,
    ) -> dict[str, Any]:
        now = _utc_now()
        document_type = _extract_document_type(record_payload, asset_row, record_type)
        source_priority = _source_priority(document_type)
        normalized_snapshot = {
            field: _normalize_snapshot_value(field, patient_snapshot.get(field))
            for field in SNAPSHOT_COLUMNS
            if field in patient_snapshot
        }

        patient_row = connection.execute(
            "SELECT * FROM patients WHERE id = ?",
            (patient_id,),
        ).fetchone()
        if patient_row is None:
            raise KeyError(f"Patient not found: {patient_id}")

        if asset_id is None:
            existing_asset = connection.execute(
                """
                SELECT asset_id
                FROM patient_assets
                WHERE patient_id = ? AND sha256 = ?
                """,
                (patient_id, asset_row["sha256"]),
            ).fetchone()

            reused = existing_asset is not None
            if existing_asset is None:
                asset_cursor = connection.execute(
                    """
                    INSERT INTO patient_assets (
                        patient_id,
                        filename,
                        content_type,
                        sha256,
                        storage_path,
                        source,
                        created_at,
                        storage_status,
                        parse_status,
                        patient_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        asset_row["filename"],
                        asset_row["content_type"],
                        asset_row["sha256"],
                        asset_row["storage_path"],
                        asset_row["source"],
                        now,
                        "available",
                        "parsed",
                        patient_version,
                    ),
                )
                asset_id = int(asset_cursor.lastrowid)
            else:
                asset_id = int(existing_asset["asset_id"])
        else:
            reused = False

        existing_snapshot = {
            field: patient_row[field]
            for field in SNAPSHOT_COLUMNS
            if field in patient_row.keys()
        }
        existing_provenance = _load_json_mapping(patient_row["snapshot_provenance_json"])
        merged_snapshot = dict(existing_snapshot)
        merged_provenance = dict(existing_provenance)
        record_snapshot_meta: dict[str, dict[str, Any]] = {}
        snapshot_contributed = False
        conflict_detected = False

        for field in SNAPSHOT_COLUMNS:
            if field not in patient_snapshot:
                continue

            incoming_value = normalized_snapshot.get(field)
            if not _is_valid_snapshot_value(incoming_value):
                record_snapshot_meta[field] = {
                    "accepted": False,
                    "conflict_detected": False,
                    "document_type": document_type,
                    "priority": source_priority,
                    "previous_value": existing_snapshot.get(field),
                    "incoming_value": incoming_value,
                    "rejected_reason": "placeholder",
                }
                continue

            current_value = existing_snapshot.get(field)
            current_meta = _load_json_mapping(merged_provenance.get(field, {}))
            current_priority = int(current_meta.get("priority", 0)) if current_meta else 0
            field_meta = {
                "accepted": False,
                "conflict_detected": False,
                "document_type": document_type,
                "priority": source_priority,
                "previous_value": current_value,
                "incoming_value": incoming_value,
                "previous_priority": current_priority,
            }

            if current_value in EMPTY_VALUES:
                merged_snapshot[field] = incoming_value
                field_meta["accepted"] = True
                snapshot_contributed = True
            elif current_value == incoming_value:
                field_meta["accepted"] = True
                snapshot_contributed = True
            else:
                conflict_detected = True
                field_meta["conflict_detected"] = True
                if source_priority >= current_priority:
                    merged_snapshot[field] = incoming_value
                    field_meta["accepted"] = True
                    snapshot_contributed = True

            if field_meta["accepted"]:
                merged_provenance[field] = {
                    "record_id": None,
                    "document_type": document_type,
                    "priority": source_priority,
                    "updated_at": now,
                    "conflict_detected": field_meta["conflict_detected"],
                }
            record_snapshot_meta[field] = field_meta

        ingest_decision = "record_and_snapshot" if snapshot_contributed else "record_only"

        existing_record = connection.execute(
            """
            SELECT record_id
            FROM patient_records
            WHERE patient_id = ? AND asset_id = ? AND record_type = ?
            ORDER BY record_id ASC
            LIMIT 1
            """,
            (patient_id, asset_id, record_type),
        ).fetchone()
        if existing_record is None:
            record_cursor = connection.execute(
                """
                INSERT INTO patient_records (
                    patient_id,
                    asset_id,
                    record_type,
                    document_type,
                    ingest_decision,
                    snapshot_contributed,
                    conflict_detected,
                    normalized_payload_json,
                    summary_text,
                    source,
                    snapshot_meta_json,
                    source_event_id,
                    patient_version,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    asset_id,
                    record_type,
                    document_type,
                    ingest_decision,
                    1 if snapshot_contributed else 0,
                    1 if conflict_detected else 0,
                    json.dumps(record_payload, ensure_ascii=False),
                    summary_text,
                    asset_row["source"],
                    json.dumps(record_snapshot_meta, ensure_ascii=False),
                    source_event_id,
                    patient_version,
                    now,
                ),
            )
            record_id = int(record_cursor.lastrowid)
        else:
            record_id = int(existing_record["record_id"])

        final_record_snapshot_meta = {
            field: {
                **field_meta,
                "record_id": record_id,
            }
            for field, field_meta in record_snapshot_meta.items()
        }
        final_provenance = {
            field: {
                **field_meta,
                "record_id": record_id,
            }
            for field, field_meta in merged_provenance.items()
        }
        connection.execute(
            """
            UPDATE patients
            SET
                updated_at = ?,
                chief_complaint = ?,
                age = ?,
                gender = ?,
                tumor_location = ?,
                mmr_status = ?,
                clinical_stage = ?,
                t_stage = ?,
                n_stage = ?,
                m_stage = ?,
                snapshot_provenance_json = ?
            WHERE id = ?
            """,
            (
                now,
                merged_snapshot.get("chief_complaint"),
                merged_snapshot.get("age"),
                merged_snapshot.get("gender"),
                merged_snapshot.get("tumor_location"),
                merged_snapshot.get("mmr_status"),
                merged_snapshot.get("clinical_stage"),
                merged_snapshot.get("t_stage"),
                merged_snapshot.get("n_stage"),
                merged_snapshot.get("m_stage"),
                json.dumps(final_provenance, ensure_ascii=False),
                patient_id,
            ),
        )
        connection.execute(
            """
            UPDATE patient_records
            SET snapshot_meta_json = ?
            WHERE record_id = ?
            """,
            (
                json.dumps(final_record_snapshot_meta, ensure_ascii=False),
                record_id,
            ),
        )

        return {
            "patient_id": patient_id,
            "asset_id": asset_id,
            "record_id": record_id,
            "reused": reused,
            "document_type": document_type,
            "ingest_decision": ingest_decision,
            "snapshot_contributed": snapshot_contributed,
            "conflict_detected": conflict_detected,
        }

    def list_patient_records(self, patient_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    record_id,
                    patient_id,
                    asset_id,
                    record_type,
                    document_type,
                    ingest_decision,
                    snapshot_contributed,
                    conflict_detected,
                    normalized_payload_json,
                    summary_text,
                    source,
                    snapshot_meta_json,
                    created_at
                FROM patient_records
                WHERE patient_id = ?
                ORDER BY record_id DESC
                """,
                (patient_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_patient_alerts(self, patient_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            patient_row = connection.execute(
                "SELECT id FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if patient_row is None:
                raise KeyError(f"Patient not found: {patient_id}")

            rows = connection.execute(
                """
                SELECT
                    record_id,
                    patient_id,
                    document_type,
                    ingest_decision,
                    snapshot_contributed,
                    conflict_detected,
                    snapshot_meta_json,
                    summary_text,
                    created_at
                FROM patient_records
                WHERE patient_id = ?
                ORDER BY record_id DESC
                """,
                (patient_id,),
            ).fetchall()

        alerts: list[dict[str, Any]] = []
        for row in rows:
            record_row = dict(row)
            meta = _load_json_mapping(record_row.get("snapshot_meta_json"))
            conflict_fields = [
                field_name
                for field_name, field_meta in meta.items()
                if isinstance(field_meta, dict) and field_meta.get("conflict_detected")
            ]
            if record_row.get("conflict_detected"):
                alerts.append(
                    _row_to_alert(
                        patient_id=patient_id,
                        record_row=record_row,
                        field_names=conflict_fields,
                        kind="conflict_detected",
                        message=(
                            f"Conflict detected on {', '.join(conflict_fields)}."
                            if conflict_fields
                            else "Conflict detected in patient snapshot."
                        ),
                    )
                )

            document_type = str(record_row.get("document_type") or "unknown")
            snapshot_contributed = bool(record_row.get("snapshot_contributed"))
            if document_type in {"unknown", "guideline_or_education"}:
                alerts.append(
                    _row_to_alert(
                        patient_id=patient_id,
                        record_row=record_row,
                        field_names=[],
                        kind="low_confidence",
                        message=f"Document type {document_type} is low confidence.",
                    )
                )
            elif not snapshot_contributed:
                alerts.append(
                    _row_to_alert(
                        patient_id=patient_id,
                        record_row=record_row,
                        field_names=[],
                        kind="not_snapshot_eligible",
                        message=f"Record {record_row['record_id']} did not update the snapshot.",
                    )
                )

        return alerts

    def list_recent_patients(self, *, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id AS patient_id,
                    status,
                    created_by_session_id,
                    updated_at,
                    tumor_location,
                    mmr_status,
                    clinical_stage
                FROM patients
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_patients(
        self,
        *,
        patient_id: int | None = None,
        tumor_location: str | None = None,
        mmr_status: str | None = None,
        clinical_stage: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        if patient_id is not None:
            clauses.append("id = ?")
            params.append(patient_id)
        if tumor_location:
            clauses.append("tumor_location = ?")
            params.append(tumor_location)
        if mmr_status:
            clauses.append("mmr_status = ?")
            params.append(mmr_status)
        if clinical_stage:
            clauses.append("clinical_stage = ?")
            params.append(clinical_stage)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            total_row = connection.execute(
                f"SELECT COUNT(*) AS total FROM patients {where_sql}",
                params,
            ).fetchone()
            rows = connection.execute(
                f"""
                SELECT
                    id AS patient_id,
                    status,
                    created_by_session_id,
                    updated_at,
                    tumor_location,
                    mmr_status,
                    clinical_stage
                FROM patients
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "total": int(total_row["total"]) if total_row is not None else 0,
        }

    def get_patient_summary_message(self, patient_id: int) -> HumanMessage | None:
        try:
            detail = self.get_patient_detail(patient_id)
        except KeyError:
            return None

        summary_parts = [f"patient_id={patient_id}"]
        if detail.get("chief_complaint") not in EMPTY_VALUES:
            summary_parts.append(f"chief_complaint={detail['chief_complaint']}")
        if detail.get("tumor_location") not in EMPTY_VALUES:
            summary_parts.append(f"tumor_location={detail['tumor_location']}")
        if detail.get("mmr_status") not in EMPTY_VALUES:
            summary_parts.append(f"mmr_status={detail['mmr_status']}")
        if detail.get("clinical_stage") not in EMPTY_VALUES:
            summary_parts.append(f"clinical_stage={detail['clinical_stage']}")
        return HumanMessage(content="Bound patient summary: " + ", ".join(summary_parts))
