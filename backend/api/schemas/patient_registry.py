from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientRegistrySearchRequest(BaseModel):
    patient_id: int | None = None
    tumor_location: str | None = None
    mmr_status: str | None = None
    clinical_stage: str | None = None
    limit: int = Field(default=20, ge=1, le=100)

    model_config = ConfigDict(extra="forbid")


class PatientRegistryItem(BaseModel):
    patient_id: int
    status: str
    created_by_session_id: str | None = None
    updated_at: str
    tumor_location: str | None = None
    mmr_status: str | None = None
    clinical_stage: str | None = None


class PatientRegistryListResponse(BaseModel):
    items: list[PatientRegistryItem] = Field(default_factory=list)
    total: int = 0


class PatientRegistryDetailResponse(BaseModel):
    patient_id: int
    status: str
    created_by_session_id: str | None = None
    created_at: str
    updated_at: str
    chief_complaint: str | None = None
    age: int | None = None
    gender: str | None = None
    tumor_location: str | None = None
    mmr_status: str | None = None
    clinical_stage: str | None = None
    t_stage: str | None = None
    n_stage: str | None = None
    m_stage: str | None = None

    @field_validator("age", mode="before")
    @classmethod
    def _normalize_age(cls, value: Any) -> int | None:
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


class PatientRegistryRecord(BaseModel):
    record_id: int
    patient_id: int
    asset_id: int
    record_type: str
    document_type: str
    ingest_decision: str
    snapshot_contributed: bool
    conflict_detected: bool
    normalized_payload_json: dict[str, Any] | list[Any] | str | None = None
    summary_text: str
    source: str
    snapshot_meta_json: dict[str, Any] | list[Any] | str | None = None
    created_at: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PatientRegistryRecord":
        payload = row.get("normalized_payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass
        snapshot_meta = row.get("snapshot_meta_json")
        if isinstance(snapshot_meta, str):
            try:
                snapshot_meta = json.loads(snapshot_meta)
            except json.JSONDecodeError:
                pass
        return cls(
            record_id=int(row["record_id"]),
            patient_id=int(row["patient_id"]),
            asset_id=int(row["asset_id"]),
            record_type=str(row["record_type"]),
            document_type=str(row.get("document_type") or "unknown"),
            ingest_decision=str(row.get("ingest_decision") or "record_only"),
            snapshot_contributed=bool(int(row.get("snapshot_contributed") or 0)),
            conflict_detected=bool(int(row.get("conflict_detected") or 0)),
            normalized_payload_json=payload,
            summary_text=str(row["summary_text"]),
            source=str(row["source"]),
            snapshot_meta_json=snapshot_meta,
            created_at=str(row["created_at"]),
        )


class PatientRegistryRecordsResponse(BaseModel):
    items: list[PatientRegistryRecord] = Field(default_factory=list)


class PatientRegistryAlert(BaseModel):
    kind: str
    message: str
    patient_id: int
    record_id: int | None = None
    field_name: str | None = None
    field_names: list[str] = Field(default_factory=list)
    document_type: str | None = None
    created_at: str | None = None


class PatientRegistryAlertsResponse(BaseModel):
    items: list[PatientRegistryAlert] = Field(default_factory=list)


class PatientRegistryDeleteResponse(BaseModel):
    patient_id: int
    deleted_records: int
    deleted_assets: int
    deleted_asset_paths: list[str] = Field(default_factory=list)
    record_ids: list[int] = Field(default_factory=list)


class PatientRegistryClearResponse(BaseModel):
    deleted_patients: int
    deleted_records: int
    deleted_assets: int
    patient_ids: list[int] = Field(default_factory=list)
    deleted_asset_paths: list[str] = Field(default_factory=list)
