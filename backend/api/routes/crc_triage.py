from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/sessions", tags=["crc-triage"])


class CrcTriageAssessmentPayload(BaseModel):
    record_type: Literal["crc_triage_assessment"]
    chief_complaint: str = Field(min_length=1)
    symptom_group: str = Field(min_length=1)
    risk_level: str = Field(min_length=1)
    disposition: str = Field(min_length=1)
    red_flags: list[str] = Field(default_factory=list)
    known_crc_signals: dict[str, Any] = Field(default_factory=dict)
    suggested_tests: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    qa_summary: list[dict[str, Any]] = Field(default_factory=list)
    node_results: list[dict[str, Any]] = Field(default_factory=list)
    protocol_state: dict[str, Any] = Field(default_factory=dict)
    patient_summary: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    source_session_id: str = Field(min_length=1)
    source_subflow: Literal["crc_triage"]
    safety_policy_version: str | None = None
    matched_rules: list[str] = Field(default_factory=list)
    hard_fail_flags: list[str] = Field(default_factory=list)
    patient_message_key: str | None = None
    assessment_id: str | None = None

    @field_validator(
        "chief_complaint",
        "symptom_group",
        "risk_level",
        "disposition",
        "patient_summary",
        "next_step",
        "source_session_id",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class SaveCrcTriageAssessmentRequest(BaseModel):
    assessment: CrcTriageAssessmentPayload


class SaveCrcTriageAssessmentResponse(BaseModel):
    patient_id: int
    patient_version: int
    projection_version: int
    event_ids: list[str]
    record_id: int
    reused: bool = False


def _runtime_dependencies(request: Request) -> tuple[Any, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    session_store = getattr(runtime, "session_store", None)
    patient_commands = getattr(runtime, "patient_command_service", None)
    if session_store is None or patient_commands is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized")
    return session_store, patient_commands


@router.post("/{session_id}/crc-triage/assessments")
async def save_crc_triage_assessment(
    session_id: str,
    request: Request,
    payload: SaveCrcTriageAssessmentRequest,
) -> SaveCrcTriageAssessmentResponse:
    session_store, patient_commands = _runtime_dependencies(request)
    meta = session_store.get_session(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if meta.active_run_id is not None:
        raise HTTPException(status_code=409, detail="SESSION_BUSY")
    if meta.scene != "patient":
        raise HTTPException(status_code=409, detail="NOT_PATIENT_SESSION")
    if meta.patient_id is None:
        raise HTTPException(status_code=409, detail="PATIENT_IDENTITY_NOT_FOUND")

    result = patient_commands.record_crc_triage_assessment(
        patient_id=meta.patient_id,
        assessment=payload.assessment.model_dump(),
        source_session_id=session_id,
    )
    if result.record_id is None:
        raise HTTPException(status_code=500, detail="CRC_TRIAGE_RECORD_NOT_CREATED")

    bump_snapshot_version = getattr(session_store, "bump_snapshot_version", None)
    if callable(bump_snapshot_version):
        bump_snapshot_version(session_id)

    return SaveCrcTriageAssessmentResponse(
        patient_id=result.patient_id,
        patient_version=result.patient_version,
        projection_version=result.projection_version,
        event_ids=result.event_ids,
        record_id=result.record_id,
        reused=result.reused,
    )
