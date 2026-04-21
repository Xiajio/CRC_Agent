from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from backend.api.adapters.state_snapshot import build_message_history, build_session_snapshot
from backend.api.schemas.responses import MessageHistoryResponse, SessionResponse
from backend.api.services.patient_registry_service import (
    PatientIdentityLockedError,
    PatientIdentityNotFoundError,
    PatientNumberConflictError,
)
from backend.api.services.session_store import InMemorySessionStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
session_store = InMemorySessionStore()
patient_registry_service: Any | None = None


class CreateSessionRequest(BaseModel):
    scene: str = Field(default="doctor")


class BindPatientRequest(BaseModel):
    patient_id: int


class UpdatePatientIdentityRequest(BaseModel):
    patient_name: str
    patient_number: str

    @field_validator("patient_name", "patient_number", mode="before")
    @classmethod
    def _strip_and_reject_blank(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value.strip()
        if not candidate:
            raise ValueError("must not be blank")
        return candidate


def load_agent_state(session_id: str) -> dict[str, Any] | None:
    del session_id
    return None


def get_runtime_metadata() -> dict[str, Any]:
    return {
        "runner_mode": "real",
        "fixture_case": None,
    }


def _get_session_meta_or_404(session_id: str):
    meta = session_store.get_session(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return meta


def _load_patient_identity(meta) -> dict[str, Any] | None:
    if meta.patient_id is None or patient_registry_service is None:
        return None
    try:
        return patient_registry_service.get_patient_identity(meta.patient_id)
    except (PatientIdentityNotFoundError, KeyError):
        return None


def _build_session_response(session_id: str) -> SessionResponse:
    meta = _get_session_meta_or_404(session_id)
    snapshot = build_session_snapshot(load_agent_state(session_id), meta)
    snapshot.patient_identity = _load_patient_identity(meta)
    return SessionResponse(
        session_id=meta.session_id,
        thread_id=meta.thread_id,
        scene=meta.scene,
        patient_id=meta.patient_id,
        snapshot_version=meta.snapshot_version,
        snapshot=snapshot,
        runtime=get_runtime_metadata(),
    )


@router.post("")
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    scene = (request.scene or "doctor").strip().lower()
    if scene not in {"patient", "doctor"}:
        raise HTTPException(status_code=422, detail="Unsupported scene")

    meta = session_store.create_session(scene=scene)
    if scene == "patient":
        if patient_registry_service is None:
            raise HTTPException(status_code=503, detail="Patient registry is not initialized")
        patient_id = patient_registry_service.create_draft_patient(
            created_by_session_id=meta.session_id,
        )
        session_store.set_patient_id(meta.session_id, patient_id, allow_replace=True)

    return _build_session_response(meta.session_id)


@router.get("/{session_id}")
async def get_session(session_id: str) -> SessionResponse:
    return _build_session_response(session_id)


@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    before: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> MessageHistoryResponse:
    meta = _get_session_meta_or_404(session_id)
    history = build_message_history(load_agent_state(session_id), before=before, limit=limit)
    return MessageHistoryResponse(
        session_id=meta.session_id,
        thread_id=meta.thread_id,
        snapshot_version=meta.snapshot_version,
        messages_total=history.messages_total,
        next_before_cursor=history.next_before_cursor,
        messages=history.messages,
    )


@router.post("/{session_id}/reset")
async def reset_session(session_id: str) -> SessionResponse:
    meta = _get_session_meta_or_404(session_id)
    if meta.active_run_id is not None:
        raise HTTPException(status_code=409, detail="Session is busy")
    session_store.rotate_thread(session_id, clear_patient_id=meta.scene == "doctor")
    session_store.bump_snapshot_version(session_id)
    return _build_session_response(session_id)


@router.patch("/{session_id}")
async def bind_session_patient(session_id: str, request: BindPatientRequest) -> SessionResponse:
    meta = _get_session_meta_or_404(session_id)
    if meta.active_run_id is not None:
        raise HTTPException(status_code=409, detail="Session is busy")
    if meta.scene != "doctor":
        raise HTTPException(status_code=409, detail="Only doctor sessions can bind patients")
    try:
        session_store.bind_patient(session_id, request.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session_store.bump_snapshot_version(session_id)
    return _build_session_response(session_id)


@router.post("/{session_id}/identity")
async def set_session_patient_identity(
    session_id: str,
    request: UpdatePatientIdentityRequest,
) -> SessionResponse:
    meta = _get_session_meta_or_404(session_id)
    if meta.active_run_id is not None:
        raise HTTPException(status_code=409, detail="Session is busy")
    if meta.scene != "patient":
        raise HTTPException(status_code=409, detail="NOT_PATIENT_SESSION")
    if meta.patient_id is None:
        raise HTTPException(status_code=409, detail="PATIENT_IDENTITY_NOT_FOUND")
    if patient_registry_service is None:
        raise HTTPException(status_code=503, detail="Patient registry is not initialized")

    try:
        patient_registry_service.set_patient_identity(
            meta.patient_id,
            request.patient_name,
            request.patient_number,
        )
    except PatientNumberConflictError as exc:
        raise HTTPException(status_code=409, detail="PATIENT_NUMBER_ALREADY_EXISTS") from exc
    except PatientIdentityLockedError as exc:
        raise HTTPException(status_code=409, detail="PATIENT_IDENTITY_LOCKED") from exc
    except PatientIdentityNotFoundError as exc:
        raise HTTPException(status_code=409, detail="PATIENT_IDENTITY_NOT_FOUND") from exc

    session_store.bump_snapshot_version(session_id)
    return _build_session_response(session_id)
