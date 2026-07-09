from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.schemas.learning_jobs import CreateLearningJobRequest
from backend.api.services.admin_release_dashboard import REPO_ROOT
from backend.api.services.learning_job_store import (
    LearningJobIntegrityError,
    LearningJobStore,
)
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import (
    LearningJobService,
    LearningJobValidationError,
)


router = APIRouter(prefix="/api/admin/learning-jobs", tags=["admin-learning-jobs"])
_LEARNING_JOB_STORE_ROOT = REPO_ROOT / "reports" / "learning_jobs"


def _timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _learning_job_service() -> LearningJobService:
    return LearningJobService(
        store=LearningJobStore(_LEARNING_JOB_STORE_ROOT),
        now=_timestamp,
    )


@router.get("")
async def get_learning_jobs() -> dict[str, Any]:
    try:
        return _learning_job_service().read_jobs()
    except Exception as exc:
        _raise_learning_job_http_error(exc)


@router.post("")
async def create_learning_job(payload: CreateLearningJobRequest) -> dict[str, Any]:
    try:
        signals = [
            LearningSignal(
                signal_id=make_learning_signal_id(signal_payload.source_ref),
                **signal_payload.model_dump(),
            )
            for signal_payload in payload.signals
        ]
        return _learning_job_service().create_job(
            signals,
            requested_by=payload.requested_by,
            idempotency_key=payload.idempotency_key,
        )
    except Exception as exc:
        _raise_learning_job_http_error(exc)


def _raise_learning_job_http_error(exc: Exception) -> None:
    if isinstance(exc, (LearningJobIntegrityError, FileExistsError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (LearningJobValidationError, TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


__all__ = ["router", "_learning_job_service", "_timestamp"]
