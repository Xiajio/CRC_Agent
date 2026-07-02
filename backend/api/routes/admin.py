from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.release_governance import (
    ReleaseGovernanceApprovalRequest,
    ReleaseGovernanceCancelRequest,
    ReleaseGovernanceCreateIntentRequest,
    ReleaseGovernanceRollbackPlanRequest,
)
from backend.api.services.admin_release_dashboard import REPO_ROOT, build_release_dashboard
from backend.api.services.release_governance_store import (
    GovernanceIntegrityError,
    ReleaseGovernanceStore,
)
from src.config import load_settings
from src.services.release_governance import (
    GovernanceConflictError,
    GovernanceValidationError,
    ReleaseGovernanceService,
)
from src.tools.manifest import build_tool_manifest_response

router = APIRouter(prefix="/api/admin", tags=["admin"])
_GOVERNANCE_STORE_ROOT = REPO_ROOT / "reports" / "release_governance"


def _web_search_enabled_from_request(request: Request) -> bool:
    runtime = getattr(request.app.state, "runtime", None)
    runtime_settings = getattr(runtime, "settings", None)
    runtime_web_search = getattr(runtime_settings, "web_search", None)
    runtime_enabled = getattr(runtime_web_search, "enabled", None)
    if runtime_enabled is not None:
        return bool(runtime_enabled)

    return bool(load_settings().web_search.enabled)


def _governance_timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _release_governance_service() -> ReleaseGovernanceService:
    return ReleaseGovernanceService(
        store=ReleaseGovernanceStore(_GOVERNANCE_STORE_ROOT),
        dashboard_loader=build_release_dashboard,
        now=_governance_timestamp,
    )


def _raise_governance_http_error(exc: Exception) -> None:
    if isinstance(exc, GovernanceConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, GovernanceIntegrityError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, FileExistsError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (GovernanceValidationError, TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _model_dump(model: Any) -> dict[str, Any]:
    return model.model_dump()


@router.get("/tools")
async def get_admin_tools(request: Request) -> dict[str, Any]:
    return build_tool_manifest_response(
        web_search_enabled=_web_search_enabled_from_request(request),
    )


@router.get("/release-dashboard")
async def get_admin_release_dashboard() -> dict[str, Any]:
    return build_release_dashboard()


@router.get("/release-governance")
async def get_admin_release_governance() -> dict[str, Any]:
    return _release_governance_service().read_governance()


@router.post("/release-governance/intents")
async def create_admin_release_intent(
    payload: ReleaseGovernanceCreateIntentRequest,
) -> dict[str, Any]:
    try:
        return _release_governance_service().create_intent(**_model_dump(payload))
    except Exception as exc:
        _raise_governance_http_error(exc)


@router.post("/release-governance/intents/{intent_id}/approvals")
async def record_admin_release_approval(
    intent_id: str,
    payload: ReleaseGovernanceApprovalRequest,
) -> dict[str, Any]:
    try:
        return _release_governance_service().record_approval(
            intent_id=intent_id,
            **_model_dump(payload),
        )
    except Exception as exc:
        _raise_governance_http_error(exc)


@router.post("/release-governance/intents/{intent_id}/rollback-plan")
async def record_admin_release_rollback_plan(
    intent_id: str,
    payload: ReleaseGovernanceRollbackPlanRequest,
) -> dict[str, Any]:
    try:
        return _release_governance_service().record_rollback_plan(
            intent_id=intent_id,
            **_model_dump(payload),
        )
    except Exception as exc:
        _raise_governance_http_error(exc)


@router.post("/release-governance/intents/{intent_id}/cancel")
async def cancel_admin_release_intent(
    intent_id: str,
    payload: ReleaseGovernanceCancelRequest,
) -> dict[str, Any]:
    try:
        return _release_governance_service().cancel_intent(
            intent_id=intent_id,
            **_model_dump(payload),
        )
    except Exception as exc:
        _raise_governance_http_error(exc)
