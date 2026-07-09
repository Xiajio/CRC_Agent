from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.research import CohortFeasibilityRequestPayload
from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


router = APIRouter(prefix="/api/admin/research", tags=["admin-research"])


def _get_registry_service(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    registry = getattr(runtime, "patient_registry_service", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Patient registry is not initialized",
        )
    return registry


@router.post("/cohort-feasibility")
async def evaluate_cohort_feasibility(
    payload: CohortFeasibilityRequestPayload,
    request: Request,
) -> dict[str, Any]:
    try:
        feasibility_request = CohortFeasibilityRequest(**payload.model_dump())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        service = CohortFeasibilityService()
        if feasibility_request.patient_level_export_requested:
            result = service.evaluate(feasibility_request, ())
        else:
            registry = _get_registry_service(request)
            records = registry.list_research_projection_records(limit=1000)
            result = service.evaluate(feasibility_request, records)
        response = result.to_dict()
        response["runtime"] = {
            "auth": "admin",
            "source": "patient_record_projection",
            "mode": "shadow_cohort_feasibility",
        }
        return response
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
