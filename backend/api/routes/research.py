from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from backend.api.schemas.research import (
    CohortFeasibilityRequestPayload,
    CreateAutoResearchRunRequest,
)
from backend.api.services.admin_release_dashboard import REPO_ROOT
from backend.api.services.auto_research_store import (
    AutoResearchRunNotFoundError,
    AutoResearchRunStore,
    AutoResearchStoreIntegrityError,
)
from src.config import load_settings
from src.contracts.auto_research import AutoResearchRequest
from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.auto_research_service import (
    AutoResearchConflictError,
    AutoResearchService,
    AutoResearchServiceUnavailableError,
    DeferredResearchReasoner,
    LLMResearchReasoner,
)
from src.services.cohort_feasibility_service import CohortFeasibilityService
from src.services.llm_service import LLMService
from src.services.pubmed_research import PubMedEvidenceRetriever


router = APIRouter(prefix="/api/admin/research", tags=["admin-research"])
_AUTO_RESEARCH_STORE_ROOT = REPO_ROOT / "reports" / "auto_research"


def _configured_research_reasoner() -> LLMResearchReasoner:
    try:
        settings = load_settings()
        if settings.llm.mode != "API":
            raise AutoResearchServiceUnavailableError(
                "Auto-research structured reasoning requires LLM_MODE=API with a "
                "function-calling compatible endpoint. In-process Local HF/VLLM "
                "models are not supported by this endpoint."
            )
        model = LLMService(settings.llm).create_chat_model()
    except AutoResearchServiceUnavailableError:
        raise
    except Exception as exc:
        raise AutoResearchServiceUnavailableError(
            "Auto-research reasoning model is unavailable. Configure LLM_MODE=API "
            "with a function-calling compatible LLM_API_BASE and, for non-local "
            "endpoints, LLM_API_KEY; then restart the backend."
        ) from exc
    return LLMResearchReasoner(model)


def _auto_research_service() -> AutoResearchService:
    return AutoResearchService(
        retriever=PubMedEvidenceRetriever(),
        reasoner=DeferredResearchReasoner(
            _configured_research_reasoner,
            provider_hint="llm:configured-lazy",
        ),
        store=AutoResearchRunStore(_AUTO_RESEARCH_STORE_ROOT),
    )


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


@router.get("/runs")
async def list_auto_research_runs() -> dict[str, Any]:
    try:
        return _auto_research_service().list_runs()
    except Exception as exc:
        _raise_auto_research_http_error(exc)


@router.get("/runs/{run_id}")
async def get_auto_research_run(run_id: str) -> dict[str, Any]:
    try:
        return _auto_research_service().get_run(run_id)
    except Exception as exc:
        _raise_auto_research_http_error(exc)


@router.post("/runs")
async def create_auto_research_run(
    payload: CreateAutoResearchRunRequest,
) -> dict[str, Any]:
    try:
        research_request = AutoResearchRequest(**payload.model_dump())
        service = _auto_research_service()
        return await run_in_threadpool(service.create_run, research_request)
    except Exception as exc:
        _raise_auto_research_http_error(exc)


def _raise_auto_research_http_error(exc: Exception) -> None:
    if isinstance(exc, AutoResearchRunNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(
        exc,
        (
            AutoResearchConflictError,
            AutoResearchStoreIntegrityError,
            FileExistsError,
        ),
    ):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, AutoResearchServiceUnavailableError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


__all__ = [
    "router",
    "_auto_research_service",
    "_get_registry_service",
]
