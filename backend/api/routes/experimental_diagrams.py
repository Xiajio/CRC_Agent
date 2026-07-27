from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.api.schemas.diagrams import (
    CompileDiagramRequest,
    CompileDiagramResponse,
)
from src.config import load_settings
from src.contracts.diagram import DiagramCompileRequest
from src.services.diagram_service import (
    DiagramGenerationError,
    DiagramOutputValidationError,
    DiagramService,
    DiagramServiceUnavailableError,
)
from src.services.llm_service import LLMService


router = APIRouter(
    prefix="/api/admin/experimental/diagrams",
    tags=["admin-experimental-diagrams"],
)


def _diagram_service() -> DiagramService:
    try:
        settings = load_settings()
        if settings.llm.mode != "API":
            raise DiagramServiceUnavailableError(
                "Diagram compilation requires LLM_MODE=API with a function-calling "
                "compatible endpoint. In-process Local HF/VLLM models are not "
                "supported by this endpoint."
            )
        model = LLMService(settings.llm).create_chat_model()
        return DiagramService(model)
    except DiagramServiceUnavailableError:
        raise
    except Exception as exc:
        raise DiagramServiceUnavailableError(
            "The diagram compilation model is unavailable. Configure LLM_MODE=API "
            "with a function-calling compatible provider and restart the backend."
        ) from exc


def _compile_diagram(request: DiagramCompileRequest) -> Any:
    return _diagram_service().compile(request)


@router.post("/compile", response_model=CompileDiagramResponse)
async def compile_diagram(payload: CompileDiagramRequest) -> CompileDiagramResponse:
    try:
        request = DiagramCompileRequest(**payload.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(_compile_diagram, request)
    except DiagramServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DiagramGenerationError, DiagramOutputValidationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return CompileDiagramResponse.model_validate(result.to_dict())
    except (AttributeError, TypeError, ValidationError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Diagram compilation returned an invalid response payload.",
        ) from exc


__all__ = ["router", "_compile_diagram", "_diagram_service"]
