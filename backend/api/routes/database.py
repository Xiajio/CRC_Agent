from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas.database import (
    DatabaseCaseDetailResponse,
    DatabaseQueryIntentRequest,
    DatabaseQueryIntentResponse,
    DatabaseSearchRequest,
    DatabaseSearchResponse,
    DatabaseUpsertRequest,
)
from backend.api.services.database_intent_service import parse_database_query
from backend.api.services.database_service import (
    DatabaseCaseNotFoundError,
    get_database_case_detail,
    get_database_stats,
    search_database_cases,
    upsert_database_case,
)

router = APIRouter(prefix="/api/database", tags=["database"])


@router.get("/stats")
async def get_stats() -> dict:
    return get_database_stats()


@router.post("/cases/search", response_model=DatabaseSearchResponse)
async def search_cases(body: DatabaseSearchRequest) -> DatabaseSearchResponse:
    return search_database_cases(body.filters, body.pagination, body.sort)


@router.get("/cases/{patient_id}", response_model=DatabaseCaseDetailResponse)
async def get_case_detail(patient_id: int) -> DatabaseCaseDetailResponse:
    try:
        payload = get_database_case_detail(patient_id)
    except DatabaseCaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc
    return DatabaseCaseDetailResponse(**payload)


@router.post("/cases/upsert", response_model=DatabaseCaseDetailResponse)
async def upsert_case(body: DatabaseUpsertRequest) -> DatabaseCaseDetailResponse:
    payload = upsert_database_case(body.record)
    return DatabaseCaseDetailResponse(**payload)


@router.post("/query-intent", response_model=DatabaseQueryIntentResponse)
async def query_intent(body: DatabaseQueryIntentRequest) -> DatabaseQueryIntentResponse:
    return parse_database_query(body.query)
