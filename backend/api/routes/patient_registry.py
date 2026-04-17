from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from backend.api.schemas.patient_registry import (
    PatientRegistryAlert,
    PatientRegistryAlertsResponse,
    PatientRegistryClearResponse,
    PatientRegistryDeleteResponse,
    PatientRegistryDetailResponse,
    PatientRegistryListResponse,
    PatientRegistryRecord,
    PatientRegistryRecordsResponse,
    PatientRegistrySearchRequest,
)

router = APIRouter(prefix="/api/patient-registry", tags=["patient-registry"])


def _get_registry_service(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    service = getattr(runtime, "patient_registry_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Patient registry is not initialized")
    return service


@router.get("/patients/recent", response_model=PatientRegistryListResponse)
async def list_recent_patients(
    request: Request,
    limit: int = Query(default=5, ge=1, le=100),
) -> PatientRegistryListResponse:
    service = _get_registry_service(request)
    items = service.list_recent_patients(limit=limit)
    return PatientRegistryListResponse(items=items, total=len(items))


@router.post("/patients/search", response_model=PatientRegistryListResponse)
async def search_patients(
    request: Request,
    body: PatientRegistrySearchRequest,
) -> PatientRegistryListResponse:
    service = _get_registry_service(request)
    payload = service.search_patients(
        patient_id=body.patient_id,
        tumor_location=body.tumor_location,
        mmr_status=body.mmr_status,
        clinical_stage=body.clinical_stage,
        limit=body.limit,
    )
    return PatientRegistryListResponse(**payload)


@router.get("/patients/{patient_id}", response_model=PatientRegistryDetailResponse)
async def get_patient_detail(request: Request, patient_id: int) -> PatientRegistryDetailResponse:
    service = _get_registry_service(request)
    try:
        payload = service.get_patient_detail(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc
    return PatientRegistryDetailResponse(**payload)


@router.get("/patients/{patient_id}/records", response_model=PatientRegistryRecordsResponse)
async def list_patient_records(request: Request, patient_id: int) -> PatientRegistryRecordsResponse:
    service = _get_registry_service(request)
    try:
        service.get_patient_detail(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc
    rows = service.list_patient_records(patient_id)
    return PatientRegistryRecordsResponse(
        items=[PatientRegistryRecord.from_row(row) for row in rows],
    )


@router.get("/patients/{patient_id}/alerts", response_model=PatientRegistryAlertsResponse)
async def list_patient_alerts(request: Request, patient_id: int) -> PatientRegistryAlertsResponse:
    service = _get_registry_service(request)
    try:
        service.get_patient_detail(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc
    rows = service.list_patient_alerts(patient_id)
    return PatientRegistryAlertsResponse(items=[PatientRegistryAlert(**row) for row in rows])


@router.delete("/patients/{patient_id}", response_model=PatientRegistryDeleteResponse)
async def delete_patient(request: Request, patient_id: int) -> PatientRegistryDeleteResponse:
    service = _get_registry_service(request)
    try:
        payload = service.delete_patient(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc
    return PatientRegistryDeleteResponse(**payload)


@router.delete("/patients", response_model=PatientRegistryClearResponse)
async def clear_registry(request: Request) -> PatientRegistryClearResponse:
    service = _get_registry_service(request)
    payload = service.clear_registry()
    return PatientRegistryClearResponse(**payload)
