from __future__ import annotations

from typing import Any

from src.services.case_excel_service import upsert_case_record
from src.services import virtual_database_service as virtual_db_module
from src.services.virtual_database_service import (
    get_case_database,
    get_case_statistics,
    get_imaging_by_patient_id,
    get_pathology_slides_by_patient_id,
)
from src.tools.card_formatter import CardFormatter

from backend.api.schemas.database import (
    DatabaseAvailableData,
    DatabaseFilters,
    DatabaseNumericStatistics,
    DatabasePagination,
    DatabaseSearchResponse,
    DatabaseSort,
    DatabaseStatsResponse,
)


class DatabaseCaseNotFoundError(Exception):
    pass


_formatter = CardFormatter()


EMPTY_STATS_PAYLOAD: dict[str, Any] = DatabaseStatsResponse().model_dump()


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalized_mmr_bucket(value: Any) -> str:
    raw = _normalized_text(value)
    if "pmmr" in raw or "mss" in raw or raw == "1":
        return "pMMR_MSS"
    if "dmmr" in raw or "msi-h" in raw or "msih" in raw or raw == "2":
        return "dMMR_MSI_H"
    return ""


def _matches_list_filter(candidate: Any, accepted_values: list[str]) -> bool:
    if not accepted_values:
        return True
    candidate_text = _normalized_text(candidate)
    normalized_values = [_normalized_text(item) for item in accepted_values if _normalized_text(item)]
    return any(value in candidate_text or candidate_text in value for value in normalized_values)


def _normalize_numeric_stats(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = value
    else:
        payload = {}

    return DatabaseNumericStatistics(
        min=payload.get("min"),
        max=payload.get("max"),
        mean=payload.get("mean"),
    ).model_dump()


def _normalize_stats_payload(value: Any) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    return DatabaseStatsResponse(
        total_cases=int(payload.get("total_cases") or 0),
        gender_distribution=dict(payload.get("gender_distribution") or {}),
        age_statistics=DatabaseNumericStatistics(**_normalize_numeric_stats(payload.get("age_statistics"))),
        tumor_location_distribution=dict(payload.get("tumor_location_distribution") or {}),
        ct_stage_distribution=dict(payload.get("ct_stage_distribution") or {}),
        mmr_status_distribution=dict(payload.get("mmr_status_distribution") or {}),
        cea_statistics=DatabaseNumericStatistics(**_normalize_numeric_stats(payload.get("cea_statistics"))),
    ).model_dump()


def _apply_filters(items: list[dict[str, Any]], filters: DatabaseFilters) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        if filters.patient_id is not None and int(item.get("patient_id") or 0) != filters.patient_id:
            continue
        if not _matches_list_filter(item.get("tumor_location"), filters.tumor_location):
            continue
        if not _matches_list_filter(item.get("ct_stage"), filters.ct_stage):
            continue
        if not _matches_list_filter(item.get("cn_stage"), filters.cn_stage):
            continue
        if not _matches_list_filter(item.get("histology_type"), filters.histology_type):
            continue
        if filters.mmr_status:
            mmr_bucket = _normalized_mmr_bucket(item.get("mmr_status"))
            if mmr_bucket not in filters.mmr_status:
                continue
        age = int(item.get("age") or 0)
        if filters.age_min is not None and age < filters.age_min:
            continue
        if filters.age_max is not None and age > filters.age_max:
            continue
        cea_level = float(item.get("cea_level") or 0.0)
        if filters.cea_max is not None and cea_level > filters.cea_max:
            continue
        if filters.family_history is not None and item.get("family_history") is not filters.family_history:
            continue
        if filters.biopsy_confirmed is not None and item.get("biopsy_confirmed") is not filters.biopsy_confirmed:
            continue
        ecog_score = item.get("ecog_score")
        if filters.ecog_min is not None:
            if ecog_score is None or int(ecog_score) < filters.ecog_min:
                continue
        if filters.ecog_max is not None:
            if ecog_score is None or int(ecog_score) > filters.ecog_max:
                continue
        results.append(item)
    return results


def _sort_key(field: str, item: dict[str, Any]) -> tuple[int, Any]:
    value = item.get(field)
    if value is None:
        return (1, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (0, str(value).lower())


def _applied_filters(filters: DatabaseFilters) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key, value in filters.model_dump().items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        applied[key] = value
    return applied


def get_database_stats() -> dict[str, Any]:
    return _normalize_stats_payload(get_case_statistics())


def search_database_cases(
    filters: DatabaseFilters,
    pagination: DatabasePagination,
    sort: DatabaseSort,
) -> DatabaseSearchResponse:
    db = get_case_database()
    items = db.get_all_cases()
    filtered = _apply_filters(items, filters)
    reverse = sort.direction == "desc"
    filtered.sort(key=lambda item: _sort_key(sort.field, item), reverse=reverse)

    total = len(filtered)
    start = (pagination.page - 1) * pagination.page_size
    end = start + pagination.page_size
    paged = filtered[start:end]

    return DatabaseSearchResponse(
        items=paged,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        applied_filters=_applied_filters(filters),
        warnings=[],
    )


def get_database_case_detail(patient_id: int) -> dict[str, Any]:
    db = get_case_database()
    case_record = db.get_case_by_id(patient_id)
    if not case_record:
        raise DatabaseCaseNotFoundError(f"Patient {patient_id} not found")

    cards: dict[str, dict[str, Any]] = {
        "patient_card": _formatter.format_patient_card(patient_id),
    }

    imaging_payload = get_imaging_by_patient_id(patient_id)
    pathology_payload = get_pathology_slides_by_patient_id(str(patient_id))

    if isinstance(imaging_payload, dict) and "error" not in imaging_payload:
        cards["imaging_card"] = _formatter.format_imaging_card(imaging_payload)
    if isinstance(pathology_payload, dict) and "error" not in pathology_payload:
        cards["pathology_slide_card"] = _formatter.format_pathology_slide_card(pathology_payload)

    available_data = DatabaseAvailableData(
        case_info=True,
        imaging="imaging_card" in cards,
        pathology_slides="pathology_slide_card" in cards,
    )

    return {
        "patient_id": str(patient_id).zfill(3),
        "case_record": case_record,
        "available_data": available_data.model_dump(),
        "cards": cards,
    }


def upsert_database_case(record: dict[str, Any]) -> dict[str, Any]:
    upsert_case_record(virtual_db_module.CLASSIFICATION_FILE, record)
    patient_id = int(record["patient_id"])
    db = get_case_database()
    db._load_data()
    return get_database_case_detail(patient_id)