from __future__ import annotations

import re
from typing import Any

from backend.api.schemas.database import DatabaseQueryIntentResponse


SUPPORTED_LOCATIONS = [
    "横结肠",
    "直肠",
    "升结肠",
    "降结肠",
    "乙状结肠",
    "盲肠",
    "肝曲",
    "脾曲",
    "结肠",
]

SUPPORTED_HISTOLOGY = [
    "高分化",
    "中分化",
    "低分化",
    "粘液腺癌",
    "印戒细胞癌",
]

UNSUPPORTED_CONCEPTS = [
    "肝转移",
    "肺转移",
    "骨转移",
    "腹膜转移",
]


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"[，,。；;、]", " ", query)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_age_range(query: str) -> tuple[int | None, int | None]:
    range_match = re.search(r"(\d{1,3})\s*(?:到|-|至)\s*(\d{1,3})\s*岁", query)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return min(start, end), max(start, end)
    return None, None


def _parse_patient_id(query: str) -> int | None:
    for pattern in (
        r"(?:患者|病人|病例)\s*(\d{1,3})",
        r"(\d{1,3})\s*号(?:患者|病人|病例)?",
    ):
        match = re.search(pattern, query)
        if match:
            return int(match.group(1))
    return None


def _parse_mmr(query: str) -> list[str]:
    lowered = query.lower().replace(" ", "")
    buckets: list[str] = []
    if "pmmr" in lowered or "mss" in lowered:
        buckets.append("pMMR_MSS")
    if "dmmr" in lowered or "msi-h" in lowered or "msih" in lowered:
        buckets.append("dMMR_MSI_H")
    return buckets


def _parse_stage_values(query: str) -> tuple[list[str], list[str]]:
    ct_values = re.findall(r"c?t\s*([0-4](?:a|b)?)", query, flags=re.IGNORECASE)
    cn_values = re.findall(r"c?n\s*([0-3](?:a|b|c)?)", query, flags=re.IGNORECASE)
    return [value.lower() for value in ct_values], [value.lower() for value in cn_values]


def _parse_list_values(query: str, supported_values: list[str]) -> list[str]:
    matched: list[str] = []
    covered_query = query
    for value in sorted(supported_values, key=len, reverse=True):
        if value in covered_query:
            matched.append(value)
            covered_query = covered_query.replace(value, " ")
    return matched


def parse_database_query(query: str) -> DatabaseQueryIntentResponse:
    normalized_query = _normalize_query(query)
    age_min, age_max = _parse_age_range(normalized_query)
    patient_id = _parse_patient_id(normalized_query)
    tumor_locations = _parse_list_values(normalized_query, SUPPORTED_LOCATIONS)
    histology_type = _parse_list_values(normalized_query, SUPPORTED_HISTOLOGY)
    mmr_status = _parse_mmr(normalized_query)
    ct_stage, cn_stage = _parse_stage_values(normalized_query)

    unsupported_terms = [term for term in UNSUPPORTED_CONCEPTS if term in normalized_query]
    warnings = []
    if unsupported_terms:
        warnings.append("当前虚拟数据库不包含转移器官字段，未执行该条件筛选。")

    filters: dict[str, Any] = {}
    if patient_id is not None:
        filters["patient_id"] = patient_id
    if age_min is not None:
        filters["age_min"] = age_min
    if age_max is not None:
        filters["age_max"] = age_max
    if tumor_locations:
        filters["tumor_location"] = tumor_locations
    if histology_type:
        filters["histology_type"] = histology_type
    if mmr_status:
        filters["mmr_status"] = mmr_status
    if ct_stage:
        filters["ct_stage"] = ct_stage
    if cn_stage:
        filters["cn_stage"] = cn_stage

    return DatabaseQueryIntentResponse(
        query=query,
        normalized_query=normalized_query,
        filters=filters,
        unsupported_terms=unsupported_terms,
        warnings=warnings,
    )
