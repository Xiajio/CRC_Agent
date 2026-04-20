from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PLACEHOLDER_TEXT = {
    "",
    "unknown",
    "not_provided",
    "pending_evaluation",
    "pending_assessment",
    "null",
    "none",
    "n/a",
    "na",
}
PROJECTION_VERSION = "patient_self_report.v1"
TRIAGE_FRESH_FIELDS = {
    "history_block.chief_complaint",
    "history_block.symptom_duration",
}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in PLACEHOLDER_TEXT:
        return None
    return text


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = _normalize_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "y", "1", "present", "positive"}:
        return True
    if lowered in {"false", "no", "n", "0", "absent", "negative"}:
        return False
    return None


def _normalize_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    text = _normalize_text(value)
    if text is None:
        return None
    if text.isdigit():
        return int(text)
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _normalize_enum(value: Any, mapping: dict[str, str]) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    return mapping.get(text.lower(), text)


def _normalize_list(value: Any) -> list[Any] | None:
    if value is None:
        return None

    items: list[Any]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        text = _normalize_text(value)
        if text is None:
            return None
        items = [part.strip() for part in text.split(",")]

    normalized: list[Any] = []
    seen: set[str] = set()
    for item in items:
        number = _normalize_number(item)
        if number is not None:
            marker = f"n:{number}"
            if marker not in seen:
                normalized.append(number)
                seen.add(marker)
            continue

        boolean = _normalize_bool(item)
        if boolean is not None:
            marker = f"b:{boolean}"
            if marker not in seen:
                normalized.append(boolean)
                seen.add(marker)
            continue

        text = _normalize_text(item)
        if text is None:
            continue
        marker = f"t:{text.lower()}"
        if marker not in seen:
            normalized.append(text.lower())
            seen.add(marker)

    if not normalized:
        return None
    return sorted(normalized, key=lambda item: str(item))


def _normalize_site(value: Any) -> str | None:
    return _normalize_enum(
        value,
        {
            "rectum": "rectum",
            "rectal": "rectum",
            "colon": "colon",
            "colonic": "colon",
        },
    )


def _normalize_gender(value: Any) -> str | None:
    return _normalize_enum(
        value,
        {
            "male": "male",
            "m": "male",
            "man": "male",
            "female": "female",
            "f": "female",
            "woman": "female",
        },
    )


def _normalize_mmr(value: Any) -> str | None:
    return _normalize_enum(
        value,
        {
            "pmmr": "pMMR",
            "dmmr": "dMMR",
            "mss": "MSS",
            "msi-h": "MSI-H",
            "msih": "MSI-H",
        },
    )


def _normalize_stage(value: Any) -> str | None:
    text = _normalize_text(value)
    return text.upper() if text is not None else None


def _resolution_value(value: Any, kind: str) -> Any:
    if kind == "bool":
        return _normalize_bool(value)
    if kind == "number":
        return _normalize_number(value)
    if kind == "gender":
        return _normalize_gender(value)
    if kind == "site":
        return _normalize_site(value)
    if kind == "mmr":
        return _normalize_mmr(value)
    if kind == "stage":
        return _normalize_stage(value)
    if kind == "list":
        return _normalize_list(value)
    return _normalize_text(value)


def _canonical_marker(value: Any, kind: str) -> Any:
    resolved = _resolution_value(value, kind)
    if resolved is None:
        return None
    if kind == "text":
        return str(resolved).lower()
    if kind == "list" and isinstance(resolved, list):
        return [str(item).lower() if isinstance(item, str) else item for item in resolved]
    return resolved


def _display_value(value: Any, status: str) -> str:
    if status == "pending":
        return "待确认"
    if status == "conflict":
        return "待确认（来源不一致）"
    if value is None:
        return "待确认"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _source_entry(source_path: str, value: Any, kind: str) -> dict[str, Any] | None:
    resolved = _resolution_value(value, kind)
    canonical = _canonical_marker(value, kind)
    if resolved is None or canonical is None:
        return None
    source_type = source_path.split(".", 1)[0]
    return {
        "source_type": source_type,
        "source_path": source_path,
        "value": value,
        "display_value": _display_value(resolved, "confirmed"),
        "canonical": canonical,
        "resolved": resolved,
    }


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _medical_card_sections(medical_card: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = _coerce_mapping(medical_card)
    data = _coerce_mapping(payload.get("data"))
    return (
        _coerce_mapping(data.get("patient_summary")),
        _coerce_mapping(data.get("patient_info")),
        _coerce_mapping(data.get("diagnosis_block")),
        _coerce_mapping(data.get("staging_block")),
        _coerce_mapping(data.get("history_block")),
    )


def _field_specs(
    patient_profile: dict[str, Any],
    findings: dict[str, Any],
    symptom_snapshot: dict[str, Any],
    medical_card: Any,
) -> dict[str, dict[str, Any]]:
    medical_summary, medical_patient_info, medical_diagnosis, medical_staging, medical_history = _medical_card_sections(medical_card)
    findings_tnm = _coerce_mapping(findings.get("tnm_staging"))
    profile_tnm = _coerce_mapping(patient_profile.get("tnm_staging"))

    return {
        "patient_info.gender": {
            "kind": "gender",
            "candidates": [
                ("patient_profile.gender", patient_profile.get("gender")),
                ("medical_card.patient_summary.gender", medical_summary.get("gender")),
                ("medical_card.patient_info.gender", medical_patient_info.get("gender")),
            ],
        },
        "patient_info.age": {
            "kind": "number",
            "candidates": [
                ("patient_profile.age", patient_profile.get("age")),
                ("medical_card.patient_summary.age", medical_summary.get("age")),
                ("medical_card.patient_info.age", medical_patient_info.get("age")),
            ],
        },
        "patient_info.ecog": {
            "kind": "number",
            "candidates": [
                ("patient_profile.ecog_score", patient_profile.get("ecog_score")),
                ("medical_card.patient_summary.ecog", medical_summary.get("ecog")),
                ("medical_card.patient_info.ecog", medical_patient_info.get("ecog")),
            ],
        },
        "patient_info.cea": {
            "kind": "number",
            "candidates": [
                ("findings.cea_level", findings.get("cea_level")),
                ("medical_card.patient_summary.cea", medical_summary.get("cea")),
                ("medical_card.patient_info.cea", medical_patient_info.get("cea")),
            ],
        },
        "diagnosis_block.confirmed": {
            "kind": "bool",
            "candidates": [
                ("findings.pathology_confirmed", findings.get("pathology_confirmed")),
                ("findings.biopsy_confirmed", findings.get("biopsy_confirmed")),
                ("patient_profile.pathology_confirmed", patient_profile.get("pathology_confirmed")),
                ("medical_card.diagnosis_block.confirmed", medical_diagnosis.get("confirmed")),
                ("medical_card.history_block.biopsy_confirmed", medical_history.get("biopsy_confirmed")),
            ],
        },
        "diagnosis_block.primary_site": {
            "kind": "site",
            "candidates": [
                ("findings.tumor_location", findings.get("tumor_location")),
                ("medical_card.diagnosis_block.primary_site", medical_diagnosis.get("primary_site")),
                ("medical_card.diagnosis_block.location", medical_diagnosis.get("location")),
            ],
        },
        "diagnosis_block.mmr_status": {
            "kind": "mmr",
            "candidates": [
                ("findings.mmr_status", findings.get("mmr_status")),
                ("patient_profile.mmr_status", patient_profile.get("mmr_status")),
                ("medical_card.diagnosis_block.mmr_status", medical_diagnosis.get("mmr_status")),
            ],
        },
        "staging_block.clinical_stage": {
            "kind": "stage",
            "candidates": [
                ("findings.clinical_stage_summary", findings.get("clinical_stage_summary")),
                ("findings.clinical_stage", findings.get("clinical_stage")),
                ("medical_card.staging_block.clinical_stage", medical_staging.get("clinical_stage")),
            ],
        },
        "staging_block.ct_stage": {
            "kind": "text",
            "candidates": [
                ("findings.tnm_staging.cT", findings_tnm.get("cT")),
                ("patient_profile.tnm_staging.cT", profile_tnm.get("cT")),
                ("medical_card.staging_block.t_stage", medical_staging.get("t_stage")),
            ],
        },
        "staging_block.cn_stage": {
            "kind": "text",
            "candidates": [
                ("findings.tnm_staging.cN", findings_tnm.get("cN")),
                ("patient_profile.tnm_staging.cN", profile_tnm.get("cN")),
                ("medical_card.staging_block.n_stage", medical_staging.get("n_stage")),
            ],
        },
        "staging_block.cm_stage": {
            "kind": "text",
            "candidates": [
                ("findings.tnm_staging.cM", findings_tnm.get("cM")),
                ("patient_profile.tnm_staging.cM", profile_tnm.get("cM")),
                ("medical_card.staging_block.m_stage", medical_staging.get("m_stage")),
            ],
        },
        "history_block.chief_complaint": {
            "kind": "text",
            "candidates": [
                ("patient_profile.chief_complaint", patient_profile.get("chief_complaint")),
                ("symptom_snapshot.chief_symptoms", symptom_snapshot.get("chief_symptoms")),
                ("medical_card.patient_summary.chief_complaint", medical_summary.get("chief_complaint")),
                ("medical_card.history_block.chief_complaint", medical_history.get("chief_complaint")),
            ],
        },
        "history_block.symptom_duration": {
            "kind": "text",
            "candidates": [
                ("symptom_snapshot.duration", symptom_snapshot.get("duration")),
                ("medical_card.history_block.symptom_duration", medical_history.get("symptom_duration")),
            ],
        },
        "history_block.family_history": {
            "kind": "bool",
            "candidates": [
                ("findings.family_history", findings.get("family_history")),
                ("medical_card.history_block.family_history", medical_history.get("family_history")),
            ],
        },
        "history_block.family_history_details": {
            "kind": "text",
            "candidates": [
                ("findings.family_history_details", findings.get("family_history_details")),
                ("medical_card.history_block.family_history_details", medical_history.get("family_history_details")),
            ],
        },
        "history_block.biopsy_confirmed": {
            "kind": "bool",
            "candidates": [
                ("findings.biopsy_confirmed", findings.get("biopsy_confirmed")),
                ("findings.pathology_confirmed", findings.get("pathology_confirmed")),
                ("patient_profile.pathology_confirmed", patient_profile.get("pathology_confirmed")),
                ("medical_card.history_block.biopsy_confirmed", medical_history.get("biopsy_confirmed")),
                ("medical_card.diagnosis_block.confirmed", medical_diagnosis.get("confirmed")),
            ],
        },
        "history_block.biopsy_details": {
            "kind": "text",
            "candidates": [
                ("findings.biopsy_details", findings.get("biopsy_details")),
                ("medical_card.history_block.biopsy_details", medical_history.get("biopsy_details")),
            ],
        },
        "history_block.risk_factors": {
            "kind": "list",
            "candidates": [
                ("findings.risk_factors", findings.get("risk_factors")),
                ("medical_card.history_block.risk_factors", medical_history.get("risk_factors")),
            ],
        },
    }


def _resolve_field(field_key: str, kind: str, candidates: list[tuple[str, Any]]) -> tuple[Any, dict[str, str], list[dict[str, Any]]]:
    entries = [entry for entry in (_source_entry(source_path, value, kind) for source_path, value in candidates) if entry is not None]
    public_candidates = [
        {
            "source_type": entry["source_type"],
            "source_path": entry["source_path"],
            "value": entry["value"],
            "display_value": entry["display_value"],
        }
        for entry in entries
    ]

    if field_key in TRIAGE_FRESH_FIELDS:
        triage_entries = [entry for entry in entries if entry["source_type"] == "symptom_snapshot"]
        if triage_entries:
            chosen = triage_entries[-1]["resolved"]
            return chosen, {"status": "confirmed", "display": _display_value(chosen, "confirmed")}, public_candidates

    if not entries:
        return None, {"status": "pending", "display": _display_value(None, "pending")}, public_candidates

    canonical_markers = []
    seen_markers: set[str] = set()
    for entry in entries:
        marker = repr(entry["canonical"])
        if marker not in seen_markers:
            seen_markers.add(marker)
            canonical_markers.append(marker)

    if len(canonical_markers) > 1:
        return None, {"status": "conflict", "display": _display_value(None, "conflict")}, public_candidates

    chosen = entries[0]["resolved"]
    return chosen, {"status": "confirmed", "display": _display_value(chosen, "confirmed")}, public_candidates


def project_patient_card(
    *,
    patient_profile: Any,
    findings: Any,
    symptom_snapshot: Any,
    medical_card: Any,
    patient_id: str = "current",
) -> dict[str, Any]:
    profile_payload = _coerce_mapping(patient_profile)
    findings_payload = _coerce_mapping(findings)
    symptom_payload = _coerce_mapping(symptom_snapshot)

    data = {
        "patient_info": {
            "gender": None,
            "age": None,
            "ecog": None,
            "cea": None,
        },
        "diagnosis_block": {
            "confirmed": None,
            "primary_site": None,
            "mmr_status": None,
        },
        "staging_block": {
            "clinical_stage": None,
            "ct_stage": None,
            "cn_stage": None,
            "cm_stage": None,
        },
        "history_block": {
            "chief_complaint": None,
            "symptom_duration": None,
            "family_history": None,
            "family_history_details": None,
            "biopsy_confirmed": None,
            "biopsy_details": None,
            "risk_factors": None,
        },
    }
    field_meta = {
        "patient_info": {},
        "diagnosis_block": {},
        "staging_block": {},
        "history_block": {},
    }
    source_candidates: dict[str, list[dict[str, Any]]] = {}

    specs = _field_specs(profile_payload, findings_payload, symptom_payload, medical_card)
    confirmed_count = 0
    conflict_count = 0
    total_fields = len(specs)

    for field_key, spec in specs.items():
        resolved, meta, candidates = _resolve_field(field_key, str(spec["kind"]), list(spec["candidates"]))
        _set_nested(data, field_key, resolved)
        section, field = field_key.split(".", 1)
        field_meta[section][field] = meta
        source_candidates[field_key] = candidates
        if meta["status"] == "confirmed":
            confirmed_count += 1
        elif meta["status"] == "conflict":
            conflict_count += 1

    summary = (
        data["history_block"]["chief_complaint"]
        or data["staging_block"]["clinical_stage"]
        or data["diagnosis_block"]["primary_site"]
        or "Patient self-report summary"
    )
    completion_ratio = round(confirmed_count / total_fields, 4) if total_fields else 0.0

    return {
        "type": "patient_card",
        "patient_id": patient_id,
        "card_meta": {
            "source_mode": "patient_self_report",
            "completion_ratio": completion_ratio,
            "conflict_count": conflict_count,
            "projection_version": PROJECTION_VERSION,
        },
        "data": data,
        "field_meta": field_meta,
        "source_candidates": source_candidates,
        "text_summary": summary,
    }


def project_patient_card_for_updates(
    *,
    patient_profile: Any,
    findings: Any,
    symptom_snapshot: Any,
    medical_card: Any,
    patient_id: str = "current",
) -> dict[str, Any]:
    return project_patient_card(
        patient_profile=patient_profile,
        findings=findings,
        symptom_snapshot=symptom_snapshot,
        medical_card=medical_card,
        patient_id=patient_id,
    )


def project_patient_self_report_card(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    payload = _coerce_mapping(state)
    return project_patient_card(
        patient_profile=payload.get("patient_profile"),
        findings=payload.get("findings"),
        symptom_snapshot=payload.get("symptom_snapshot"),
        medical_card=payload.get("medical_card"),
        patient_id=str(payload.get("current_patient_id") or "current"),
    )
