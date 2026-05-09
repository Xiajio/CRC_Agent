"""Shared patient identity helpers for graph nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..state import CRCAgentState


def first_present(*values: Any) -> Any | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_case_database_patient_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.zfill(3) if text.isdigit() else text


def normalize_registry_patient_id(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mapping_value(source: Mapping[str, Any] | None, key: str) -> Any | None:
    if not isinstance(source, Mapping):
        return None
    return source.get(key)


def resolve_registry_patient_id(
    state: CRCAgentState,
    findings: Mapping[str, Any] | None = None,
    *,
    candidate_values: Sequence[Any] = (),
) -> int | None:
    return normalize_registry_patient_id(
        first_present(
            *candidate_values,
            getattr(state, "registry_patient_id", None),
            _mapping_value(findings, "registry_patient_id"),
        )
    )


def resolve_legacy_current_patient_id(
    state: CRCAgentState,
    findings: Mapping[str, Any] | None = None,
) -> str | None:
    value = first_present(
        getattr(state, "current_patient_id", None),
        _mapping_value(findings, "current_patient_id"),
    )
    return str(value) if value is not None else None


def resolve_case_database_patient_id(
    state: CRCAgentState,
    findings: Mapping[str, Any] | None = None,
    *,
    candidate_values: Sequence[Any] = (),
    trailing_candidates: Sequence[Any] = (),
    include_registry: bool = True,
    include_case_database: bool = True,
    include_current: bool = True,
) -> str | None:
    values: list[Any] = list(candidate_values)

    if include_registry:
        registry_patient_id = resolve_registry_patient_id(state, findings)
        if registry_patient_id is not None:
            values.append(str(registry_patient_id))

    if include_case_database:
        values.extend(
            [
                getattr(state, "case_database_patient_id", None),
                _mapping_value(findings, "case_database_patient_id"),
            ]
        )

    if include_current:
        values.append(resolve_legacy_current_patient_id(state, findings))

    values.extend(trailing_candidates)
    return normalize_case_database_patient_id(first_present(*values))


def apply_split_identity(
    state: CRCAgentState,
    return_dict: dict[str, Any],
    *,
    findings: Mapping[str, Any] | None = None,
    case_database_candidate_values: Sequence[Any] = (),
    case_database_trailing_candidates: Sequence[Any] = (),
    include_registry_for_case: bool = True,
    include_case_database: bool = True,
    include_current: bool = True,
) -> dict[str, Any]:
    findings_source = findings if findings is not None else return_dict.get("findings")
    if not isinstance(findings_source, Mapping):
        findings_source = state.findings or {}

    registry_patient_id = resolve_registry_patient_id(state, findings_source)
    if registry_patient_id is not None:
        return_dict["registry_patient_id"] = registry_patient_id
        if isinstance(return_dict.get("findings"), dict):
            return_dict["findings"]["registry_patient_id"] = registry_patient_id

    explicit_case_id = normalize_case_database_patient_id(return_dict.get("case_database_patient_id"))
    case_database_patient_id = explicit_case_id or resolve_case_database_patient_id(
        state,
        findings_source,
        candidate_values=case_database_candidate_values,
        trailing_candidates=case_database_trailing_candidates,
        include_registry=include_registry_for_case,
        include_case_database=include_case_database,
        include_current=include_current,
    )
    if case_database_patient_id is not None:
        return_dict["case_database_patient_id"] = case_database_patient_id
        if isinstance(return_dict.get("findings"), dict):
            return_dict["findings"]["case_database_patient_id"] = case_database_patient_id

    return return_dict


__all__ = [
    "apply_split_identity",
    "first_present",
    "normalize_case_database_patient_id",
    "normalize_registry_patient_id",
    "resolve_case_database_patient_id",
    "resolve_legacy_current_patient_id",
    "resolve_registry_patient_id",
]
