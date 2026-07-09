from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, TypeAlias


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

ResearchAssetType = Literal["cohort_feasibility", "ethics_review_item"]
ResearchAssetStatus = Literal["candidate", "needs_review", "blocked", "reviewed"]
CohortFeasibilityStatus = Literal[
    "feasible_for_review",
    "needs_review",
    "insufficient_data",
    "blocked_by_governance",
]
ReviewType = Literal[
    "research_ethics_review",
    "pi_review",
    "data_governance_review",
]

ASSET_TYPES: tuple[ResearchAssetType, ...] = (
    "cohort_feasibility",
    "ethics_review_item",
)
ASSET_STATUSES: tuple[ResearchAssetStatus, ...] = (
    "candidate",
    "needs_review",
    "blocked",
    "reviewed",
)
FEASIBILITY_STATUSES: tuple[CohortFeasibilityStatus, ...] = (
    "feasible_for_review",
    "needs_review",
    "insufficient_data",
    "blocked_by_governance",
)
REVIEW_TYPES: tuple[ReviewType, ...] = (
    "research_ethics_review",
    "pi_review",
    "data_governance_review",
)
REVIEW_ITEM_STATUSES = (
    "pending",
    "in_review",
    "approved",
    "rejected",
    "blocked",
)
PATIENT_IDENTIFIER_KEYS = frozenset(
    {
        "patient_id",
        "patient_ids",
        "patient_identifier",
        "patient_identifiers",
        "patient_name",
        "patient_names",
        "patient_number",
        "patient_numbers",
        "medical_record_number",
        "medical_record_numbers",
        "mrn",
        "mrns",
        "record_id",
        "record_ids",
        "session_id",
        "session_ids",
    }
)


@dataclass(frozen=True)
class ResearchAsset:
    asset_id: str
    asset_type: ResearchAssetType
    title: str
    status: ResearchAssetStatus
    created_by: str
    created_at: str
    source_refs: list[dict[str, JsonValue]]
    governance_refs: list[str]

    def __post_init__(self) -> None:
        _validate_allowed("asset_type", self.asset_type, ASSET_TYPES)
        _validate_allowed("status", self.status, ASSET_STATUSES)
        object.__setattr__(
            self,
            "source_refs",
            _validate_json_dict_list("source_refs", self.source_refs),
        )
        object.__setattr__(
            self,
            "governance_refs",
            _validate_string_list("governance_refs", self.governance_refs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "title": self.title,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "source_refs": _copy_json_value(self.source_refs),
            "governance_refs": list(self.governance_refs),
        }


@dataclass(frozen=True)
class CohortFeasibilityRequest:
    request_id: str
    project_id: str
    question: str
    cohort_criteria: dict[str, JsonValue]
    data_scope: dict[str, JsonValue]
    version_refs: dict[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cohort_criteria",
            _validate_json_dict("cohort_criteria", self.cohort_criteria),
        )
        object.__setattr__(
            self,
            "data_scope",
            _validate_json_dict("data_scope", self.data_scope),
        )
        object.__setattr__(
            self,
            "version_refs",
            _validate_json_dict("version_refs", self.version_refs),
        )
        _reject_patient_identifier_keys(self.cohort_criteria)

        if self.data_scope.get("source") != "patient_record_projection":
            raise ValueError("data_scope.source must be patient_record_projection")
        export_requested = self.data_scope.get("patient_level_export_requested")
        if not isinstance(export_requested, bool):
            raise ValueError("patient_level_export_requested must be boolean")
        if self.data_scope.get("deidentified_only") is not True:
            raise ValueError("deidentified_only must be true")
        if not self.required_features:
            raise ValueError("required_features must not be empty")

    @property
    def required_features(self) -> list[str]:
        features = self.cohort_criteria.get("required_features")
        if not isinstance(features, list):
            return []
        if not all(isinstance(feature, str) and feature for feature in features):
            return []
        return list(features)

    @property
    def patient_level_export_requested(self) -> bool:
        return bool(self.data_scope["patient_level_export_requested"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "project_id": self.project_id,
            "question": self.question,
            "cohort_criteria": _copy_json_value(self.cohort_criteria),
            "data_scope": _copy_json_value(self.data_scope),
            "version_refs": _copy_json_value(self.version_refs),
        }


@dataclass(frozen=True)
class VariableCoverage:
    covered_count: int
    coverage_ratio: float
    source_fact_types: list[str]
    reviewed_status_mix: dict[str, int]

    def __post_init__(self) -> None:
        _validate_non_negative_int("covered_count", self.covered_count)
        if self.covered_count < 0:
            raise ValueError("covered_count must be non-negative")
        if (
            isinstance(self.coverage_ratio, bool)
            or not isinstance(self.coverage_ratio, (int, float))
            or not math.isfinite(float(self.coverage_ratio))
            or not 0 <= self.coverage_ratio <= 1
        ):
            raise ValueError("coverage_ratio must be between 0 and 1")
        object.__setattr__(
            self,
            "source_fact_types",
            _validate_string_list("source_fact_types", self.source_fact_types),
        )
        object.__setattr__(
            self,
            "reviewed_status_mix",
            _validate_count_map("reviewed_status_mix", self.reviewed_status_mix),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_count": self.covered_count,
            "coverage_ratio": self.coverage_ratio,
            "source_fact_types": list(self.source_fact_types),
            "reviewed_status_mix": dict(self.reviewed_status_mix),
        }


@dataclass(frozen=True)
class ReviewQueueItem:
    review_item_id: str
    review_type: ReviewType
    status: str
    trigger: str
    scope: dict[str, JsonValue]
    required_checks: list[str]

    def __post_init__(self) -> None:
        _validate_allowed("review_type", self.review_type, REVIEW_TYPES)
        _validate_allowed("status", self.status, REVIEW_ITEM_STATUSES)
        object.__setattr__(
            self,
            "scope",
            _validate_json_dict("scope", self.scope),
        )
        object.__setattr__(
            self,
            "required_checks",
            _validate_string_list("required_checks", self.required_checks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_item_id": self.review_item_id,
            "review_type": self.review_type,
            "status": self.status,
            "trigger": self.trigger,
            "scope": _copy_json_value(self.scope),
            "required_checks": list(self.required_checks),
        }


@dataclass(frozen=True)
class CohortFeasibilityResult:
    result_id: str
    request_id: str
    project_id: str
    status: CohortFeasibilityStatus
    estimated_count: int
    variable_coverage: dict[str, VariableCoverage]
    missing_key_variables: list[str]
    unmapped_required_features: list[str]
    bias_warnings: list[str]
    requires_review: bool
    review_queue_items: list[ReviewQueueItem]
    patient_level_rows_returned: bool

    def __post_init__(self) -> None:
        _validate_allowed("status", self.status, FEASIBILITY_STATUSES)
        _validate_non_negative_int("estimated_count", self.estimated_count)
        if self.estimated_count < 0:
            raise ValueError("estimated_count must be non-negative")
        if self.patient_level_rows_returned is not False:
            raise ValueError("patient_level_rows_returned must be false")
        if not isinstance(self.variable_coverage, dict):
            raise ValueError("variable_coverage must be a dict")
        if not all(
            isinstance(key, str) and isinstance(value, VariableCoverage)
            for key, value in self.variable_coverage.items()
        ):
            raise ValueError("variable_coverage must map strings to VariableCoverage")
        object.__setattr__(self, "variable_coverage", dict(self.variable_coverage))
        object.__setattr__(
            self,
            "missing_key_variables",
            _validate_string_list("missing_key_variables", self.missing_key_variables),
        )
        object.__setattr__(
            self,
            "unmapped_required_features",
            _validate_string_list(
                "unmapped_required_features",
                self.unmapped_required_features,
            ),
        )
        object.__setattr__(
            self,
            "bias_warnings",
            _validate_string_list("bias_warnings", self.bias_warnings),
        )
        if not isinstance(self.requires_review, bool):
            raise ValueError("requires_review must be boolean")
        if not isinstance(self.review_queue_items, list):
            raise ValueError("review_queue_items must be a list")
        if not all(isinstance(item, ReviewQueueItem) for item in self.review_queue_items):
            raise ValueError("review_queue_items must contain ReviewQueueItem values")
        object.__setattr__(self, "review_queue_items", list(self.review_queue_items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "request_id": self.request_id,
            "project_id": self.project_id,
            "status": self.status,
            "estimated_count": self.estimated_count,
            "variable_coverage": {
                key: coverage.to_dict()
                for key, coverage in self.variable_coverage.items()
            },
            "missing_key_variables": list(self.missing_key_variables),
            "unmapped_required_features": list(self.unmapped_required_features),
            "bias_warnings": list(self.bias_warnings),
            "requires_review": self.requires_review,
            "review_queue_items": [
                item.to_dict() for item in self.review_queue_items
            ],
            "patient_level_rows_returned": self.patient_level_rows_returned,
        }


def make_research_asset_id(prefix: str, seed: str) -> str:
    return _stable_id(prefix, seed)


def make_review_item_id(request_id: str, review_type: str) -> str:
    return _stable_id("review_item", f"{request_id}:{review_type}")


def _stable_id(prefix: str, seed: str) -> str:
    stable_hash = hashlib.sha256(
        json.dumps(
            {"prefix": prefix, "seed": seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:8]
    return f"{prefix}_{stable_hash}"


def _validate_allowed(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {', '.join(allowed)}")


def _reject_patient_identifier_keys(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key.lower() in PATIENT_IDENTIFIER_KEYS:
                raise ValueError("patient identifiers are not allowed")
            _reject_patient_identifier_keys(nested_value)
        return
    if isinstance(value, list):
        for item in value:
            _reject_patient_identifier_keys(item)


def _validate_json_dict(field_name: str, value: Any) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    _validate_json_value(value)
    return _copy_json_value(value)


def _validate_json_dict_list(
    field_name: str,
    value: Any,
) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must be a list of dicts")
    _validate_json_value(value)
    return _copy_json_value(value)


def _validate_string_list(field_name: str, value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return list(value)


def _validate_count_map(field_name: str, value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    for key, count in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        _validate_non_negative_int(f"{field_name} counts", count)
    return dict(value)


def _validate_non_negative_int(field_name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _copy_json_value(value: JsonValue) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_json_value(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("value must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("value must be JSON-safe")
            _validate_json_value(item)
        return
    raise TypeError("value must be JSON-safe")


__all__ = [
    "CohortFeasibilityRequest",
    "CohortFeasibilityResult",
    "ResearchAsset",
    "ReviewQueueItem",
    "VariableCoverage",
    "make_research_asset_id",
    "make_review_item_id",
]
