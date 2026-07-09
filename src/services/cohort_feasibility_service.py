from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.contracts.clinical_assertion import (
    ClinicalAssertion,
    EvidenceRef,
    NormalizedFact,
    make_assertion_id,
)
from src.contracts.research_asset import (
    CohortFeasibilityRequest,
    CohortFeasibilityResult,
    ReviewQueueItem,
    VariableCoverage,
    make_research_asset_id,
    make_review_item_id,
)
from src.services.clinical_assertion_projection import (
    PROJECTION_VERSION,
    project_clinical_assertions_from_records,
)


FeatureMatcher = tuple[tuple[str, str | None], ...]

FEATURE_MATCHERS: dict[str, FeatureMatcher] = {
    "rectal_bleeding": (
        ("condition_signal", "rectal_bleeding"),
        ("symptom", "rectal_bleeding"),
    ),
    "weight_loss": (
        ("condition_signal", "weight_loss"),
        ("symptom", "weight_loss"),
    ),
    "disposition": (("risk_disposition", "disposition"),),
    "matched_safety_rule": (("safety_rule_match", None),),
    "colonoscopy_status": (
        ("test_status", "colonoscopy_status"),
        ("document_fact", "colonoscopy_status"),
    ),
    "pathology_result": (("document_fact", "pathology_result"),),
}

DOCUMENT_FACT_FIELDS = frozenset({"colonoscopy_status", "pathology_result"})


@dataclass(frozen=True)
class _Observation:
    patient_id: str
    fact_type: str
    fact_name: str
    reviewed_status: str


class CohortFeasibilityService:
    def evaluate(
        self,
        request: CohortFeasibilityRequest,
        records: Iterable[Mapping[str, Any]],
    ) -> CohortFeasibilityResult:
        if request.patient_level_export_requested:
            return self._blocked_by_governance(request)

        materialized_records = [dict(record) for record in records]
        triage_assertions = project_clinical_assertions_from_records(materialized_records)
        document_assertions = _project_document_fact_assertions(materialized_records)
        observations = [
            _Observation(
                patient_id=assertion.patient_id,
                fact_type=assertion.normalized_fact.type,
                fact_name=assertion.normalized_fact.name,
                reviewed_status=assertion.reviewed_status,
            )
            for assertion in [*triage_assertions, *document_assertions]
        ]

        patient_ids = {
            str(record["patient_id"])
            for record in materialized_records
            if record.get("patient_id") is not None
        }
        estimated_count = len(patient_ids)
        status = "needs_review" if estimated_count > 0 else "insufficient_data"
        coverage = _build_variable_coverage(
            request.required_features,
            observations,
            estimated_count,
        )
        missing = [
            feature
            for feature in request.required_features
            if coverage[feature].coverage_ratio <= 0.5
        ]
        unmapped = [
            feature
            for feature in request.required_features
            if feature not in FEATURE_MATCHERS
        ]
        warnings = _bias_warnings(missing=missing, unmapped=unmapped)

        return CohortFeasibilityResult(
            result_id=make_research_asset_id("cohort_feasibility", request.request_id),
            request_id=request.request_id,
            project_id=request.project_id,
            status=status,
            estimated_count=estimated_count,
            variable_coverage=coverage,
            missing_key_variables=missing,
            unmapped_required_features=unmapped,
            bias_warnings=warnings,
            requires_review=True,
            review_queue_items=[_review_item(request, status="pending")],
            patient_level_rows_returned=False,
        )

    def _blocked_by_governance(
        self,
        request: CohortFeasibilityRequest,
    ) -> CohortFeasibilityResult:
        coverage = {
            feature: VariableCoverage(
                covered_count=0,
                coverage_ratio=0,
                source_fact_types=[],
                reviewed_status_mix={},
            )
            for feature in request.required_features
        }
        return CohortFeasibilityResult(
            result_id=make_research_asset_id("cohort_feasibility", request.request_id),
            request_id=request.request_id,
            project_id=request.project_id,
            status="blocked_by_governance",
            estimated_count=0,
            variable_coverage=coverage,
            missing_key_variables=request.required_features,
            unmapped_required_features=[
                feature
                for feature in request.required_features
                if feature not in FEATURE_MATCHERS
            ],
            bias_warnings=[
                "Patient-level export requires ethics and data-governance approval before registry records are inspected."
            ],
            requires_review=True,
            review_queue_items=[
                _review_item(
                    request,
                    status="blocked",
                    trigger="patient_level_export_requested",
                )
            ],
            patient_level_rows_returned=False,
        )


def _build_variable_coverage(
    required_features: list[str],
    observations: list[_Observation],
    estimated_count: int,
) -> dict[str, VariableCoverage]:
    by_feature: dict[str, dict[str, list[_Observation]]] = {
        feature: defaultdict(list) for feature in required_features
    }
    for observation in observations:
        for feature in required_features:
            if _matches_feature(feature, observation):
                by_feature[feature][observation.patient_id].append(observation)

    coverage: dict[str, VariableCoverage] = {}
    for feature in required_features:
        covered_patients = by_feature[feature]
        covered_count = len(covered_patients)
        source_fact_types = sorted(
            {
                observation.fact_type
                for patient_observations in covered_patients.values()
                for observation in patient_observations
            }
        )
        reviewed_status_mix: dict[str, int] = {}
        for patient_observations in covered_patients.values():
            statuses = {observation.reviewed_status for observation in patient_observations}
            for status in statuses:
                reviewed_status_mix[status] = reviewed_status_mix.get(status, 0) + 1
        coverage[feature] = VariableCoverage(
            covered_count=covered_count,
            coverage_ratio=(
                covered_count / estimated_count if estimated_count > 0 else 0
            ),
            source_fact_types=source_fact_types,
            reviewed_status_mix=reviewed_status_mix,
        )
    return coverage


def _matches_feature(feature: str, observation: _Observation) -> bool:
    matchers = FEATURE_MATCHERS.get(feature)
    if matchers is None:
        return False
    return any(
        observation.fact_type == fact_type
        and (fact_name is None or observation.fact_name == fact_name)
        for fact_type, fact_name in matchers
    )


def _project_document_fact_assertions(
    records: list[Mapping[str, Any]],
) -> list[ClinicalAssertion]:
    assertions: list[ClinicalAssertion] = []
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id is None:
            continue
        payload = _load_payload(record.get("normalized_payload_json"))
        record_id = _record_id(record)
        for field in sorted(DOCUMENT_FACT_FIELDS):
            value = payload.get(field)
            if value in (None, ""):
                continue
            fact = NormalizedFact(type="document_fact", name=field, value=value)
            evidence_refs = [
                EvidenceRef(
                    kind="patient_record",
                    id=record_id,
                    field=f"payload.{field}",
                )
            ]
            assertions.append(
                ClinicalAssertion(
                    assertion_id=make_assertion_id(
                        "patient_upload",
                        patient_id,
                        record_id,
                        fact,
                        evidence_refs,
                    ),
                    patient_id=str(patient_id),
                    session_id=None,
                    source="patient_upload",
                    source_record_id=record_id,
                    source_assessment_id=None,
                    normalized_fact=fact,
                    evidence_refs=evidence_refs,
                    confidence="structured_document_projection",
                    reviewed_status="unreviewed",
                    safety_policy_version=None,
                    created_from_projection_version=PROJECTION_VERSION,
                )
            )
    return assertions


def _review_item(
    request: CohortFeasibilityRequest,
    *,
    status: str,
    trigger: str = "patient_level_data_used_for_cohort_feasibility",
) -> ReviewQueueItem:
    return ReviewQueueItem(
        review_item_id=make_review_item_id(request.request_id, "research_ethics_review"),
        review_type="research_ethics_review",
        status=status,
        trigger=trigger,
        scope={
            "project_id": request.project_id,
            "request_id": request.request_id,
            "data_minimization": "aggregate_only",
            "patient_level_export_requested": request.patient_level_export_requested,
        },
        required_checks=[
            "authorization_basis",
            "deidentification_strategy",
            "data_minimization",
            "irb_or_local_policy_need",
        ],
    )


def _bias_warnings(*, missing: list[str], unmapped: list[str]) -> list[str]:
    warnings: list[str] = []
    if missing:
        warnings.append(
            "Coverage at or below 50% may bias feasibility estimates for: "
            + ", ".join(missing)
        )
    if unmapped:
        warnings.append(
            "Required features are not mapped to the current projection: "
            + ", ".join(unmapped)
        )
    return warnings


def _load_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if value is None:
        return ""
    return str(value)


__all__ = ["CohortFeasibilityService"]
