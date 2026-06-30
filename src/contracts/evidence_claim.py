from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
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

EffectDirection = Literal["benefit", "harm", "neutral", "mixed", "uncertain"]
EvidenceGrade = Literal[
    "rct",
    "observational",
    "systematic_review",
    "meta_analysis",
    "guideline",
    "expert_consensus",
    "case_report",
    "preclinical",
    "unknown",
]
RiskOfBias = Literal["low", "moderate", "high", "unclear", "not_applicable"]
ReviewStatus = Literal[
    "candidate",
    "needs_review",
    "rejected",
]
LocalGuidelineConflict = Literal["none", "possible", "conflict", "unknown"]
ApplicabilityToCrcContext = Literal["direct", "partial", "indirect", "unknown"]
DeltaType = Literal[
    "support",
    "conflict",
    "updates",
    "supersedes",
    "duplicates",
    "safety_signal",
]
DeltaSeverity = Literal["informational", "monitor", "review_required", "block"]
IsolationZone = Literal[
    "external_literature_search",
    "project_evidence_pool",
    "clinical_rag_index",
]

EFFECT_DIRECTIONS: tuple[EffectDirection, ...] = (
    "benefit",
    "harm",
    "neutral",
    "mixed",
    "uncertain",
)
EVIDENCE_GRADES: tuple[EvidenceGrade, ...] = (
    "rct",
    "observational",
    "systematic_review",
    "meta_analysis",
    "guideline",
    "expert_consensus",
    "case_report",
    "preclinical",
    "unknown",
)
RISK_OF_BIAS_LEVELS: tuple[RiskOfBias, ...] = (
    "low",
    "moderate",
    "high",
    "unclear",
    "not_applicable",
)
REVIEW_STATUSES: tuple[ReviewStatus, ...] = (
    "candidate",
    "needs_review",
    "rejected",
)
LOCAL_GUIDELINE_CONFLICTS: tuple[LocalGuidelineConflict, ...] = (
    "none",
    "possible",
    "conflict",
    "unknown",
)
APPLICABILITY_TO_CRC_CONTEXTS: tuple[ApplicabilityToCrcContext, ...] = (
    "direct",
    "partial",
    "indirect",
    "unknown",
)
DELTA_TYPES: tuple[DeltaType, ...] = (
    "support",
    "conflict",
    "updates",
    "supersedes",
    "duplicates",
    "safety_signal",
)
DELTA_SEVERITIES: tuple[DeltaSeverity, ...] = (
    "informational",
    "monitor",
    "review_required",
    "block",
)
ISOLATION_ZONES: tuple[IsolationZone, ...] = (
    "external_literature_search",
    "project_evidence_pool",
    "clinical_rag_index",
)
CLINICAL_RAG_APPROVAL_STATUSES = frozenset(
    {
        "approved_for_clinical_rag",
        "clinical_rag_approved",
        "clinical_rag_index",
    }
)


@dataclass(frozen=True)
class SourceQuality:
    is_guideline: bool
    is_systematic_review: bool
    is_preprint: bool
    is_retracted: bool

    def __post_init__(self) -> None:
        for field_name in (
            "is_guideline",
            "is_systematic_review",
            "is_preprint",
            "is_retracted",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")

    def to_dict(self) -> dict[str, bool]:
        return {
            "is_guideline": self.is_guideline,
            "is_systematic_review": self.is_systematic_review,
            "is_preprint": self.is_preprint,
            "is_retracted": self.is_retracted,
        }


@dataclass(frozen=True)
class SourceSpan:
    page: int | None = None
    section: str | None = None
    quote: str | None = None

    def __post_init__(self) -> None:
        if self.page is not None and (not isinstance(self.page, int) or self.page < 1):
            raise ValueError("source_span.page must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "page": self.page,
                "section": self.section,
                "quote": self.quote,
            }
        )


@dataclass(frozen=True)
class PaperCandidate:
    source_id: str
    title: str
    url: str
    publication_year: int | None
    venue: str | None
    candidate_summary: str
    retrieval_query: str
    retrieval_timestamp: str
    source_quality: SourceQuality
    extracted_claims: list[dict[str, JsonValue]]

    def __post_init__(self) -> None:
        _require_non_empty("source_id", self.source_id)
        _require_non_empty("title", self.title)
        _require_non_empty("url", self.url)
        if self.publication_year is not None and self.publication_year < 1:
            raise ValueError("publication_year must be positive")
        if not isinstance(self.source_quality, SourceQuality):
            raise TypeError("source_quality must be SourceQuality")
        for index, claim in enumerate(self.extracted_claims):
            if not isinstance(claim, dict):
                raise TypeError("extracted_claims must contain dictionaries")
            validate_json_safe(claim, path=f"extracted_claims[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "source_id": self.source_id,
                "title": self.title,
                "url": self.url,
                "publication_year": self.publication_year,
                "venue": self.venue,
                "candidate_summary": self.candidate_summary,
                "retrieval_query": self.retrieval_query,
                "retrieval_timestamp": self.retrieval_timestamp,
                "source_quality": self.source_quality.to_dict(),
                "extracted_claims": [
                    _copy_json_safe(claim, path=f"extracted_claims[{index}]")
                    for index, claim in enumerate(self.extracted_claims)
                ],
            }
        )


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    source_id: str
    claim_text: str
    population: str
    intervention: str | None
    comparator: str | None
    outcome: str
    effect_direction: EffectDirection
    effect_size: str | None
    uncertainty: str | None
    evidence_grade: EvidenceGrade
    study_design: str
    sample_size: int | None
    risk_of_bias: RiskOfBias
    source_quality: SourceQuality
    local_guideline_conflict: LocalGuidelineConflict
    applicability_to_crc_context: ApplicabilityToCrcContext
    source_span: SourceSpan
    review_status: ReviewStatus
    created_from: str

    def __post_init__(self) -> None:
        _require_non_empty("claim_id", self.claim_id)
        _require_non_empty("source_id", self.source_id)
        _require_non_empty("claim_text", self.claim_text)
        _require_non_empty("population", self.population)
        _require_non_empty("outcome", self.outcome)
        _require_non_empty("study_design", self.study_design)
        _require_non_empty("created_from", self.created_from)
        _validate_choice("effect_direction", self.effect_direction, EFFECT_DIRECTIONS)
        _validate_choice("evidence_grade", self.evidence_grade, EVIDENCE_GRADES)
        _validate_choice("risk_of_bias", self.risk_of_bias, RISK_OF_BIAS_LEVELS)
        _validate_choice(
            "local_guideline_conflict",
            self.local_guideline_conflict,
            LOCAL_GUIDELINE_CONFLICTS,
        )
        _validate_choice(
            "applicability_to_crc_context",
            self.applicability_to_crc_context,
            APPLICABILITY_TO_CRC_CONTEXTS,
        )
        _validate_review_status(self.review_status)
        if self.sample_size is not None and (
            not isinstance(self.sample_size, int) or self.sample_size < 1
        ):
            raise ValueError("sample_size must be a positive integer when provided")
        if not isinstance(self.source_quality, SourceQuality):
            raise TypeError("source_quality must be SourceQuality")
        if not isinstance(self.source_span, SourceSpan):
            raise TypeError("source_span must be SourceSpan")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "claim_id": self.claim_id,
                "source_id": self.source_id,
                "claim_text": self.claim_text,
                "population": self.population,
                "intervention": self.intervention,
                "comparator": self.comparator,
                "outcome": self.outcome,
                "effect_direction": self.effect_direction,
                "effect_size": self.effect_size,
                "uncertainty": self.uncertainty,
                "evidence_grade": self.evidence_grade,
                "study_design": self.study_design,
                "sample_size": self.sample_size,
                "risk_of_bias": self.risk_of_bias,
                "source_quality": self.source_quality.to_dict(),
                "local_guideline_conflict": self.local_guideline_conflict,
                "applicability_to_crc_context": self.applicability_to_crc_context,
                "source_span": self.source_span.to_dict(),
                "review_status": self.review_status,
                "created_from": self.created_from,
            }
        )


@dataclass(frozen=True)
class EvidenceDelta:
    delta_id: str
    claim_id: str
    related_claim_id: str
    delta_type: DeltaType
    summary: str
    severity: DeltaSeverity
    recommended_action: str

    def __post_init__(self) -> None:
        _require_non_empty("delta_id", self.delta_id)
        _require_non_empty("claim_id", self.claim_id)
        _require_non_empty("related_claim_id", self.related_claim_id)
        _require_non_empty("summary", self.summary)
        _require_non_empty("recommended_action", self.recommended_action)
        _validate_choice("delta_type", self.delta_type, DELTA_TYPES)
        _validate_choice("severity", self.severity, DELTA_SEVERITIES)

    def to_dict(self) -> dict[str, str]:
        return {
            "delta_id": self.delta_id,
            "claim_id": self.claim_id,
            "related_claim_id": self.related_claim_id,
            "delta_type": self.delta_type,
            "summary": self.summary,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class IsolationCheck:
    check_id: str
    zone: IsolationZone
    subject_id: str
    passed: bool
    forbidden_behavior: str
    promotion_gate: str
    notes: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("check_id", self.check_id)
        _require_non_empty("subject_id", self.subject_id)
        _require_non_empty("forbidden_behavior", self.forbidden_behavior)
        _require_non_empty("promotion_gate", self.promotion_gate)
        _validate_choice("zone", self.zone, ISOLATION_ZONES)
        if type(self.passed) is not bool:
            raise TypeError("passed must be bool")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "check_id": self.check_id,
                "zone": self.zone,
                "subject_id": self.subject_id,
                "passed": self.passed,
                "forbidden_behavior": self.forbidden_behavior,
                "promotion_gate": self.promotion_gate,
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class LiteratureHarnessRun:
    run_id: str
    retrieval_timestamp: str
    candidates: list[PaperCandidate]
    claims: list[EvidenceClaim]
    deltas: list[EvidenceDelta]
    isolation_checks: list[IsolationCheck]
    created_from: str
    clinical_rag_ingest_allowed: bool = False
    summary: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("retrieval_timestamp", self.retrieval_timestamp)
        _require_non_empty("created_from", self.created_from)
        if type(self.clinical_rag_ingest_allowed) is not bool:
            raise TypeError("clinical_rag_ingest_allowed must be bool")
        if self.clinical_rag_ingest_allowed:
            raise ValueError(
                "clinical_rag_ingest_allowed must remain False in Step 10"
            )
        if self.summary is not None:
            validate_json_safe(self.summary, path="summary")
        for candidate in self.candidates:
            if not isinstance(candidate, PaperCandidate):
                raise TypeError("candidates must contain PaperCandidate")
        for claim in self.claims:
            if not isinstance(claim, EvidenceClaim):
                raise TypeError("claims must contain EvidenceClaim")
        for delta in self.deltas:
            if not isinstance(delta, EvidenceDelta):
                raise TypeError("deltas must contain EvidenceDelta")
        for check in self.isolation_checks:
            if not isinstance(check, IsolationCheck):
                raise TypeError("isolation_checks must contain IsolationCheck")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "run_id": self.run_id,
                "retrieval_timestamp": self.retrieval_timestamp,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "claims": [claim.to_dict() for claim in self.claims],
                "deltas": [delta.to_dict() for delta in self.deltas],
                "isolation_checks": [
                    check.to_dict() for check in self.isolation_checks
                ],
                "clinical_rag_ingest_allowed": self.clinical_rag_ingest_allowed,
                "summary": _run_summary(self) if self.summary is None else self.summary,
                "created_from": self.created_from,
            }
        )


def make_claim_id(
    *,
    source_id: str,
    claim_text: str,
    population: str,
    intervention: str | None,
    comparator: str | None,
    outcome: str,
    effect_direction: str,
    source_span: SourceSpan,
) -> str:
    if not isinstance(source_span, SourceSpan):
        raise TypeError("source_span must be SourceSpan")
    payload = {
        "source_id": source_id,
        "claim_text": claim_text,
        "population": population,
        "intervention": intervention,
        "comparator": comparator,
        "outcome": outcome,
        "effect_direction": effect_direction,
        "source_span": source_span.to_dict(),
    }
    stable_hash = _stable_hash(payload)
    return f"claim_{_slug(source_id)}_{_slug(outcome)}_{stable_hash}"


def make_delta_id(
    *,
    claim_id: str,
    related_claim_id: str,
    delta_type: str,
) -> str:
    payload = {
        "claim_id": claim_id,
        "related_claim_id": related_claim_id,
        "delta_type": delta_type,
    }
    stable_hash = _stable_hash(payload)
    return (
        f"delta_{_slug(claim_id)}_{_slug(related_claim_id)}_"
        f"{_slug(delta_type)}_{stable_hash}"
    )


def validate_json_safe(value: JsonValue, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"{path} must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must contain only string keys")
            validate_json_safe(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} must be JSON-safe")


def _copy_json_safe(value: JsonValue, *, path: str) -> JsonValue:
    validate_json_safe(value, path=path)
    if isinstance(value, list):
        return [
            _copy_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _copy_json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    return value


def _run_summary(run: LiteratureHarnessRun) -> dict[str, int]:
    return {
        "candidate_count": len(run.candidates),
        "claim_count": len(run.claims),
        "delta_count": len(run.deltas),
        "isolation_check_count": len(run.isolation_checks),
    }


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    validate_json_safe(payload)
    stable_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:8]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "unknown"


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_choice(
    field_name: str,
    value: str,
    allowed_values: tuple[str, ...],
) -> None:
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} must be one of: {allowed}")


def _validate_review_status(value: str) -> None:
    if value in CLINICAL_RAG_APPROVAL_STATUSES:
        raise ValueError(
            "review_status cannot approve clinical RAG ingestion in Step 10"
        )
    _validate_choice("review_status", value, REVIEW_STATUSES)


def _omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


__all__ = [
    "APPLICABILITY_TO_CRC_CONTEXTS",
    "CLINICAL_RAG_APPROVAL_STATUSES",
    "DELTA_SEVERITIES",
    "DELTA_TYPES",
    "EFFECT_DIRECTIONS",
    "EVIDENCE_GRADES",
    "ISOLATION_ZONES",
    "LOCAL_GUIDELINE_CONFLICTS",
    "REVIEW_STATUSES",
    "RISK_OF_BIAS_LEVELS",
    "ApplicabilityToCrcContext",
    "DeltaSeverity",
    "DeltaType",
    "EffectDirection",
    "EvidenceClaim",
    "EvidenceDelta",
    "EvidenceGrade",
    "IsolationCheck",
    "IsolationZone",
    "JsonValue",
    "LiteratureHarnessRun",
    "LocalGuidelineConflict",
    "PaperCandidate",
    "ReviewStatus",
    "RiskOfBias",
    "SourceQuality",
    "SourceSpan",
    "make_claim_id",
    "make_delta_id",
    "validate_json_safe",
]
