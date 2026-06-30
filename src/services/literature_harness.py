from __future__ import annotations

from collections import OrderedDict
from typing import Any

from src.contracts.evidence_claim import (
    EvidenceClaim,
    EvidenceDelta,
    IsolationCheck,
    LiteratureHarnessRun,
    PaperCandidate,
    SourceQuality,
    SourceSpan,
    make_claim_id,
    make_delta_id,
)


RUN_LEVEL = "L0_shadow"
DEFAULT_EVIDENCE_INDEX_VERSION = "rag_crc_guideline_20260620"
NEGATIVE_OR_CONFLICTING_DIRECTIONS = frozenset(
    {"harm", "neutral", "inconclusive", "conflicting"}
)
PAIR_CONFLICT_DIRECTIONS = frozenset({"harm", "neutral", "conflicting"})
LOCAL_GUIDELINE_CONFLICTS_REQUIRING_REVIEW = frozenset({"possible", "conflict"})
ISOLATION_INPUT_KEYS = (
    "clinical_rag_claim_ids",
    "patient_default_claim_ids",
    "doctor_default_claim_ids",
)


def build_literature_harness_run(
    *,
    run_id: str,
    claim_pack: dict[str, Any],
) -> dict[str, Any]:
    validation_errors: list[str] = []
    if not isinstance(claim_pack, dict):
        validation_errors.append("claim_pack must be a dictionary")
        claim_pack = {}

    claim_pack_version = _string_or_default(
        claim_pack.get("claim_pack_id"),
        "unknown_claim_pack",
    )
    _validate_required_string(
        claim_pack=claim_pack,
        field_name="claim_pack_id",
        validation_errors=validation_errors,
    )
    evidence_index_version = _string_or_default(
        claim_pack.get("evidence_index_version"),
        DEFAULT_EVIDENCE_INDEX_VERSION,
    )
    _validate_required_string(
        claim_pack=claim_pack,
        field_name="evidence_index_version",
        validation_errors=validation_errors,
    )
    expected_min_negative_or_conflicting = _expected_min_negative_or_conflicting(
        claim_pack,
        validation_errors,
    )
    paper_candidates = _paper_candidates_from_pack(claim_pack, validation_errors)
    claims = _claims_from_candidates(
        claim_pack_version=claim_pack_version,
        paper_candidates=paper_candidates,
        validation_errors=validation_errors,
    )
    negative_or_conflicting_claims = sum(
        1
        for claim in claims
        if claim.effect_direction in NEGATIVE_OR_CONFLICTING_DIRECTIONS
    )
    deltas = _build_deltas(claims)
    isolation_checks = _build_isolation_checks(
        claims=claims,
        claim_pack=claim_pack,
        expected_min_negative_or_conflicting=expected_min_negative_or_conflicting,
        actual_negative_or_conflicting=negative_or_conflicting_claims,
        validation_errors=validation_errors,
    )
    failed_isolation_checks = [
        check for check in isolation_checks if check.passed is False
    ]
    release_decision = _release_decision(
        validation_errors=validation_errors,
        failed_isolation_checks=failed_isolation_checks,
        deltas=deltas,
    )
    summary = {
        "paper_candidates": len(paper_candidates),
        "claims": len(claims),
        "deltas": len(deltas),
        "negative_or_conflicting_claims": negative_or_conflicting_claims,
        "isolation_violations": len(failed_isolation_checks),
    }

    run = LiteratureHarnessRun(
        run_id=run_id,
        run_level=RUN_LEVEL,
        claim_pack_version=claim_pack_version,
        evidence_index_version=evidence_index_version,
        summary=summary,
        claims=claims,
        deltas=deltas,
        isolation_checks=isolation_checks,
        release_decision=release_decision,
        validation_errors=validation_errors,
    )
    return run.to_dict()


def _paper_candidates_from_pack(
    claim_pack: dict[str, Any],
    validation_errors: list[str],
) -> list[PaperCandidate]:
    if "paper_candidates" not in claim_pack:
        validation_errors.append("paper_candidates is required")
        return []
    raw_candidates = claim_pack["paper_candidates"]
    if not isinstance(raw_candidates, list):
        validation_errors.append("paper_candidates must be a list")
        return []

    paper_candidates: list[PaperCandidate] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            validation_errors.append(f"paper_candidates[{index}] must be a dictionary")
            continue
        try:
            paper_candidates.append(_paper_candidate_from_payload(candidate))
        except (KeyError, TypeError, ValueError) as exc:
            validation_errors.append(f"paper_candidates[{index}]: {exc}")
    return paper_candidates


def _paper_candidate_from_payload(payload: dict[str, Any]) -> PaperCandidate:
    return PaperCandidate(
        source_id=payload["source_id"],
        title=payload["title"],
        url=payload["url"],
        publication_year=payload.get("publication_year"),
        venue=payload.get("venue"),
        candidate_summary=payload["candidate_summary"],
        retrieval_query=payload["retrieval_query"],
        retrieval_timestamp=payload["retrieval_timestamp"],
        source_quality=_source_quality_from_payload(payload["source_quality"]),
        extracted_claims=payload["extracted_claims"],
    )


def _claims_from_candidates(
    *,
    claim_pack_version: str,
    paper_candidates: list[PaperCandidate],
    validation_errors: list[str],
) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for candidate in paper_candidates:
        for index, raw_claim in enumerate(candidate.extracted_claims):
            try:
                claims.append(
                    _evidence_claim_from_payload(
                        claim_pack_version=claim_pack_version,
                        candidate=candidate,
                        payload=raw_claim,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                validation_errors.append(
                    f"{candidate.source_id}.extracted_claims[{index}]: {exc}"
                )
    return claims


def _evidence_claim_from_payload(
    *,
    claim_pack_version: str,
    candidate: PaperCandidate,
    payload: dict[str, Any],
) -> EvidenceClaim:
    if "review_status" in payload:
        raise ValueError("review_status is harness-owned and must not be provided")
    if "source_span" not in payload:
        raise KeyError("source_span")
    source_span = _source_span_from_payload(payload["source_span"])
    claim_id = make_claim_id(
        source_id=candidate.source_id,
        claim_text=payload["claim_text"],
        population=payload["population"],
        intervention=payload.get("intervention"),
        comparator=payload.get("comparator"),
        outcome=payload["outcome"],
        effect_direction=payload["effect_direction"],
        source_span=source_span,
    )
    return EvidenceClaim(
        claim_id=claim_id,
        source_id=candidate.source_id,
        claim_text=payload["claim_text"],
        population=payload["population"],
        intervention=payload.get("intervention"),
        comparator=payload.get("comparator"),
        outcome=payload["outcome"],
        effect_direction=payload["effect_direction"],
        effect_size=payload.get("effect_size"),
        uncertainty=payload.get("uncertainty"),
        evidence_grade=payload["evidence_grade"],
        study_design=payload["study_design"],
        sample_size=payload.get("sample_size"),
        risk_of_bias=payload["risk_of_bias"],
        source_quality=candidate.source_quality,
        local_guideline_conflict=payload["local_guideline_conflict"],
        applicability_to_crc_context=payload["applicability_to_crc_context"],
        source_span=source_span,
        review_status=_review_status(candidate.source_quality, payload),
        created_from=claim_pack_version,
    )


def _source_quality_from_payload(payload: dict[str, Any]) -> SourceQuality:
    return SourceQuality(
        is_guideline=payload["is_guideline"],
        is_systematic_review=payload["is_systematic_review"],
        is_preprint=payload["is_preprint"],
        is_retracted=payload["is_retracted"],
    )


def _source_span_from_payload(payload: Any) -> SourceSpan:
    if not isinstance(payload, dict):
        raise TypeError("source_span must be a dictionary")
    return SourceSpan(
        page=payload.get("page"),
        section=payload.get("section"),
        quote=payload.get("quote"),
    )


def _review_status(
    source_quality: SourceQuality,
    claim_payload: dict[str, Any],
) -> str:
    if source_quality.is_retracted:
        return "rejected"
    if (
        claim_payload.get("effect_direction") in NEGATIVE_OR_CONFLICTING_DIRECTIONS
        or claim_payload.get("risk_of_bias") == "high"
        or source_quality.is_preprint
        or claim_payload.get("local_guideline_conflict")
        in LOCAL_GUIDELINE_CONFLICTS_REQUIRING_REVIEW
    ):
        return "needs_review"
    return "candidate"


def _build_deltas(claims: list[EvidenceClaim]) -> list[EvidenceDelta]:
    deltas: list[EvidenceDelta] = []

    for claim in claims:
        if claim.effect_direction in NEGATIVE_OR_CONFLICTING_DIRECTIONS:
            deltas.append(_negative_evidence_delta(claim))
        if claim.effect_direction == "harm":
            deltas.append(_safety_signal_delta(claim))
        if claim.local_guideline_conflict in LOCAL_GUIDELINE_CONFLICTS_REQUIRING_REVIEW:
            deltas.append(_local_guideline_conflict_delta(claim))
        if _quality_warning_reasons(claim):
            deltas.append(_quality_warning_delta(claim))

    deltas.extend(_pair_conflict_deltas(claims))
    return _deduplicate_deltas(deltas)


def _negative_evidence_delta(claim: EvidenceClaim) -> EvidenceDelta:
    return _delta(
        claim=claim,
        related_claim_id=None,
        delta_type="negative_evidence",
        summary=(
            f"{claim.effect_direction} evidence was extracted for "
            f"{claim.outcome} and must remain visible in shadow review."
        ),
        severity="review_required",
        recommended_action="human_evidence_review",
    )


def _safety_signal_delta(claim: EvidenceClaim) -> EvidenceDelta:
    return _delta(
        claim=claim,
        related_claim_id=None,
        delta_type="safety_signal",
        summary=f"Harm signal detected for {claim.outcome}.",
        severity="review_required",
        recommended_action="safety_review_before_promotion",
    )


def _local_guideline_conflict_delta(claim: EvidenceClaim) -> EvidenceDelta:
    return _delta(
        claim=claim,
        related_claim_id=None,
        delta_type="conflict",
        summary=(
            f"Local guideline conflict status is "
            f"{claim.local_guideline_conflict} for {claim.outcome}."
        ),
        severity="review_required",
        recommended_action="compare_with_local_guideline",
    )


def _quality_warning_delta(claim: EvidenceClaim) -> EvidenceDelta:
    reasons = _quality_warning_reasons(claim)
    severity = "block_promotion" if claim.source_quality.is_retracted else "review_required"
    return _delta(
        claim=claim,
        related_claim_id=None,
        delta_type="retraction_or_quality_warning",
        summary=f"Source quality warning: {', '.join(reasons)}.",
        severity=severity,
        recommended_action=(
            "remove_from_promotion_queue"
            if severity == "block_promotion"
            else "human_evidence_review"
        ),
    )


def _pair_conflict_deltas(claims: list[EvidenceClaim]) -> list[EvidenceDelta]:
    benefits_by_context: dict[tuple[str, str, str, str], list[EvidenceClaim]] = {}
    for claim in claims:
        if claim.effect_direction != "benefit":
            continue
        benefits_by_context.setdefault(_claim_context(claim), []).append(claim)

    deltas: list[EvidenceDelta] = []
    for claim in claims:
        if claim.effect_direction not in PAIR_CONFLICT_DIRECTIONS:
            continue
        for benefit_claim in benefits_by_context.get(_claim_context(claim), []):
            deltas.append(
                _delta(
                    claim=claim,
                    related_claim_id=benefit_claim.claim_id,
                    delta_type="conflict",
                    summary=(
                        f"{claim.effect_direction} evidence conflicts with "
                        f"benefit evidence for {claim.outcome}."
                    ),
                    severity="review_required",
                    recommended_action="compare_claims_before_promotion",
                )
            )
    return deltas


def _quality_warning_reasons(claim: EvidenceClaim) -> list[str]:
    reasons: list[str] = []
    if claim.source_quality.is_retracted:
        reasons.append("retracted_source")
    if claim.source_quality.is_preprint:
        reasons.append("preprint_source")
    if claim.risk_of_bias == "high":
        reasons.append("high_risk_of_bias")
    return reasons


def _delta(
    *,
    claim: EvidenceClaim,
    related_claim_id: str | None,
    delta_type: str,
    summary: str,
    severity: str,
    recommended_action: str,
) -> EvidenceDelta:
    return EvidenceDelta(
        delta_id=make_delta_id(
            claim_id=claim.claim_id,
            related_claim_id=related_claim_id,
            delta_type=delta_type,
        ),
        claim_id=claim.claim_id,
        related_claim_id=related_claim_id,
        delta_type=delta_type,
        summary=summary,
        severity=severity,
        recommended_action=recommended_action,
    )


def _claim_context(claim: EvidenceClaim) -> tuple[str, str, str, str]:
    return (
        _normalize(claim.population),
        _normalize(claim.outcome),
        _normalize_optional(claim.intervention),
        _normalize_optional(claim.comparator),
    )


def _build_isolation_checks(
    *,
    claims: list[EvidenceClaim],
    claim_pack: dict[str, Any],
    expected_min_negative_or_conflicting: int,
    actual_negative_or_conflicting: int,
    validation_errors: list[str],
) -> list[IsolationCheck]:
    isolation_inputs = _isolation_inputs_from_pack(claim_pack, validation_errors)

    return [
        _isolation_check(
            check_id="no_candidate_in_clinical_rag",
            claim_ids=claims,
            forbidden_claim_ids=isolation_inputs["clinical_rag_claim_ids"],
        ),
        _isolation_check(
            check_id="no_candidate_in_patient_default_path",
            claim_ids=claims,
            forbidden_claim_ids=isolation_inputs["patient_default_claim_ids"],
        ),
        _isolation_check(
            check_id="no_candidate_in_doctor_default_path",
            claim_ids=claims,
            forbidden_claim_ids=isolation_inputs["doctor_default_claim_ids"],
        ),
        IsolationCheck(
            check_id="negative_evidence_preserved",
            passed=actual_negative_or_conflicting
            >= expected_min_negative_or_conflicting,
            details={
                "expected_min_negative_or_conflicting": (
                    expected_min_negative_or_conflicting
                ),
                "actual_negative_or_conflicting": actual_negative_or_conflicting,
            },
        ),
    ]


def _isolation_check(
    *,
    check_id: str,
    claim_ids: list[EvidenceClaim],
    forbidden_claim_ids: list[str],
) -> IsolationCheck:
    harness_claim_ids = {claim.claim_id for claim in claim_ids}
    leaked_claim_ids = [
        claim_id
        for claim_id in forbidden_claim_ids
        if claim_id in harness_claim_ids
    ]
    return IsolationCheck(
        check_id=check_id,
        passed=not leaked_claim_ids,
        details={"leaked_claim_ids": leaked_claim_ids},
    )


def _release_decision(
    *,
    validation_errors: list[str],
    failed_isolation_checks: list[IsolationCheck],
    deltas: list[EvidenceDelta],
) -> str:
    if validation_errors:
        return "block"
    if failed_isolation_checks:
        return "block"
    if any(delta.severity == "block_promotion" for delta in deltas):
        return "block"
    return "shadow_only"


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return default


def _validate_required_string(
    *,
    claim_pack: dict[str, Any],
    field_name: str,
    validation_errors: list[str],
) -> None:
    if field_name not in claim_pack:
        validation_errors.append(f"{field_name} is required")
        return
    value = claim_pack[field_name]
    if not isinstance(value, str) or not value.strip():
        validation_errors.append(f"{field_name} must be a non-empty string")


def _expected_min_negative_or_conflicting(
    claim_pack: dict[str, Any],
    validation_errors: list[str],
) -> int:
    field_name = "expected_min_negative_or_conflicting"
    if field_name not in claim_pack:
        validation_errors.append(f"{field_name} is required")
        return 0
    value = claim_pack[field_name]
    if type(value) is not int or value < 0:
        validation_errors.append(f"{field_name} must be a non-negative integer")
        return 0
    return value


def _isolation_inputs_from_pack(
    claim_pack: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, list[str]]:
    isolation_inputs: dict[str, list[str]] = {
        key: []
        for key in ISOLATION_INPUT_KEYS
    }
    if "isolation_inputs" not in claim_pack:
        validation_errors.append("isolation_inputs is required")
        return isolation_inputs

    raw_inputs = claim_pack["isolation_inputs"]
    if not isinstance(raw_inputs, dict):
        validation_errors.append("isolation_inputs must be a dictionary")
        return isolation_inputs

    for key in ISOLATION_INPUT_KEYS:
        if key not in raw_inputs:
            validation_errors.append(f"isolation_inputs.{key} is required")
            continue
        value = raw_inputs[key]
        if not isinstance(value, list):
            validation_errors.append(f"isolation_inputs.{key} must be a list")
            continue
        if not all(isinstance(item, str) for item in value):
            validation_errors.append(
                f"isolation_inputs.{key} must contain only strings"
            )
        isolation_inputs[key] = [item for item in value if isinstance(item, str)]
    return isolation_inputs


def _deduplicate_deltas(deltas: list[EvidenceDelta]) -> list[EvidenceDelta]:
    by_delta_id: OrderedDict[str, EvidenceDelta] = OrderedDict()
    for delta in deltas:
        by_delta_id.setdefault(delta.delta_id, delta)
    return list(by_delta_id.values())


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_optional(value: str | None) -> str:
    if value is None:
        return ""
    return _normalize(value)


__all__ = ["build_literature_harness_run"]
