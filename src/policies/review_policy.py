"""Pure review-signal normalization and review decisions."""

from __future__ import annotations

from typing import Any

from .constants import (
    MAX_DECISION_RETRIES,
    MAX_EVALUATION_RETRIES,
)
from .types import DegradedSignal, ReviewDecision, ReviewSignal, TurnFacts


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_reasons(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        return (_normalize_text(value),) if _normalize_text(value) else ()

    reasons: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            reasons.append(text)
    return tuple(reasons)


def _normalize_scores(report: dict[str, Any] | None) -> dict[str, int]:
    report = report or {}
    return {
        "factual_accuracy": _safe_int(report.get("factual_accuracy"), default=3),
        "citation_accuracy": _safe_int(report.get("citation_accuracy"), default=3),
        "completeness": _safe_int(report.get("completeness"), default=3),
        "safety": _safe_int(report.get("safety"), default=3),
    }


def _inline_anchor_count(citation_report: dict[str, Any] | None) -> int:
    citation_report = citation_report or {}
    value = citation_report.get("inline_anchor_count")
    if value is not None:
        return _safe_int(value, default=0)

    notes = _normalize_text(citation_report.get("notes"))
    marker = "inline_anchors="
    if marker not in notes:
        return 0
    tail = notes.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return _safe_int("".join(digits), default=0)


def _stable_guideline_support(citation_report: dict[str, Any] | None) -> bool:
    citation_report = citation_report or {}
    return _normalize_bool(citation_report.get("stable_guideline_rag_support"))


def _evaluator_retryable(report: dict[str, Any] | None, citation_report: dict[str, Any] | None) -> bool:
    if not report:
        return False

    scores = _normalize_scores(report)
    raw_verdict = _normalize_text(report.get("verdict")).upper()
    verdict = "FAIL" if raw_verdict == "FAIL" or any(score < 3 for score in scores.values()) else "PASS"
    citation_report = citation_report or {}

    if scores["factual_accuracy"] < 3 or scores["safety"] < 3 or scores["completeness"] < 3:
        return True

    if scores["citation_accuracy"] < 3:
        if _normalize_bool(citation_report.get("needs_more_sources")):
            return True
        return verdict == "FAIL"

    return verdict == "FAIL" and _normalize_bool(citation_report.get("needs_more_sources"))


def build_critic_review_signal(
    *,
    verdict: Any,
    feedback: Any = "",
    retryable: bool | None = None,
    reasons: Any = None,
    degraded: DegradedSignal | dict[str, Any] | None = None,
) -> ReviewSignal:
    normalized_verdict = _normalize_text(verdict).upper() or "APPROVED"
    degraded_signal = (
        degraded
        if isinstance(degraded, DegradedSignal)
        else DegradedSignal.model_validate(degraded or {})
    )
    normalized_reasons = _normalize_reasons(reasons)
    if not normalized_reasons and normalized_verdict == "REJECTED":
        normalized_reasons = _normalize_reasons(feedback)

    return ReviewSignal(
        source="critic",
        verdict=normalized_verdict,
        retryable=(normalized_verdict == "REJECTED") if retryable is None else bool(retryable),
        reasons=normalized_reasons,
        degraded=degraded_signal,
        feedback=_normalize_text(feedback),
    )


def build_evaluator_review_signal(
    report: dict[str, Any] | None,
    *,
    citation_report: dict[str, Any] | None = None,
) -> ReviewSignal:
    report = dict(report or {})
    citation_report = dict(citation_report or {})
    scores = _normalize_scores(report)
    raw_verdict = _normalize_text(report.get("verdict")).upper() or "PASS"
    verdict = "FAIL" if raw_verdict == "FAIL" or any(score < 3 for score in scores.values()) else "PASS"
    coverage_score = _safe_int(citation_report.get("coverage_score"), default=0)
    inline_anchor_count = _inline_anchor_count(citation_report)

    degraded = DegradedSignal(
        is_degraded=_normalize_bool(report.get("degraded")),
        reason=_normalize_text(report.get("degraded_reason")),
        fallback_value=raw_verdict or None,
    )

    reasons: list[str] = []
    for key, score in scores.items():
        if score < 3:
            reasons.append(key)
    if verdict == "FAIL" and not reasons:
        reasons.append("verdict_fail")
    if degraded.is_degraded and degraded.reason:
        reasons.append(f"degraded:{degraded.reason}")

    return ReviewSignal(
        source="evaluator",
        verdict=verdict,
        retryable=_evaluator_retryable(report, citation_report),
        reasons=_normalize_reasons(reasons),
        degraded=degraded,
        feedback=_normalize_text(report.get("feedback")),
        scores=scores,
        coverage_score=coverage_score,
        inline_anchor_count=inline_anchor_count,
        notes=_normalize_text(citation_report.get("notes")),
    )


def _review_decision(
    action: str,
    rule_name: str,
    rationale: str,
    signal: ReviewSignal,
) -> ReviewDecision:
    return ReviewDecision(
        action=action,
        rule_name=rule_name,
        rationale=rationale,
        review_signal=signal,
    )


def decide_after_critic(facts: TurnFacts, signal: ReviewSignal) -> ReviewDecision:
    if facts.iteration_count >= MAX_DECISION_RETRIES:
        return _review_decision(
            "finalize",
            "critic_retry_cap_finalize",
            "decision retry cap reached",
            signal,
        )

    if facts.decision_strategy == "template_fast":
        return _review_decision(
            "finalize",
            "critic_template_fast_finalize",
            "template fast decisions do not retry critic review",
            signal,
        )

    if facts.decision_strategy == "rag_guideline" and facts.stable_guideline_rag_support:
        return _review_decision(
            "finalize",
            "critic_stable_guideline_finalize",
            "stable guideline support bypasses critic retry",
            signal,
        )

    if signal.verdict == "REJECTED" and signal.retryable:
        return _review_decision(
            "retry_decision",
            "critic_rejected_retry",
            "critic rejected the decision and retry remains available",
            signal,
        )

    if signal.degraded.is_degraded:
        return _review_decision(
            "finalize",
            "critic_approved_degraded",
            "critic approval came from degraded fallback handling",
            signal,
        )

    return _review_decision(
        "finalize",
        "critic_finalize",
        "critic approval does not require another decision pass",
        signal,
    )


def decide_after_evaluator(facts: TurnFacts, signal: ReviewSignal) -> ReviewDecision:
    if signal.verdict != "FAIL":
        rule_name = "evaluator_pass_degraded" if signal.degraded.is_degraded else "evaluator_pass"
        rationale = (
            "evaluator pass came from degraded fallback handling"
            if signal.degraded.is_degraded
            else "evaluator passed the decision"
        )
        return _review_decision("finalize", rule_name, rationale, signal)

    if facts.stable_guideline_rag_support:
        return _review_decision(
            "finalize",
            "evaluator_fail_stable_guideline_finalize",
            "stable guideline support suppresses evaluator retry",
            signal,
        )

    if signal.retryable and facts.evaluation_retry_count < MAX_EVALUATION_RETRIES:
        return _review_decision(
            "retry_decision",
            "evaluator_fail_retry",
            "evaluator failure remains actionable within retry cap",
            signal,
        )

    if signal.retryable:
        return _review_decision(
            "finalize",
            "evaluator_fail_retry_cap_finalize",
            "evaluator retry cap reached",
            signal,
        )

    return _review_decision(
        "finalize",
        "evaluator_fail_finalize",
        "evaluator failure is not actionable for retry",
        signal,
    )


__all__ = [
    "build_critic_review_signal",
    "build_evaluator_review_signal",
    "decide_after_critic",
    "decide_after_evaluator",
]
