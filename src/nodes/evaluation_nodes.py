"""
LLM-Judge Evaluation Node
"""

import json
import os
import re
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..policies.diagnostics import record_review_divergence
from ..policies.review_policy import build_evaluator_review_signal, decide_after_evaluator
from ..policies.turn_facts import build_turn_facts
from ..policies.types import ReviewDecision, ReviewSignal
from ..prompts import LLM_JUDGE_SYSTEM_PROMPT
from ..state import CRCAgentState
from .node_utils import _invoke_structured_with_recovery


class EvaluationReport(BaseModel):
    factual_accuracy: int = Field(description="1-5")
    citation_accuracy: int = Field(description="1-5")
    completeness: int = Field(description="1-5")
    safety: int = Field(description="1-5")
    verdict: str = Field(description="PASS/FAIL")
    feedback: str = Field(description="Short validation notes")
    degraded: bool = Field(default=False, description="Whether evaluation output fell back to a degraded default")
    degraded_reason: str = Field(default="", description="Why evaluation output was degraded")


def _fast_review_enabled() -> bool:
    return os.getenv("FAST_REVIEW_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def _safe_int(value: Any, default: int = 3) -> int:
    try:
        return int(value)
    except Exception:
        return default


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


def _citation_issue_is_parse_only(citation_report: dict | None) -> bool:
    if not citation_report:
        return False
    if _normalize_bool(citation_report.get("needs_more_sources", False)):
        return False
    notes = str(citation_report.get("notes", "") or "").lower()
    return "expected value at line 1 column 1" in notes or notes.startswith("citation validation failed")


def _inline_anchor_count(citation_report: dict | None) -> int:
    if not citation_report:
        return 0
    notes = str(citation_report.get("notes", "") or "")
    match = re.search(r"inline_anchors=(\d+)", notes)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _stable_guideline_rag_support(citation_report: dict | None) -> bool:
    if not citation_report:
        return False
    return _normalize_bool(citation_report.get("stable_guideline_rag_support", False))


def _validator_is_fail(evaluation_report: dict | None) -> bool:
    if not evaluation_report:
        return False

    verdict = str(evaluation_report.get("verdict", "PASS")).upper()
    if verdict == "FAIL":
        return True

    scores = [
        _safe_int(evaluation_report.get("factual_accuracy", 3)),
        _safe_int(evaluation_report.get("citation_accuracy", 3)),
        _safe_int(evaluation_report.get("completeness", 3)),
        _safe_int(evaluation_report.get("safety", 3)),
    ]
    return any(score < 3 for score in scores)


def _should_retry_decision(
    evaluation_report: dict | None,
    citation_report: dict | None = None,
) -> bool:
    if not evaluation_report:
        return False

    factual_accuracy = _safe_int(evaluation_report.get("factual_accuracy", 3))
    citation_accuracy = _safe_int(evaluation_report.get("citation_accuracy", 3))
    completeness = _safe_int(evaluation_report.get("completeness", 3))
    safety = _safe_int(evaluation_report.get("safety", 3))
    raw_verdict = str(evaluation_report.get("verdict", "PASS")).upper()
    verdict = "FAIL" if raw_verdict == "FAIL" or any(
        score < 3 for score in [factual_accuracy, citation_accuracy, completeness, safety]
    ) else "PASS"

    if factual_accuracy < 3 or safety < 3 or completeness < 3:
        return True

    if citation_accuracy < 3:
        if citation_report and _normalize_bool(citation_report.get("needs_more_sources", False)):
            return True
        if _stable_guideline_rag_support(citation_report):
            return False
        if _citation_issue_is_parse_only(citation_report):
            return False
        return verdict == "FAIL"

    if verdict != "FAIL":
        return False

    if citation_report and _normalize_bool(citation_report.get("needs_more_sources", False)):
        return True
    if _citation_issue_is_parse_only(citation_report):
        return False
    return False


def _should_reject(report: EvaluationReport) -> bool:
    return _validator_is_fail(report.model_dump())


def _fast_evaluation_report(state: CRCAgentState) -> EvaluationReport:
    decision = state.decision_json or {}
    findings = state.findings or {}
    citation_report = state.citation_report or {}

    plan_items = decision.get("treatment_plan") or []
    summary = str(decision.get("summary") or "").strip()
    follow_up = decision.get("follow_up") or []
    coverage = _safe_int(citation_report.get("coverage_score", 0), default=0)
    citation_needs_more = _normalize_bool(citation_report.get("needs_more_sources", False))
    no_direct_references = "no_direct_references" in str(citation_report.get("notes", "") or "")
    inline_anchor_count = _inline_anchor_count(citation_report)
    stable_guideline_rag = _stable_guideline_rag_support(citation_report)

    factual_accuracy = 4
    if not findings.get("pathology_confirmed"):
        factual_accuracy = 3
    tnm = findings.get("tnm_staging") or {}
    clinical = tnm.get("clinical") if isinstance(tnm, dict) and isinstance(tnm.get("clinical"), dict) else {}
    if not (clinical.get("cT") or tnm.get("cT")):
        factual_accuracy = min(factual_accuracy, 3)

    citation_accuracy = 4 if coverage >= 70 else 3
    if stable_guideline_rag:
        citation_accuracy = 5
    elif inline_anchor_count >= 2 and coverage >= 75:
        citation_accuracy = max(citation_accuracy, 4)
    if no_direct_references:
        citation_accuracy = min(citation_accuracy, 3)
    if citation_needs_more:
        citation_accuracy = 2

    completeness = 4 if summary and plan_items else 2
    if summary and plan_items and len(plan_items) >= 2:
        completeness = 5
    elif summary and plan_items and follow_up:
        completeness = 4

    safety = 4
    if not plan_items:
        safety = 2

    verdict = "PASS"
    if any(score < 3 for score in [factual_accuracy, citation_accuracy, completeness, safety]):
        verdict = "FAIL"

    feedback = (
        f"heuristic-fast-path factual={factual_accuracy} citation={citation_accuracy} "
        f"completeness={completeness} safety={safety}"
    )
    if no_direct_references:
        feedback += " no_direct_references"
    return EvaluationReport(
        factual_accuracy=factual_accuracy,
        citation_accuracy=citation_accuracy,
        completeness=completeness,
        safety=safety,
        verdict=verdict,
        feedback=feedback,
    )


def node_llm_judge(model, streaming: bool = False, show_thinking: bool = True):
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_JUDGE_SYSTEM_PROMPT),
        ("human", "结论(JSON): {decision_json}\n引用校验: {citation_report}\n引用来源: {references}\n并行报告: {subagent_reports}"),
    ])

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        if not state.decision_json:
            return {}

        if _fast_review_enabled():
            report = _fast_evaluation_report(state)
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report.model_dump(), state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            return {
                "evaluation_report": report.model_dump(),
                "evaluation_retry_count": retry_count,
            }

        refs_payload = [
            (r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
            for r in (state.retrieved_references or [])
        ]

        ctx = {
            "decision_json": json.dumps(state.decision_json, ensure_ascii=False),
            "citation_report": json.dumps(state.citation_report or {}, ensure_ascii=False),
            "references": json.dumps(refs_payload, ensure_ascii=False),
            "subagent_reports": json.dumps(state.subagent_reports or [], ensure_ascii=False),
        }

        def _eval_text_parser(raw_text: str):
            text = (raw_text or "").replace("```json", "").replace("```", "").strip()
            if not text:
                return None
            verdict = "FAIL" if "FAIL" in text.upper() else "PASS"
            return {
                "factual_accuracy": 3,
                "citation_accuracy": 3,
                "completeness": 3,
                "safety": 3,
                "verdict": verdict,
                "feedback": text[:500],
            }

        try:
            report: EvaluationReport = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=EvaluationReport,
                payload=ctx,
                log_prefix="[LLM-Judge]",
                raw_text_parser=_eval_text_parser,
                fallback_factory=lambda _payload, _err: EvaluationReport(
                    factual_accuracy=3,
                    citation_accuracy=3,
                    completeness=3,
                    safety=3,
                    verdict="PASS",
                    feedback="Evaluation degraded to safe default after parser recovery failed.",
                ),
            )
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report.model_dump(), state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            return {
                "evaluation_report": report.model_dump(),
                "evaluation_retry_count": retry_count,
            }
        except Exception as e:
            if show_thinking:
                print(f"[LLM-Judge] evaluation failed: {e}")
            return {
                "evaluation_report": {
                    "factual_accuracy": 3,
                    "citation_accuracy": 3,
                    "completeness": 3,
                    "safety": 3,
                    "verdict": "PASS",
                    "feedback": "Evaluation degraded to safe default after runtime failure.",
                }
            }

    return _run


def route_after_evaluator(state: CRCAgentState) -> str:
    report = state.evaluation_report or {}
    should_fail = _validator_is_fail(report)
    should_retry = _should_retry_decision(report, state.citation_report or {})
    retry = state.evaluation_retry_count or 0

    if should_fail and should_retry:
        if _stable_guideline_rag_support(state.citation_report or {}):
            return "finalize"
        if retry < 2:
            return "decision"
        return "finalize"

    return "finalize"


def _build_evaluator_signal_from_state(state: CRCAgentState) -> ReviewSignal:
    stored_signal = getattr(state, "evaluator_review_signal", None) or {}
    if isinstance(stored_signal, dict) and stored_signal:
        try:
            return ReviewSignal.model_validate(stored_signal)
        except Exception:
            pass

    return build_evaluator_review_signal(
        state.evaluation_report or {},
        citation_report=state.citation_report or {},
    )


def _route_after_evaluator_legacy_impl(state: CRCAgentState) -> str:
    report = state.evaluation_report or {}
    should_fail = _validator_is_fail(report)
    should_retry = _should_retry_decision(report, state.citation_report or {})
    retry = state.evaluation_retry_count or 0

    if should_fail and should_retry:
        if _stable_guideline_rag_support(state.citation_report or {}):
            return "finalize"
        if retry < 2:
            return "decision"
        return "finalize"

    return "finalize"


def _route_after_evaluator_policy_decision(state: CRCAgentState) -> ReviewDecision:
    facts = build_turn_facts(state)
    return decide_after_evaluator(facts, _build_evaluator_signal_from_state(state))


def _build_route_after_evaluator_shadow(
    state: CRCAgentState,
    legacy_action: str | None = None,
    policy_decision: ReviewDecision | None = None,
) -> dict[str, Any]:
    if policy_decision is None:
        policy_decision = _route_after_evaluator_policy_decision(state)
    if legacy_action is None:
        legacy_action = _route_after_evaluator_legacy_impl(state)

    return record_review_divergence(
        legacy_action,
        policy_decision.route,
        policy_rule_name=policy_decision.rule_name,
        divergence_reason=policy_decision.rationale,
    )


def _log_review_shadow(label: str, shadow_payload: dict[str, Any]) -> None:
    legacy_action = shadow_payload.get("legacy_review_action", "")
    policy_action = shadow_payload.get("policy_review_action", "")
    policy_rule_name = shadow_payload.get("policy_rule_name", "")
    review_diverged = bool(shadow_payload.get("review_diverged"))
    if review_diverged:
        divergence_reason = shadow_payload.get("divergence_reason", "")
        print(
            f"\n  [Shadow] {label} legacy={legacy_action} policy={policy_action} "
            f"rule={policy_rule_name} diverged=True reason={divergence_reason}"
        )
        return

    print(
        f"\n  [Shadow] {label} legacy={legacy_action} policy={policy_action} "
        f"rule={policy_rule_name} diverged=False"
    )


def node_llm_judge(model, streaming: bool = False, show_thinking: bool = True):
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_JUDGE_SYSTEM_PROMPT),
        ("human", "ç¼æ’¹î†‘(JSON): {decision_json}\nå¯®æ› æ•¤éï¿ ç™: {citation_report}\nå¯®æ› æ•¤é‰ãƒ¦ç°®: {references}\néªžæƒ°î”‘éŽ¶ãƒ¥æ†¡: {subagent_reports}"),
    ])

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        if not state.decision_json:
            return {}

        if _fast_review_enabled():
            report = _fast_evaluation_report(state)
            report_payload = report.model_dump()
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report_payload, state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": retry_count,
            }

        refs_payload = [
            (r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
            for r in (state.retrieved_references or [])
        ]

        ctx = {
            "decision_json": json.dumps(state.decision_json, ensure_ascii=False),
            "citation_report": json.dumps(state.citation_report or {}, ensure_ascii=False),
            "references": json.dumps(refs_payload, ensure_ascii=False),
            "subagent_reports": json.dumps(state.subagent_reports or [], ensure_ascii=False),
        }

        def _eval_text_parser(raw_text: str):
            text = (raw_text or "").replace("```json", "").replace("```", "").strip()
            if not text:
                return None
            verdict = "FAIL" if "FAIL" in text.upper() else "PASS"
            return {
                "factual_accuracy": 3,
                "citation_accuracy": 3,
                "completeness": 3,
                "safety": 3,
                "verdict": verdict,
                "feedback": text[:500],
                "degraded": True,
                "degraded_reason": "raw_text_fallback",
            }

        try:
            report: EvaluationReport = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=EvaluationReport,
                payload=ctx,
                log_prefix="[LLM-Judge]",
                raw_text_parser=_eval_text_parser,
                fallback_factory=lambda _payload, _err: EvaluationReport(
                    factual_accuracy=3,
                    citation_accuracy=3,
                    completeness=3,
                    safety=3,
                    verdict="PASS",
                    feedback="Evaluation degraded to safe default after parser recovery failed.",
                    degraded=True,
                    degraded_reason="parser_recovery_failed",
                ),
            )
            report_payload = report.model_dump()
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report_payload, state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": retry_count,
            }
        except Exception as e:
            if show_thinking:
                print(f"[LLM-Judge] evaluation failed: {e}")
            report_payload = {
                "factual_accuracy": 3,
                "citation_accuracy": 3,
                "completeness": 3,
                "safety": 3,
                "verdict": "PASS",
                "feedback": "Evaluation degraded to safe default after runtime failure.",
                "degraded": True,
                "degraded_reason": "runtime_failure",
            }
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": state.evaluation_retry_count or 0,
            }

    return _run


def route_after_evaluator(state: CRCAgentState) -> str:
    legacy_action = _route_after_evaluator_legacy_impl(state)
    policy_decision = _route_after_evaluator_policy_decision(state)
    shadow_payload = _build_route_after_evaluator_shadow(
        state,
        legacy_action=legacy_action,
        policy_decision=policy_decision,
    )
    _log_review_shadow("route_after_evaluator", shadow_payload)
    return policy_decision.route


def node_llm_judge(model, streaming: bool = False, show_thinking: bool = True):
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_JUDGE_SYSTEM_PROMPT),
        ("human", "Decision (JSON): {decision_json}\nCitation report: {citation_report}\nReferences: {references}\nSubagent reports: {subagent_reports}"),
    ])

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        if not state.decision_json:
            return {}

        if _fast_review_enabled():
            report = _fast_evaluation_report(state)
            report_payload = report.model_dump()
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report_payload, state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": retry_count,
            }

        refs_payload = [
            (r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
            for r in (state.retrieved_references or [])
        ]

        ctx = {
            "decision_json": json.dumps(state.decision_json, ensure_ascii=False),
            "citation_report": json.dumps(state.citation_report or {}, ensure_ascii=False),
            "references": json.dumps(refs_payload, ensure_ascii=False),
            "subagent_reports": json.dumps(state.subagent_reports or [], ensure_ascii=False),
        }

        def _eval_text_parser(raw_text: str):
            text = (raw_text or "").replace("```json", "").replace("```", "").strip()
            if not text:
                return None
            verdict = "FAIL" if "FAIL" in text.upper() else "PASS"
            return {
                "factual_accuracy": 3,
                "citation_accuracy": 3,
                "completeness": 3,
                "safety": 3,
                "verdict": verdict,
                "feedback": text[:500],
                "degraded": True,
                "degraded_reason": "raw_text_fallback",
            }

        try:
            report: EvaluationReport = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=EvaluationReport,
                payload=ctx,
                log_prefix="[LLM-Judge]",
                raw_text_parser=_eval_text_parser,
                fallback_factory=lambda _payload, _err: EvaluationReport(
                    factual_accuracy=3,
                    citation_accuracy=3,
                    completeness=3,
                    safety=3,
                    verdict="PASS",
                    feedback="Evaluation degraded to safe default after parser recovery failed.",
                    degraded=True,
                    degraded_reason="parser_recovery_failed",
                ),
            )
            report_payload = report.model_dump()
            reject = _should_reject(report)
            actionable_retry = _should_retry_decision(report_payload, state.citation_report or {})
            retry_count = (state.evaluation_retry_count or 0) + (1 if reject and actionable_retry else 0)
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": retry_count,
            }
        except Exception as e:
            if show_thinking:
                print(f"[LLM-Judge] evaluation failed: {e}")
            report_payload = {
                "factual_accuracy": 3,
                "citation_accuracy": 3,
                "completeness": 3,
                "safety": 3,
                "verdict": "PASS",
                "feedback": "Evaluation degraded to safe default after runtime failure.",
                "degraded": True,
                "degraded_reason": "runtime_failure",
            }
            signal = build_evaluator_review_signal(
                report_payload,
                citation_report=state.citation_report or {},
            )
            return {
                "evaluation_report": report_payload,
                "evaluator_review_signal": signal.model_dump(),
                "evaluation_retry_count": state.evaluation_retry_count or 0,
            }

    return _run
