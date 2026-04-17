"""Deterministic normalization helpers for routing/review policy facts."""

from __future__ import annotations

import json
import re
from typing import Any

from .constants import STABLE_GUIDELINE_MIN_COVERAGE, STABLE_GUIDELINE_MIN_INLINE_ANCHORS
from .tool_targets import classify_pending_step_target
from .types import DerivedRoutingFlags, TurnFacts
from ..state import CRCAgentState


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


def _normalize_str_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, set):
        items = sorted(value, key=lambda item: _normalize_text(item))
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = _normalize_text(value)
        return (text,) if text else ()

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return tuple(normalized)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _latest_user_text(state: CRCAgentState) -> str:
    for message in reversed(state.messages or []):
        content = getattr(message, "content", None)
        if content is None:
            continue
        message_type = getattr(message, "type", message.__class__.__name__).lower()
        if "human" in message_type:
            return _normalize_text(content)
    return ""


_TRIAGE_SWITCH_MARKERS = (
    "改问",
    "换个",
    "切换",
    "另外想问",
    "我想改问",
    "我想换",
    "改成",
    "不想继续",
    "先不聊这个",
    "问别的",
)

_TRIAGE_SWITCH_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "case_database_query": ("数据库", "病例", "病历"),
    "knowledge_query": ("知识", "科普", "原理", "为什么", "是什么"),
    "treatment_decision": ("治疗", "方案", "手术", "化疗", "放疗", "靶向", "免疫", "用药"),
    "imaging_query": ("影像", "ct", "mri", "片子"),
    "imaging_analysis": ("影像", "ct", "mri", "片子"),
    "pathology_analysis": ("病理", "切片", "活检"),
    "general_chat": ("天气", "聊天", "闲聊", "笑话"),
    "off_topic_redirect": ("天气", "聊天", "闲聊", "笑话"),
}


def _looks_like_triage_switch_request(user_text: str, user_intent: str) -> bool:
    compact = "".join(_normalize_text(user_text).split()).lower()
    if not compact:
        return False
    if any(marker in compact for marker in _TRIAGE_SWITCH_MARKERS):
        return True
    return any(keyword.lower() in compact for keyword in _TRIAGE_SWITCH_INTENT_KEYWORDS.get(user_intent, ()))


def _triage_explicit_switch_request(state: CRCAgentState, findings: dict[str, Any], user_intent: str) -> bool:
    if _normalize_bool(findings.get("triage_explicit_switch_request")):
        return True
    if not _normalize_bool(findings.get("triage_switch_prompt_active")):
        return False
    if _normalize_text(state.encounter_track or findings.get("encounter_track")) != "outpatient_triage":
        return False
    if not _normalize_bool(findings.get("active_inquiry")):
        return False
    return _looks_like_triage_switch_request(_latest_user_text(state), user_intent)


def _first_pending_step(state: CRCAgentState) -> Any:
    for step in state.current_plan or []:
        if getattr(step, "status", None) == "pending":
            return step
    return None


def _has_pending_parallel_group(state: CRCAgentState) -> bool:
    for step in state.current_plan or []:
        if getattr(step, "status", None) != "pending":
            continue
        if _normalize_text(getattr(step, "parallel_group", "")):
            return True
    return False


def _inline_anchor_count(citation_report: dict[str, Any]) -> int:
    if not citation_report:
        return 0

    explicit_value = citation_report.get("inline_anchor_count")
    if explicit_value is not None:
        return _safe_int(explicit_value, default=0)

    notes = _normalize_text(citation_report.get("notes"))
    match = re.search(r"inline_anchors=(\d+)", notes)
    if not match:
        return 0
    return _safe_int(match.group(1), default=0)


def _decision_anchor_count(state: CRCAgentState) -> int:
    decision = _normalize_dict(state.decision_json)
    if not decision:
        return 0
    text = json.dumps(decision, ensure_ascii=False)
    return len(re.findall(r"\[\[Source:[^|\]]+\|Page:\d+\]\]", text))


def _citation_report_has_stable_guideline_support(citation_report: dict[str, Any]) -> bool:
    return _normalize_bool(citation_report.get("stable_guideline_rag_support"))


def _stable_guideline_rag_support(state: CRCAgentState, citation_report: dict[str, Any]) -> bool:
    if _citation_report_has_stable_guideline_support(citation_report):
        return True
    if citation_report:
        return False

    findings = _normalize_dict(state.findings)
    if _normalize_text(findings.get("decision_strategy")) != "rag_guideline":
        return False
    if len(state.retrieved_references or []) < 2:
        return False

    decision = _normalize_dict(state.decision_json)
    if not decision.get("summary") or not decision.get("treatment_plan"):
        return False
    return _decision_anchor_count(state) >= STABLE_GUIDELINE_MIN_INLINE_ANCHORS


def _normalize_evaluator_scores(evaluation_report: dict[str, Any]) -> dict[str, int]:
    return {
        "factual_accuracy": _safe_int(evaluation_report.get("factual_accuracy"), default=3),
        "citation_accuracy": _safe_int(evaluation_report.get("citation_accuracy"), default=3),
        "completeness": _safe_int(evaluation_report.get("completeness"), default=3),
        "safety": _safe_int(evaluation_report.get("safety"), default=3),
    }


def _evaluator_actionable_retry(
    evaluation_report: dict[str, Any],
    citation_report: dict[str, Any],
) -> bool:
    if not evaluation_report:
        return False

    scores = _normalize_evaluator_scores(evaluation_report)
    verdict = _normalize_text(evaluation_report.get("verdict")).upper()

    if scores["factual_accuracy"] < 3 or scores["safety"] < 3 or scores["completeness"] < 3:
        return True

    if scores["citation_accuracy"] < 3:
        if _normalize_bool(citation_report.get("needs_more_sources")):
            return True
        if _citation_report_has_stable_guideline_support(citation_report):
            return False
        return verdict == "FAIL"

    if verdict != "FAIL":
        return False

    if _normalize_bool(citation_report.get("needs_more_sources")):
        return True
    return False


def _evaluator_degraded(evaluation_report: dict[str, Any]) -> bool:
    return _normalize_bool(evaluation_report.get("degraded"))


def _decision_exists(state: CRCAgentState) -> bool:
    return bool(_normalize_dict(state.decision_json))


def _pathology_confirmed(state: CRCAgentState) -> bool:
    findings = _normalize_dict(state.findings)
    profile = state.patient_profile
    profile_confirmed = _normalize_bool(getattr(profile, "pathology_confirmed", False)) if profile else False
    return any(
        (
            _normalize_bool(state.pathology_confirmed),
            _normalize_bool(findings.get("pathology_confirmed")),
            profile_confirmed,
        )
    )


def _patient_profile_locked(state: CRCAgentState) -> bool:
    return _normalize_bool(getattr(state.patient_profile, "is_locked", False))


def _stage_complete(state: CRCAgentState) -> bool:
    profile_stage = getattr(state.patient_profile, "tnm_staging", {}) if state.patient_profile else {}
    findings = _normalize_dict(state.findings)
    finding_stage = _normalize_dict(findings.get("tnm_staging"))
    clinical_stage = _normalize_dict(finding_stage.get("clinical"))
    stage = {
        "cT": _normalize_text(profile_stage.get("cT") or clinical_stage.get("cT") or finding_stage.get("cT")),
        "cN": _normalize_text(profile_stage.get("cN") or clinical_stage.get("cN") or finding_stage.get("cN")),
        "cM": _normalize_text(profile_stage.get("cM") or clinical_stage.get("cM") or finding_stage.get("cM")),
    }
    return all(stage.values())


def _needs_full_decision(state: CRCAgentState, user_intent: str) -> bool:
    if _normalize_text(state.critic_verdict).upper() == "REJECTED":
        return True
    if user_intent == "treatment_decision":
        return True
    if _pathology_confirmed(state) and not _decision_exists(state):
        return True
    return False


def build_turn_facts(state: CRCAgentState) -> TurnFacts:
    findings = _normalize_dict(state.findings)
    citation_report = _normalize_dict(state.citation_report)
    evaluation_report = _normalize_dict(state.evaluation_report)
    user_intent = _normalize_text(findings.get("user_intent"))
    pending_step = _first_pending_step(state)
    pending_step_tool = _normalize_text(getattr(pending_step, "tool_needed", ""))
    pending_step_target = classify_pending_step_target(
        pending_step_tool,
        _normalize_text(getattr(pending_step, "assignee", "")),
    )
    sub_tasks = _normalize_str_sequence(findings.get("sub_tasks"))
    decision_exists = _decision_exists(state)
    pathology_confirmed = _pathology_confirmed(state)
    stage_complete = _stage_complete(state)
    patient_profile_locked = _patient_profile_locked(state)
    can_fast_pass_decision = pathology_confirmed and patient_profile_locked and stage_complete
    decision_strategy = _normalize_text(findings.get("decision_strategy"))
    evaluator_scores = _normalize_evaluator_scores(evaluation_report)

    return TurnFacts(
        user_intent=user_intent,
        sub_tasks=sub_tasks,
        multi_task_mode=_normalize_bool(findings.get("multi_task_mode")) or len(sub_tasks) > 1,
        has_plan=bool(state.current_plan),
        pending_step_tool=pending_step_tool,
        pending_step_target=pending_step_target,
        has_parallel_group=_has_pending_parallel_group(state),
        active_inquiry=_normalize_bool(findings.get("active_inquiry")),
        active_field=_normalize_text(findings.get("active_field")),
        pending_patient_data=_normalize_text(findings.get("pending_patient_data")),
        pending_patient_id=_normalize_text(findings.get("pending_patient_id")),
        encounter_track=_normalize_text(state.encounter_track or findings.get("encounter_track")),
        clinical_stage=_normalize_text(state.clinical_stage or findings.get("clinical_stage")),
        triage_switch_prompt_active=_normalize_bool(findings.get("triage_switch_prompt_active")),
        triage_explicit_switch_request=_triage_explicit_switch_request(state, findings, user_intent),
        has_missing_critical_data=bool(state.missing_critical_data),
        missing_critical_data_count=len(state.missing_critical_data or []),
        pathology_confirmed=pathology_confirmed,
        stage_complete=stage_complete,
        tumor_location=_normalize_text(findings.get("tumor_location")),
        patient_profile_locked=patient_profile_locked,
        needs_full_decision=_needs_full_decision(state, user_intent),
        decision_exists=decision_exists,
        decision_strategy=decision_strategy or "full",
        iteration_count=_safe_int(state.iteration_count, default=0),
        rejection_count=_safe_int(state.rejection_count, default=0),
        evaluation_retry_count=_safe_int(state.evaluation_retry_count, default=0),
        critic_verdict=_normalize_text(state.critic_verdict).upper(),
        citation_coverage_score=_safe_int(citation_report.get("coverage_score"), default=0),
        citation_needs_more_sources=_normalize_bool(citation_report.get("needs_more_sources")),
        stable_guideline_rag_support=_stable_guideline_rag_support(state, citation_report),
        evaluator_verdict=_normalize_text(evaluation_report.get("verdict")).upper(),
        evaluator_scores=evaluator_scores,
        evaluator_actionable_retry=_evaluator_actionable_retry(evaluation_report, citation_report),
        evaluator_degraded=_evaluator_degraded(evaluation_report),
    )


def derive_routing_flags(facts: TurnFacts) -> DerivedRoutingFlags:
    can_fast_pass_decision = (
        facts.pathology_confirmed
        and facts.patient_profile_locked
        and facts.stage_complete
    )

    return DerivedRoutingFlags(
        is_degraded=facts.evaluator_degraded,
        has_guideline_support=facts.stable_guideline_rag_support,
        has_inline_citations=False,
        can_fast_pass_decision=can_fast_pass_decision,
        should_shortcut_to_general_chat=facts.user_intent in {"general_chat", "off_topic_redirect", "greeting", "thanks"},
        should_end_turn_for_inquiry=facts.active_inquiry and facts.has_missing_critical_data,
    )


__all__ = ["build_turn_facts", "derive_routing_flags"]
