"""Intent classification and routing nodes."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
from ..services.provider_capabilities import resolve_provider_capabilities
from ..state import CRCAgentState
from .general_nodes import _get_recent_conversation_history
from .node_utils import (
    _build_pinned_context,
    _clean_and_validate_json,
    _extract_first_json_object,
    _latest_user_text,
    _unwrap_nested_json,
)


class IntentDecision(BaseModel):
    """Structured intent output from the classifier LLM."""

    category: Literal[
        "imaging_analysis",
        "pathology_analysis",
        "imaging_query",
        "case_database_query",
        "clinical_assessment",
        "treatment_decision",
        "knowledge_query",
        "general_chat",
        "off_topic_redirect",
        "multi_task",
    ]

    sub_tasks: Optional[
        List[
            Literal[
                "imaging_analysis",
                "pathology_analysis",
                "imaging_query",
                "case_database_query",
                "clinical_assessment",
                "treatment_decision",
                "knowledge_query",
            ]
        ]
    ] = None

    requires_context: Optional[bool] = None
    correction_suggestion: Optional[str] = None
    reasoning: str = Field(default="")


_TRIAGE_SWITCH_MARKERS = (
    "\u6539\u95ee",
    "\u6362\u4e2a",
    "\u5207\u6362",
    "\u53e6\u5916\u60f3\u95ee",
    "\u6211\u60f3\u6539\u95ee",
    "\u6211\u60f3\u6362",
    "\u6539\u6210",
    "\u4e0d\u60f3\u7ee7\u7eed",
    "\u5148\u4e0d\u804a\u8fd9\u4e2a",
    "\u95ee\u522b\u7684",
)

_META_CAPABILITY_QUERIES = {
    "你有什么用",
    "你有啥用",
    "你能做什么",
    "你会什么",
    "你是谁",
    "介绍一下你自己",
    "自我介绍",
    "你的功能是什么",
    "你可以做什么",
}

_TRIAGE_SWITCH_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "case_database_query": ("\u6570\u636e\u5e93", "\u75c5\u4f8b", "\u75c5\u5386"),
    "knowledge_query": ("\u77e5\u8bc6", "\u79d1\u666e", "\u539f\u7406", "\u4e3a\u4ec0\u4e48", "\u662f\u4ec0\u4e48"),
    "treatment_decision": ("\u6cbb\u7597", "\u65b9\u6848", "\u624b\u672f", "\u5316\u7597", "\u653e\u7597", "\u9776\u5411", "\u514d\u75ab", "\u7528\u836f"),
    "imaging_query": ("\u5f71\u50cf", "ct", "mri", "\u7247\u5b50"),
    "imaging_analysis": ("\u5f71\u50cf", "ct", "mri", "\u7247\u5b50"),
    "pathology_analysis": ("\u75c5\u7406", "\u5207\u7247", "\u6d3b\u68c0"),
    "general_chat": ("\u5929\u6c14", "\u804a\u5929", "\u95f2\u804a", "\u7b11\u8bdd"),
    "off_topic_redirect": ("\u5929\u6c14", "\u804a\u5929", "\u95f2\u804a", "\u7b11\u8bdd"),
}

_REPORT_DRAFT_EXACT_MARKERS = (
    "\u75c5\u4f8b\u6458\u8981\u8349\u7a3f",
    "\u4f1a\u8bca\u62a5\u544a\u8349\u7a3f",
    "\u4ea4\u63a5\u8bb0\u5f55\u8349\u7a3f",
    "\u75c5\u4f8b/\u62a5\u544a\u8349\u7a3f",
    "\u62a5\u544a\u8349\u7a3f",
    "case summary draft",
    "report draft",
)

_REPORT_DRAFT_DOCUMENT_MARKERS = (
    "\u75c5\u4f8b",
    "\u75c5\u5386",
    "\u62a5\u544a",
    "\u4f1a\u8bca",
    "\u4ea4\u63a5\u8bb0\u5f55",
    "case",
    "report",
    "summary",
)

_REPORT_DRAFT_ACTION_MARKERS = (
    "\u8bf7\u751f\u6210",
    "\u751f\u6210",
    "\u8f93\u51fa",
    "\u6574\u7406",
    "\u64b0\u5199",
    "draft",
    "write",
    "generate",
)

_DOCUMENT_DRAFT_MARKERS = _REPORT_DRAFT_EXACT_MARKERS + (
    "\u8349\u7a3f",
    "\u6587\u4e66",
    "\u6a21\u677f",
    "\u5199\u4e00\u4efd",
)

_CASE_SUMMARY_MARKERS = (
    "\u75c5\u4f8b\u6458\u8981",
    "\u75c5\u5386\u6458\u8981",
    "\u75c5\u60c5\u6458\u8981",
    "\u603b\u7ed3\u75c5\u4f8b",
    "\u6574\u7406\u75c5\u4f8b",
    "\u75c5\u60c5\u603b\u7ed3",
)

_MISSING_INFO_GUIDANCE_MARKERS = (
    "\u8fd8\u7f3a\u4ec0\u4e48",
    "\u7f3a\u4ec0\u4e48\u8d44\u6599",
    "\u9700\u8981\u8865\u5145\u4ec0\u4e48",
    "\u8fd8\u9700\u8981\u54ea\u4e9b",
    "\u9700\u8981\u54ea\u4e9b\u8d44\u6599",
    "\u8d44\u6599\u9f50\u4e0d\u9f50",
)

_EXPLANATION_MARKERS = (
    "\u5e2e\u6211\u770b\u770b",
    "\u770b\u770b",
    "\u89e3\u91ca",
    "\u4ec0\u4e48\u610f\u601d",
    "\u600e\u4e48\u7406\u89e3",
    "\u75c5\u7406\u600e\u4e48\u770b",
    "\u62a5\u544a\u600e\u4e48\u770b",
    "\u89e3\u8bfb",
)

_TREATMENT_DECISION_MARKERS = (
    "\u6cbb\u7597\u65b9\u6848",
    "\u6cbb\u7597\u5efa\u8bae",
    "\u600e\u4e48\u6cbb",
    "\u5982\u4f55\u6cbb\u7597",
    "\u4e0b\u4e00\u6b65",
    "\u4e0b\u4e00\u6b65\u600e\u4e48\u529e",
    "\u540e\u7eed\u600e\u4e48\u529e",
    "\u9700\u8981\u624b\u672f",
    "\u9700\u8981\u5316\u7597",
    "\u9700\u8981\u653e\u7597",
    "\u7528\u836f",
    "\u65b9\u6848\u600e\u4e48\u5b9a",
)

_STAGING_DECISION_MARKERS = (
    "\u5206\u671f",
    "tnm",
    "\u4e34\u5e8a\u5206\u671f",
    "\u75c5\u7406\u5206\u671f",
    "\u8bc4\u4f30\u5206\u671f",
)

_SYMPTOM_TRIAGE_MARKERS = (
    "\u8179\u75db",
    "\u4fbf\u8840",
    "\u53d1\u70ed",
    "\u5455\u5410",
    "\u8179\u6cfb",
    "\u75bc",
    "\u4e0d\u8212\u670d",
)


def _has_any_marker(compact_text: str, markers: tuple[str, ...]) -> bool:
    return any(marker.lower() in compact_text for marker in markers)


def _compact_lower_text(text: str) -> str:
    return "".join((text or "").strip().split()).lower()


def _looks_like_report_draft_request(user_text: str) -> bool:
    compact = _compact_lower_text(user_text)
    if not compact:
        return False
    if any(marker in compact for marker in _REPORT_DRAFT_EXACT_MARKERS):
        return True
    return (
        ("\u8349\u7a3f" in compact or "draft" in compact)
        and any(marker in compact for marker in _REPORT_DRAFT_DOCUMENT_MARKERS)
        and any(marker in compact for marker in _REPORT_DRAFT_ACTION_MARKERS)
    )


def _clinical_task_profile_from_text(
    user_text: str,
    intent: str,
    sub_tasks: list[str] | None = None,
) -> dict[str, Any]:
    compact = _compact_lower_text(user_text)
    tasks = set(sub_tasks or [])

    if _has_any_marker(compact, _MISSING_INFO_GUIDANCE_MARKERS):
        return {
            "task_type": "missing_info_guidance",
            "requires_complete_case": False,
            "missing_info_policy": "guide_collection",
            "response_mode": "guided_collection",
            "reason": "deterministic_rule:missing_info_guidance",
        }

    if (
        intent != "treatment_decision"
        and "treatment_decision" not in tasks
        and _has_any_marker(compact, _EXPLANATION_MARKERS)
    ):
        return {
            "task_type": "explain_existing_info",
            "requires_complete_case": False,
            "missing_info_policy": "answer_with_gaps",
            "response_mode": "partial_explanation",
            "reason": "deterministic_rule:explain_existing_info",
        }

    if (
        intent == "treatment_decision"
        or "treatment_decision" in tasks
        or _has_any_marker(compact, _TREATMENT_DECISION_MARKERS)
    ):
        return {
            "task_type": "treatment_decision",
            "requires_complete_case": True,
            "missing_info_policy": "hard_inquiry",
            "response_mode": "decision_blocked",
            "reason": "deterministic_rule:treatment_or_next_step_decision",
        }

    if _has_any_marker(compact, _STAGING_DECISION_MARKERS):
        return {
            "task_type": "staging_assessment",
            "requires_complete_case": True,
            "missing_info_policy": "hard_inquiry",
            "response_mode": "decision_blocked",
            "reason": "deterministic_rule:staging_assessment",
        }

    if _looks_like_report_draft_request(user_text) or _has_any_marker(compact, _DOCUMENT_DRAFT_MARKERS):
        return {
            "task_type": "document_draft",
            "requires_complete_case": False,
            "missing_info_policy": "answer_with_gaps",
            "response_mode": "case_summary_with_gaps",
            "reason": "deterministic_rule:document_draft",
        }

    if _has_any_marker(compact, _CASE_SUMMARY_MARKERS):
        return {
            "task_type": "case_summary",
            "requires_complete_case": False,
            "missing_info_policy": "answer_with_gaps",
            "response_mode": "case_summary_with_gaps",
            "reason": "deterministic_rule:case_summary",
        }

    if _has_any_marker(compact, _EXPLANATION_MARKERS):
        return {
            "task_type": "explain_existing_info",
            "requires_complete_case": False,
            "missing_info_policy": "answer_with_gaps",
            "response_mode": "partial_explanation",
            "reason": "deterministic_rule:explain_existing_info",
        }

    if intent == "clinical_assessment" and _has_any_marker(compact, _SYMPTOM_TRIAGE_MARKERS):
        return {
            "task_type": "symptom_triage",
            "requires_complete_case": False,
            "missing_info_policy": "none",
            "response_mode": "clinical_answer",
            "reason": "deterministic_rule:symptom_triage",
        }

    return {
        "task_type": "general_clinical_question",
        "requires_complete_case": False,
        "missing_info_policy": "none",
        "response_mode": "general_with_gaps",
        "reason": "deterministic_rule:general_clinical_question",
    }


def _looks_like_triage_switch_request(user_text: str, intent: str) -> bool:
    compact = _compact_lower_text(user_text)
    if not compact:
        return False
    if any(marker in compact for marker in _TRIAGE_SWITCH_MARKERS):
        return True
    return any(keyword.lower() in compact for keyword in _TRIAGE_SWITCH_INTENT_KEYWORDS.get(intent, ()))


def _parse_intent_from_raw_response(raw_response: Any) -> IntentDecision:
    """Parse model raw text into IntentDecision with tolerant JSON extraction."""
    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    parsed = _clean_and_validate_json(content)
    if parsed is None:
        parsed = _extract_first_json_object(content)
    if parsed is None:
        raise ValueError("intent parse failed")

    parsed = _unwrap_nested_json(
        parsed,
        ["category", "sub_tasks", "requires_context", "correction_suggestion", "reasoning"],
    )
    return IntentDecision(**parsed)


def node_intent_classifier(model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    """Intent classification node with robust structured-output recovery."""

    del streaming
    intent_prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFIER_SYSTEM_PROMPT)
    capabilities = resolve_provider_capabilities(
        model_name=str(getattr(model, "model_name", "") or getattr(model, "model", "") or ""),
        base_url=str(getattr(model, "openai_api_base", "") or getattr(model, "base_url", "") or ""),
    )
    classifier_chain = None
    if capabilities.structured_output_strategy != "raw_first":
        classifier_chain = intent_prompt | model.with_structured_output(IntentDecision).bind(temperature=0)

    def _is_active_outpatient_triage(state: CRCAgentState) -> bool:
        current_findings = state.findings or {}
        current_track = state.encounter_track or current_findings.get("encounter_track")
        return current_track == "outpatient_triage" and bool(current_findings.get("active_inquiry"))

    def _has_explicit_triage_switch_request(intent: str, user_text: str, state: CRCAgentState) -> bool:
        current_findings = state.findings or {}
        if not _is_active_outpatient_triage(state):
            return False
        if not bool(current_findings.get("triage_switch_prompt_active")):
            return False
        return _looks_like_triage_switch_request(user_text, intent)

    def _track_runtime_resets(intent: str, preserve_outpatient_triage: bool) -> Dict[str, Any]:
        if preserve_outpatient_triage or intent not in {"general_chat", "knowledge_query", "off_topic_redirect"}:
            return {}

        return {
            "encounter_track": None,
            "clinical_entry_reason": None,
            "entry_explanation_shown": False,
            "triage_risk_level": None,
            "triage_disposition": None,
            "triage_suggested_tests": [],
            "triage_summary": None,
            "triage_card": None,
            "symptom_snapshot": {},
        }

    def _base_findings(
        state: CRCAgentState,
        intent: str,
        preserve_outpatient_triage: bool,
        explicit_switch_request: bool,
        *,
        user_text: str,
        sub_tasks: list[str] | None = None,
        clear_inquiry: bool = False,
    ) -> Dict[str, Any]:
        profile = _clinical_task_profile_from_text(user_text, intent, sub_tasks)
        findings_update: Dict[str, Any] = {
            "user_intent": intent,
            "plan_followup": False,
            "multi_task_mode": False,
            "clinical_task_profile": profile,
            "requires_complete_case": profile["requires_complete_case"],
            "missing_info_policy": profile["missing_info_policy"],
            "response_mode": profile["response_mode"],
        }
        if _is_active_outpatient_triage(state):
            findings_update["triage_explicit_switch_request"] = explicit_switch_request
        if clear_inquiry:
            findings_update.update(
                {
                    "active_inquiry": False,
                    "active_field": None,
                    "inquiry_message": None,
                    "inquiry_type": None,
                }
            )
        if not preserve_outpatient_triage and intent in {"general_chat", "knowledge_query", "off_topic_redirect"}:
            findings_update.update(
                {
                    "active_inquiry": False,
                    "active_field": None,
                    "inquiry_message": None,
                    "pending_patient_data": None,
                    "pending_patient_id": None,
                    "encounter_track": None,
                    "clinical_entry_reason": None,
                    "entry_explanation_shown": False,
                    "triage_risk_level": None,
                    "triage_disposition": None,
                    "triage_suggested_tests": [],
                    "triage_summary": None,
                    "triage_card": None,
                    "symptom_snapshot": {},
                }
            )
        return findings_update

    def _run(state: CRCAgentState):
        user_text = _latest_user_text(state) or ""
        text_lower = user_text.strip().lower()
        text_compact = "".join(user_text.strip().split())
        preserve_outpatient_triage = _is_active_outpatient_triage(state)

        def _fast_path_updates(intent: str, *, clear_inquiry: bool = False) -> Dict[str, Any]:
            findings_update = _base_findings(
                state,
                intent,
                preserve_outpatient_triage,
                False,
                user_text=user_text,
                clear_inquiry=clear_inquiry,
            )
            updates = {
                "findings": findings_update,
                "clinical_stage": "Intent",
                "error": None,
            }
            if not preserve_outpatient_triage and findings_update.get("active_inquiry") is False:
                updates["missing_critical_data"] = []
            updates.update(_track_runtime_resets(intent, preserve_outpatient_triage))
            return updates

        # lightweight fast-paths to save tokens and avoid unnecessary model calls
        if text_lower in {"", " ", "\n", "\t"}:
            return _fast_path_updates("off_topic_redirect")

        if text_lower in {"hi", "hello", "hey"} or text_compact in {
            "\u4f60\u597d",
            "\u60a8\u597d",
            "\u54c8\u55bd",
            "\u54c8\u56c9",
            "\u55e8",
            "\u5728\u5417",
            "\u5728\u55ce",
        }:
            return _fast_path_updates("general_chat")

        if text_compact in _META_CAPABILITY_QUERIES:
            return _fast_path_updates("general_chat")

        if text_compact in {
            "\u8c22\u8c22",
            "\u8b1d\u8b1d",
            "\u591a\u8c22",
            "\u591a\u8b1d",
            "thx",
            "thanks",
            "thankyou",
            "thankyou!",
            "thankyou.",
        }:
            return _fast_path_updates("general_chat")

        if any(k in text_lower for k in ["chat history", "conversation history", "chat log"]):
            return _fast_path_updates("general_chat")

        if _looks_like_report_draft_request(user_text):
            return _fast_path_updates("general_chat", clear_inquiry=True)

        findings = state.findings or {}
        ctx = {
            "user_input": user_text,
            "has_diagnosis": "Yes" if findings.get("pathology_confirmed") else "No",
            "has_treatment_plan": "Yes" if state.decision_json else "No",
            "registry_patient_id": getattr(state, "registry_patient_id", None) or "None",
            "case_database_patient_id": (
                getattr(state, "case_database_patient_id", None)
                or findings.get("case_database_patient_id")
                or getattr(state, "current_patient_id", None)
                or "None"
            ),
            "recent_conversation": _get_recent_conversation_history(state, max_turns=3),
            "summary_memory": state.summary_memory or "",
            "pinned_context": _build_pinned_context(state),
        }

        result: Optional[IntentDecision] = None
        try:
            if show_thinking:
                print(f"[Intent] Analyzing: '{user_text[:30]}...'")

            if getattr(model, "_llm_type", "") in {"local-hf", "local-hf-with-tools"}:
                raw_response = (intent_prompt | model.bind(temperature=0)).invoke(ctx)
                result = _parse_intent_from_raw_response(raw_response)
            else:
                if capabilities.structured_output_strategy == "raw_first":
                    raw_response = (intent_prompt | model.bind(temperature=0)).invoke(ctx)
                    result = _parse_intent_from_raw_response(raw_response)
                else:
                    try:
                        result = classifier_chain.invoke(ctx)
                    except Exception:
                        raw_response = (intent_prompt | model.bind(temperature=0)).invoke(ctx)
                        result = _parse_intent_from_raw_response(raw_response)

            intent = result.category
            reasoning = result.reasoning

            if intent == "multi_task" and not result.sub_tasks:
                intent = "clinical_assessment"

            if show_thinking:
                print(f"[Intent] Routed to: {intent} | Reason: {reasoning}")

        except Exception as e:
            print(f"[Intent Fail] LLM Routing failed: {e}")
            fallback_profile = _clinical_task_profile_from_text(user_text, "general_chat")
            if fallback_profile["requires_complete_case"]:
                intent = (
                    "treatment_decision"
                    if fallback_profile["task_type"] == "treatment_decision"
                    else "clinical_assessment"
                )
            else:
                intent = "general_chat"

        explicit_switch_request = _has_explicit_triage_switch_request(intent, user_text, state)
        profile_sub_tasks = result.sub_tasks if result is not None else None
        findings_update: Dict[str, Any] = _base_findings(
            state,
            intent,
            preserve_outpatient_triage,
            explicit_switch_request,
            user_text=user_text,
            sub_tasks=profile_sub_tasks,
        )

        if result is not None:
            if result.correction_suggestion:
                findings_update["intent_correction"] = result.correction_suggestion
            if result.requires_context is not None:
                findings_update["requires_context"] = bool(result.requires_context)

            if intent == "multi_task" and result.sub_tasks:
                findings_update["sub_tasks"] = result.sub_tasks
                findings_update["multi_task_mode"] = True

        updates = {
            "findings": findings_update,
            "clinical_stage": "Intent",
            "error": None,
        }
        if not preserve_outpatient_triage and findings_update.get("active_inquiry") is False:
            updates["missing_critical_data"] = []
        updates.update(_track_runtime_resets(intent, preserve_outpatient_triage))
        return updates

    return _run


def route_by_intent(state: CRCAgentState) -> str:
    """Route to downstream node by classified intent."""

    intent = (state.findings or {}).get("user_intent", "assessment")

    if intent == "imaging_analysis":
        return "rad_agent"
    if intent == "pathology_analysis":
        return "path_agent"
    if intent == "imaging_query":
        return "case_database"
    if intent == "multi_task":
        return "assessment"
    if intent == "general_chat":
        return "general_chat"
    if intent == "off_topic_redirect":
        return "general_chat"
    if intent == "knowledge_query":
        return "knowledge"
    if intent == "treatment_decision":
        return "decision"
    if intent == "case_database_query":
        return "case_database"

    return "assessment"
