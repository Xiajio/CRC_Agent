"""Intent classification and routing nodes."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
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


def _compact_lower_text(text: str) -> str:
    return "".join((text or "").strip().split()).lower()


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
    ) -> Dict[str, Any]:
        findings_update: Dict[str, Any] = {
            "user_intent": intent,
            "plan_followup": False,
            "multi_task_mode": False,
        }
        if _is_active_outpatient_triage(state):
            findings_update["triage_explicit_switch_request"] = explicit_switch_request
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

        # lightweight fast-paths to save tokens and avoid unnecessary model calls
        if text_lower in {"", " ", "\n", "\t"}:
            updates = {
                "findings": _base_findings(state, "off_topic_redirect", preserve_outpatient_triage, False),
                "clinical_stage": "Intent",
                "error": None,
            }
            updates.update(_track_runtime_resets("off_topic_redirect", preserve_outpatient_triage))
            return updates

        if text_lower in {"hi", "hello", "hey"} or text_compact in {
            "\u4f60\u597d",
            "\u60a8\u597d",
            "\u54c8\u55bd",
            "\u54c8\u56c9",
            "\u55e8",
            "\u5728\u5417",
            "\u5728\u55ce",
        }:
            return {
                "findings": _base_findings(state, "general_chat", preserve_outpatient_triage, False),
                "clinical_stage": "Intent",
                "error": None,
            }

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
            return {
                "findings": _base_findings(state, "general_chat", preserve_outpatient_triage, False),
                "clinical_stage": "Intent",
                "error": None,
            }

        if any(k in text_lower for k in ["chat history", "conversation history", "chat log"]):
            return {
                "findings": _base_findings(state, "general_chat", preserve_outpatient_triage, False),
                "clinical_stage": "Intent",
                "error": None,
            }

        ctx = {
            "user_input": user_text,
            "has_diagnosis": "Yes" if (state.findings or {}).get("pathology_confirmed") else "No",
            "has_treatment_plan": "Yes" if state.decision_json else "No",
            "current_patient_id": state.current_patient_id or "None",
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
            intent = "general_chat" if len(user_text) < 10 else "clinical_assessment"

        explicit_switch_request = _has_explicit_triage_switch_request(intent, user_text, state)
        findings_update: Dict[str, Any] = _base_findings(
            state,
            intent,
            preserve_outpatient_triage,
            explicit_switch_request,
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
