from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from ..state import CRCAgentState
from .node_utils import _latest_user_text


TRIAGE_FIELD_ORDER = [
    "duration",
    "bleeding",
    "bowel_change",
    "weight_loss",
    "fever",
]

TRIAGE_QUESTION_MAP: dict[str, str] = {
    "duration": "腹痛/腹泻/便秘是从什么时候开始的？大概持续了多久？",
    "bleeding": "最近有没有便血或黑便？如果有，大概是什么颜色？",
    "bowel_change": "最近排便习惯有没有变化，比如腹泻、便秘、次数增多或变细？",
    "weight_loss": "最近有没有明显消瘦、乏力，或者贫血样表现？",
    "fever": "最近有没有发热、呕吐，或者停止排气排便？",
}

TRIAGE_CLARIFICATION_MAP: dict[str, str] = {
    "duration": "我还没识别出持续时间，请直接回答例如“4小时”“3天”“2周”“1个月”。",
    "bleeding": "请直接回答“有/没有便血或黑便”；如果有，也可以补充“鲜红/暗红/发黑”。",
    "bowel_change": "请直接回答最近排便是否有变化，例如“腹泻”“便秘”“没有变化”。",
    "weight_loss": "请直接回答最近是否有消瘦、乏力或贫血样表现，例如“有”或“没有”。",
    "fever": "请直接回答最近是否有发热、呕吐，或停止排气排便，例如“有”或“没有”。",
}

TRIAGE_SWITCH_PROMPT_TEMPLATE = (
    "如果您想继续当前门诊分诊，请先直接回答当前问题：{prompt}\n"
    "如果你想切换到别的目的，也可以直接告诉我，比如改问治疗方案、病例数据库、检查结果解读或其他问题。"
)

NEGATION_PREFIXES = ("无", "没有", "没", "未见", "否认", "不伴", "并无", "未")
POSITIVE_PREFIXES = ("有", "出现", "伴有", "存在")
OBVIOUS_OFF_TOPIC_INTENTS = {
    "off_topic_redirect",
    "knowledge_query",
    "case_database_query",
    "imaging_query",
    "imaging_analysis",
    "pathology_analysis",
}
OFF_TOPIC_CHAT_KEYWORDS = ("天气", "笑话", "闲聊", "聊天")
SHORT_DURATION_UNITS = "分钟|小时|天|周|星期|个月|月|年"
CHINESE_NUMERALS = "零一二两三四五六七八九十半"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_user_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


TRIAGE_SWITCH_MARKERS = (
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

TRIAGE_SWITCH_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "case_database_query": ("\u6570\u636e\u5e93", "\u75c5\u4f8b", "\u75c5\u5386"),
    "knowledge_query": ("\u77e5\u8bc6", "\u79d1\u666e", "\u539f\u7406", "\u4e3a\u4ec0\u4e48", "\u662f\u4ec0\u4e48"),
    "treatment_decision": ("\u6cbb\u7597", "\u65b9\u6848", "\u624b\u672f", "\u5316\u7597", "\u653e\u7597", "\u9776\u5411", "\u514d\u75ab", "\u7528\u836f"),
    "imaging_query": ("\u5f71\u50cf", "ct", "mri", "\u7247\u5b50"),
    "imaging_analysis": ("\u5f71\u50cf", "ct", "mri", "\u7247\u5b50"),
    "pathology_analysis": ("\u75c5\u7406", "\u5207\u7247", "\u6d3b\u68c0"),
    "general_chat": ("\u5929\u6c14", "\u804a\u5929", "\u95f2\u804a", "\u7b11\u8bdd"),
    "off_topic_redirect": ("\u5929\u6c14", "\u804a\u5929", "\u95f2\u804a", "\u7b11\u8bdd"),
}


def _looks_like_explicit_triage_switch(user_text: str, user_intent: str) -> bool:
    compact = _normalize_user_text(user_text).lower()
    if not compact:
        return False
    if any(marker in compact for marker in TRIAGE_SWITCH_MARKERS):
        return True
    return any(keyword.lower() in compact for keyword in TRIAGE_SWITCH_INTENT_KEYWORDS.get(user_intent, ()))


def _has_negated_keyword(text: str, keyword: str) -> bool:
    compact = _normalize_user_text(text)
    escaped = re.escape(keyword)
    for negation in NEGATION_PREFIXES:
        pattern = rf"(?:也|都)?{re.escape(negation)}[^，。,.；;、]{{0,8}}{escaped}"
        if re.search(pattern, compact):
            return True
    return False


def _has_positive_keyword(text: str, keyword: str) -> bool:
    compact = _normalize_user_text(text)
    escaped = re.escape(keyword)
    if _has_negated_keyword(compact, keyword):
        return False
    prefix_group = "|".join(map(re.escape, POSITIVE_PREFIXES))
    if re.search(rf"(?:{prefix_group}){escaped}", compact):
        return True
    return keyword in compact


def _extract_boolean(text: str, keywords: tuple[str, ...]) -> bool | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None

    for keyword in keywords:
        if _has_negated_keyword(normalized, keyword):
            return False

    for keyword in keywords:
        if _has_positive_keyword(normalized, keyword):
            return True

    return None


def _extract_duration(text: str) -> str | None:
    compact = _normalize_user_text(text)
    if not compact:
        return None

    for literal in ("半小时", "半天", "半个月", "半年"):
        if literal in compact:
            return literal

    quantity_pattern = rf"((?:\d+|[{CHINESE_NUMERALS}]+)(?:个)?(?:{SHORT_DURATION_UNITS}))"
    match = re.search(quantity_pattern, compact)
    if match:
        return match.group(1)

    if compact in {"今天", "昨天", "前天"}:
        return compact

    relative_pattern = r"((?:今天|昨天|前天)(?:开始|起|以来))"
    match = re.search(relative_pattern, compact)
    if match:
        return match.group(1)

    nearby_pattern = rf"((?:近|最近)(?:\d+|[{CHINESE_NUMERALS}]+)(?:天|周|星期|个月|月|年))"
    match = re.search(nearby_pattern, compact)
    if match:
        return match.group(1)

    for literal in ("一周", "两周", "一个月"):
        if literal in compact:
            return literal

    return None


def _extract_bleeding_detail(text: str) -> str | None:
    compact = _normalize_user_text(text)
    for detail in ("鲜红", "暗红", "发黑", "黑便"):
        if detail in compact:
            return detail
    return None


def _extract_bowel_change(text: str) -> bool | None:
    compact = _normalize_user_text(text)
    negative_patterns = (
        "没有变化",
        "没变化",
        "正常",
        "没有腹泻",
        "没有便秘",
        "没腹泻",
        "没便秘",
    )
    if any(pattern in compact for pattern in negative_patterns):
        return False
    positive_keywords = (
        "腹泻",
        "便秘",
        "排便习惯改变",
        "排便次数增多",
        "大便变细",
        "排便变化",
    )
    if any(keyword in compact for keyword in positive_keywords):
        return True
    return None


def _extract_weight_loss(text: str) -> bool | None:
    return _extract_boolean(text, ("消瘦", "体重下降", "乏力", "贫血"))


def _extract_fever(text: str) -> bool | None:
    return _extract_boolean(text, ("发热", "发烧", "呕吐", "停止排气", "停止排便", "不排气", "不排便"))


def _conversation_user_text(state: CRCAgentState) -> str:
    parts: list[str] = []
    for message in state.messages or []:
        if message.__class__.__name__.lower().startswith("human"):
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                parts.append(content)
    return "\n".join(parts)


def _update_symptom_snapshot(state: CRCAgentState, user_text: str) -> dict[str, Any]:
    existing = dict(state.symptom_snapshot or (state.findings or {}).get("symptom_snapshot") or {})
    latest_text = str(user_text or "").strip()
    compact = _normalize_user_text(latest_text)

    if latest_text and not existing.get("chief_symptoms"):
        existing["chief_symptoms"] = latest_text

    duration = _extract_duration(compact)
    if duration:
        existing["duration"] = duration

    bleeding = _extract_boolean(compact, ("便血", "黑便"))
    if bleeding is not None:
        existing["bleeding"] = bleeding

    bleeding_detail = _extract_bleeding_detail(compact)
    if bleeding_detail:
        existing["bleeding_detail"] = bleeding_detail

    bowel_change = _extract_bowel_change(compact)
    if bowel_change is not None:
        existing["bowel_change"] = bowel_change

    weight_loss = _extract_weight_loss(compact)
    if weight_loss is not None:
        existing["weight_loss"] = weight_loss

    fever = _extract_fever(compact)
    if fever is not None:
        existing["fever"] = fever

    risk_flags = dict(existing.get("risk_flags") or {})
    for key in ("bleeding", "bowel_change", "weight_loss", "fever"):
        if key in existing:
            risk_flags[key] = existing[key]
    existing["risk_flags"] = risk_flags
    return existing


def _triage_from_symptoms(text: str) -> tuple[str, str, list[str], dict[str, bool], dict[str, bool]]:
    compact = _normalize_user_text(text)
    symptoms = {
        "pain": any(keyword in compact for keyword in ("腹痛", "肚子痛", "隐痛", "绞痛")),
        "bleeding": _extract_boolean(compact, ("便血", "黑便")) is True,
        "bowel_change": _extract_bowel_change(compact) is True,
        "weight_loss": _extract_weight_loss(compact) is True,
        "fever": _extract_fever(compact) is True,
        "obstruction": any(keyword in compact for keyword in ("停止排气排便", "停止排气", "停止排便", "不排气", "不排便")),
        "massive_bleeding": any(keyword in compact for keyword in ("大量便血", "出血很多")),
    }
    known_signals = {name: enabled for name, enabled in symptoms.items() if enabled}

    if symptoms["obstruction"] or symptoms["massive_bleeding"]:
        return "high", "emergency", ["血常规", "腹部影像", "急诊评估"], symptoms, known_signals

    if symptoms["bleeding"] or symptoms["weight_loss"] or symptoms["fever"]:
        return "medium", "urgent_gi_clinic", ["血常规", "粪便常规/潜血", "基础门诊检查"], symptoms, known_signals

    if symptoms["pain"] or symptoms["bowel_change"]:
        return "medium", "routine_gi_clinic", ["血常规", "粪便常规/潜血", "基础门诊检查"], symptoms, known_signals

    return "low", "observe_followup", ["观察随访"], symptoms, known_signals


def _next_triage_question(symptom_snapshot: dict[str, Any], disposition: str) -> tuple[list[str], str | None, str | None]:
    if disposition == "emergency":
        return [], None, None

    pending_fields = [field for field in TRIAGE_FIELD_ORDER if symptom_snapshot.get(field) is None]
    if not pending_fields:
        return [], None, None

    next_field = pending_fields[0]
    return pending_fields, next_field, TRIAGE_QUESTION_MAP[next_field]


def _risk_label(risk_level: str) -> str:
    return {"high": "高风险", "medium": "中风险", "low": "低风险"}.get(risk_level, risk_level)


def _disposition_label(disposition: str) -> str:
    return {
        "emergency": "急诊",
        "urgent_gi_clinic": "尽快消化门诊",
        "routine_gi_clinic": "常规消化门诊",
        "observe_followup": "观察随访",
    }.get(disposition, disposition)


def _triage_progress_made(previous_snapshot: dict[str, Any], next_snapshot: dict[str, Any]) -> bool:
    tracked_fields = set(TRIAGE_FIELD_ORDER) | {"bleeding_detail"}
    return any(previous_snapshot.get(field) != next_snapshot.get(field) for field in tracked_fields)


def _is_obvious_off_topic_followup(user_intent: str | None, user_text: str) -> bool:
    normalized_intent = str(user_intent or "")
    if normalized_intent in OBVIOUS_OFF_TOPIC_INTENTS:
        return True

    normalized_text = _normalize_user_text(user_text)
    return any(keyword in normalized_text for keyword in OFF_TOPIC_CHAT_KEYWORDS)


def _triage_prompt_text(field: str | None, *, clarification: bool = False) -> str:
    if not field:
        return ""
    if clarification:
        return TRIAGE_CLARIFICATION_MAP.get(field, TRIAGE_QUESTION_MAP.get(field, ""))
    return TRIAGE_QUESTION_MAP.get(field, "")


def _triage_switch_prompt(field: str | None) -> str:
    prompt = _triage_prompt_text(field, clarification=True) or "请先直接回答当前分诊问题。"
    return TRIAGE_SWITCH_PROMPT_TEMPLATE.format(prompt=prompt)


def _published_triage_fields(
    *,
    active_inquiry: bool,
    risk_level: str,
    disposition: str,
    tests: list[str],
    summary: str,
    symptom_snapshot: dict[str, Any],
) -> tuple[str | None, str | None, list[str], str | None, dict[str, Any] | None]:
    if active_inquiry:
        return None, None, [], None, None

    triage_card = {
        "type": "triage_card",
        "title": "门诊分诊",
        "risk_level": risk_level,
        "disposition": disposition,
        "suggested_tests": list(tests),
        "summary": summary,
        "chief_symptoms": str(symptom_snapshot.get("chief_symptoms", ""))[:120],
        "symptom_snapshot": symptom_snapshot,
    }
    return risk_level, disposition, list(tests), summary, triage_card


def node_outpatient_triage(
    model: Any = None,
    *,
    streaming: bool = False,
    show_thinking: bool = True,
    **_: Any,
) -> Runnable:
    del model, streaming

    def _run(state: CRCAgentState) -> dict[str, Any]:
        user_text = _latest_user_text(state) or ""
        combined_user_text = _conversation_user_text(state) or user_text
        previous_findings = dict(state.findings or {})
        previous_snapshot = dict(state.symptom_snapshot or previous_findings.get("symptom_snapshot") or {})
        previous_field = previous_findings.get("triage_current_field")
        previous_no_progress_count = int(previous_findings.get("triage_no_progress_count") or 0)
        previous_switch_prompt_active = bool(previous_findings.get("triage_switch_prompt_active"))
        user_intent = previous_findings.get("user_intent")

        risk_level, disposition, tests, symptoms, known_crc_signals = _triage_from_symptoms(combined_user_text)
        symptom_snapshot = _update_symptom_snapshot(state, user_text)
        symptom_snapshot["risk_flags"] = {**dict(previous_snapshot.get("risk_flags") or {}), **symptoms}
        if not symptom_snapshot.get("chief_symptoms"):
            symptom_snapshot["chief_symptoms"] = user_text

        pending_fields, next_field, next_question = _next_triage_question(symptom_snapshot, disposition)
        active_inquiry = bool(next_question)
        progress_made = _triage_progress_made(previous_snapshot, symptom_snapshot)
        same_field_stalled = active_inquiry and bool(next_field) and next_field == previous_field and not progress_made
        obvious_off_topic = active_inquiry and _is_obvious_off_topic_followup(user_intent, user_text)

        if progress_made:
            triage_no_progress_count = 0
        elif same_field_stalled or obvious_off_topic or previous_switch_prompt_active:
            triage_no_progress_count = previous_no_progress_count + 1
        else:
            triage_no_progress_count = 0

        summary = {
            "high": "存在需要尽快急诊评估的危险信号。",
            "medium": "建议尽快完成基础门诊评估并结合检查结果进一步判断。",
            "low": "目前更适合观察随访，但如果症状持续或加重需要及时就诊。",
        }.get(risk_level, "建议结合门诊评估进一步判断。")

        inquiry_message = ""
        triage_switch_prompt_active = False

        if active_inquiry:
            if obvious_off_topic or previous_switch_prompt_active or triage_no_progress_count >= 3:
                triage_switch_prompt_active = True
                inquiry_message = _triage_switch_prompt(next_field)
                message = inquiry_message
            elif same_field_stalled:
                inquiry_message = _triage_prompt_text(next_field, clarification=True)
                message = f"我还没有识别出当前分诊所需信息。\n{inquiry_message}"
            else:
                inquiry_message = next_question or ""
                message = f"为了继续门诊分诊，我先追问 1 个关键问题：\n{inquiry_message}"
        else:
            triage_no_progress_count = 0
            message = (
                "关键信息已基本补齐，先给出门诊分诊建议。\n\n"
                f"当前风险：{_risk_label(risk_level)}\n"
                f"建议去向：{_disposition_label(disposition)}\n"
                f"建议检查：{', '.join(tests)}\n\n"
                "门诊分诊结果仅用于下一步就诊方向参考，如症状明显加重请及时线下就医。"
            )

        if show_thinking:
            print(f"[Outpatient Triage] risk={risk_level} disposition={disposition}")

        published_risk_level, published_disposition, published_tests, published_summary, published_triage_card = _published_triage_fields(
            active_inquiry=active_inquiry,
            risk_level=risk_level,
            disposition=disposition,
            tests=tests,
            summary=summary,
            symptom_snapshot=symptom_snapshot,
        )

        findings = {
            **previous_findings,
            "encounter_track": "outpatient_triage",
            "clinical_entry_reason": "symptom_based_triage",
            "entry_explanation_shown": False,
            "known_crc_signals": known_crc_signals,
            "triage_risk_level": published_risk_level,
            "triage_disposition": published_disposition,
            "triage_suggested_tests": published_tests,
            "triage_summary": published_summary,
            "symptom_snapshot": symptom_snapshot,
            "triage_card": published_triage_card,
            "active_inquiry": active_inquiry,
            "inquiry_type": "outpatient_triage",
            "inquiry_message": inquiry_message if active_inquiry else "",
            "triage_pending_fields": pending_fields,
            "triage_current_field": next_field,
            "triage_no_progress_count": triage_no_progress_count,
            "triage_switch_prompt_active": triage_switch_prompt_active,
            "triage_explicit_switch_request": False,
        }

        return {
            "encounter_track": "outpatient_triage",
            "clinical_entry_reason": "symptom_based_triage",
            "entry_explanation_shown": False,
            "known_crc_signals": known_crc_signals,
            "triage_risk_level": published_risk_level,
            "triage_disposition": published_disposition,
            "triage_suggested_tests": published_tests,
            "triage_summary": published_summary,
            "symptom_snapshot": symptom_snapshot,
            "triage_card": published_triage_card,
            "missing_critical_data": [],
            "findings": findings,
            "messages": [AIMessage(content=message)],
            "clinical_stage": "Inquiry_Pending" if active_inquiry else "Outpatient_Triage",
            "error": None,
        }

    return _run


def _looks_like_gi_vague_discomfort(text: str) -> bool:
    compact = _normalize_user_text(text)
    if not compact:
        return False
    return bool(
        re.search(r"(?:肠胃|胃肠|胃|腹部|肚子).{0,4}(?:不舒服|不适|难受)", compact)
    )


def _looks_like_outpatient_triage(text: str) -> bool:
    compact = _normalize_user_text(text)
    if not compact:
        return False

    if any(
        keyword in compact
        for keyword in (
            "腹痛",
            "肚子痛",
            "隐痛",
            "便血",
            "黑便",
            "腹泻",
            "便秘",
            "消瘦",
            "发热",
            "大便变细",
            "排便变化",
        )
    ):
        return True

    return _looks_like_gi_vague_discomfort(compact)


def node_clinical_entry_resolver(
    model: Any = None,
    *,
    streaming: bool = False,
    show_thinking: bool = True,
    **_: Any,
) -> Runnable:
    del model, streaming

    def _run(state: CRCAgentState) -> dict[str, Any]:
        findings = dict(state.findings or {})
        encounter_track = state.encounter_track or findings.get("encounter_track")
        user_text = _latest_user_text(state) or ""
        explicit_switch_request = bool(findings.get("triage_switch_prompt_active")) and (
            bool(findings.get("triage_explicit_switch_request"))
            or _looks_like_explicit_triage_switch(user_text, str(findings.get("user_intent") or ""))
        )

        if encounter_track == "outpatient_triage" and findings.get("active_inquiry") and not explicit_switch_request:
            return {
                "encounter_track": "outpatient_triage",
                "triage_risk_level": None,
                "triage_disposition": None,
                "triage_suggested_tests": [],
                "triage_summary": None,
                "triage_card": None,
                "findings": {
                    **findings,
                    "encounter_track": "outpatient_triage",
                    "triage_risk_level": None,
                    "triage_disposition": None,
                    "triage_suggested_tests": [],
                    "triage_summary": None,
                    "triage_card": None,
                },
                "clinical_stage": "Clinical_Entry",
                "error": None,
            }

        next_track = "outpatient_triage" if _looks_like_outpatient_triage(user_text) else "crc_clinical"
        if explicit_switch_request:
            next_track = "crc_clinical"

        if show_thinking:
            print(f"[Clinical Entry] route={next_track}")

        next_findings = {
            **findings,
            "encounter_track": next_track,
            "clinical_entry_reason": "symptom_based_triage" if next_track == "outpatient_triage" else "clinical_assessment",
            "entry_explanation_shown": False,
        }
        if explicit_switch_request:
            next_findings.update(
                {
                    "active_inquiry": False,
                    "inquiry_type": None,
                    "inquiry_message": None,
                    "triage_pending_fields": [],
                    "triage_current_field": None,
                    "triage_no_progress_count": 0,
                    "triage_switch_prompt_active": False,
                    "triage_explicit_switch_request": False,
                }
            )

        return {
            "encounter_track": next_track,
            "clinical_entry_reason": "symptom_based_triage" if next_track == "outpatient_triage" else "clinical_assessment",
            "entry_explanation_shown": False,
            "findings": next_findings,
            "clinical_stage": "Clinical_Entry",
            "error": None,
        }

    return _run


def route_after_clinical_entry(state: CRCAgentState) -> str:
    encounter_track = state.encounter_track or (state.findings or {}).get("encounter_track")
    if encounter_track == "outpatient_triage":
        return "outpatient_triage"
    return "assessment"


def route_after_outpatient_triage(state: CRCAgentState) -> str:
    del state
    return "end_turn"


__all__ = [
    "TRIAGE_FIELD_ORDER",
    "TRIAGE_QUESTION_MAP",
    "TRIAGE_CLARIFICATION_MAP",
    "TRIAGE_SWITCH_PROMPT_TEMPLATE",
    "_extract_boolean",
    "_extract_duration",
    "_triage_from_symptoms",
    "node_clinical_entry_resolver",
    "node_outpatient_triage",
    "route_after_clinical_entry",
    "route_after_outpatient_triage",
]


