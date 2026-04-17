from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes.router import route_after_intent
from src.policies.turn_facts import build_turn_facts
from src.nodes.triage_nodes import (
    TRIAGE_QUESTION_MAP,
    _extract_boolean,
    _extract_duration,
    node_clinical_entry_resolver,
    _triage_from_symptoms,
    node_outpatient_triage,
)
from src.state import CRCAgentState


def _build_followup_state(
    previous_result: dict,
    *messages: str,
    findings_updates: dict | None = None,
) -> CRCAgentState:
    findings = dict(previous_result["findings"])
    if findings_updates:
        findings.update(findings_updates)
    return CRCAgentState(
        messages=[HumanMessage(content=text) for text in messages],
        encounter_track="outpatient_triage",
        symptom_snapshot=previous_result["symptom_snapshot"],
        findings=findings,
    )


def test_extract_duration_supports_short_units() -> None:
    assert _extract_duration("持续4个小时了") == "4个小时"
    assert _extract_duration("已经45分钟了") == "45分钟"
    assert _extract_duration("已经3天了") == "3天"
    assert _extract_duration("近2周总是这样") == "2周"
    assert _extract_duration("差不多一个月") == "一个月"


def test_extract_boolean_respects_colloquial_negation() -> None:
    bleeding_keywords = ("便血", "黑便")
    fever_keywords = ("发热", "发烧", "呕吐")

    assert _extract_boolean("没便血", bleeding_keywords) is False
    assert _extract_boolean("没有便血", bleeding_keywords) is False
    assert _extract_boolean("没发热", fever_keywords) is False
    assert _extract_boolean("没有发热", fever_keywords) is False


def test_triage_from_symptoms_ignores_negated_alarm_keywords() -> None:
    risk_level, disposition, _, _, _ = _triage_from_symptoms("我最近腹痛，但是没有便血，也没发热。")

    assert risk_level == "medium"
    assert disposition == "routine_gi_clinic"


def test_duration_answer_advances_to_next_triage_question() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    first_result = triage(CRCAgentState(messages=[HumanMessage(content="我最近左下腹部有点隐痛")]))

    second_result = triage(
        _build_followup_state(first_result, "我最近左下腹部有点隐痛", "持续4个小时了"),
    )

    assert second_result["symptom_snapshot"]["duration"] == "4个小时"
    assert second_result["findings"]["triage_current_field"] == "bleeding"
    assert TRIAGE_QUESTION_MAP["duration"] not in second_result["messages"][0].content
    assert TRIAGE_QUESTION_MAP["bleeding"] in second_result["messages"][0].content


def test_combined_answer_updates_multiple_fields_and_skips_ahead() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    first_result = triage(CRCAgentState(messages=[HumanMessage(content="我最近左下腹部有点隐痛")]))

    second_result = triage(
        _build_followup_state(
            first_result,
            "我最近左下腹部有点隐痛",
            "持续4个小时了，没有便血，也没发热",
        ),
    )

    snapshot = second_result["symptom_snapshot"]

    assert snapshot["duration"] == "4个小时"
    assert snapshot["bleeding"] is False
    assert snapshot["fever"] is False
    assert second_result["findings"]["triage_current_field"] == "bowel_change"
    assert TRIAGE_QUESTION_MAP["bowel_change"] in second_result["messages"][0].content


def test_unrecognized_followup_uses_clarification_prompt_instead_of_repeating_question() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    first_result = triage(CRCAgentState(messages=[HumanMessage(content="我最近左下腹部有点隐痛")]))

    second_result = triage(
        _build_followup_state(first_result, "我最近左下腹部有点隐痛", "就是不太舒服"),
    )

    message = second_result["messages"][0].content

    assert second_result["findings"]["triage_current_field"] == "duration"
    assert TRIAGE_QUESTION_MAP["duration"] not in message
    assert "4小时" in message
    assert "3天" in message
    assert "2周" in message
    assert "1个月" in message


def test_third_no_progress_followup_switches_to_goal_confirmation_prompt() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    first_result = triage(CRCAgentState(messages=[HumanMessage(content="我最近左下腹部有点隐痛")]))

    second_result = triage(
        _build_followup_state(
            first_result,
            "我最近左下腹部有点隐痛",
            "就是不太舒服",
            findings_updates={"user_intent": "general_chat"},
        ),
    )
    third_result = triage(
        _build_followup_state(
            second_result,
            "我最近左下腹部有点隐痛",
            "就是不太舒服",
            "还是不太舒服",
            findings_updates={"user_intent": "general_chat"},
        ),
    )
    fourth_result = triage(
        _build_followup_state(
            third_result,
            "我最近左下腹部有点隐痛",
            "就是不太舒服",
            "还是不太舒服",
            "我也说不清",
            findings_updates={"user_intent": "general_chat"},
        ),
    )

    assert "4小时" in second_result["messages"][0].content
    assert "4小时" in third_result["messages"][0].content
    assert "如果您想继续当前门诊分诊" in fourth_result["messages"][0].content
    assert fourth_result["findings"]["triage_switch_prompt_active"] is True
    assert fourth_result["findings"]["triage_no_progress_count"] == 3


def test_obvious_off_topic_followup_prompts_for_goal_confirmation_without_exiting_triage() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    first_result = triage(CRCAgentState(messages=[HumanMessage(content="我最近左下腹部有点隐痛")]))

    second_result = triage(
        _build_followup_state(
            first_result,
            "我最近左下腹部有点隐痛",
            "今天天气怎么样",
            findings_updates={"user_intent": "off_topic_redirect"},
        ),
    )

    assert second_result["findings"]["triage_current_field"] == "duration"
    assert second_result["findings"]["triage_switch_prompt_active"] is True
    assert "如果您想继续当前门诊分诊" in second_result["messages"][0].content


def test_active_inquiry_routes_back_to_triage_until_switch_prompt_is_confirmed() -> None:
    stay_in_triage = CRCAgentState(
        messages=[HumanMessage(content="就是不太舒服")],
        encounter_track="outpatient_triage",
        findings={
            "user_intent": "knowledge_query",
            "active_inquiry": True,
            "triage_switch_prompt_active": False,
            "triage_current_field": "duration",
            "triage_pending_fields": ["duration", "bleeding"],
            "inquiry_message": "我还没识别出持续时间，请直接回答例如“4小时”“3天”“2周”“1个月”。",
            "encounter_track": "outpatient_triage",
        },
    )
    assert route_after_intent(stay_in_triage) == "clinical_entry_resolver"

    confirmed_switch = CRCAgentState(
        messages=[HumanMessage(content="我想改问治疗方案")],
        encounter_track="outpatient_triage",
        findings={
            "user_intent": "treatment_decision",
            "active_inquiry": True,
            "triage_switch_prompt_active": True,
            "triage_current_field": "duration",
            "triage_pending_fields": ["duration", "bleeding"],
            "inquiry_message": "如果您想继续当前门诊分诊，请直接回答持续时间；如果您想改问别的问题，也可以直接告诉我。",
            "encounter_track": "outpatient_triage",
        },
    )
    assert route_after_intent(confirmed_switch) == "clinical_entry_resolver"

    confirmed_database_switch = CRCAgentState(
        messages=[HumanMessage(content="我想改问病例数据库")],
        encounter_track="outpatient_triage",
        findings={
            "user_intent": "case_database_query",
            "active_inquiry": True,
            "triage_switch_prompt_active": True,
            "triage_current_field": "duration",
            "triage_pending_fields": ["duration", "bleeding"],
            "inquiry_message": "如果您想继续当前门诊分诊，请直接回答持续时间；如果您想改问别的问题，也可以直接告诉我。",
            "encounter_track": "outpatient_triage",
        },
    )
    assert route_after_intent(confirmed_database_switch) == "case_database"

def test_turn_facts_detect_explicit_triage_switch_request_from_latest_user_text() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="\u6211\u60f3\u6539\u95ee\u6cbb\u7597\u65b9\u6848")],
        encounter_track="outpatient_triage",
        findings={
            "user_intent": "treatment_decision",
            "active_inquiry": True,
            "triage_switch_prompt_active": True,
            "encounter_track": "outpatient_triage",
        },
    )

    facts = build_turn_facts(state)

    assert facts.triage_explicit_switch_request is True


def test_clinical_entry_resolver_exits_triage_after_explicit_switch_request() -> None:
    resolver = node_clinical_entry_resolver(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="\u6211\u60f3\u6539\u95ee\u6cbb\u7597\u65b9\u6848")],
        encounter_track="outpatient_triage",
        findings={
            "user_intent": "treatment_decision",
            "active_inquiry": True,
            "triage_switch_prompt_active": True,
            "triage_explicit_switch_request": False,
            "triage_current_field": "bleeding",
            "triage_pending_fields": ["bleeding", "bowel_change"],
            "inquiry_message": "\u8bf7\u76f4\u63a5\u56de\u7b54\u6709/\u6ca1\u6709\u4fbf\u8840\u6216\u9ed1\u4fbf\u3002",
            "encounter_track": "outpatient_triage",
        },
    )

    result = resolver(state)

    assert result["encounter_track"] == "crc_clinical"
    assert result["findings"]["active_inquiry"] is False
    assert result["findings"]["triage_switch_prompt_active"] is False
    assert result["findings"]["triage_current_field"] is None

