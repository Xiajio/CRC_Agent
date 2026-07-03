from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes.triage_nodes import node_clinical_entry_resolver, node_outpatient_triage
from src.services.crc_triage_flow import CrcTriageAnswer, advance_crc_triage, start_crc_triage_state
from src.state import CRCAgentState


def _run_outpatient_triage(user_message: str) -> dict:
    triage = node_outpatient_triage(show_thinking=False)
    state = CRCAgentState(messages=[HumanMessage(content=user_message)])
    return triage(state)


def test_outpatient_triage_hides_risk_and_cards_while_inquiry_is_active() -> None:
    result = _run_outpatient_triage("我最近有点腹痛")

    message = result["messages"][0].content
    findings = result["findings"]

    assert "当前风险" not in message
    assert "建议去向" not in message
    assert "建议检查" not in message
    assert "继续门诊分诊" in message

    assert result["triage_risk_level"] is None
    assert result["triage_disposition"] is None
    assert result["triage_suggested_tests"] == []
    assert result["triage_summary"] is None
    assert result["triage_card"] is None

    assert findings["active_inquiry"] is True
    assert findings["triage_risk_level"] is None
    assert findings["triage_disposition"] is None
    assert findings["triage_suggested_tests"] == []
    assert findings["triage_summary"] is None
    assert findings["triage_card"] is None
    assert "crc_protocol_assessment" not in findings
    assert "crc_protocol_assessment" not in result
    assert result["clinical_stage"] == "Inquiry_Pending"


def test_outpatient_triage_restores_risk_and_card_after_required_fields_are_complete() -> None:
    result = _run_outpatient_triage("腹痛3天，最近有腹泻，没有便血，也没有消瘦和发热。")

    message = result["messages"][0].content
    findings = result["findings"]

    assert "当前风险" in message
    assert "建议去向" in message
    assert "建议检查" in message
    assert "关键信息已基本补齐" in message

    assert result["triage_risk_level"] == "medium"
    assert result["triage_disposition"] == "routine_gi_clinic"
    assert result["triage_suggested_tests"]
    assert result["triage_summary"]
    assert result["triage_card"]["type"] == "triage_card"

    assert findings["active_inquiry"] is False
    assert findings["triage_risk_level"] == "medium"
    assert findings["triage_disposition"] == "routine_gi_clinic"
    assert findings["triage_card"]["type"] == "triage_card"
    assert "crc_protocol_assessment" not in findings
    assert "crc_protocol_assessment" not in result
    assert result["clinical_stage"] == "Outpatient_Triage"


def test_outpatient_triage_keeps_emergency_response_immediate() -> None:
    result = _run_outpatient_triage("我现在大量便血，而且已经停止排气排便。")

    message = result["messages"][0].content

    assert "当前风险" in message
    assert "建议去向" in message
    assert "急诊" in message
    assert result["findings"]["active_inquiry"] is False
    assert result["triage_risk_level"] == "high"
    assert result["triage_card"]["disposition"] == "emergency"


def test_clinical_entry_resolver_clears_stale_triage_advice_during_active_inquiry() -> None:
    resolver = node_clinical_entry_resolver(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="已经持续两天了")],
        encounter_track="outpatient_triage",
        triage_risk_level="medium",
        triage_disposition="urgent_gi_clinic",
        triage_suggested_tests=["血常规"],
        triage_summary="旧摘要",
        symptom_snapshot={"chief_symptoms": "腹痛"},
        findings={
            "encounter_track": "outpatient_triage",
            "active_inquiry": True,
            "inquiry_type": "outpatient_triage",
            "inquiry_message": "请补充持续时间",
            "triage_risk_level": "medium",
            "triage_disposition": "urgent_gi_clinic",
            "triage_suggested_tests": ["血常规"],
            "triage_summary": "旧摘要",
            "triage_card": {"type": "triage_card", "risk_level": "medium"},
            "crc_protocol_assessment": {"action": "archive_triage"},
            "symptom_snapshot": {"chief_symptoms": "腹痛"},
        },
    )

    result = resolver(state)
    findings = result["findings"]

    assert result["triage_risk_level"] is None
    assert result["triage_disposition"] is None
    assert result["triage_suggested_tests"] == []
    assert result["triage_summary"] is None

    assert findings["active_inquiry"] is True
    assert findings["triage_risk_level"] is None
    assert findings["triage_disposition"] is None
    assert findings["triage_suggested_tests"] == []
    assert findings["triage_summary"] is None
    assert findings["triage_card"] is None
    assert "crc_protocol_assessment" not in findings


def test_crc_subflow_routes_generic_start_text_to_outpatient_triage() -> None:
    resolver = node_clinical_entry_resolver(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="我想进行 CRC 专项预问诊")],
        patient_subflow="crc_triage",
        crc_triage={"action": "start"},
        findings={},
    )

    result = resolver(state)

    assert result["encounter_track"] == "outpatient_triage"
    assert result["clinical_entry_reason"] == "crc_triage"
    assert result["findings"]["patient_subflow"] == "crc_triage"
    assert result["findings"]["source_subflow"] == "crc_triage"


def test_crc_subflow_uses_crc_client_aligned_first_question() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="我想进行 CRC 专项预问诊，请按结构化问题引导我完成。")],
        patient_subflow="crc_triage",
        source_subflow="crc_triage",
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {"action": "start"},
        },
        registry_patient_id=7,
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["stage"] == "vitals"
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert result["clinical_stage"] == "Inquiry_Pending"
    assert "最近有没有出现头晕、眼前发黑、意识模糊" in result["messages"][0].content
    card = result["messages"][0].additional_kwargs["triage_question_card"]
    assert card["type"] == "triage_question_card"
    assert card["version"] == 1
    assert card["question_id"] == "vitals_shock_or_consciousness"
    assert card["field_key"] == "vitals_shock_or_consciousness"
    assert card["prompt"] == findings["crc_triage_state"]["current_question"]["text"]
    assert card["selection_mode"] == "single"
    assert card["allow_other"] is False
    assert card["source_subflow"] == "crc_triage"
    assert card["options"][0] == {"id": "option_0", "label": "没有", "submit_text": "没有"}
    assert all(set(option) == {"id", "label", "submit_text"} for option in card["options"])


def test_crc_subflow_answer_advances_protocol_question() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="没有")],
        patient_subflow="crc_triage",
        source_subflow="crc_triage",
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {
                "action": "answer",
                "question_id": "vitals_shock_or_consciousness",
            },
            "crc_triage_state": {
                "stage": "vitals",
                "identity": {"source": "langg_registry", "registry_patient_id": 7, "crc_client_local_id": None},
                "current_question": {
                    "id": "vitals_shock_or_consciousness",
                    "stage": "vitals",
                    "text": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
                    "options": ("没有", "有", "不清楚"),
                    "askable": True,
                    "terminal": False,
                },
                "active_inquiry": True,
                "qa_summary": [],
                "node_results": [],
                "miss_count": 0,
            },
        },
        registry_patient_id=7,
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_heart_or_breathing"
    assert findings["crc_triage_state"]["qa_summary"][0]["answer"] == "没有"
    assert "心慌、胸闷、喘不上气" in result["messages"][0].content


def test_crc_subflow_answer_uses_top_level_question_context_when_findings_context_missing() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="没有")],
        source_subflow="crc_triage",
        crc_triage={
            "action": "answer",
            "question_id": "vitals_shock_or_consciousness",
        },
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": None,
            "crc_triage_state": {
                "stage": "vitals",
                "identity": {"source": "langg_registry", "registry_patient_id": 7, "crc_client_local_id": None},
                "current_question": {
                    "id": "vitals_shock_or_consciousness",
                    "stage": "vitals",
                    "text": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
                    "options": ("没有", "有", "不清楚"),
                    "askable": True,
                    "terminal": False,
                },
                "active_inquiry": True,
                "qa_summary": [],
                "node_results": [],
                "miss_count": 0,
            },
        },
        registry_patient_id=7,
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_heart_or_breathing"
    assert findings["crc_triage_state"]["qa_summary"][0]["question_id"] == "vitals_shock_or_consciousness"
    assert findings["crc_triage"]["question_id"] == "vitals_shock_or_consciousness"


def test_crc_subflow_answer_uses_top_level_question_context_when_findings_context_empty() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="没有")],
        source_subflow="crc_triage",
        crc_triage={
            "action": "answer",
            "question_id": "vitals_shock_or_consciousness",
        },
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {},
            "crc_triage_state": {
                "stage": "vitals",
                "identity": {"source": "langg_registry", "registry_patient_id": 7, "crc_client_local_id": None},
                "current_question": {
                    "id": "vitals_shock_or_consciousness",
                    "stage": "vitals",
                    "text": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
                    "options": ("没有", "有", "不清楚"),
                    "askable": True,
                    "terminal": False,
                },
                "active_inquiry": True,
                "qa_summary": [],
                "node_results": [],
                "miss_count": 0,
            },
        },
        registry_patient_id=7,
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_heart_or_breathing"
    assert findings["crc_triage_state"]["qa_summary"][0]["question_id"] == "vitals_shock_or_consciousness"
    assert findings["crc_triage_state"]["qa_summary"][0]["answer"] == "没有"


def _crc_triage_answer_state(*, user_text: str, crc_triage: dict[str, object]) -> CRCAgentState:
    return CRCAgentState(
        messages=[HumanMessage(content=user_text)],
        source_subflow="crc_triage",
        crc_triage=crc_triage,
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": crc_triage,
            "crc_triage_state": {
                "stage": "vitals",
                "identity": {"source": "langg_registry", "registry_patient_id": 7, "crc_client_local_id": None},
                "current_question": {
                    "id": "vitals_shock_or_consciousness",
                    "stage": "vitals",
                    "text": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
                    "options": ("没有", "有", "不清楚"),
                    "askable": True,
                    "terminal": False,
                },
                "active_inquiry": True,
                "qa_summary": [],
                "node_results": [],
                "miss_count": 0,
            },
        },
        registry_patient_id=7,
    )


def test_crc_subflow_answer_uses_nested_triage_interaction_question_id() -> None:
    state = _crc_triage_answer_state(
        user_text="没有",
        crc_triage={
            "action": "answer",
            "triage_interaction": {
                "question_id": "vitals_shock_or_consciousness",
                "field_key": "vitals_shock_or_consciousness",
                "selection_mode": "single",
                "selected_option_ids": ["option_0"],
                "other_text": None,
            },
        },
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_heart_or_breathing"
    assert findings["crc_triage"]["question_id"] == "vitals_shock_or_consciousness"


def test_crc_subflow_answer_falls_back_to_current_question_when_question_id_missing() -> None:
    state = _crc_triage_answer_state(
        user_text="没有",
        crc_triage={"action": "answer"},
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    findings = result["findings"]
    assert findings["crc_triage_state"]["current_question"]["id"] == "vitals_heart_or_breathing"
    assert findings["crc_triage"]["question_id"] == "vitals_shock_or_consciousness"
    assert findings["crc_triage_state"]["qa_summary"][0]["answer"] == "没有"


def test_completed_crc_subflow_marks_findings_source_subflow() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="腹痛3天，最近有腹泻，没有便血，也没有消瘦和发热。")],
        patient_subflow="crc_triage",
        crc_triage={"action": "answer"},
    )

    result = triage(state)

    assert result["findings"]["active_inquiry"] is True
    assert result["findings"]["source_subflow"] == "crc_triage"
    assert result["findings"]["patient_subflow"] == "crc_triage"
    assert result["source_subflow"] == "crc_triage"
    assert result["findings"]["crc_triage_state"]["current_question"]["id"] == "vitals_shock_or_consciousness"


def test_active_crc_subflow_emits_question_card_without_legacy_protocol_assessment() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="腹痛3天，最近有腹泻，没有便血，也没有消瘦和发热。")],
        patient_subflow="crc_triage",
        crc_triage={"action": "answer"},
    )

    result = triage(state)

    assert "crc_protocol_assessment" not in result["findings"]
    assert "crc_protocol_assessment" not in result
    card = result["messages"][0].additional_kwargs["triage_question_card"]
    assert card["source_subflow"] == "crc_triage"
    assert card["question_id"] == "vitals_shock_or_consciousness"
    assert card["prompt"] == "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？"


def test_crc_subflow_final_answer_attaches_assessment_to_findings_state() -> None:
    crc_triage_state = start_crc_triage_state(registry_patient_id=7)
    for question_id, answer in [
        ("vitals_shock_or_consciousness", "没有"),
        ("vitals_heart_or_breathing", "没有"),
        ("red_flags_weight_or_bleeding", "没有"),
        ("red_flags_pain_or_obstruction", "没有"),
        ("symptom_cluster_chief", "大便习惯改变"),
        ("differential_duration", "1个月以上"),
    ]:
        crc_triage_state = advance_crc_triage(
            crc_triage_state,
            CrcTriageAnswer(question_id=question_id, answer_text=answer),
        )

    state = CRCAgentState(
        messages=[HumanMessage(content="没有做过")],
        patient_subflow="crc_triage",
        source_subflow="crc_triage",
        crc_triage={
            "action": "answer",
            "question_id": "tests_recent_exam",
        },
        findings={
            "patient_subflow": "crc_triage",
            "source_subflow": "crc_triage",
            "crc_triage": {
                "action": "answer",
                "question_id": "tests_recent_exam",
            },
            "crc_triage_state": crc_triage_state,
        },
        registry_patient_id=7,
    )

    result = node_outpatient_triage(show_thinking=False)(state)

    final_crc_state = result["findings"]["crc_triage_state"]
    assessment = final_crc_state["assessment"]
    assert result["clinical_stage"] == "Outpatient_Triage"
    assert final_crc_state["stage"] == "final"
    assert assessment["record_type"] == "crc_triage_assessment"
    assert assessment["node_results"]
    assert assessment["missing_information"] == ["内镜或粪便潜血等辅助检查结果"]


def test_crc_subflow_blocks_archive_when_endoscopy_lacks_key_finding() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    state = CRCAgentState(
        messages=[
            HumanMessage(
                content=(
                    "腹痛3天，没有便血，排便习惯没有明显变化，没有消瘦，也没有发热。"
                    "我做过肠镜，但现在不知道具体结果。"
                )
            )
        ],
        patient_subflow="crc_triage",
        crc_triage={"action": "answer"},
    )

    result = triage(state)

    assert "crc_protocol_assessment" not in result["findings"]
    assert result["findings"]["active_inquiry"] is True
    assert result["findings"]["crc_triage_state"]["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert "最近有没有出现头晕、眼前发黑、意识模糊" in result["messages"][0].content

