from __future__ import annotations

import pytest

from src.services.crc_triage_flow import (
    CRC_TRIAGE_QUESTIONS,
    CRC_TRIAGE_STAGE_SEQUENCE,
    CrcTriageAnswer,
    advance_crc_triage,
    start_crc_triage_state,
)


EXPECTED_QUESTION_CONTRACTS = (
    {
        "id": "vitals_shock_or_consciousness",
        "stage": "vitals",
        "options": ("没有", "有", "不清楚"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "vitals_heart_or_breathing",
        "stage": "vitals",
        "options": ("没有", "有", "不清楚"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "red_flags_weight_or_bleeding",
        "stage": "red_flags",
        "options": ("没有", "有", "不清楚"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "red_flags_pain_or_obstruction",
        "stage": "red_flags",
        "options": ("没有", "有", "不清楚"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "symptom_cluster_chief",
        "stage": "symptom_cluster",
        "options": ("腹痛腹胀", "排便改变", "出血或黑便", "恶心呕吐", "其他"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "differential_duration",
        "stage": "differential",
        "options": ("正在好转", "基本稳定", "逐渐加重"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "tests_recent_exam",
        "stage": "tests",
        "options": ("没有做过", "做过但结果不清楚", "做过且有结果"),
        "askable": True,
        "terminal": False,
    },
    {
        "id": "final_ready",
        "stage": "final",
        "options": ("可以", "我还要补充", "暂时结束"),
        "askable": False,
        "terminal": True,
    },
)


def test_crc_triage_stage_sequence_matches_crc_client_reference() -> None:
    assert CRC_TRIAGE_STAGE_SEQUENCE == (
        "identity",
        "vitals",
        "red_flags",
        "symptom_cluster",
        "differential",
        "tests",
        "final",
    )


def test_crc_triage_question_contracts_are_table_driven_and_mark_terminal_sentinel() -> None:
    assert len(CRC_TRIAGE_QUESTIONS) == len(EXPECTED_QUESTION_CONTRACTS)

    for question, expected in zip(CRC_TRIAGE_QUESTIONS, EXPECTED_QUESTION_CONTRACTS):
        assert question["id"] == expected["id"]
        assert question["stage"] == expected["stage"]
        assert question["options"] == expected["options"]
        assert question["askable"] is expected["askable"]
        assert question["terminal"] is expected["terminal"]


def test_exported_question_contracts_do_not_allow_global_mutation() -> None:
    with pytest.raises(TypeError):
        CRC_TRIAGE_QUESTIONS[0]["id"] = "mutated"

    with pytest.raises(TypeError):
        CRC_TRIAGE_QUESTIONS[0]["options"][0] = "mutated"


def test_mutating_returned_state_question_does_not_mutate_global_contract() -> None:
    state = start_crc_triage_state(registry_patient_id=7)
    state["current_question"]["id"] = "mutated"
    state["current_question"]["options"] = ("mutated",)

    fresh_state = start_crc_triage_state(registry_patient_id=7)

    assert CRC_TRIAGE_QUESTIONS[0]["id"] == "vitals_shock_or_consciousness"
    assert CRC_TRIAGE_QUESTIONS[0]["options"] == ("没有", "有", "不清楚")
    assert fresh_state["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert fresh_state["current_question"]["options"] == ("没有", "有", "不清楚")


def test_start_state_uses_langg_patient_identity_and_asks_one_vitals_question() -> None:
    state = start_crc_triage_state(registry_patient_id=7)

    assert state["stage"] == "vitals"
    assert state["identity"] == {
        "source": "langg_registry",
        "registry_patient_id": 7,
        "crc_client_local_id": None,
    }
    assert state["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert state["current_question"]["text"] == "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？"
    assert state["current_question"]["options"] == ("没有", "有", "不清楚")
    assert state["current_question"]["askable"] is True
    assert state["current_question"]["terminal"] is False
    assert state["active_inquiry"] is True
    assert state["qa_summary"] == []
    assert state["node_results"] == []
    assert state["miss_count"] == 0


def test_crc_triage_question_chain_uses_exact_contract_ids() -> None:
    assert tuple(question["id"] for question in CRC_TRIAGE_QUESTIONS) == (
        *(item["id"] for item in EXPECTED_QUESTION_CONTRACTS),
    )


def test_answering_current_question_records_qa_and_advances_one_question() -> None:
    state = start_crc_triage_state(registry_patient_id=7)

    next_state = advance_crc_triage(
        state,
        CrcTriageAnswer(question_id="vitals_shock_or_consciousness", answer_text="没有"),
    )

    assert next_state["qa_summary"] == [
        {
            "stage": "vitals",
            "question_id": "vitals_shock_or_consciousness",
            "question": "最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
            "answer": "没有",
        }
    ]
    assert next_state["current_question"]["id"] == "vitals_heart_or_breathing"
    assert next_state["active_inquiry"] is True


def test_vitals_stage_completion_adds_node_result_card() -> None:
    state = start_crc_triage_state(registry_patient_id=7)
    state = advance_crc_triage(
        state,
        CrcTriageAnswer(question_id="vitals_shock_or_consciousness", answer_text="没有"),
    )
    state = advance_crc_triage(
        state,
        CrcTriageAnswer(question_id="vitals_heart_or_breathing", answer_text="没有"),
    )

    assert state["stage"] == "red_flags"
    assert len(state["node_results"]) == 1
    assert state["node_results"][-1] == {
        "stage": "vitals",
        "title": "节点1：生命体征评估",
        "risk_level": "生命体征平稳",
        "summary": "未识别到意识异常、休克表现、明显心率或呼吸异常。",
        "next_step": "进入节点2：全系统危险信号筛查。",
    }


def test_triage_advances_through_askable_question_chain_to_final_state_without_displaying_sentinel() -> None:
    state = start_crc_triage_state(registry_patient_id=7)
    askable_question_ids = tuple(
        item["id"] for item in EXPECTED_QUESTION_CONTRACTS if item["askable"]
    )

    for question_id in askable_question_ids:
        assert state["current_question"]["id"] == question_id
        assert state["current_question"]["askable"] is True
        assert state["current_question"]["terminal"] is False
        state = advance_crc_triage(
            state,
            CrcTriageAnswer(question_id=question_id, answer_text="没有"),
        )

    assert state["stage"] == "final"
    assert state["current_question"] is None
    assert state["active_inquiry"] is False
    assert "final_ready" not in [item["question_id"] for item in state["qa_summary"]]
    assert [item["question_id"] for item in state["qa_summary"]] == list(askable_question_ids)


def test_final_state_contains_assessment_payload_fields() -> None:
    state = start_crc_triage_state(registry_patient_id=7)
    for question_id, answer in [
        ("vitals_shock_or_consciousness", "没有"),
        ("vitals_heart_or_breathing", "没有"),
        ("red_flags_weight_or_bleeding", "没有"),
        ("red_flags_pain_or_obstruction", "没有"),
        ("symptom_cluster_chief", "大便习惯改变"),
        ("differential_duration", "1个月以上"),
        ("tests_recent_exam", "没有做过"),
    ]:
        state = advance_crc_triage(
            state,
            CrcTriageAnswer(question_id=question_id, answer_text=answer),
        )

    assert state["stage"] == "final"
    assert state["active_inquiry"] is False
    assert state["assessment"]["record_type"] == "crc_triage_assessment"
    assert state["assessment"]["source_subflow"] == "crc_triage"
    assert state["assessment"]["qa_summary"][0]["question_id"] == "vitals_shock_or_consciousness"
    assert state["assessment"]["missing_information"] == ["内镜或粪便潜血等辅助检查结果"]
    assert state["assessment"]["node_results"] == state["node_results"]
    assert state["assessment"]["suggested_tests"] == ["血常规", "粪便潜血", "肠镜或结肠镜相关检查"]
    assert state["assessment"]["patient_summary"] == "已完成 CRC 专项预问诊，建议结合辅助检查结果进一步判断。"
    assert state["assessment"]["next_step"] == "上传或携带近期检查结果，必要时预约消化专科门诊。"


def test_final_state_with_recent_test_results_uses_high_urgent_assessment_path() -> None:
    state = start_crc_triage_state(registry_patient_id=7)
    for question_id, answer in [
        ("vitals_shock_or_consciousness", "没有"),
        ("vitals_heart_or_breathing", "没有"),
        ("red_flags_weight_or_bleeding", "没有"),
        ("red_flags_pain_or_obstruction", "没有"),
        ("symptom_cluster_chief", "大便习惯改变"),
        ("differential_duration", "1个月以上"),
        ("tests_recent_exam", "做过且有结果"),
    ]:
        state = advance_crc_triage(
            state,
            CrcTriageAnswer(question_id=question_id, answer_text=answer),
        )

    assert state["stage"] == "final"
    assert state["assessment"]["missing_information"] == []
    assert state["assessment"]["risk_level"] == "high"
    assert state["assessment"]["disposition"] == "urgent_gi_clinic"
    assert state["assessment"]["node_results"] == state["node_results"]
    assert state["assessment"]["suggested_tests"] == ["血常规", "粪便潜血", "肠镜或结肠镜相关检查"]
    assert state["assessment"]["patient_summary"] == "已完成 CRC 专项预问诊，建议结合辅助检查结果进一步判断。"
    assert state["assessment"]["next_step"] == "上传或携带近期检查结果，必要时预约消化专科门诊。"


def test_stale_question_answer_is_accepted_as_free_text_without_advancing_wrong_question() -> None:
    state = start_crc_triage_state(registry_patient_id=7)

    next_state = advance_crc_triage(
        state,
        CrcTriageAnswer(question_id="older_question", answer_text="我想补充一下，最近没有便血"),
    )

    assert next_state["current_question"]["id"] == "vitals_shock_or_consciousness"
    assert next_state["qa_summary"][-1]["question_id"] == "free_text"
    assert next_state["qa_summary"][-1]["answer"] == "我想补充一下，最近没有便血"
