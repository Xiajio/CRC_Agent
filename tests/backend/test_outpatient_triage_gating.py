from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes.triage_nodes import node_clinical_entry_resolver, node_outpatient_triage
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



