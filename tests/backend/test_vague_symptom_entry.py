from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.nodes.assessment_nodes import node_assessment
from src.nodes.triage_nodes import node_clinical_entry_resolver, node_outpatient_triage
from src.state import CRCAgentState


class _UnusedModel:
    def with_structured_output(self, _schema):
        def _unexpected_invoke(_payload):
            raise AssertionError("Structured assessment chain should not run in fast-rule regression tests.")

        return RunnableLambda(_unexpected_invoke)


def _run_clinical_entry(user_message: str) -> dict:
    resolver = node_clinical_entry_resolver(show_thinking=False)
    return resolver(
        CRCAgentState(
            messages=[HumanMessage(content=user_message)],
            findings={"user_intent": "clinical_assessment"},
        )
    )


def _run_assessment(user_message: str, encounter_track: str, findings: dict) -> dict:
    assessment = node_assessment(model=_UnusedModel(), tools=[], show_thinking=False)
    return assessment(
        CRCAgentState(
            messages=[HumanMessage(content=user_message)],
            encounter_track=encounter_track,
            findings=findings,
        )
    )


def test_gi_vague_discomfort_routes_to_outpatient_triage_without_pathology_prompt() -> None:
    resolver_result = _run_clinical_entry("我肠胃有点不舒服")

    assert resolver_result["encounter_track"] == "outpatient_triage"
    assert resolver_result["clinical_entry_reason"] == "symptom_based_triage"

    triage = node_outpatient_triage(show_thinking=False)
    triage_result = triage(
        CRCAgentState(
            messages=[HumanMessage(content="我肠胃有点不舒服")],
            encounter_track=resolver_result["encounter_track"],
            findings=resolver_result["findings"],
        )
    )

    message = triage_result["messages"][0].content

    assert triage_result["findings"]["active_inquiry"] is True
    assert triage_result["findings"]["inquiry_type"] == "outpatient_triage"
    assert triage_result["findings"]["triage_current_field"] == "duration"
    assert "病理" not in message
    assert "继续门诊分诊" in message


def test_abdominal_vague_discomfort_also_routes_to_outpatient_triage() -> None:
    resolver_result = _run_clinical_entry("我腹部有点不适")

    assert resolver_result["encounter_track"] == "outpatient_triage"
    assert resolver_result["clinical_entry_reason"] == "symptom_based_triage"


def test_non_gi_vague_discomfort_avoids_pathology_required_prompt() -> None:
    resolver_result = _run_clinical_entry("我浑身有点不舒服")

    assert resolver_result["encounter_track"] == "crc_clinical"

    assessment_result = _run_assessment(
        "我浑身有点不舒服",
        resolver_result["encounter_track"],
        resolver_result["findings"],
    )

    message = assessment_result["messages"][0].content

    assert assessment_result["findings"]["inquiry_type"] == "symptom_inquiry"
    assert assessment_result["findings"]["active_inquiry"] is True
    assert "病理确诊信息缺失提醒" not in message
    assert "病理" not in message
    assert "结直肠癌" in message


def test_explicit_crc_question_stays_on_crc_assessment_path() -> None:
    resolver_result = _run_clinical_entry("术后病理怎么看")

    assert resolver_result["encounter_track"] == "crc_clinical"
    assert resolver_result["clinical_entry_reason"] == "clinical_assessment"
