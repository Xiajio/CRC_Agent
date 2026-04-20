from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes import assessment_nodes
from src.state import CRCAgentState, PatientProfile


def test_patient_assessment_emits_self_report_patient_card(monkeypatch) -> None:
    def _fake_assessment(*, model, tools, streaming=False, show_thinking=True):
        del model, tools, streaming, show_thinking

        def _run(state: CRCAgentState) -> dict:
            return {
                "findings": {
                    **dict(state.findings or {}),
                    "tumor_location": "rectum",
                    "clinical_stage_summary": "cT2N0M0",
                },
                "clinical_stage": "Assessment",
            }

        return _run

    monkeypatch.setattr(assessment_nodes, "node_assessment", _fake_assessment)
    runner = assessment_nodes.node_patient_assessment(model=object(), tools=[], show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="I have rectal bleeding")],
        patient_profile=PatientProfile(age=39, gender="female", chief_complaint="rectal bleeding"),
    )

    result = runner(state)

    assert result["patient_card"]["type"] == "patient_card"
    assert result["patient_card"]["card_meta"]["source_mode"] == "patient_self_report"
    assert result["patient_card"]["card_meta"]["projection_version"]
    assert result["patient_card"]["data"]["patient_info"]["age"] == 39
    assert result["patient_card"]["data"]["history_block"]["risk_factors"] is None
    assert result["patient_card"]["field_meta"]["patient_info"]["age"]["display"] == "39"
    assert result["findings"]["encounter_track"] == "patient_assessment"
