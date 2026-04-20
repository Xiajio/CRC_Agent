from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes.triage_nodes import node_outpatient_triage
from src.state import CRCAgentState, PatientProfile


def test_outpatient_triage_emits_self_report_patient_card() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    state = CRCAgentState(
        messages=[HumanMessage(content="I have abdominal pain and blood in stool for 2 weeks")],
        patient_profile=PatientProfile(age=57, gender="male", chief_complaint="blood in stool"),
    )

    result = triage(state)

    assert result["patient_card"]["type"] == "patient_card"
    assert result["patient_card"]["card_meta"]["source_mode"] == "patient_self_report"
    assert result["patient_card"]["card_meta"]["projection_version"]
    assert result["patient_card"]["data"]["patient_info"]["age"] == 57
    assert result["patient_card"]["field_meta"]["patient_info"]["age"]["display"] == "57"
    assert result["findings"]["encounter_track"] == "outpatient_triage"
