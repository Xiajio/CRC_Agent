from __future__ import annotations

import json
from pathlib import Path

from src.services.clinical_safety_policy import (
    compare_disposition,
    evaluate_clinical_safety_policy,
    load_clinical_safety_policy,
)
from src.services.crc_triage_flow import (
    CrcTriageAnswer,
    advance_crc_triage,
    start_crc_triage_state,
)


FIXTURE_PATH = Path("tests/fixtures/crc_mutation_pack_v0.json")
EXPECTED_CASE_IDS = [
    "rectal_bleeding_age_escalation",
    "possible_obstruction",
    "self_diagnosis_hemorrhoids_with_weight_loss",
    "missing_endoscopy_backfill",
    "topic_switch_resume_crc_state",
]


def _load_mutation_pack() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_crc_mutation_pack_replays_clinical_safety_policy_cases() -> None:
    mutation_pack = _load_mutation_pack()
    policy = load_clinical_safety_policy()

    assert mutation_pack["case_pack_id"] == "crc_mutation_pack_v0"
    assert mutation_pack["clinical_safety_policy_version"] == policy.policy_id
    assert [case["case_id"] for case in mutation_pack["cases"]] == EXPECTED_CASE_IDS

    for case in mutation_pack["cases"]:
        if case["case_id"] == "topic_switch_resume_crc_state":
            continue

        facts = {**case["base_input"], **case["mutation"]}
        actual = evaluate_clinical_safety_policy(facts, policy=policy)
        expected = case["expected"]

        if "disposition" in expected:
            assert actual["disposition"] == expected["disposition"]
        if "disposition_minimum" in expected:
            assert (
                compare_disposition(
                    actual["disposition"],
                    expected["disposition_minimum"],
                    policy,
                )
                >= 0
            )
        if "patient_message_key" in expected:
            assert actual["patient_message_key"] == expected["patient_message_key"]


def test_crc_mutation_pack_topic_switch_preserves_current_crc_question() -> None:
    mutation_pack = _load_mutation_pack()
    case = next(
        case
        for case in mutation_pack["cases"]
        if case["case_id"] == "topic_switch_resume_crc_state"
    )

    state = start_crc_triage_state(registry_patient_id=7)
    first_question_id = state["current_question"]["id"]
    state = advance_crc_triage(
        state,
        CrcTriageAnswer(question_id=first_question_id, answer_text="没有"),
    )
    current_question_id = state["current_question"]["id"]

    next_state = advance_crc_triage(
        state,
        CrcTriageAnswer(
            question_id="non_current_question",
            answer_text=case["mutation"]["off_topic_message"],
        ),
    )

    assert next_state["current_question"]["id"] == current_question_id
    assert next_state["qa_summary"][-1]["question_id"] == "free_text"
