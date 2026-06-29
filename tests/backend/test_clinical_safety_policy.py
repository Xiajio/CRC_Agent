from pathlib import Path

from src.services.clinical_safety_policy import (
    compare_disposition,
    evaluate_clinical_safety_policy,
    load_clinical_safety_policy,
    merge_policy_disposition,
)


CONFIG_PATH = Path("config/safety_policy.yaml")


def test_load_clinical_safety_policy_loads_expected_metadata():
    policy = load_clinical_safety_policy(CONFIG_PATH)

    assert policy.policy_id == "crc_safety_policy_v0"
    assert policy.applies_to == "patient_crc_triage"
    assert policy.version == "2026-06-29.0"
    assert policy.status == "draft"
    assert policy.severity_order == [
        "emergency",
        "urgent",
        "backfill",
        "routine",
    ]
    assert policy.fallback == {
        "missing_required_input": "ask_targeted_follow_up",
        "rule_conflict": "choose_highest_severity",
        "tool_failure": "safe_message_and_human_review",
    }
    assert all("hard_fail_if_missed" in rule for rule in policy.rules)
    assert all("hard_fail" not in rule for rule in policy.rules)


def test_load_clinical_safety_policy_default_path_works_from_backend_cwd(
    monkeypatch,
):
    monkeypatch.chdir(Path("backend"))

    policy = load_clinical_safety_policy()

    assert policy.policy_id == "crc_safety_policy_v0"


def test_compare_disposition_uses_configured_severity_order():
    policy = load_clinical_safety_policy(CONFIG_PATH)

    assert compare_disposition("emergency", "urgent", policy) > 0
    assert compare_disposition("urgent", "backfill", policy) > 0
    assert compare_disposition("backfill", "routine", policy) > 0
    assert compare_disposition("routine", "emergency", policy) < 0
    assert compare_disposition("urgent", "urgent", policy) == 0


def test_rectal_bleeding_age_escalates_to_urgent_with_hard_fail():
    result = evaluate_clinical_safety_policy(
        {
            "age": 62,
            "rectal_bleeding": True,
        },
        load_clinical_safety_policy(CONFIG_PATH),
    )

    assert result["disposition"] == "urgent"
    assert result["matched_rules"] == ["rectal_bleeding_age_escalation"]
    assert result["safety_policy_version"] == "crc_safety_policy_v0"
    assert result["hard_fail_flags"] == ["rectal_bleeding_age_escalation"]
    assert result["patient_message_key"] == "urgent_clinical_review"


def test_rectal_bleeding_weight_loss_escalates_to_urgent_with_hard_fail():
    result = evaluate_clinical_safety_policy(
        {
            "age": 42,
            "rectal_bleeding": True,
            "weight_loss": True,
        },
        load_clinical_safety_policy(CONFIG_PATH),
    )

    assert result["disposition"] == "urgent"
    assert "rectal_bleeding_weight_loss_escalation" in result["matched_rules"]
    assert (
        "rectal_bleeding_weight_loss_escalation"
        in result["hard_fail_flags"]
    )
    assert result["patient_message_key"] == "urgent_clinical_review"


def test_bowel_obstruction_red_flags_escalate_to_emergency():
    result = evaluate_clinical_safety_policy(
        {
            "vomiting": True,
            "obstipation": True,
            "severe_abdominal_pain": True,
        },
        load_clinical_safety_policy(CONFIG_PATH),
    )

    assert result["disposition"] == "emergency"
    assert result["matched_rules"] == ["bowel_obstruction_red_flag"]
    assert result["hard_fail_flags"] == ["bowel_obstruction_red_flag"]
    assert result["patient_message_key"] == "seek_emergency_care"


def test_self_diagnosis_with_red_flag_review_backfills_without_hard_fail():
    result = evaluate_clinical_safety_policy(
        {
            "age": 42,
            "rectal_bleeding": True,
            "user_explanation": "可能是痔疮",
            "endoscopy_status": "normal",
            "fecal_occult_blood_test": "negative",
        },
        load_clinical_safety_policy(CONFIG_PATH),
    )

    assert result["disposition"] == "backfill"
    assert "self_diagnosis_with_red_flag_review" in result["matched_rules"]
    assert result["hard_fail_flags"] == []
    assert result["patient_message_key"] == "do_not_close_red_flag_as_benign"


def test_missing_recent_test_results_backfills_when_no_higher_rule_matches():
    result = evaluate_clinical_safety_policy(
        {
            "age": 42,
            "rectal_bleeding": False,
        },
        load_clinical_safety_policy(CONFIG_PATH),
    )

    assert result["disposition"] == "backfill"
    assert result["matched_rules"] == ["missing_endoscopy_backfill"]
    assert result["hard_fail_flags"] == []
    assert result["patient_message_key"] == "prepare_recent_test_results"


def test_merge_policy_disposition_never_lowers_existing_urgent_path():
    policy = load_clinical_safety_policy(CONFIG_PATH)

    assert (
        merge_policy_disposition("urgent_gi_clinic", "backfill", policy)
        == "urgent_gi_clinic"
    )
    assert (
        merge_policy_disposition("complete_basic_tests", "urgent", policy)
        == "urgent_gi_clinic"
    )
    assert (
        merge_policy_disposition("routine_gi_followup", "emergency", policy)
        == "emergency"
    )
