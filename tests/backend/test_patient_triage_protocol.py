from __future__ import annotations

from src.services.patient_triage_protocol import (
    assess_fatal_risk,
    assess_final_referral,
    assess_first_referral,
    assess_red_flags,
)


def test_assess_fatal_risk_forces_emergency_at_score_threshold() -> None:
    result = assess_fatal_risk(abnormal_vitals_count=2, has_shock_signs=True)

    assert result == {
        "node": "fatal_risk_assessment",
        "score": 80,
        "risk_level": "1级强制急诊",
        "disposition": "emergency",
        "action": "immediate_emergency",
        "can_continue": False,
    }


def test_assess_fatal_risk_routes_extreme_risk_before_emergency_threshold() -> None:
    result = assess_fatal_risk(abnormal_vitals_count=0, has_shock_signs=True)

    assert result["score"] == 50
    assert result["risk_level"] == "极高危"
    assert result["disposition"] == "urgent_assessment"
    assert result["action"] == "continue_red_flags"
    assert result["can_continue"] is True


def test_assess_fatal_risk_continues_when_vitals_are_stable() -> None:
    result = assess_fatal_risk(abnormal_vitals_count=1, has_shock_signs=False)

    assert result["score"] == 15
    assert result["risk_level"] == "生命体征平稳"
    assert result["disposition"] == "continue_triage"
    assert result["action"] == "continue_red_flags"
    assert result["can_continue"] is True


def test_assess_red_flags_routes_suspected_non_gi_fatal_to_emergency_or_specialty() -> None:
    result = assess_red_flags(
        red_flags_count=0,
        suspected_non_gi_fatal=True,
    )

    assert result["risk_level"] == "极高危"
    assert result["disposition"] == "emergency_or_specialty"
    assert result["action"] == "emergency_or_specialty"
    assert result["can_continue"] is False


def test_assess_red_flags_routes_two_or_more_flags_to_urgent_gi() -> None:
    result = assess_red_flags(
        red_flags_count=2,
        suspected_non_gi_fatal=False,
    )

    assert result["risk_level"] == "高危"
    assert result["disposition"] == "urgent_gi_clinic"
    assert result["action"] == "urgent_gi_clinic"
    assert result["can_continue"] is False


def test_assess_red_flags_backfills_relevant_unasked_flags_for_mid_risk() -> None:
    result = assess_red_flags(
        red_flags_count=1,
        suspected_non_gi_fatal=False,
        symptom_group="A",
        asked_flags=["黑便"],
        resume_node="节点3",
    )

    assert result["risk_level"] == "中危"
    assert result["disposition"] == "backfill_required"
    assert result["action"] == "backfill_node2"
    assert result["can_continue"] is False
    assert result["relevant_unasked_flags"] == ["便血", "持续腹痛", "停止排气排便"]
    assert result["resume_node"] == "节点3"


def test_assess_red_flags_reassessment_resumes_original_node() -> None:
    result = assess_red_flags(
        red_flags_count=1,
        suspected_non_gi_fatal=False,
        symptom_group="A",
        asked_flags=["黑便"],
        is_mid_risk_reassessment=True,
        resume_node="节点4",
    )

    assert result["risk_level"] == "中危"
    assert result["disposition"] == "continue_assessment"
    assert result["action"] == "resume_original_node"
    assert result["can_continue"] is True
    assert result["resume_node"] == "节点4"
    assert "relevant_unasked_flags" not in result


def test_assess_red_flags_without_flags_continues_symptom_cluster() -> None:
    result = assess_red_flags(
        red_flags_count=0,
        suspected_non_gi_fatal=False,
    )

    assert result["risk_level"] == "低危"
    assert result["disposition"] == "continue_symptom_cluster"
    assert result["action"] == "continue_symptom_cluster"
    assert result["can_continue"] is True


def test_assess_first_referral_backfills_mid_risk_before_referral() -> None:
    result = assess_first_referral(
        symptom_group="B",
        gi_confidence=60,
        suspected_disease="炎症性肠病待排",
        recommended_department="消化内科",
        referral_level="3级建议排查后转诊",
        relevant_unasked_flags=["便血"],
    )

    assert result["risk_level"] == "中危"
    assert result["disposition"] == "backfill_required"
    assert result["action"] == "backfill_node2"
    assert result["resume_node"] == "节点4"
    assert result["can_continue"] is False
    assert result["relevant_unasked_flags"] == ["便血"]


def test_assess_first_referral_continues_when_backfill_rule_is_not_met() -> None:
    result = assess_first_referral(
        symptom_group="B",
        gi_confidence=85,
        suspected_disease="结直肠肿瘤待排",
        recommended_department="消化内科",
        referral_level="2级建议专科转诊",
    )

    assert result["risk_level"] == "待检查解读"
    assert result["disposition"] == "continue_test_interpretation"
    assert result["action"] == "continue_test_interpretation"
    assert result["can_continue"] is True


def test_assess_final_referral_backfills_relevant_unasked_flags_before_archive() -> None:
    result = assess_final_referral(
        has_abnormal_tests=False,
        final_gi_confidence=40,
        need_mdt=False,
        mentioned_endoscopy=False,
        has_endoscopy_key_finding=False,
        relevant_unasked_flags=["便血"],
    )

    assert result["risk_level"] == "中危"
    assert result["disposition"] == "backfill_required"
    assert result["action"] == "backfill_node2"
    assert result["resume_node"] == "节点5"
    assert result["can_archive"] is False
    assert result["relevant_unasked_flags"] == ["便血"]


def test_assess_final_referral_blocks_when_endoscopy_lacks_key_finding() -> None:
    result = assess_final_referral(
        has_abnormal_tests=False,
        final_gi_confidence=40,
        need_mdt=False,
        mentioned_endoscopy=True,
        has_endoscopy_key_finding=False,
    )

    assert result["risk_level"] == "信息不足"
    assert result["disposition"] == "collect_endoscopy_finding"
    assert result["action"] == "need_endoscopy_finding"
    assert result["can_archive"] is False


def test_assess_final_referral_routes_mdt_to_specialist_archive() -> None:
    result = assess_final_referral(
        has_abnormal_tests=False,
        final_gi_confidence=40,
        need_mdt=True,
        mentioned_endoscopy=False,
        has_endoscopy_key_finding=False,
    )

    assert result["risk_level"] == "高危"
    assert result["disposition"] == "mdt_or_specialist"
    assert result["action"] == "archive_triage"
    assert result["can_archive"] is True


def test_assess_final_referral_routes_abnormal_tests_to_urgent_gi() -> None:
    result = assess_final_referral(
        has_abnormal_tests=True,
        final_gi_confidence=45,
        need_mdt=False,
        mentioned_endoscopy=False,
        has_endoscopy_key_finding=False,
    )

    assert result["risk_level"] == "高危"
    assert result["disposition"] == "urgent_gi_clinic"
    assert result["action"] == "archive_triage"
    assert result["can_archive"] is True


def test_assess_final_referral_routes_high_confidence_to_urgent_gi() -> None:
    result = assess_final_referral(
        has_abnormal_tests=False,
        final_gi_confidence=75,
        need_mdt=False,
        mentioned_endoscopy=False,
        has_endoscopy_key_finding=False,
    )

    assert result["risk_level"] == "高危"
    assert result["disposition"] == "urgent_gi_clinic"
    assert result["action"] == "archive_triage"
    assert result["can_archive"] is True


def test_assess_final_referral_routes_low_risk_to_routine_followup() -> None:
    result = assess_final_referral(
        has_abnormal_tests=False,
        final_gi_confidence=45,
        need_mdt=False,
        mentioned_endoscopy=False,
        has_endoscopy_key_finding=False,
    )

    assert result["risk_level"] == "中低危"
    assert result["disposition"] == "routine_gi_followup"
    assert result["action"] == "archive_triage"
    assert result["can_archive"] is True
