from __future__ import annotations

from typing import Any


RELEVANT_RED_FLAGS_BY_SYMPTOM_GROUP: dict[str, tuple[str, ...]] = {
    "A": ("黑便", "便血", "持续腹痛", "停止排气排便"),
    "B": ("便血", "黑便", "发热", "体重下降", "夜间腹泻"),
    "C": ("吞咽困难", "呕血", "黑便", "进行性消瘦"),
    "D": ("黄疸", "发热", "持续右上腹痛", "意识改变"),
    "E": ("反复呕吐", "脱水", "呕血", "剧烈腹痛"),
    "F": ("贫血", "便血", "排便习惯改变", "进行性消瘦"),
}


def _relevant_unasked_flags(
    symptom_group: str | None,
    asked_flags: list[str] | None,
) -> list[str]:
    group_key = symptom_group.upper() if symptom_group else ""
    asked = set(asked_flags or [])
    return [flag for flag in RELEVANT_RED_FLAGS_BY_SYMPTOM_GROUP.get(group_key, ()) if flag not in asked]


def assess_fatal_risk(abnormal_vitals_count: int, has_shock_signs: bool) -> dict[str, Any]:
    score = max(0, int(abnormal_vitals_count)) * 15 + (50 if has_shock_signs else 0)
    result: dict[str, Any] = {
        "node": "fatal_risk_assessment",
        "score": score,
    }

    if score >= 80:
        result.update(
            {
                "risk_level": "1级强制急诊",
                "disposition": "emergency",
                "action": "immediate_emergency",
                "can_continue": False,
            }
        )
    elif score >= 50:
        result.update(
            {
                "risk_level": "极高危",
                "disposition": "urgent_assessment",
                "action": "continue_red_flags",
                "can_continue": True,
            }
        )
    else:
        result.update(
            {
                "risk_level": "生命体征平稳",
                "disposition": "continue_triage",
                "action": "continue_red_flags",
                "can_continue": True,
            }
        )

    return result


def assess_red_flags(
    red_flags_count: int,
    suspected_non_gi_fatal: bool,
    symptom_group: str | None = None,
    asked_flags: list[str] | None = None,
    is_mid_risk_reassessment: bool = False,
    resume_node: str | None = None,
) -> dict[str, Any]:
    if suspected_non_gi_fatal:
        return {
            "node": "red_flags_assessment",
            "risk_level": "极高危",
            "disposition": "emergency_or_specialty",
            "action": "emergency_or_specialty",
            "can_continue": False,
        }

    if red_flags_count >= 2:
        return {
            "node": "red_flags_assessment",
            "risk_level": "高危",
            "disposition": "urgent_gi_clinic",
            "action": "urgent_gi_clinic",
            "can_continue": False,
        }

    if red_flags_count == 1:
        relevant_unasked_flags = _relevant_unasked_flags(symptom_group, asked_flags)
        if relevant_unasked_flags and not is_mid_risk_reassessment:
            return {
                "node": "red_flags_assessment",
                "risk_level": "中危",
                "disposition": "backfill_required",
                "action": "backfill_node2",
                "can_continue": False,
                "relevant_unasked_flags": relevant_unasked_flags,
                "resume_node": resume_node or "节点4",
            }

        result: dict[str, Any] = {
            "node": "red_flags_assessment",
            "risk_level": "中危",
            "disposition": "continue_assessment",
            "action": "resume_original_node",
            "can_continue": True,
        }
        if resume_node is not None:
            result["resume_node"] = resume_node
        return result

    return {
        "node": "red_flags_assessment",
        "risk_level": "低危",
        "disposition": "continue_symptom_cluster",
        "action": "continue_symptom_cluster",
        "can_continue": True,
    }


def assess_first_referral(
    symptom_group: str,
    gi_confidence: int,
    suspected_disease: str,
    recommended_department: str | None,
    referral_level: str,
    relevant_unasked_flags: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "node": "first_referral_assessment",
        "symptom_group": symptom_group,
        "gi_confidence": gi_confidence,
        "suspected_disease": suspected_disease,
        "recommended_department": recommended_department,
        "referral_level": referral_level,
    }

    if "3级建议排查后转诊" in referral_level and 45 <= gi_confidence <= 75:
        result.update(
            {
                "risk_level": "中危",
                "disposition": "backfill_required",
                "action": "backfill_node2",
                "can_continue": False,
                "resume_node": "节点4",
                "relevant_unasked_flags": list(relevant_unasked_flags or []),
            }
        )
    else:
        result.update(
            {
                "risk_level": "待检查解读",
                "disposition": "continue_test_interpretation",
                "action": "continue_test_interpretation",
                "can_continue": True,
            }
        )

    return result


def assess_final_referral(
    has_abnormal_tests: bool,
    final_gi_confidence: int,
    need_mdt: bool,
    mentioned_endoscopy: bool,
    has_endoscopy_key_finding: bool,
    relevant_unasked_flags: list[str] | None = None,
) -> dict[str, Any]:
    if relevant_unasked_flags:
        return {
            "node": "final_referral_assessment",
            "risk_level": "中危",
            "disposition": "backfill_required",
            "action": "backfill_node2",
            "can_archive": False,
            "resume_node": "节点5",
            "relevant_unasked_flags": list(relevant_unasked_flags),
        }

    if mentioned_endoscopy and not has_endoscopy_key_finding:
        return {
            "node": "final_referral_assessment",
            "risk_level": "信息不足",
            "disposition": "collect_endoscopy_finding",
            "action": "need_endoscopy_finding",
            "can_archive": False,
        }

    if need_mdt:
        return {
            "node": "final_referral_assessment",
            "risk_level": "高危",
            "disposition": "mdt_or_specialist",
            "action": "archive_triage",
            "can_archive": True,
        }

    if has_abnormal_tests or final_gi_confidence >= 70:
        return {
            "node": "final_referral_assessment",
            "risk_level": "高危",
            "disposition": "urgent_gi_clinic",
            "action": "archive_triage",
            "can_archive": True,
        }

    return {
        "node": "final_referral_assessment",
        "risk_level": "中低危",
        "disposition": "routine_gi_followup",
        "action": "archive_triage",
        "can_archive": True,
    }


__all__ = [
    "RELEVANT_RED_FLAGS_BY_SYMPTOM_GROUP",
    "assess_fatal_risk",
    "assess_red_flags",
    "assess_first_referral",
    "assess_final_referral",
]
