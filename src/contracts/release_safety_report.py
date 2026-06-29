from __future__ import annotations

from typing import Any


ROLLBACK_TARGET = "agent_policy_20260624_0"


def build_release_safety_report(
    *,
    report_id: str,
    harness_run: dict[str, Any],
) -> dict[str, Any]:
    hard_fail_types = _hard_fail_types(harness_run.get("hard_fails", []))

    return {
        "report_id": report_id,
        "change_type": ["clinical_safety_policy", "crc_persistence"],
        "version_chain": {
            "agent_policy_version": harness_run["agent_policy_version"],
            "clinical_safety_policy_version": harness_run[
                "clinical_safety_policy_version"
            ],
            "evidence_index_version": harness_run["evidence_index_version"],
            "judge_rubric_version": harness_run["judge_rubric_version"],
        },
        "harness_runs": [harness_run["run_id"]],
        "hard_fail_summary": {
            "count": len(hard_fail_types),
            "types": hard_fail_types,
        },
        "release_decision": "block" if hard_fail_types else "feature_flag_or_pass",
        "rollback_target": ROLLBACK_TARGET,
    }


def _hard_fail_types(hard_fails: Any) -> list[str]:
    types: list[str] = []
    for item in hard_fails or []:
        if isinstance(item, dict):
            fail_type = item.get("type")
        else:
            fail_type = item
        if isinstance(fail_type, str) and fail_type and fail_type not in types:
            types.append(fail_type)
    return types
