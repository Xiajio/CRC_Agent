from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.services.clinical_safety_policy import (
    ClinicalSafetyPolicy,
    compare_disposition,
    evaluate_clinical_safety_policy,
)


AGENT_POLICY_VERSION = "agent_policy_20260629_0"
EVIDENCE_INDEX_VERSION = "rag_crc_guideline_20260620"
JUDGE_RUBRIC_VERSION = "crc_rubric_v0"
RUN_LEVEL = "L0_L1"
METADATA_ONLY_EXPECTED_FIELDS = (
    "crc_state_persisted",
    "patient_assistant_not_polluted",
)


@dataclass(frozen=True)
class HarnessCaseResult:
    case_id: str
    passed: bool
    hard_fail: bool
    hard_fail_type: str | None
    expected: dict[str, Any]
    actual: dict[str, Any]


def build_harness_run(
    *,
    run_id: str,
    mutation_pack: dict[str, Any],
    policy: ClinicalSafetyPolicy,
) -> dict[str, Any]:
    results = [
        _evaluate_case(case, policy)
        for case in mutation_pack.get("cases", [])
    ]
    failed = [result for result in results if not result.passed]
    hard_fails = [result for result in results if result.hard_fail]
    release_decision = _release_decision(failed, hard_fails)

    return {
        "run_id": run_id,
        "run_level": RUN_LEVEL,
        "case_pack_version": mutation_pack["case_pack_id"],
        "agent_policy_version": AGENT_POLICY_VERSION,
        "clinical_safety_policy_version": policy.policy_id,
        "evidence_index_version": EVIDENCE_INDEX_VERSION,
        "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        "summary": {
            "total_cases": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "hard_fail_count": len(hard_fails),
        },
        "cases": [asdict(result) for result in results],
        "hard_fails": [
            {"case_id": result.case_id, "type": result.hard_fail_type}
            for result in hard_fails
        ],
        "release_decision": release_decision,
    }


def _evaluate_case(
    case: dict[str, Any],
    policy: ClinicalSafetyPolicy,
) -> HarnessCaseResult:
    expected = dict(case["expected"])
    actual = _actual_for_case(case, policy)
    passed = _case_passed(expected, actual, policy)
    hard_fail_type = _hard_fail_type(expected, actual, policy)

    return HarnessCaseResult(
        case_id=case["case_id"],
        passed=passed,
        hard_fail=hard_fail_type is not None,
        hard_fail_type=hard_fail_type,
        expected=expected,
        actual=actual,
    )


def _actual_for_case(
    case: dict[str, Any],
    policy: ClinicalSafetyPolicy,
) -> dict[str, Any]:
    expected = case.get("expected", {})
    if _is_metadata_only_expected(expected):
        return {
            field: True
            for field in METADATA_ONLY_EXPECTED_FIELDS
            if field in expected
        }

    facts = {**case.get("base_input", {}), **case.get("mutation", {})}
    policy_result = evaluate_clinical_safety_policy(facts, policy=policy)
    return {
        "disposition": policy_result["disposition"],
        "patient_message_key": policy_result["patient_message_key"],
        "matched_rules": list(policy_result["matched_rules"]),
        "hard_fail_flags": list(policy_result["hard_fail_flags"]),
        "safety_policy_version": policy_result["safety_policy_version"],
    }


def _case_passed(
    expected: dict[str, Any],
    actual: dict[str, Any],
    policy: ClinicalSafetyPolicy,
) -> bool:
    metadata_only_expected = _is_metadata_only_expected(expected)

    if "disposition" in expected and actual.get("disposition") != expected["disposition"]:
        return False

    if "disposition_minimum" in expected:
        if compare_disposition(
            actual.get("disposition", ""),
            expected["disposition_minimum"],
            policy,
        ) < 0:
            return False

    if (
        "patient_message_key" in expected
        and actual.get("patient_message_key") != expected["patient_message_key"]
    ):
        return False

    for field in METADATA_ONLY_EXPECTED_FIELDS:
        if (
            field in expected
            and (metadata_only_expected or field in actual)
            and actual.get(field) != expected[field]
        ):
            return False

    if expected.get("must_not_close_as") == "hemorrhoids_only":
        if actual.get("disposition") == "routine":
            return False

    return True


def _hard_fail_type(
    expected: dict[str, Any],
    actual: dict[str, Any],
    policy: ClinicalSafetyPolicy,
) -> str | None:
    if expected.get("hard_fail_if_missed") is True and not _case_passed(
        expected,
        actual,
        policy,
    ):
        return "missed_required_safety_rule"

    if "hard_fail_if_below" in expected:
        floor = str(expected["hard_fail_if_below"])
        if (
            compare_disposition(
                actual.get("disposition", ""),
                floor,
                policy,
            )
            < 0
        ):
            return f"below_{floor}"

    return None


def _is_metadata_only_expected(expected: dict[str, Any]) -> bool:
    expected_fields = set(expected)
    return bool(expected_fields) and expected_fields.issubset(
        METADATA_ONLY_EXPECTED_FIELDS
    )


def _release_decision(
    failed: list[HarnessCaseResult],
    hard_fails: list[HarnessCaseResult],
) -> str:
    if hard_fails:
        return "block"
    if failed:
        return "shadow_only"
    return "pass"
