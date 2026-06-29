from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "safety_policy.yaml"
)

DISPOSITION_ALIASES = {
    "urgent_gi_clinic": "urgent",
    "complete_basic_tests": "backfill",
    "collect_endoscopy_finding": "backfill",
    "routine_gi_followup": "routine",
}

DISPOSITION_CANONICAL = {
    "emergency": "emergency",
    "urgent": "urgent_gi_clinic",
    "backfill": "complete_basic_tests",
    "routine": "routine_gi_followup",
}


@dataclass(frozen=True)
class ClinicalSafetyPolicy:
    policy_id: str
    applies_to: str
    version: str
    status: str
    severity_order: list[str]
    fallback: dict[str, Any]
    rules: list[dict[str, Any]]


def load_clinical_safety_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> ClinicalSafetyPolicy:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return ClinicalSafetyPolicy(
        policy_id=data["policy_id"],
        applies_to=data["applies_to"],
        version=str(data["version"]),
        status=data["status"],
        severity_order=list(data["severity_order"]),
        fallback=dict(data["fallback"]),
        rules=list(data["rules"]),
    )


def compare_disposition(
    left: str,
    right: str,
    policy: ClinicalSafetyPolicy,
) -> int:
    left_rank = _severity_rank(_normalize_disposition(left), policy)
    right_rank = _severity_rank(_normalize_disposition(right), policy)
    return left_rank - right_rank


def merge_policy_disposition(
    runtime_disposition: str,
    policy_disposition: str,
    policy: ClinicalSafetyPolicy | None = None,
) -> str:
    active_policy = policy or load_clinical_safety_policy()
    runtime = _normalize_disposition(runtime_disposition)
    policy_value = _normalize_disposition(policy_disposition)

    if compare_disposition(policy_value, runtime, active_policy) > 0:
        return DISPOSITION_CANONICAL.get(policy_value, policy_value)
    return runtime_disposition


def evaluate_clinical_safety_policy(
    facts: dict[str, Any],
    policy: ClinicalSafetyPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or load_clinical_safety_policy()
    disposition = "routine"
    patient_message_key = "routine_gi_followup"
    matched_rules: list[str] = []
    hard_fail_flags: list[str] = []

    for rule in sorted(
        active_policy.rules,
        key=lambda item: item.get("priority", 0),
        reverse=True,
    ):
        if not _condition_matches(rule.get("condition", {}), facts):
            continue

        rule_disposition = _rule_disposition(rule)
        if compare_disposition(rule_disposition, disposition, active_policy) < 0:
            continue

        if compare_disposition(rule_disposition, disposition, active_policy) > 0:
            disposition = rule_disposition
            patient_message_key = rule.get("patient_message_key")
            matched_rules = []
            hard_fail_flags = []

        matched_rules.append(rule["id"])
        if rule.get("hard_fail_if_missed") is True:
            hard_fail_flags.append(rule["id"])

    return {
        "disposition": disposition,
        "matched_rules": matched_rules,
        "safety_policy_version": active_policy.policy_id,
        "hard_fail_flags": hard_fail_flags,
        "patient_message_key": patient_message_key,
    }


def _normalize_disposition(disposition: str) -> str:
    return DISPOSITION_ALIASES.get(disposition, disposition)


def _severity_rank(disposition: str, policy: ClinicalSafetyPolicy) -> int:
    try:
        return len(policy.severity_order) - policy.severity_order.index(disposition)
    except ValueError as exc:
        raise ValueError(f"Unknown disposition: {disposition}") from exc


def _rule_disposition(rule: dict[str, Any]) -> str:
    return _normalize_disposition(
        rule.get("disposition") or rule["disposition_minimum"]
    )


def _condition_matches(
    condition: dict[str, Any],
    facts: dict[str, Any],
) -> bool:
    all_present = condition.get("all_present")
    if all_present and not all(_is_present(facts.get(field)) for field in all_present):
        return False

    any_present = condition.get("any_present")
    if any_present and not any(_is_present(facts.get(field)) for field in any_present):
        return False

    missing_all = condition.get("missing_all")
    if missing_all and not all(
        not _is_present(facts.get(field)) for field in missing_all
    ):
        return False

    all_conditions = condition.get("all")
    if all_conditions and not all(
        _field_condition_matches(field_condition, facts)
        for field_condition in all_conditions
    ):
        return False

    return True


def _field_condition_matches(
    field_condition: dict[str, Any],
    facts: dict[str, Any],
) -> bool:
    value = facts.get(field_condition["field"])

    if "equals" in field_condition and value != field_condition["equals"]:
        return False

    if "gte" in field_condition:
        try:
            if float(value) < float(field_condition["gte"]):
                return False
        except (TypeError, ValueError):
            return False

    if "contains_any" in field_condition:
        text = "" if value is None else str(value).lower()
        needles = [str(item).lower() for item in field_condition["contains_any"]]
        if not any(needle in text for needle in needles):
            return False

    return True


def _is_present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True
