from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SAFETY_POLICY_PATH = REPO_ROOT / "config" / "safety_policy.yaml"
SOURCE_PATH = "config/safety_policy.yaml"
NOTE = "read-only projection; not editable from admin UI"


def _as_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _condition_summary(condition: object) -> str | None:
    if not isinstance(condition, dict):
        return None

    if isinstance(condition.get("any_present"), list) or isinstance(
        condition.get("all_present"),
        list,
    ):
        parts: list[str] = []
        any_present = _as_string_list(condition.get("any_present"))
        all_present = _as_string_list(condition.get("all_present"))
        if any_present:
            parts.append(f"any: {', '.join(any_present)}")
        if all_present:
            parts.append(f"all: {', '.join(all_present)}")
        return "; ".join(parts) if parts else None

    if isinstance(condition.get("missing_all"), list):
        missing_all = _as_string_list(condition.get("missing_all"))
        return f"missing all: {', '.join(missing_all)}" if missing_all else None

    if isinstance(condition.get("all"), list):
        return "all structured criteria"

    return None


def _project_rule(rule: object) -> dict[str, Any] | None:
    if not isinstance(rule, dict):
        return None

    rule_id = rule.get("id")
    priority = rule.get("priority")
    disposition = rule.get("disposition") or rule.get("disposition_minimum")
    hard_fail_if_missed = rule.get("hard_fail_if_missed")
    if (
        not isinstance(rule_id, str)
        or not isinstance(priority, int)
        or isinstance(priority, bool)
        or not isinstance(disposition, str)
        or not isinstance(hard_fail_if_missed, bool)
    ):
        return None

    projected: dict[str, Any] = {
        "id": rule_id,
        "priority": priority,
        "disposition": disposition,
        "hard_fail_if_missed": hard_fail_if_missed,
        "group": "safety",
    }
    summary = _condition_summary(rule.get("condition"))
    if summary:
        projected["condition_summary"] = summary
    return projected


def build_admin_rules_projection(policy_path: Path | None = None) -> dict[str, Any]:
    resolved_path = policy_path or SAFETY_POLICY_PATH
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("safety policy must be a mapping")

    raw_rules = payload.get("rules")
    rules = [_project_rule(rule) for rule in raw_rules] if isinstance(raw_rules, list) else []

    return {
        "policy_id": _as_string(payload.get("policy_id")),
        "version": _as_string(payload.get("version")),
        "status": _as_string(payload.get("status")),
        "applies_to": _as_string(payload.get("applies_to")),
        "severity_order": _as_string_list(payload.get("severity_order")),
        "rules": [rule for rule in rules if rule is not None],
        "source_path": SOURCE_PATH,
        "note": NOTE,
    }
