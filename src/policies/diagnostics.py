"""Pure shadow diagnostics for comparing legacy and policy outcomes."""

from __future__ import annotations

from typing import Any

from .types import RouteDecision


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def record_route_divergence(
    legacy_route: str,
    policy_decision: RouteDecision,
    *,
    divergence_reason: str | None = None,
) -> dict[str, Any]:
    """Return a plain shadow payload comparing a legacy route to a policy route."""
    policy_route = _normalize_text(policy_decision.route)
    legacy_route = _normalize_text(legacy_route)
    route_diverged = legacy_route != policy_route

    reason = ""
    if route_diverged:
        reason = (
            _normalize_text(divergence_reason)
            or _normalize_text(policy_decision.notes)
            or "policy route differs from legacy route"
        )

    return {
        "legacy_route": legacy_route,
        "policy_route": policy_route,
        "route_diverged": route_diverged,
        "divergence_reason": reason,
        "policy_rule_name": _normalize_text(policy_decision.rule_name),
    }


def record_review_divergence(
    legacy_review_action: str,
    policy_review_action: str,
    *,
    policy_rule_name: str = "",
    divergence_reason: str | None = None,
) -> dict[str, Any]:
    """Return a plain shadow payload comparing a legacy review action to a policy action."""
    legacy_review_action = _normalize_text(legacy_review_action)
    policy_review_action = _normalize_text(policy_review_action)
    review_diverged = legacy_review_action != policy_review_action

    reason = ""
    if review_diverged:
        reason = (
            _normalize_text(divergence_reason)
            or "policy review action differs from legacy review action"
        )

    return {
        "legacy_review_action": legacy_review_action,
        "policy_review_action": policy_review_action,
        "review_diverged": review_diverged,
        "divergence_reason": reason,
        "policy_rule_name": _normalize_text(policy_rule_name),
    }


__all__ = ["record_route_divergence", "record_review_divergence"]
