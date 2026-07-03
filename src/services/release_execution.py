from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_execution_store import ReleaseExecutionStore
from src.contracts.release_execution import (
    FEATURE_FLAG_NAME,
    FEATURE_FLAG_SCOPE,
    FeatureFlagState,
    ReleaseExecutionRequest,
    ReleaseExecutionResult,
    canonical_execution_payload_hash,
    make_release_execution_id,
    make_release_execution_result_id,
)


class ReleaseExecutionPreflightError(ValueError):
    """Raised when controlled release execution gates are not satisfied."""


class ReleaseExecutionConflictError(ValueError):
    """Raised when execution conflicts with existing release state."""


class ReleaseExecutionService:
    def __init__(
        self,
        *,
        store: ReleaseExecutionStore,
        governance_loader: Callable[[], dict[str, Any]],
        dashboard_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._governance_loader = governance_loader
        self._dashboard_loader = dashboard_loader
        self._now = now

    def read_execution(self) -> dict[str, Any]:
        governance = self._governance_loader()
        dashboard = self._dashboard_loader()
        state = self._store.read_state()
        active_intent = self._active_intent(governance)
        intent_id = active_intent.get("intent_id") if active_intent else None
        rollback_plan = self._rollback_plan(governance)
        rollback_plan_id = (
            rollback_plan.get("rollback_plan_id")
            if rollback_plan is not None
            else None
        )

        return {
            "governance": self._governance_summary(governance),
            "preflight": {
                "release": self._preflight_result(
                    self._release_preflight_reasons(
                        governance=governance,
                        dashboard=dashboard,
                        state=state,
                        intent_id=intent_id,
                        expected_rollback_plan_id=rollback_plan_id,
                        include_existing_release=True,
                    )
                ),
                "rollback": self._preflight_result(
                    self._rollback_preflight_reasons(
                        governance=governance,
                        state=state,
                        intent_id=intent_id,
                        expected_rollback_plan_id=rollback_plan_id,
                    )
                ),
            },
            "feature_flag_state": deepcopy(state.feature_flag_state),
            "requests": [request.to_dict() for request in state.requests],
            "results": [result.to_dict() for result in state.results],
            "audit_events": [event.to_dict() for event in state.audit_events],
            "integrity": deepcopy(state.integrity),
            "runtime": {
                "auth": "admin",
                "source": "reports/release_execution",
                "mode": "controlled_local_execution",
            },
        }

    def execute_release(
        self,
        *,
        intent_id: str,
        requested_by: str,
        idempotency_key: str,
        reason: str,
        expected_rollback_plan_id: str,
    ) -> dict[str, Any]:
        return self._execute(
            action="release",
            intent_id=intent_id,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            reason=reason,
            expected_rollback_plan_id=expected_rollback_plan_id,
        )

    def execute_rollback(
        self,
        *,
        intent_id: str,
        requested_by: str,
        idempotency_key: str,
        reason: str,
        expected_rollback_plan_id: str,
    ) -> dict[str, Any]:
        return self._execute(
            action="rollback",
            intent_id=intent_id,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            reason=reason,
            expected_rollback_plan_id=expected_rollback_plan_id,
        )

    def _execute(
        self,
        *,
        action: str,
        intent_id: str,
        requested_by: str,
        idempotency_key: str,
        reason: str,
        expected_rollback_plan_id: str,
    ) -> dict[str, Any]:
        governance = self._governance_loader()
        dashboard = self._dashboard_loader()
        state = self._store.read_state()
        timestamp = self._now()
        active_intent = self._active_intent(governance)
        rollback_plan = self._rollback_plan(governance)
        rollback_target = (
            active_intent.get("rollback_target")
            if active_intent is not None
            else None
        )
        request = ReleaseExecutionRequest(
            execution_id=make_release_execution_id(
                intent_id,
                action,
                idempotency_key,
            ),
            intent_id=intent_id,
            action=action,
            requested_by=requested_by,
            requested_at=timestamp,
            idempotency_key=idempotency_key,
            reason=reason,
            expected_governance_hash=self._expected_governance_hash(
                governance,
                dashboard,
            ),
            expected_rollback_plan_id=expected_rollback_plan_id,
            target_flag_state={
                "flag_name": FEATURE_FLAG_NAME,
                "enabled": action == "release",
                "scope": FEATURE_FLAG_SCOPE,
            },
            rollback_target=(
                str(rollback_target)
                if action == "rollback" and rollback_target is not None
                else None
            ),
        )

        idempotent_match = self._store.find_by_idempotency_key(
            action,
            idempotency_key,
        )
        if idempotent_match is not None:
            self._store.assert_idempotent_request_matches(request)
            return self.read_execution()

        if action == "release":
            reasons = self._release_preflight_reasons(
                governance=governance,
                dashboard=dashboard,
                state=state,
                intent_id=intent_id,
                expected_rollback_plan_id=expected_rollback_plan_id,
                include_existing_release=True,
            )
        else:
            reasons = self._rollback_preflight_reasons(
                governance=governance,
                state=state,
                intent_id=intent_id,
                expected_rollback_plan_id=expected_rollback_plan_id,
            )
        if reasons:
            raise ReleaseExecutionPreflightError("; ".join(reasons))

        flag_state = FeatureFlagState(
            flag_name=FEATURE_FLAG_NAME,
            enabled=action == "release",
            scope=FEATURE_FLAG_SCOPE,
            source_intent_id=intent_id,
            source_execution_id=request.execution_id,
            rollback_target=str(rollback_target),
            updated_by=requested_by,
            updated_at=timestamp,
        )
        result = ReleaseExecutionResult(
            result_id=make_release_execution_result_id(request.execution_id),
            execution_id=request.execution_id,
            intent_id=intent_id,
            action=action,
            status="succeeded",
            started_at=timestamp,
            finished_at=timestamp,
            actor=requested_by,
            previous_flag_state=deepcopy(state.feature_flag_state),
            new_flag_state=flag_state.to_dict(),
            failure_reason=None,
        )
        self._store.write_successful_execution(
            request,
            result,
            flag_state,
            timestamp=timestamp,
        )
        return self.read_execution()

    def _release_preflight_reasons(
        self,
        *,
        governance: dict[str, Any],
        dashboard: dict[str, Any],
        state: Any,
        intent_id: str | None,
        expected_rollback_plan_id: str | None,
        include_existing_release: bool,
    ) -> list[str]:
        reasons = self._shared_governance_reasons(governance, state)
        active_intent = self._active_intent(governance)
        if active_intent is None:
            reasons.append("no active governance intent")
            return reasons
        if intent_id is not None and active_intent.get("intent_id") != intent_id:
            reasons.append("submitted intent does not match active intent")
        if active_intent.get("target_scope") != FEATURE_FLAG_SCOPE:
            reasons.append("target_scope must be feature_flag_candidate")
        if active_intent.get("derived_status") != "approved":
            reasons.append("active intent is not approved")
        if not self._required_approvals_complete(governance):
            reasons.append("required approvals are incomplete")
        rollback_plan = self._rollback_plan(governance)
        if rollback_plan is None or rollback_plan.get("status") != "accepted":
            reasons.append("accepted rollback plan is missing")
        elif (
            expected_rollback_plan_id is not None
            and rollback_plan.get("rollback_plan_id") != expected_rollback_plan_id
        ):
            reasons.append("expected rollback plan id mismatch")
        reasons.extend(self._dashboard_drift_reasons(active_intent, dashboard))
        if include_existing_release and self._successful_release_exists(
            state,
            str(active_intent.get("intent_id")),
        ):
            reasons.append("release already succeeded for this intent")
        return reasons

    def _rollback_preflight_reasons(
        self,
        *,
        governance: dict[str, Any],
        state: Any,
        intent_id: str | None,
        expected_rollback_plan_id: str | None,
    ) -> list[str]:
        reasons = self._shared_governance_reasons(governance, state)
        rollback_plan = self._rollback_plan(governance)
        if rollback_plan is None or rollback_plan.get("status") != "accepted":
            reasons.append("accepted rollback plan is missing")
        elif (
            expected_rollback_plan_id is not None
            and rollback_plan.get("rollback_plan_id") != expected_rollback_plan_id
        ):
            reasons.append("expected rollback plan id mismatch")
        if intent_id is None:
            reasons.append("no active governance intent")
            return reasons
        if not self._successful_release_exists(state, intent_id):
            reasons.append("no successful release execution exists for this intent")
        current_flag = state.feature_flag_state
        if not (
            isinstance(current_flag, dict)
            and current_flag.get("enabled") is True
            and current_flag.get("source_intent_id") == intent_id
        ):
            reasons.append("current feature flag is not enabled for this intent")
        return reasons

    def _shared_governance_reasons(
        self,
        governance: dict[str, Any],
        state: Any,
    ) -> list[str]:
        reasons: list[str] = []
        if self._integrity_status(governance.get("integrity")) != "verified":
            reasons.append("governance integrity is not verified")
        if self._integrity_status(state.integrity) != "verified":
            reasons.append("execution integrity is not verified")
        return reasons

    def _dashboard_drift_reasons(
        self,
        active_intent: dict[str, Any],
        dashboard: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        release_run = self._first_run(dashboard, "release_safety")
        if release_run is None or release_run.get("run_id") != active_intent.get(
            "source_release_report_id"
        ):
            reasons.append("dashboard release report drifted")
        if dashboard.get("release_decision") != active_intent.get(
            "release_decision_snapshot"
        ):
            reasons.append("dashboard release decision drifted")
        if dashboard.get("rollback_target") != active_intent.get("rollback_target"):
            reasons.append("dashboard rollback_target drifted")
        if dashboard.get("version_chain") != active_intent.get("version_chain"):
            reasons.append("dashboard version_chain drifted")
        summary = dashboard.get("summary")
        hard_fail_count = (
            summary.get("hard_fail_count") if isinstance(summary, dict) else None
        )
        if hard_fail_count != 0:
            reasons.append("hard_fail_count is not zero")
        literature_run = self._first_run(dashboard, "literature_shadow_harness")
        if literature_run is None or literature_run.get("status") != "shadow_only":
            reasons.append("literature status is not shadow_only")
        return reasons

    def _governance_summary(self, governance: dict[str, Any]) -> dict[str, Any]:
        active_intent = self._active_intent(governance)
        rollback_plan = self._rollback_plan(governance)
        return {
            "active_intent_id": (
                active_intent.get("intent_id") if active_intent else None
            ),
            "derived_status": (
                active_intent.get("derived_status") if active_intent else None
            ),
            "required_approvals_complete": self._required_approvals_complete(
                governance
            ),
            "rollback_plan_id": (
                rollback_plan.get("rollback_plan_id") if rollback_plan else None
            ),
        }

    def _expected_governance_hash(
        self,
        governance: dict[str, Any],
        dashboard: dict[str, Any],
    ) -> str:
        return canonical_execution_payload_hash(
            {
                "active_intent": self._active_intent(governance),
                "required_approvals": governance.get("required_approvals", []),
                "rollback_plan": self._rollback_plan(governance),
                "dashboard_snapshot": self._dashboard_snapshot(dashboard),
            }
        )

    def _dashboard_snapshot(self, dashboard: dict[str, Any]) -> dict[str, Any]:
        summary = dashboard.get("summary")
        literature_run = self._first_run(dashboard, "literature_shadow_harness")
        return {
            "version_chain": deepcopy(dashboard.get("version_chain")),
            "release_decision": dashboard.get("release_decision"),
            "rollback_target": dashboard.get("rollback_target"),
            "hard_fail_count": (
                summary.get("hard_fail_count") if isinstance(summary, dict) else None
            ),
            "literature_status": (
                literature_run.get("status") if literature_run is not None else None
            ),
        }

    def _required_approvals_complete(self, governance: dict[str, Any]) -> bool:
        approvals = governance.get("required_approvals")
        return (
            isinstance(approvals, list)
            and bool(approvals)
            and all(
                isinstance(approval, dict)
                and approval.get("status") == "approved"
                for approval in approvals
            )
        )

    def _active_intent(self, governance: dict[str, Any]) -> dict[str, Any] | None:
        active_intent = governance.get("active_intent")
        return active_intent if isinstance(active_intent, dict) else None

    def _rollback_plan(self, governance: dict[str, Any]) -> dict[str, Any] | None:
        rollback_plan = governance.get("rollback_plan")
        return rollback_plan if isinstance(rollback_plan, dict) else None

    def _integrity_status(self, integrity: Any) -> str | None:
        return integrity.get("status") if isinstance(integrity, dict) else None

    def _successful_release_exists(self, state: Any, intent_id: str) -> bool:
        return any(
            result.intent_id == intent_id
            and result.action == "release"
            and result.status == "succeeded"
            for result in state.results
        )

    def _first_run(
        self,
        dashboard: dict[str, Any],
        kind: str,
    ) -> dict[str, Any] | None:
        runs = dashboard.get("runs")
        if not isinstance(runs, list):
            return None
        for run in runs:
            if isinstance(run, dict) and run.get("kind") == kind:
                return run
        return None

    def _preflight_result(self, reasons: list[str]) -> dict[str, Any]:
        return {"allowed": not reasons, "reasons": reasons}


__all__ = [
    "ReleaseExecutionConflictError",
    "ReleaseExecutionPreflightError",
    "ReleaseExecutionService",
]
