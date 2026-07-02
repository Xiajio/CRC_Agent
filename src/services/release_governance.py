from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_governance_store import ReleaseGovernanceStore
from src.contracts.release_governance import (
    ReleaseApproval,
    ReleaseIntent,
    ReleaseRollbackPlan,
    make_release_approval_id,
    make_release_intent_id,
    make_release_rollback_plan_id,
)


class GovernanceValidationError(ValueError):
    """Raised when a requested governance action violates release rules."""


class GovernanceConflictError(ValueError):
    """Raised when a requested governance action conflicts with active state."""


class ReleaseGovernanceService:
    def __init__(
        self,
        *,
        store: ReleaseGovernanceStore,
        dashboard_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._dashboard_loader = dashboard_loader
        self._now = now

    def read_governance(self) -> dict[str, Any]:
        dashboard = self._dashboard_loader()
        state = self._store.read_state()
        intents = [
            {
                **intent.to_dict(),
                "derived_status": self._derived_status(intent, state),
            }
            for intent in state.intents
        ]
        active_intent = self._active_intent(intents)
        active_scope = (
            active_intent["target_scope"] if active_intent is not None else "shadow"
        )
        active_intent_id = (
            active_intent["intent_id"] if active_intent is not None else None
        )

        return {
            "dashboard_snapshot": self._dashboard_snapshot(dashboard),
            "intents": intents,
            "active_intent": active_intent,
            "approvals": [
                approval.to_dict() for approval in state.approvals
            ],
            "required_approvals": self._required_approvals(
                target_scope=active_scope,
                intent_id=active_intent_id,
                approvals=state.approvals,
            ),
            "rollback_plan": self._latest_rollback_plan(
                active_intent_id,
                state.rollback_plans,
            ),
            "audit_events": [
                event.to_dict() for event in state.audit_events
            ],
            "integrity": deepcopy(state.integrity),
            "disabled_execution_actions": self._disabled_execution_actions(),
            "runtime": {
                "auth": "admin",
                "source": "reports/release_governance",
                "mode": "audit_only",
            },
        }

    def create_intent(
        self,
        *,
        requested_by: str,
        target_scope: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        self._require_non_empty("reason", reason)
        dashboard = self._dashboard_loader()
        release_run = self._required_run(dashboard, "release_safety")
        p0_runs = self._runs_by_kind(dashboard, "p0_crc_harness")
        if not p0_runs:
            raise GovernanceValidationError("p0_crc_harness run is required")

        source_release_report_id = self._required_string(
            release_run,
            "run_id",
            "release_safety run_id is required",
        )
        source_report_path = self._required_string(
            release_run,
            "source_path",
            "release_safety source_path is required",
        )
        release_decision = self._required_string(
            dashboard,
            "release_decision",
            "release_decision is required",
        )
        rollback_target = self._required_string(
            dashboard,
            "rollback_target",
            "rollback_target is required",
        )
        blocking_summary = self._summary(dashboard)
        hard_fail_count = self._int_value(
            blocking_summary.get("hard_fail_count"),
            field_name="hard_fail_count",
        )

        if status == "pending_approval":
            if hard_fail_count > 0:
                raise GovernanceValidationError(
                    "hard fails prevent pending approval"
                )
            if release_decision == "block":
                raise GovernanceValidationError(
                    "release decision blocks pending approval"
                )

        literature_run = self._first_run(dashboard, "literature_shadow_harness")
        if target_scope == "feature_flag_candidate":
            literature_status = (
                literature_run.get("status") if literature_run is not None else None
            )
            if literature_status != "shadow_only":
                raise GovernanceValidationError(
                    "literature run must be shadow_only"
                )

        state = self._store.read_state()
        active_same_source = [
            intent
            for intent in state.intents
            if intent.source_release_report_id == source_release_report_id
            and self._derived_status(intent, state) != "cancelled"
        ]
        if active_same_source:
            raise GovernanceConflictError("active intent already exists")

        timestamp = self._now()
        intent = ReleaseIntent(
            intent_id=make_release_intent_id(source_release_report_id),
            source_release_report_id=source_release_report_id,
            source_report_path=source_report_path,
            harness_run_ids=[
                self._required_string(
                    run,
                    "run_id",
                    "p0_crc_harness run_id is required",
                )
                for run in p0_runs
            ],
            literature_run_id=(
                self._required_string(
                    literature_run,
                    "run_id",
                    "literature_shadow_harness run_id is required",
                )
                if literature_run is not None
                else None
            ),
            version_chain=self._version_chain(dashboard),
            release_decision_snapshot=release_decision,
            rollback_target=rollback_target,
            requested_by=requested_by,
            requested_at=timestamp,
            target_scope=target_scope,
            status=status,
            blocking_summary=blocking_summary,
        )
        self._store.write_intent(
            intent,
            actor=requested_by,
            timestamp=timestamp,
        )
        return intent.to_dict()

    def record_approval(
        self,
        *,
        intent_id: str,
        approver_role: str,
        decision: str,
        reason: str,
        signed_by: str,
    ) -> dict[str, Any]:
        intent = self._existing_non_cancelled_intent(intent_id)
        timestamp = self._now()
        approval = ReleaseApproval(
            approval_id=make_release_approval_id(
                intent_id,
                approver_role,
                timestamp,
            ),
            intent_id=intent_id,
            approver_role=approver_role,
            decision=decision,
            reason=reason,
            signed_by=signed_by,
            signed_at=timestamp,
            required=approver_role in self._required_roles(intent.target_scope),
        )
        self._store.write_approval(
            approval,
            actor=signed_by,
            timestamp=timestamp,
        )
        return approval.to_dict()

    def record_rollback_plan(
        self,
        *,
        intent_id: str,
        owner: str,
        status: str,
        verification_steps: list[str],
    ) -> dict[str, Any]:
        intent = self._existing_non_cancelled_intent(intent_id)
        timestamp = self._now()
        plan = ReleaseRollbackPlan(
            rollback_plan_id=make_release_rollback_plan_id(intent_id, timestamp),
            intent_id=intent_id,
            rollback_target=intent.rollback_target,
            owner=owner,
            status=status,
            verification_steps=verification_steps,
            created_at=timestamp,
        )
        self._store.write_rollback_plan(
            plan,
            actor=owner,
            timestamp=timestamp,
        )
        return plan.to_dict()

    def cancel_intent(
        self,
        *,
        intent_id: str,
        actor: str,
        reason: str,
    ) -> None:
        self._require_non_empty("reason", reason)
        self._existing_non_cancelled_intent(intent_id)
        self._store.append_cancel_event(
            intent_id=intent_id,
            actor=actor,
            reason=reason,
            timestamp=self._now(),
        )

    def _existing_non_cancelled_intent(self, intent_id: str) -> ReleaseIntent:
        state = self._store.read_state()
        for intent in state.intents:
            if intent.intent_id == intent_id:
                if self._derived_status(intent, state) == "cancelled":
                    raise GovernanceValidationError(
                        "cancelled intent cannot be modified"
                    )
                return intent
        raise GovernanceValidationError("release intent not found")

    def _derived_status(
        self,
        intent: ReleaseIntent,
        state: Any,
    ) -> str:
        if any(
            event.intent_id == intent.intent_id
            and event.event_type == "intent_cancelled"
            for event in state.audit_events
        ):
            return "cancelled"

        latest_by_role = self._latest_approvals_by_role(
            intent.intent_id,
            state.approvals,
        )
        if any(
            approval.decision == "reject"
            for approval in latest_by_role.values()
        ):
            return "rejected"
        required_roles = self._required_roles(intent.target_scope)
        if all(
            latest_by_role.get(role) is not None
            and latest_by_role[role].decision == "approve"
            for role in required_roles
        ):
            return "approved"
        return intent.status

    def _active_intent(
        self,
        intents: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        active = [
            intent
            for intent in intents
            if intent["derived_status"] != "cancelled"
        ]
        if not active:
            return None
        return max(active, key=lambda item: item["requested_at"])

    def _required_approvals(
        self,
        *,
        target_scope: str,
        intent_id: str | None,
        approvals: list[ReleaseApproval],
    ) -> list[dict[str, Any]]:
        latest_by_role = (
            self._latest_approvals_by_role(intent_id, approvals)
            if intent_id is not None
            else {}
        )
        return [
            self._required_approval_read(role, latest_by_role.get(role))
            for role in self._required_roles(target_scope)
        ]

    def _required_approval_read(
        self,
        role: str,
        approval: ReleaseApproval | None,
    ) -> dict[str, Any]:
        if approval is None:
            return {
                "role": role,
                "status": "missing",
                "latest_decision": None,
            }
        status_by_decision = {
            "approve": "approved",
            "reject": "rejected",
            "request_changes": "changes_requested",
        }
        return {
            "role": role,
            "status": status_by_decision[approval.decision],
            "latest_decision": approval.decision,
            "approval_id": approval.approval_id,
            "signed_by": approval.signed_by,
            "signed_at": approval.signed_at,
        }

    def _latest_approvals_by_role(
        self,
        intent_id: str,
        approvals: list[ReleaseApproval],
    ) -> dict[str, ReleaseApproval]:
        latest: dict[str, ReleaseApproval] = {}
        for approval in approvals:
            if approval.intent_id == intent_id:
                current = latest.get(approval.approver_role)
                if current is None or approval.signed_at >= current.signed_at:
                    latest[approval.approver_role] = approval
        return latest

    def _latest_rollback_plan(
        self,
        intent_id: str | None,
        rollback_plans: list[ReleaseRollbackPlan],
    ) -> dict[str, Any] | None:
        if intent_id is None:
            return None
        matching = [
            plan for plan in rollback_plans if plan.intent_id == intent_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda plan: plan.created_at).to_dict()

    def _dashboard_snapshot(self, dashboard: dict[str, Any]) -> dict[str, Any]:
        summary = self._summary(dashboard)
        literature_run = self._first_run(dashboard, "literature_shadow_harness")
        return {
            "version_chain": self._version_chain(dashboard),
            "release_decision": dashboard.get("release_decision"),
            "rollback_target": dashboard.get("rollback_target"),
            "hard_fail_count": summary.get("hard_fail_count"),
            "literature_isolation_violations": summary.get(
                "literature_isolation_violations"
            ),
            "clinical_rag_ingest_enabled": summary.get(
                "clinical_rag_ingest_enabled"
            ),
            "literature_status": (
                literature_run.get("status") if literature_run is not None else None
            ),
        }

    def _required_roles(self, target_scope: str) -> tuple[str, ...]:
        if target_scope == "feature_flag_candidate":
            return (
                "release_manager",
                "clinical_safety_reviewer",
                "evidence_reviewer",
            )
        return ("release_manager", "clinical_safety_reviewer")

    def _disabled_execution_actions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "execute_release",
                "label": "Execute release",
                "disabled": True,
                "reason": (
                    "Step 12 records governance only; release execution "
                    "requires a later execution-path design."
                ),
            },
            {
                "id": "execute_rollback",
                "label": "Execute rollback",
                "disabled": True,
                "reason": (
                    "Rollback execution requires a later execution-path design."
                ),
            },
        ]

    def _required_run(
        self,
        dashboard: dict[str, Any],
        kind: str,
    ) -> dict[str, Any]:
        run = self._first_run(dashboard, kind)
        if run is None:
            raise GovernanceValidationError(f"{kind} run is required")
        return run

    def _first_run(
        self,
        dashboard: dict[str, Any],
        kind: str,
    ) -> dict[str, Any] | None:
        runs = self._runs_by_kind(dashboard, kind)
        return runs[0] if runs else None

    def _runs_by_kind(
        self,
        dashboard: dict[str, Any],
        kind: str,
    ) -> list[dict[str, Any]]:
        runs = dashboard.get("runs")
        if not isinstance(runs, list):
            return []
        return [
            run
            for run in runs
            if isinstance(run, dict) and run.get("kind") == kind
        ]

    def _summary(self, dashboard: dict[str, Any]) -> dict[str, Any]:
        summary = dashboard.get("summary")
        if not isinstance(summary, dict):
            raise GovernanceValidationError("dashboard summary is required")
        return deepcopy(summary)

    def _version_chain(self, dashboard: dict[str, Any]) -> dict[str, Any]:
        version_chain = dashboard.get("version_chain")
        if not isinstance(version_chain, dict):
            raise GovernanceValidationError("version_chain is required")
        return deepcopy(version_chain)

    def _required_string(
        self,
        payload: dict[str, Any],
        key: str,
        error_message: str,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise GovernanceValidationError(error_message)
        return value

    def _int_value(self, value: Any, *, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise GovernanceValidationError(f"{field_name} must be an integer")
        return value

    def _require_non_empty(self, field_name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise GovernanceValidationError(
                f"{field_name} must be a non-empty string"
            )


__all__ = [
    "GovernanceConflictError",
    "GovernanceValidationError",
    "ReleaseGovernanceService",
]
