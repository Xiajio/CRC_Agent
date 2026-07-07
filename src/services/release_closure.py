from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_closure_store import (
    ReleaseClosureIntegrityError,
    ReleaseClosureStore,
)
from src.contracts.release_closure import (
    CLOSURE_STATUSES,
    ReleaseClosureGate,
    ReleaseClosureGateCheck,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    canonical_closure_payload_hash,
    make_release_closure_id,
    make_release_evidence_package_id,
)


class ReleaseClosureValidationError(ValueError):
    """Raised when a closure request payload is invalid."""


class ReleaseClosureConflictError(ValueError):
    """Raised when closure is blocked by current release state."""


class ReleaseClosureService:
    def __init__(
        self,
        *,
        store: ReleaseClosureStore,
        dashboard_loader: Callable[[], dict[str, Any]],
        governance_loader: Callable[[], dict[str, Any]],
        execution_loader: Callable[[], dict[str, Any]],
        monitoring_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._dashboard_loader = dashboard_loader
        self._governance_loader = governance_loader
        self._execution_loader = execution_loader
        self._monitoring_loader = monitoring_loader
        self._now = now

    def read_closure(self) -> dict[str, object]:
        dashboard = self._dashboard_loader()
        governance = self._governance_loader()
        execution = self._execution_loader()
        monitoring = self._monitoring_loader()
        store_state = self._store.read_state()
        latest_release = self._latest_successful_release(execution)
        latest_rollback = (
            None
            if latest_release is None
            else self._latest_successful_rollback(
                execution,
                str(latest_release.get("intent_id")),
            )
        )
        current_release_execution_id = (
            None
            if latest_release is None
            else str(latest_release.get("execution_id"))
        )
        latest_closure = self._latest_closure_for_release(
            store_state.closures,
            current_release_execution_id,
        )
        latest_package = self._latest_package_for_release(
            store_state.evidence_packages,
            current_release_execution_id,
        )
        integrity = self._integrity_model(
            dashboard=dashboard,
            governance=governance,
            execution=execution,
            monitoring=monitoring,
            store_integrity=store_state.integrity,
        )
        gate = self._derive_gate(
            latest_release=latest_release,
            monitoring=monitoring,
            integrity=integrity,
        )
        status = self._status_from_state(
            latest_release=latest_release,
            latest_closure=latest_closure,
            gate=gate,
        )
        closure_gate = self._gate_for_read_status(gate, status)

        return {
            "status": status,
            "latest_release": self._release_summary(
                latest_release=latest_release,
                latest_rollback=latest_rollback,
            ),
            "closure_gate": closure_gate.to_dict(),
            "latest_closure": (
                None if latest_closure is None else latest_closure.to_dict()
            ),
            "latest_evidence_package": (
                None if latest_package is None else latest_package.to_dict()
            ),
            "closures": [closure.to_dict() for closure in store_state.closures],
            "evidence_packages": [
                package.to_dict() for package in store_state.evidence_packages
            ],
            "integrity": integrity,
            "runtime": {
                "auth": "admin",
                "source": "reports/release_closure",
                "mode": "post_release_closure",
            },
        }

    def record_closure(
        self,
        *,
        intent_id: str,
        release_execution_id: str,
        closure_status: str,
        closed_by: str,
        rationale: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if closure_status not in CLOSURE_STATUSES:
            raise ReleaseClosureValidationError("unknown closure status")

        dashboard = self._dashboard_loader()
        governance = self._governance_loader()
        execution = self._execution_loader()
        monitoring = self._monitoring_loader()
        store_state = self._store.read_state()
        integrity = self._integrity_model(
            dashboard=dashboard,
            governance=governance,
            execution=execution,
            monitoring=monitoring,
            store_integrity=store_state.integrity,
        )
        latest_release = self._latest_successful_release(execution)
        if integrity.get("status") != "verified":
            raise ReleaseClosureConflictError("release closure integrity failed")
        try:
            idempotent_match = self._store.find_closure_by_idempotency_key(
                idempotency_key
            )
        except ReleaseClosureIntegrityError as exc:
            raise ReleaseClosureConflictError(
                "release closure integrity failed"
            ) from exc

        if latest_release is None:
            raise ReleaseClosureConflictError("no successful release execution exists")
        if (
            latest_release.get("intent_id") != intent_id
            or latest_release.get("execution_id") != release_execution_id
        ):
            raise ReleaseClosureConflictError(
                "closure must reference the latest successful release execution"
            )

        rollback = self._latest_successful_rollback(execution, intent_id)
        warning_conflict = self._warning_alert_conflict(
            monitoring=monitoring,
            closure_status=closure_status,
        )
        if warning_conflict is not None:
            raise ReleaseClosureConflictError(warning_conflict)
        if closure_status == "rolled_back":
            if rollback is None:
                raise ReleaseClosureConflictError(
                    "successful rollback is required for rolled_back closure"
                )
        elif rollback is not None:
            raise ReleaseClosureConflictError(
                "accepted closure is blocked after rollback"
            )

        gate = self._derive_gate(
            latest_release=latest_release,
            monitoring=monitoring,
            integrity=integrity,
            closure_status=closure_status,
            rollback=rollback,
        )
        if not gate.allowed:
            raise ReleaseClosureConflictError("; ".join(gate.reasons))

        timestamp = (
            idempotent_match.closure.closed_at
            if idempotent_match is not None
            else self._now()
        )
        closure = self._build_closure_record(
            intent_id=intent_id,
            release_execution_id=release_execution_id,
            rollback_execution_id=(
                None if rollback is None else str(rollback.get("execution_id"))
            ),
            closure_status=closure_status,
            closed_by=closed_by,
            closed_at=timestamp,
            rationale=rationale,
            idempotency_key=idempotency_key,
            dashboard=dashboard,
            governance=governance,
            execution=execution,
            monitoring=monitoring,
        )
        package = self._build_evidence_package(
            closure=closure,
            generated_by=closed_by,
            generated_at=timestamp,
        )
        try:
            self._store.assert_idempotent_closure_matches(closure, package)
        except ReleaseClosureIntegrityError as exc:
            raise ReleaseClosureConflictError(
                "release closure integrity failed"
            ) from exc
        except FileExistsError as exc:
            raise ReleaseClosureConflictError("idempotency payload mismatch") from exc
        if idempotent_match is not None:
            return self.read_closure()
        self._store.write_closure_with_package(closure, package, timestamp=timestamp)
        return self.read_closure()

    def _release_summary(
        self,
        *,
        latest_release: dict[str, Any] | None,
        latest_rollback: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if latest_release is None:
            return None
        return {
            "intent_id": latest_release.get("intent_id"),
            "release_execution_id": latest_release.get("execution_id"),
            "released_at": self._release_timestamp(latest_release),
            "rollback_execution_id": (
                None
                if latest_rollback is None
                else latest_rollback.get("execution_id")
            ),
            "rolled_back_at": (
                None
                if latest_rollback is None
                else self._release_timestamp(latest_rollback)
            ),
        }

    def _integrity_model(
        self,
        *,
        dashboard: dict[str, Any],
        governance: dict[str, Any],
        execution: dict[str, Any],
        monitoring: dict[str, Any],
        store_integrity: dict[str, Any],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        for name, model in (
            ("dashboard", dashboard),
            ("governance", governance),
            ("execution", execution),
            ("monitoring", monitoring),
        ):
            if self._integrity_status(model.get("integrity")) != "verified":
                warnings.append(f"{name} integrity is not verified")
        store_warnings = store_integrity.get("warnings", [])
        if isinstance(store_warnings, list):
            warnings.extend(str(item) for item in store_warnings)
        return {
            "status": "verified" if not warnings else "failed",
            "warnings": warnings,
        }

    def _derive_gate(
        self,
        *,
        latest_release: dict[str, Any] | None,
        monitoring: dict[str, Any],
        integrity: dict[str, Any],
        closure_status: str | None = None,
        rollback: dict[str, Any] | None = None,
    ) -> ReleaseClosureGate:
        reasons: list[str] = []
        checks: list[ReleaseClosureGateCheck] = []

        release_reason = (
            "successful release execution is available"
            if latest_release is not None
            else "no successful release execution exists"
        )
        release_status = "pass" if latest_release is not None else "fail"
        checks.append(
            ReleaseClosureGateCheck(
                name="latest_successful_release",
                status=release_status,
                reason=release_reason,
            )
        )
        if latest_release is None:
            reasons.append("no successful release execution exists")

        missing_required_checks = self._missing_required_checks(monitoring)
        checks.append(
            ReleaseClosureGateCheck(
                name="required_monitoring_checks_complete",
                status="pass" if not missing_required_checks else "fail",
                reason=(
                    "All required monitoring checks are present."
                    if not missing_required_checks
                    else "required monitoring checks are missing"
                ),
            )
        )
        if missing_required_checks:
            reasons.append("required monitoring checks are missing")

        has_active_critical_alerts = self._has_active_critical_alerts(monitoring)
        checks.append(
            ReleaseClosureGateCheck(
                name="active_critical_monitoring_alerts",
                status="pass" if not has_active_critical_alerts else "fail",
                reason=(
                    "No active critical monitoring alerts remain."
                    if not has_active_critical_alerts
                    else "active critical monitoring alerts exist"
                ),
            )
        )
        if has_active_critical_alerts:
            reasons.append("active critical monitoring alerts exist")

        has_rollback_candidate = self._rollback_candidate_id(monitoring) is not None
        candidate_blocks = has_rollback_candidate and not (
            closure_status == "rolled_back" and rollback is not None
        )
        checks.append(
            ReleaseClosureGateCheck(
                name="rollback_trigger_candidate",
                status="pass" if not candidate_blocks else "fail",
                reason=(
                    "No rollback trigger candidate is active."
                    if not candidate_blocks
                    else "rollback trigger candidate exists"
                ),
            )
        )
        if candidate_blocks:
            reasons.append("rollback trigger candidate exists")

        integrity_failed = integrity.get("status") != "verified"
        checks.append(
            ReleaseClosureGateCheck(
                name="closure_integrity",
                status="pass" if not integrity_failed else "fail",
                reason=(
                    "Closure service inputs and store integrity are verified."
                    if not integrity_failed
                    else "release closure integrity failed"
                ),
            )
        )
        if integrity_failed:
            reasons.append("release closure integrity failed")

        status = "ready_to_close" if not reasons else "blocked"
        if latest_release is None and reasons == ["no successful release execution exists"]:
            status = "idle"
        return ReleaseClosureGate(
            allowed=not reasons,
            status=status,
            reasons=reasons,
            checks=checks,
        )

    def _build_closure_record(
        self,
        *,
        intent_id: str,
        release_execution_id: str,
        rollback_execution_id: str | None,
        closure_status: str,
        closed_by: str,
        closed_at: str,
        rationale: str,
        idempotency_key: str,
        dashboard: dict[str, Any],
        governance: dict[str, Any],
        execution: dict[str, Any],
        monitoring: dict[str, Any],
    ) -> ReleaseClosureRecord:
        closure_id = make_release_closure_id(release_execution_id, idempotency_key)
        evidence_package_id = make_release_evidence_package_id(closure_id)
        active_alerts = self._active_alert_ids(monitoring)
        acknowledged_alerts = self._acknowledged_alert_ids(monitoring)
        unresolved_alerts = [
            alert_id for alert_id in active_alerts if alert_id not in acknowledged_alerts
        ]
        return ReleaseClosureRecord(
            closure_id=closure_id,
            intent_id=intent_id,
            release_execution_id=release_execution_id,
            rollback_execution_id=rollback_execution_id,
            closure_status=closure_status,
            closed_by=closed_by,
            closed_at=closed_at,
            rationale=rationale,
            monitoring_snapshot_hash=self._snapshot_hash(monitoring),
            dashboard_snapshot_hash=self._snapshot_hash(dashboard),
            governance_snapshot_hash=self._snapshot_hash(governance),
            execution_snapshot_hash=self._snapshot_hash(execution),
            required_check_ids=self._required_check_ids(monitoring),
            acknowledged_alert_ids=acknowledged_alerts,
            unresolved_alert_ids=unresolved_alerts,
            rollback_trigger_candidate_id=self._rollback_candidate_id(monitoring),
            evidence_package_id=evidence_package_id,
            idempotency_key=idempotency_key,
        )

    def _build_evidence_package(
        self,
        *,
        closure: ReleaseClosureRecord,
        generated_by: str,
        generated_at: str,
    ) -> ReleaseEvidencePackage:
        summary_map = {
            "accepted": "Release accepted and closure recorded.",
            "accepted_with_observations": "Release accepted with observations and closure recorded.",
            "rolled_back": "Release rollback completed and closure recorded.",
        }
        return ReleaseEvidencePackage(
            package_id=closure.evidence_package_id,
            closure_id=closure.closure_id,
            intent_id=closure.intent_id,
            release_execution_id=closure.release_execution_id,
            rollback_execution_id=closure.rollback_execution_id,
            generated_by=generated_by,
            generated_at=generated_at,
            closure_status=closure.closure_status,
            summary=summary_map[closure.closure_status],
            source_refs=[
                "GET /api/admin/release-dashboard",
                "GET /api/admin/release-governance",
                "GET /api/admin/release-execution",
                "GET /api/admin/release-monitoring",
            ],
            artifact_refs=[
                f"reports/release_closure/closures/{closure.closure_id}.json",
            ],
            snapshot_hashes={
                "dashboard": closure.dashboard_snapshot_hash,
                "governance": closure.governance_snapshot_hash,
                "execution": closure.execution_snapshot_hash,
                "monitoring": closure.monitoring_snapshot_hash,
            },
        )

    def _snapshot_hash(self, model: dict[str, Any]) -> str:
        return canonical_closure_payload_hash(deepcopy(model))

    def _status_from_state(
        self,
        *,
        latest_release: dict[str, Any] | None,
        latest_closure: ReleaseClosureRecord | None,
        gate: ReleaseClosureGate,
    ) -> str:
        if latest_closure is not None:
            if latest_closure.closure_status == "rolled_back":
                return "rolled_back_closed"
            return "closed"
        if latest_release is None:
            return "idle"
        return gate.status

    def _gate_for_read_status(
        self,
        gate: ReleaseClosureGate,
        status: str,
    ) -> ReleaseClosureGate:
        if status not in {"closed", "rolled_back_closed"}:
            return gate
        return ReleaseClosureGate(
            allowed=False,
            status=status,
            reasons=list(gate.reasons),
            checks=list(gate.checks),
        )

    def _latest_successful_release(
        self,
        execution: dict[str, Any],
    ) -> dict[str, Any] | None:
        results = execution.get("results")
        if not isinstance(results, list):
            return None
        releases = [
            result
            for result in results
            if isinstance(result, dict)
            and result.get("action") == "release"
            and result.get("status") == "succeeded"
        ]
        if not releases:
            return None
        return deepcopy(max(releases, key=self._result_sort_key))

    def _latest_successful_rollback(
        self,
        execution: dict[str, Any],
        intent_id: str,
    ) -> dict[str, Any] | None:
        results = execution.get("results")
        if not isinstance(results, list):
            return None
        rollbacks = [
            result
            for result in results
            if isinstance(result, dict)
            and result.get("intent_id") == intent_id
            and result.get("action") == "rollback"
            and result.get("status") == "succeeded"
        ]
        if not rollbacks:
            return None
        return deepcopy(max(rollbacks, key=self._result_sort_key))

    def _latest_closure_for_release(
        self,
        closures: list[ReleaseClosureRecord],
        release_execution_id: str | None,
    ) -> ReleaseClosureRecord | None:
        if not closures or release_execution_id is None:
            return None
        matching = [
            closure
            for closure in closures
            if closure.release_execution_id == release_execution_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda item: item.closed_at)

    def _latest_package_for_release(
        self,
        packages: list[ReleaseEvidencePackage],
        release_execution_id: str | None,
    ) -> ReleaseEvidencePackage | None:
        if not packages or release_execution_id is None:
            return None
        matching = [
            package
            for package in packages
            if package.release_execution_id == release_execution_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda item: item.generated_at)

    def _result_sort_key(self, result: dict[str, Any]) -> str:
        return str(
            result.get("finished_at")
            or result.get("started_at")
            or result.get("updated_at")
            or ""
        )

    def _release_timestamp(self, release: dict[str, Any]) -> str:
        return self._result_sort_key(release) or "release_closure_source_state"

    def _integrity_status(self, integrity: Any) -> str | None:
        return integrity.get("status") if isinstance(integrity, dict) else None

    def _active_intent(self, governance: dict[str, Any]) -> dict[str, Any] | None:
        active_intent = governance.get("active_intent")
        return active_intent if isinstance(active_intent, dict) else None

    def _missing_required_checks(self, monitoring: dict[str, Any]) -> list[dict[str, Any]]:
        required_checks = monitoring.get("required_checks")
        if not isinstance(required_checks, list):
            return [{"check_type": "required_checks", "status": "missing"}]
        return [
            check
            for check in required_checks
            if isinstance(check, dict) and check.get("status") == "missing"
        ]

    def _has_active_critical_alerts(self, monitoring: dict[str, Any]) -> bool:
        alerts = monitoring.get("alerts")
        if not isinstance(alerts, list):
            return False
        return any(
            isinstance(alert, dict)
            and alert.get("severity") == "critical"
            and alert.get("status") == "active"
            for alert in alerts
        )

    def _rollback_candidate_id(self, monitoring: dict[str, Any]) -> str | None:
        candidate = monitoring.get("rollback_trigger_candidate")
        if not isinstance(candidate, dict):
            return None
        candidate_id = candidate.get("candidate_id")
        return candidate_id if isinstance(candidate_id, str) and candidate_id else None

    def _required_check_ids(self, monitoring: dict[str, Any]) -> list[str]:
        required_checks = monitoring.get("required_checks")
        if not isinstance(required_checks, list):
            return []
        check_ids: list[str] = []
        for item in required_checks:
            if not isinstance(item, dict):
                continue
            check_id = item.get("latest_check_id")
            if isinstance(check_id, str) and check_id:
                check_ids.append(check_id)
        return check_ids

    def _active_alert_ids(self, monitoring: dict[str, Any]) -> list[str]:
        alerts = monitoring.get("alerts")
        if not isinstance(alerts, list):
            return []
        alert_ids: list[str] = []
        for item in alerts:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "active":
                continue
            alert_id = item.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                alert_ids.append(alert_id)
        return alert_ids

    def _warning_alert_conflict(
        self,
        *,
        monitoring: dict[str, Any],
        closure_status: str,
    ) -> str | None:
        active_warning_alert_ids = self._active_warning_alert_ids(monitoring)
        if not active_warning_alert_ids:
            return None
        acknowledged_alert_ids = set(self._acknowledged_alert_ids(monitoring))
        unresolved_warning_alert_ids = [
            alert_id
            for alert_id in active_warning_alert_ids
            if alert_id not in acknowledged_alert_ids
        ]
        if closure_status == "accepted":
            return "active warning monitoring alerts exist"
        if (
            closure_status == "accepted_with_observations"
            and unresolved_warning_alert_ids
        ):
            return "accepted_with_observations requires acknowledged active warning alerts"
        return None

    def _active_warning_alert_ids(self, monitoring: dict[str, Any]) -> list[str]:
        alerts = monitoring.get("alerts")
        if not isinstance(alerts, list):
            return []
        alert_ids: list[str] = []
        for item in alerts:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "active" or item.get("severity") != "warning":
                continue
            alert_id = item.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                alert_ids.append(alert_id)
        return alert_ids

    def _acknowledged_alert_ids(self, monitoring: dict[str, Any]) -> list[str]:
        acknowledgements = monitoring.get("acknowledgements")
        if not isinstance(acknowledgements, list):
            return []
        alert_ids: list[str] = []
        for item in acknowledgements:
            if not isinstance(item, dict):
                continue
            alert_id = item.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                alert_ids.append(alert_id)
        return alert_ids

__all__ = [
    "ReleaseClosureConflictError",
    "ReleaseClosureService",
    "ReleaseClosureValidationError",
]
