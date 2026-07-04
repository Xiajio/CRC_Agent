from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.contracts.release_monitoring import (
    MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS,
    MONITORING_CHECK_STATUSES,
    MONITORING_CHECK_TYPES,
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringAlert,
    ReleaseMonitoringCheck,
    ReleaseRollbackTriggerCandidate,
    make_monitoring_acknowledgement_id,
    make_monitoring_alert_id,
    make_monitoring_check_id,
    make_rollback_trigger_candidate_id,
)


class ReleaseMonitoringValidationError(ValueError):
    """Raised when a monitoring mutation payload is invalid."""


class ReleaseMonitoringConflictError(ValueError):
    """Raised when monitoring mutation conflicts with release state."""


REQUIRED_CHECK_TYPES = (
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
)


class ReleaseMonitoringService:
    def __init__(
        self,
        *,
        store: ReleaseMonitoringStore,
        execution_loader: Callable[[], dict[str, Any]],
        governance_loader: Callable[[], dict[str, Any]],
        dashboard_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._execution_loader = execution_loader
        self._governance_loader = governance_loader
        self._dashboard_loader = dashboard_loader
        self._now = now

    def read_monitoring(self) -> dict[str, Any]:
        return self._build_read_model()

    def record_check(
        self,
        *,
        intent_id: str,
        execution_id: str,
        check_type: str,
        status: str,
        observed_by: str,
        summary: str,
        evidence_refs: list[str],
        metrics: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        if check_type not in MONITORING_CHECK_TYPES:
            raise ReleaseMonitoringValidationError("unknown monitoring check_type")
        if status not in MONITORING_CHECK_STATUSES:
            raise ReleaseMonitoringValidationError("unknown monitoring check status")

        execution = self._execution_loader()
        latest_release = self._latest_successful_release(execution)
        if latest_release is None:
            raise ReleaseMonitoringConflictError(
                "no successful release execution exists"
            )
        if (
            latest_release.get("intent_id") != intent_id
            or latest_release.get("execution_id") != execution_id
        ):
            raise ReleaseMonitoringConflictError(
                "check must reference the latest successful release execution"
            )
        if (
            check_type != "manual_operator_note"
            and self._successful_rollback_exists(execution, intent_id)
        ):
            raise ReleaseMonitoringConflictError(
                "non-manual checks cannot be recorded after rollback"
            )

        idempotent_match = self._store.find_check_by_idempotency_key(
            check_type,
            idempotency_key,
        )
        timestamp = (
            idempotent_match.check.observed_at
            if idempotent_match is not None
            else self._now()
        )
        check = ReleaseMonitoringCheck(
            check_id=make_monitoring_check_id(
                execution_id,
                check_type,
                idempotency_key,
            ),
            intent_id=intent_id,
            execution_id=execution_id,
            check_type=check_type,
            status=status,
            observed_by=observed_by,
            observed_at=timestamp,
            summary=summary,
            evidence_refs=evidence_refs,
            metrics=metrics,
            idempotency_key=idempotency_key,
        )

        if idempotent_match is not None:
            self._store.assert_idempotent_check_matches(check)
            return self.read_monitoring()

        self._store.write_check(check, timestamp=timestamp)
        return self.read_monitoring()

    def acknowledge_alert(
        self,
        *,
        alert_id: str,
        acknowledged_by: str,
        disposition: str,
        reason: str,
    ) -> dict[str, Any]:
        if disposition not in MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS:
            raise ReleaseMonitoringValidationError(
                "unknown monitoring acknowledgement disposition"
            )

        current_model = self._build_read_model()
        alert = next(
            (item for item in current_model["alerts"] if item["alert_id"] == alert_id),
            None,
        )
        if alert is None:
            raise ReleaseMonitoringConflictError(
                "alert does not exist in current monitoring model"
            )

        timestamp = self._now()
        acknowledgement = ReleaseMonitoringAcknowledgement(
            acknowledgement_id=make_monitoring_acknowledgement_id(
                alert_id,
                f"{timestamp}:{acknowledged_by}:{disposition}:{reason}",
            ),
            alert_id=alert_id,
            intent_id=alert["intent_id"],
            execution_id=alert["execution_id"],
            acknowledged_by=acknowledged_by,
            acknowledged_at=timestamp,
            disposition=disposition,
            reason=reason,
        )
        self._store.write_acknowledgement(acknowledgement, timestamp=timestamp)
        return self.read_monitoring()

    def _build_read_model(self) -> dict[str, Any]:
        execution = self._execution_loader()
        governance = self._governance_loader()
        dashboard = self._dashboard_loader()
        state = self._store.read_state()
        latest_release = self._latest_successful_release(execution)
        rolled_back = (
            latest_release is not None
            and self._successful_rollback_exists(
                execution,
                str(latest_release.get("intent_id")),
            )
        )
        status = (
            "idle"
            if latest_release is None
            else "rolled_back"
            if rolled_back
            else "monitoring"
        )
        checks = [check.to_dict() for check in state.checks]
        acknowledgements = [
            acknowledgement.to_dict()
            for acknowledgement in state.acknowledgements
        ]
        required_checks = (
            self._required_checks(latest_release, checks)
            if status == "monitoring"
            else []
        )
        alerts = self._derive_alerts(
            latest_release=latest_release,
            status=status,
            execution=execution,
            governance=governance,
            dashboard=dashboard,
            checks=checks,
            required_checks=required_checks,
            acknowledgements=acknowledgements,
            monitoring_integrity=state.integrity,
        )
        rollback_trigger_candidate = (
            None
            if status != "monitoring"
            else self._derive_rollback_candidate(
                latest_release=latest_release,
                governance=governance,
                alerts=alerts,
                acknowledgements=acknowledgements,
            )
        )
        return {
            "status": status,
            "latest_release": deepcopy(latest_release),
            "required_checks": required_checks,
            "checks": checks,
            "alerts": alerts,
            "rollback_trigger_candidate": rollback_trigger_candidate,
            "acknowledgements": acknowledgements,
            "integrity": deepcopy(state.integrity),
            "runtime": {
                "auth": "admin",
                "source": "reports/release_monitoring",
                "mode": "post_release_monitoring",
            },
        }

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
        return deepcopy(
            max(
                releases,
                key=lambda result: str(
                    result.get("finished_at") or result.get("started_at") or ""
                ),
            )
        )

    def _successful_rollback_exists(
        self,
        execution: dict[str, Any],
        intent_id: str,
    ) -> bool:
        results = execution.get("results")
        if not isinstance(results, list):
            return False
        return any(
            isinstance(result, dict)
            and result.get("intent_id") == intent_id
            and result.get("action") == "rollback"
            and result.get("status") == "succeeded"
            for result in results
        )

    def _required_checks(
        self,
        latest_release: dict[str, Any] | None,
        checks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if latest_release is None:
            return []
        execution_id = latest_release.get("execution_id")
        summaries: list[dict[str, Any]] = []
        for check_type in REQUIRED_CHECK_TYPES:
            matching_checks = [
                check
                for check in checks
                if check.get("execution_id") == execution_id
                and check.get("check_type") == check_type
            ]
            latest_check = (
                max(matching_checks, key=lambda check: str(check.get("observed_at")))
                if matching_checks
                else None
            )
            summaries.append(
                {
                    "check_type": check_type,
                    "status": (
                        str(latest_check.get("status"))
                        if latest_check is not None
                        else "missing"
                    ),
                    "check_id": (
                        latest_check.get("check_id")
                        if latest_check is not None
                        else None
                    ),
                    "observed_at": (
                        latest_check.get("observed_at")
                        if latest_check is not None
                        else None
                    ),
                }
            )
        return summaries

    def _derive_alerts(
        self,
        *,
        latest_release: dict[str, Any] | None,
        status: str,
        execution: dict[str, Any],
        governance: dict[str, Any],
        dashboard: dict[str, Any],
        checks: list[dict[str, Any]],
        required_checks: list[dict[str, Any]],
        acknowledgements: list[dict[str, Any]],
        monitoring_integrity: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if status == "idle" and not self._flag_enabled_without_matching_release(
            execution,
            latest_release,
        ):
            return []
        alerts: list[dict[str, Any]] = []
        if latest_release is not None and status == "monitoring":
            for item in required_checks:
                if item["status"] == "missing":
                    alerts.append(
                        self._alert(
                            latest_release,
                            severity="warning",
                            category="missing_required_check",
                            discriminator=str(item["check_type"]),
                            message=(
                                f"Required post-release check is missing: "
                                f"{item['check_type']}."
                            ),
                            source_check_ids=[],
                            recommended_action="investigate",
                            acknowledgements=acknowledgements,
                        )
                    )

            for check in checks:
                if (
                    check.get("execution_id") == latest_release.get("execution_id")
                    and check.get("status") == "fail"
                ):
                    check_alert = self._failed_check_alert(
                        latest_release,
                        check,
                        acknowledgements,
                    )
                    if check_alert is not None:
                        alerts.append(check_alert)

        if self._integrity_status(execution.get("integrity")) != "verified":
            release = latest_release or self._release_from_flag_state(execution)
            if release is not None:
                alerts.append(
                    self._alert(
                        release,
                        severity="critical",
                        category="execution_integrity_failed",
                        discriminator="execution_integrity",
                        message="Release execution integrity is not verified.",
                        source_check_ids=[],
                        recommended_action="execute_step13_rollback",
                        acknowledgements=acknowledgements,
                    )
                )

        if self._integrity_status(monitoring_integrity) != "verified":
            release = latest_release or self._release_from_flag_state(execution)
            if release is not None:
                alerts.append(
                    self._alert(
                        release,
                        severity="critical",
                        category="execution_integrity_failed",
                        discriminator="monitoring_integrity",
                        message="Release monitoring integrity is not verified.",
                        source_check_ids=[],
                        recommended_action="investigate",
                        acknowledgements=acknowledgements,
                    )
                )

        if latest_release is not None:
            alerts.extend(
                self._dashboard_drift_alerts(
                    latest_release,
                    governance,
                    dashboard,
                    acknowledgements,
                )
            )

        if self._flag_enabled_without_matching_release(execution, latest_release):
            release = latest_release or self._release_from_flag_state(execution)
            if release is not None:
                alerts.append(
                    self._alert(
                        release,
                        severity="critical",
                        category="feature_flag_state_mismatch",
                        discriminator="enabled_without_matching_release",
                        message=(
                            "Current feature flag state is enabled without a "
                            "matching successful release execution."
                        ),
                        source_check_ids=[],
                        recommended_action="prepare_rollback",
                        acknowledgements=acknowledgements,
                    )
                )
        return sorted(
            self._dedupe_alerts(alerts),
            key=lambda alert: (alert["severity"] != "critical", alert["alert_id"]),
        )

    def _failed_check_alert(
        self,
        latest_release: dict[str, Any],
        check: dict[str, Any],
        acknowledgements: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        check_type = str(check.get("check_type"))
        if check_type == "execution_integrity":
            return self._alert(
                latest_release,
                severity="critical",
                category="execution_integrity_failed",
                discriminator=check_type,
                message="Release execution integrity check failed.",
                source_check_ids=[str(check.get("check_id"))],
                recommended_action="execute_step13_rollback",
                acknowledgements=acknowledgements,
                created_at=str(check.get("observed_at") or self._now()),
            )
        if check_type == "governance_drift":
            return self._alert(
                latest_release,
                severity="warning",
                category="governance_drift",
                discriminator=check_type,
                message="Release governance drift check failed.",
                source_check_ids=[str(check.get("check_id"))],
                recommended_action="investigate",
                acknowledgements=acknowledgements,
                created_at=str(check.get("observed_at") or self._now()),
            )
        if check_type in {"p0_harness_replay", "literature_isolation"}:
            recommended_action = "execute_step13_rollback"
        elif check_type == "doctor_review_smoke":
            recommended_action = "prepare_rollback"
        else:
            recommended_action = "investigate"
        severity = (
            "critical"
            if check_type
            in {"p0_harness_replay", "literature_isolation", "doctor_review_smoke"}
            else "warning"
        )
        return self._alert(
            latest_release,
            severity=severity,
            category="post_release_check_failed",
            discriminator=check_type,
            message=f"Post-release check failed: {check_type}.",
            source_check_ids=[str(check.get("check_id"))],
            recommended_action=recommended_action,
            acknowledgements=acknowledgements,
            created_at=str(check.get("observed_at") or self._now()),
        )

    def _derive_rollback_candidate(
        self,
        *,
        latest_release: dict[str, Any] | None,
        governance: dict[str, Any],
        alerts: list[dict[str, Any]],
        acknowledgements: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if latest_release is None:
            return None
        rollback_plan = self._rollback_plan(governance)
        if rollback_plan is None or rollback_plan.get("status") != "accepted":
            return None
        latest_acknowledgement = self._latest_acknowledgement_by_alert(
            acknowledgements
        )
        trigger_alert_ids = [
            alert["alert_id"]
            for alert in alerts
            if alert.get("severity") == "critical"
            and alert.get("recommended_action") == "execute_step13_rollback"
            and latest_acknowledgement.get(alert["alert_id"], {}).get("disposition")
            != "false_positive"
        ]
        if not trigger_alert_ids:
            return None
        candidate = ReleaseRollbackTriggerCandidate(
            candidate_id=make_rollback_trigger_candidate_id(
                str(latest_release["execution_id"]),
                trigger_alert_ids,
            ),
            intent_id=str(latest_release["intent_id"]),
            execution_id=str(latest_release["execution_id"]),
            source_alert_ids=trigger_alert_ids,
            recommended_action="execute_step13_rollback",
            rollback_plan_id=str(rollback_plan["rollback_plan_id"]),
            rollback_target=str(rollback_plan["rollback_target"]),
            reason=(
                "A critical post-release alert requires Step13 rollback while "
                "the release remains active."
            ),
            created_at=self._now(),
        )
        return candidate.to_dict()

    def _latest_acknowledgement_by_alert(
        self,
        acknowledgements: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for acknowledgement in sorted(
            acknowledgements,
            key=lambda item: str(item.get("acknowledged_at")),
        ):
            alert_id = acknowledgement.get("alert_id")
            if isinstance(alert_id, str):
                latest[alert_id] = acknowledgement
        return latest

    def _alert_status(
        self,
        alert_id: str,
        acknowledgements: list[dict[str, Any]],
    ) -> str:
        return (
            "acknowledged"
            if alert_id in self._latest_acknowledgement_by_alert(acknowledgements)
            else "active"
        )

    def _dashboard_drift_alerts(
        self,
        latest_release: dict[str, Any],
        governance: dict[str, Any],
        dashboard: dict[str, Any],
        acknowledgements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        active_intent = self._active_intent(governance)
        if active_intent is None:
            return [
                self._alert(
                    latest_release,
                    severity="critical",
                    category="governance_drift",
                    discriminator="active_intent",
                    message="Active release governance intent is missing.",
                    source_check_ids=[],
                    recommended_action="prepare_rollback",
                    acknowledgements=acknowledgements,
                )
            ]

        drift_reasons: list[str] = []
        if active_intent.get("intent_id") != latest_release.get("intent_id"):
            drift_reasons.append("active intent does not match latest release")
        if dashboard.get("release_decision") != active_intent.get(
            "release_decision_snapshot"
        ):
            drift_reasons.append("dashboard release decision drifted")
        if dashboard.get("rollback_target") != active_intent.get("rollback_target"):
            drift_reasons.append("dashboard rollback_target drifted")
        if dashboard.get("version_chain") != active_intent.get("version_chain"):
            drift_reasons.append("dashboard version_chain drifted")
        summary = dashboard.get("summary")
        if not isinstance(summary, dict) or summary.get("hard_fail_count") != 0:
            drift_reasons.append("hard_fail_count is not zero")
        literature_run = self._first_run(dashboard, "literature_shadow_harness")
        if literature_run is None or literature_run.get("status") != "shadow_only":
            drift_reasons.append("literature status is not shadow_only")

        if self._integrity_status(governance.get("integrity")) != "verified":
            drift_reasons.append("governance integrity is not verified")
        if not drift_reasons:
            return []
        return [
            self._alert(
                latest_release,
                severity="warning",
                category="governance_drift",
                discriminator=";".join(sorted(drift_reasons)),
                message="; ".join(drift_reasons),
                source_check_ids=[],
                recommended_action="investigate",
                acknowledgements=acknowledgements,
            )
        ]

    def _alert(
        self,
        latest_release: dict[str, Any],
        *,
        severity: str,
        category: str,
        discriminator: str,
        message: str,
        source_check_ids: list[str],
        recommended_action: str,
        acknowledgements: list[dict[str, Any]],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        alert_id = make_monitoring_alert_id(
            str(latest_release["execution_id"]),
            category,
            discriminator,
        )
        alert = ReleaseMonitoringAlert(
            alert_id=alert_id,
            intent_id=str(latest_release["intent_id"]),
            execution_id=str(latest_release["execution_id"]),
            severity=severity,
            category=category,
            status=self._alert_status(alert_id, acknowledgements),
            message=message,
            source_check_ids=source_check_ids,
            recommended_action=recommended_action,
            created_at=created_at or self._now(),
        )
        return alert.to_dict()

    def _dedupe_alerts(self, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for alert in alerts:
            deduped[alert["alert_id"]] = alert
        return list(deduped.values())

    def _flag_enabled_without_matching_release(
        self,
        execution: dict[str, Any],
        latest_release: dict[str, Any] | None,
    ) -> bool:
        flag_state = execution.get("feature_flag_state")
        if not isinstance(flag_state, dict) or flag_state.get("enabled") is not True:
            return False
        if latest_release is None:
            return True
        return not (
            flag_state.get("source_intent_id") == latest_release.get("intent_id")
            and flag_state.get("source_execution_id")
            == latest_release.get("execution_id")
        )

    def _release_from_flag_state(
        self,
        execution: dict[str, Any],
    ) -> dict[str, Any] | None:
        flag_state = execution.get("feature_flag_state")
        if not isinstance(flag_state, dict):
            return None
        intent_id = flag_state.get("source_intent_id")
        execution_id = flag_state.get("source_execution_id")
        if (
            isinstance(intent_id, str)
            and intent_id.strip()
            and isinstance(execution_id, str)
            and execution_id.strip()
        ):
            return {"intent_id": intent_id, "execution_id": execution_id}
        if flag_state.get("enabled") is not True:
            return None
        return {
            "intent_id": (
                intent_id
                if isinstance(intent_id, str) and intent_id.strip()
                else "local_feature_flag_state"
            ),
            "execution_id": (
                execution_id
                if isinstance(execution_id, str) and execution_id.strip()
                else "local_feature_flag_state_unmatched"
            ),
        }

    def _active_intent(self, governance: dict[str, Any]) -> dict[str, Any] | None:
        active_intent = governance.get("active_intent")
        return active_intent if isinstance(active_intent, dict) else None

    def _rollback_plan(self, governance: dict[str, Any]) -> dict[str, Any] | None:
        rollback_plan = governance.get("rollback_plan")
        return rollback_plan if isinstance(rollback_plan, dict) else None

    def _integrity_status(self, integrity: Any) -> str | None:
        return integrity.get("status") if isinstance(integrity, dict) else None

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


__all__ = [
    "REQUIRED_CHECK_TYPES",
    "ReleaseMonitoringConflictError",
    "ReleaseMonitoringService",
    "ReleaseMonitoringValidationError",
]
