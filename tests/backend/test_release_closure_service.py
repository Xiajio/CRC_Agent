from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.api.services.release_closure_store import ReleaseClosureStore
from src.services.release_closure import (
    ReleaseClosureConflictError,
    ReleaseClosureService,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
ROLLBACK_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_rollback_10ca7caa"


def dashboard() -> dict[str, Any]:
    return {"summary": {"hard_fail_count": 0}, "integrity": {"status": "verified", "warnings": []}}


def governance() -> dict[str, Any]:
    return {
        "active_intent": {"intent_id": INTENT_ID, "rollback_target": "agent_policy_20260624_0"},
        "integrity": {"status": "verified", "warnings": []},
    }


def execution(*, rollback: bool = False) -> dict[str, Any]:
    results = [
        {
            "execution_id": RELEASE_EXECUTION_ID,
            "intent_id": INTENT_ID,
            "action": "release",
            "status": "succeeded",
            "finished_at": "2026-07-07T09:00:00+08:00",
        }
    ]
    if rollback:
        results.append(
            {
                "execution_id": ROLLBACK_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "rollback",
                "status": "succeeded",
                "finished_at": "2026-07-07T09:30:00+08:00",
            }
        )
    return {"results": results, "integrity": {"status": "verified", "warnings": []}}


def monitoring(
    *,
    missing: bool = False,
    critical: bool = False,
    warning_acknowledged: bool = False,
    rollback_candidate: bool = False,
) -> dict[str, Any]:
    required_checks = [
        {"check_type": "execution_integrity", "status": "pass", "latest_check_id": "check-execution", "reason": "ok"},
        {"check_type": "governance_drift", "status": "pass", "latest_check_id": "check-governance", "reason": "ok"},
    ]
    if missing:
        required_checks[1] = {"check_type": "governance_drift", "status": "missing", "latest_check_id": None, "reason": "missing"}
    alerts = []
    acknowledgements = []
    if critical:
        alerts.append(
            {
                "alert_id": "release_monitor_alert_critical",
                "intent_id": INTENT_ID,
                "execution_id": RELEASE_EXECUTION_ID,
                "severity": "critical",
                "status": "active",
                "recommended_action": "execute_step13_rollback",
            }
        )
    if warning_acknowledged:
        alerts.append(
            {
                "alert_id": "release_monitor_alert_warning",
                "intent_id": INTENT_ID,
                "execution_id": RELEASE_EXECUTION_ID,
                "severity": "warning",
                "status": "acknowledged",
                "recommended_action": "investigate",
            }
        )
        acknowledgements.append({"alert_id": "release_monitor_alert_warning", "disposition": "accepted_risk"})
    return {
        "status": "monitoring",
        "required_checks": required_checks,
        "checks": [{"check_id": item["latest_check_id"]} for item in required_checks if item["latest_check_id"]],
        "alerts": alerts,
        "acknowledgements": acknowledgements,
        "rollback_trigger_candidate": {"candidate_id": "candidate-1"} if rollback_candidate else None,
        "integrity": {"status": "verified", "warnings": []},
    }


def service(
    tmp_path: Path,
    *,
    execution_model: dict[str, Any] | None = None,
    monitoring_model: dict[str, Any] | None = None,
) -> ReleaseClosureService:
    return ReleaseClosureService(
        store=ReleaseClosureStore(tmp_path),
        dashboard_loader=dashboard,
        governance_loader=governance,
        execution_loader=lambda: execution_model if execution_model is not None else execution(),
        monitoring_loader=lambda: monitoring_model if monitoring_model is not None else monitoring(),
        now=lambda: "2026-07-07T10:00:00+08:00",
    )


def test_closure_is_ready_when_required_checks_pass(tmp_path: Path) -> None:
    model = service(tmp_path).read_closure()

    assert model["status"] == "ready_to_close"
    assert model["closure_gate"]["allowed"] is True


def test_closure_blocked_when_required_checks_missing(tmp_path: Path) -> None:
    model = service(tmp_path, monitoring_model=monitoring(missing=True)).read_closure()

    assert model["status"] == "blocked"
    assert "required monitoring checks are missing" in model["closure_gate"]["reasons"]


def test_record_accepted_closure_writes_package(tmp_path: Path) -> None:
    app = service(tmp_path)

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted",
        closed_by="release_manager",
        rationale="Required checks passed.",
        idempotency_key="close-1",
    )

    assert model["status"] == "closed"
    assert model["latest_closure"]["closure_status"] == "accepted"
    assert model["latest_evidence_package"]["closure_id"] == model["latest_closure"]["closure_id"]


def test_record_closure_rejects_active_critical_alert(tmp_path: Path) -> None:
    app = service(tmp_path, monitoring_model=monitoring(critical=True))

    with pytest.raises(ReleaseClosureConflictError, match="active critical monitoring alerts"):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-1",
        )


def test_rolled_back_closure_requires_successful_rollback(tmp_path: Path) -> None:
    app = service(tmp_path, execution_model=execution(rollback=True))

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="rolled_back",
        closed_by="release_manager",
        rationale="Rollback completed after monitoring trigger.",
        idempotency_key="close-rollback-1",
    )

    assert model["status"] == "rolled_back_closed"
    assert model["latest_closure"]["rollback_execution_id"] == ROLLBACK_EXECUTION_ID
