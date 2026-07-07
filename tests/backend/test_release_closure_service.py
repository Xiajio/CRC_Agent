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
RELEASE_EXECUTION_ID = (
    "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
)
LATER_RELEASE_EXECUTION_ID = (
    "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_9f61d7ce"
)
ROLLBACK_EXECUTION_ID = (
    "release_exec_release_intent_release_safety_20260629_001_6da729a0_rollback_10ca7caa"
)


def dashboard(*, integrity_status: str = "verified") -> dict[str, Any]:
    return {
        "summary": {"hard_fail_count": 0},
        "integrity": {"status": integrity_status, "warnings": []},
    }


def governance(*, integrity_status: str = "verified") -> dict[str, Any]:
    return {
        "active_intent": {
            "intent_id": INTENT_ID,
            "rollback_target": "agent_policy_20260624_0",
        },
        "integrity": {"status": integrity_status, "warnings": []},
    }


def execution(
    *,
    release: bool = True,
    rollback: bool = False,
    integrity_status: str = "verified",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if release:
        results.append(
            {
                "execution_id": RELEASE_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "release",
                "status": "succeeded",
                "finished_at": "2026-07-07T09:00:00+08:00",
            }
        )
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
    return {"results": results, "integrity": {"status": integrity_status, "warnings": []}}


def execution_history(
    results: list[dict[str, Any]],
    *,
    integrity_status: str = "verified",
) -> dict[str, Any]:
    return {
        "results": results,
        "integrity": {"status": integrity_status, "warnings": []},
    }


def monitoring(
    *,
    missing: bool = False,
    critical: bool = False,
    warning_active: bool = False,
    warning_acknowledged: bool = False,
    rollback_candidate: bool = False,
    integrity_status: str = "verified",
) -> dict[str, Any]:
    required_checks = [
        {
            "check_type": "execution_integrity",
            "status": "pass",
            "latest_check_id": "check-execution",
            "reason": "ok",
        },
        {
            "check_type": "governance_drift",
            "status": "pass",
            "latest_check_id": "check-governance",
            "reason": "ok",
        },
    ]
    if missing:
        required_checks[1] = {
            "check_type": "governance_drift",
            "status": "missing",
            "latest_check_id": None,
            "reason": "missing",
        }

    alerts: list[dict[str, Any]] = []
    acknowledgements: list[dict[str, Any]] = []
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
    if warning_active or warning_acknowledged:
        alerts.append(
            {
                "alert_id": "release_monitor_alert_warning",
                "intent_id": INTENT_ID,
                "execution_id": RELEASE_EXECUTION_ID,
                "severity": "warning",
                "status": "active" if warning_active else "acknowledged",
                "recommended_action": "investigate",
            }
        )
    if warning_acknowledged:
        acknowledgements.append(
            {
                "alert_id": "release_monitor_alert_warning",
                "disposition": "accepted_risk",
            }
        )

    return {
        "status": "monitoring",
        "required_checks": required_checks,
        "checks": [
            {"check_id": item["latest_check_id"]}
            for item in required_checks
            if item["latest_check_id"]
        ],
        "alerts": alerts,
        "acknowledgements": acknowledgements,
        "rollback_trigger_candidate": (
            {"candidate_id": "candidate-1"} if rollback_candidate else None
        ),
        "integrity": {"status": integrity_status, "warnings": []},
    }


def service(
    tmp_path: Path,
    *,
    store: ReleaseClosureStore | None = None,
    dashboard_model: dict[str, Any] | None = None,
    governance_model: dict[str, Any] | None = None,
    execution_model: dict[str, Any] | None = None,
    monitoring_model: dict[str, Any] | None = None,
) -> ReleaseClosureService:
    return ReleaseClosureService(
        store=store if store is not None else ReleaseClosureStore(tmp_path),
        dashboard_loader=lambda: dashboard_model if dashboard_model is not None else dashboard(),
        governance_loader=(
            lambda: governance_model if governance_model is not None else governance()
        ),
        execution_loader=lambda: execution_model if execution_model is not None else execution(),
        monitoring_loader=(
            lambda: monitoring_model if monitoring_model is not None else monitoring()
        ),
        now=lambda: "2026-07-07T10:00:00+08:00",
    )


def test_read_closure_returns_latest_release_contract(tmp_path: Path) -> None:
    model = service(tmp_path, execution_model=execution(rollback=True)).read_closure()

    assert model["latest_release"] == {
        "intent_id": INTENT_ID,
        "release_execution_id": RELEASE_EXECUTION_ID,
        "released_at": "2026-07-07T09:00:00+08:00",
        "rollback_execution_id": ROLLBACK_EXECUTION_ID,
        "rolled_back_at": "2026-07-07T09:30:00+08:00",
    }


def test_closure_is_ready_when_required_checks_pass(tmp_path: Path) -> None:
    model = service(tmp_path).read_closure()

    assert model["status"] == "ready_to_close"
    assert model["closure_gate"]["allowed"] is True


def test_closure_gate_reports_no_successful_release_execution(tmp_path: Path) -> None:
    model = service(tmp_path, execution_model=execution(release=False)).read_closure()

    assert model["status"] == "idle"
    assert "no successful release execution exists" in model["closure_gate"]["reasons"]


def test_closure_blocked_when_required_checks_missing(tmp_path: Path) -> None:
    model = service(tmp_path, monitoring_model=monitoring(missing=True)).read_closure()

    assert model["status"] == "blocked"
    assert "required monitoring checks are missing" in model["closure_gate"]["reasons"]


def test_closure_blocked_when_rollback_candidate_exists(tmp_path: Path) -> None:
    model = service(
        tmp_path,
        monitoring_model=monitoring(rollback_candidate=True),
    ).read_closure()

    assert model["status"] == "blocked"
    assert "rollback trigger candidate exists" in model["closure_gate"]["reasons"]


def test_closure_blocked_when_integrity_failed(tmp_path: Path) -> None:
    model = service(
        tmp_path,
        dashboard_model=dashboard(integrity_status="failed"),
    ).read_closure()

    assert model["status"] == "blocked"
    assert "release closure integrity failed" in model["closure_gate"]["reasons"]


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


def test_read_closure_scopes_latest_fields_and_status_to_latest_successful_release(
    tmp_path: Path,
) -> None:
    store = ReleaseClosureStore(tmp_path)
    app = service(tmp_path, store=store)

    app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted",
        closed_by="release_manager",
        rationale="Older release was closed.",
        idempotency_key="close-release-a",
    )

    newer_execution = {
        "results": [
            {
                "execution_id": RELEASE_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "release",
                "status": "succeeded",
                "finished_at": "2026-07-07T09:00:00+08:00",
            },
            {
                "execution_id": LATER_RELEASE_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "release",
                "status": "succeeded",
                "finished_at": "2026-07-07T11:00:00+08:00",
            },
        ],
        "integrity": {"status": "verified", "warnings": []},
    }

    model = service(
        tmp_path,
        store=store,
        execution_model=newer_execution,
    ).read_closure()

    assert model["status"] == "ready_to_close"
    assert model["latest_release"]["release_execution_id"] == LATER_RELEASE_EXECUTION_ID
    assert model["latest_closure"] is None
    assert model["latest_evidence_package"] is None
    assert model["closure_gate"]["allowed"] is True


def test_read_closure_ignores_older_rollback_for_latest_release_cycle(
    tmp_path: Path,
) -> None:
    model = service(
        tmp_path,
        execution_model=execution_history(
            [
                {
                    "execution_id": RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:00:00+08:00",
                },
                {
                    "execution_id": ROLLBACK_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "rollback",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:30:00+08:00",
                },
                {
                    "execution_id": LATER_RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T11:00:00+08:00",
                },
            ]
        ),
    ).read_closure()

    assert model["status"] == "ready_to_close"
    assert model["latest_release"] == {
        "intent_id": INTENT_ID,
        "release_execution_id": LATER_RELEASE_EXECUTION_ID,
        "released_at": "2026-07-07T11:00:00+08:00",
        "rollback_execution_id": None,
        "rolled_back_at": None,
    }
    assert model["latest_closure"] is None
    assert model["closure_gate"]["allowed"] is True


def test_record_closure_rejects_when_no_successful_release_exists(tmp_path: Path) -> None:
    app = service(tmp_path, execution_model=execution(release=False))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="no successful release execution exists",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-1",
        )


def test_record_closure_rejects_active_critical_alert(tmp_path: Path) -> None:
    app = service(tmp_path, monitoring_model=monitoring(critical=True))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="active critical monitoring alerts exist",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-1",
        )


def test_record_accepted_closure_rejects_active_warning_alerts(tmp_path: Path) -> None:
    app = service(tmp_path, monitoring_model=monitoring(warning_active=True))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="active warning monitoring alerts exist",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-warning-1",
        )


def test_record_accepted_with_observations_requires_acknowledged_warning_alerts(
    tmp_path: Path,
) -> None:
    app = service(tmp_path, monitoring_model=monitoring(warning_active=True))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="accepted_with_observations requires acknowledged active warning alerts",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted_with_observations",
            closed_by="release_manager",
            rationale="Warnings were reviewed.",
            idempotency_key="close-warning-2",
        )


def test_record_accepted_with_observations_succeeds_when_warning_is_acknowledged(
    tmp_path: Path,
) -> None:
    app = service(
        tmp_path,
        monitoring_model=monitoring(
            warning_active=True,
            warning_acknowledged=True,
        ),
    )

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted_with_observations",
        closed_by="release_manager",
        rationale="Warnings were reviewed.",
        idempotency_key="close-warning-3",
    )

    assert model["status"] == "closed"
    assert model["latest_closure"]["closure_status"] == "accepted_with_observations"
    assert model["latest_closure"]["acknowledged_alert_ids"] == [
        "release_monitor_alert_warning"
    ]
    assert model["latest_closure"]["unresolved_alert_ids"] == []


def test_record_closure_rejects_rollback_trigger_candidate_for_non_rollback_status(
    tmp_path: Path,
) -> None:
    app = service(tmp_path, monitoring_model=monitoring(rollback_candidate=True))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="rollback trigger candidate exists",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-candidate-1",
        )


def test_rolled_back_closure_requires_successful_rollback(tmp_path: Path) -> None:
    app = service(tmp_path)

    with pytest.raises(
        ReleaseClosureConflictError,
        match="successful rollback is required for rolled_back closure",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="rolled_back",
            closed_by="release_manager",
            rationale="Rollback completed after monitoring trigger.",
            idempotency_key="close-rollback-1",
        )


def test_record_accepted_closure_is_blocked_after_successful_rollback(
    tmp_path: Path,
) -> None:
    app = service(tmp_path, execution_model=execution(rollback=True))

    with pytest.raises(
        ReleaseClosureConflictError,
        match="accepted closure is blocked after rollback",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-after-rollback-1",
        )


def test_record_accepted_closure_succeeds_for_newer_release_despite_older_rollback(
    tmp_path: Path,
) -> None:
    app = service(
        tmp_path,
        execution_model=execution_history(
            [
                {
                    "execution_id": RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:00:00+08:00",
                },
                {
                    "execution_id": ROLLBACK_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "rollback",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:30:00+08:00",
                },
                {
                    "execution_id": LATER_RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T11:00:00+08:00",
                },
            ]
        ),
    )

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=LATER_RELEASE_EXECUTION_ID,
        closure_status="accepted",
        closed_by="release_manager",
        rationale="Latest release passed closure gates.",
        idempotency_key="close-release-b-accepted",
    )

    assert model["status"] == "closed"
    assert model["latest_release"]["release_execution_id"] == LATER_RELEASE_EXECUTION_ID
    assert model["latest_release"]["rollback_execution_id"] is None
    assert model["latest_closure"]["release_execution_id"] == LATER_RELEASE_EXECUTION_ID
    assert model["latest_closure"]["closure_status"] == "accepted"


def test_rolled_back_closure_rejects_older_rollback_for_previous_release_cycle(
    tmp_path: Path,
) -> None:
    app = service(
        tmp_path,
        execution_model=execution_history(
            [
                {
                    "execution_id": RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:00:00+08:00",
                },
                {
                    "execution_id": ROLLBACK_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "rollback",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T09:30:00+08:00",
                },
                {
                    "execution_id": LATER_RELEASE_EXECUTION_ID,
                    "intent_id": INTENT_ID,
                    "action": "release",
                    "status": "succeeded",
                    "finished_at": "2026-07-07T11:00:00+08:00",
                },
            ]
        ),
    )

    with pytest.raises(
        ReleaseClosureConflictError,
        match="successful rollback is required for rolled_back closure",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=LATER_RELEASE_EXECUTION_ID,
            closure_status="rolled_back",
            closed_by="release_manager",
            rationale="Rollback completed for latest release.",
            idempotency_key="close-release-b-rollback",
        )


def test_rolled_back_closure_allows_successful_rollback_despite_active_candidate(
    tmp_path: Path,
) -> None:
    app = service(
        tmp_path,
        execution_model=execution(rollback=True),
        monitoring_model=monitoring(rollback_candidate=True),
    )

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="rolled_back",
        closed_by="release_manager",
        rationale="Rollback completed after monitoring trigger.",
        idempotency_key="close-rollback-2",
    )

    assert model["status"] == "rolled_back_closed"
    assert model["latest_closure"]["rollback_execution_id"] == ROLLBACK_EXECUTION_ID
    assert model["latest_release"]["rollback_execution_id"] == ROLLBACK_EXECUTION_ID


def test_idempotent_retry_returns_current_model_when_source_state_is_unchanged(
    tmp_path: Path,
) -> None:
    store = ReleaseClosureStore(tmp_path)
    current_monitoring = monitoring(warning_active=True, warning_acknowledged=True)
    app = service(tmp_path, store=store, monitoring_model=current_monitoring)

    first = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted_with_observations",
        closed_by="release_manager",
        rationale="Warnings were reviewed.",
        idempotency_key="close-idempotent-1",
    )
    second = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted_with_observations",
        closed_by="release_manager",
        rationale="Warnings were reviewed.",
        idempotency_key="close-idempotent-1",
    )

    assert second == first


def test_idempotent_retry_conflicts_when_source_state_changes(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    current_monitoring = monitoring(warning_active=True, warning_acknowledged=True)
    app = service(tmp_path, store=store, monitoring_model=current_monitoring)

    app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted_with_observations",
        closed_by="release_manager",
        rationale="Warnings were reviewed.",
        idempotency_key="close-idempotent-2",
    )
    current_monitoring["required_checks"][0]["reason"] = "changed after first write"

    with pytest.raises(
        ReleaseClosureConflictError,
        match="idempotency payload mismatch",
    ):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted_with_observations",
            closed_by="release_manager",
            rationale="Warnings were reviewed.",
            idempotency_key="close-idempotent-2",
        )
