from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.services.release_monitoring import ReleaseMonitoringService


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
ROLLBACK_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_rollback_10ca7caa"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


def governance() -> dict[str, Any]:
    return {
        "active_intent": {
            "intent_id": INTENT_ID,
            "target_scope": "feature_flag_candidate",
            "derived_status": "approved",
            "rollback_target": "agent_policy_20260624_0",
            "source_release_report_id": "release_safety_20260629_001",
            "version_chain": {"agent_policy_version": "agent_policy_20260629_0"},
            "release_decision_snapshot": "feature_flag_or_pass",
        },
        "required_approvals": [{"role": "release_manager", "status": "approved"}],
        "rollback_plan": {
            "rollback_plan_id": ROLLBACK_PLAN_ID,
            "intent_id": INTENT_ID,
            "rollback_target": "agent_policy_20260624_0",
            "status": "accepted",
        },
        "integrity": {"status": "verified", "warnings": []},
    }


def dashboard() -> dict[str, Any]:
    return {
        "version_chain": {"agent_policy_version": "agent_policy_20260629_0"},
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_20260624_0",
        "summary": {"hard_fail_count": 0},
        "runs": [{"kind": "literature_shadow_harness", "status": "shadow_only"}],
    }


def execution(released: bool = True, rolled_back: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if released:
        results.append(
            {
                "result_id": "release_result_1",
                "execution_id": EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "release",
                "status": "succeeded",
                "started_at": "2026-07-03T09:00:00+08:00",
                "finished_at": "2026-07-03T09:00:00+08:00",
                "actor": "release_manager",
                "previous_flag_state": None,
                "new_flag_state": {
                    "flag_name": "doctor_review_cockpit_v0",
                    "enabled": True,
                    "scope": "feature_flag_candidate",
                    "source_intent_id": INTENT_ID,
                    "source_execution_id": EXECUTION_ID,
                    "rollback_target": "agent_policy_20260624_0",
                    "updated_by": "release_manager",
                    "updated_at": "2026-07-03T09:00:00+08:00",
                },
                "failure_reason": None,
            }
        )
    if rolled_back:
        results.append(
            {
                "result_id": "rollback_result_1",
                "execution_id": ROLLBACK_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "rollback",
                "status": "succeeded",
                "started_at": "2026-07-03T12:00:00+08:00",
                "finished_at": "2026-07-03T12:00:00+08:00",
                "actor": "release_manager",
                "previous_flag_state": None,
                "new_flag_state": {
                    "flag_name": "doctor_review_cockpit_v0",
                    "enabled": False,
                    "scope": "feature_flag_candidate",
                    "source_intent_id": INTENT_ID,
                    "source_execution_id": ROLLBACK_EXECUTION_ID,
                    "rollback_target": "agent_policy_20260624_0",
                    "updated_by": "release_manager",
                    "updated_at": "2026-07-03T12:00:00+08:00",
                },
                "failure_reason": None,
            }
        )
    return {
        "feature_flag_state": results[-1]["new_flag_state"] if results else None,
        "results": results,
        "integrity": {"status": "verified", "warnings": []},
    }


def service(
    tmp_path: Path,
    exec_state: dict[str, Any] | None = None,
    gov_state: dict[str, Any] | None = None,
    dash_state: dict[str, Any] | None = None,
) -> ReleaseMonitoringService:
    return ReleaseMonitoringService(
        store=ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring"),
        execution_loader=lambda: exec_state if exec_state is not None else execution(),
        governance_loader=lambda: gov_state if gov_state is not None else governance(),
        dashboard_loader=lambda: dash_state if dash_state is not None else dashboard(),
        now=lambda: "2026-07-03T11:00:00+08:00",
    )


def test_monitoring_idle_before_successful_release(tmp_path: Path) -> None:
    model = service(tmp_path, exec_state=execution(released=False)).read_monitoring()

    assert model["status"] == "idle"
    assert model["alerts"] == []
    assert model["rollback_trigger_candidate"] is None


def test_missing_required_checks_create_warning_alerts(tmp_path: Path) -> None:
    model = service(tmp_path).read_monitoring()

    assert model["status"] == "monitoring"
    assert any(
        item["check_type"] == "p0_harness_replay" and item["status"] == "missing"
        for item in model["required_checks"]
    )
    assert any(alert["category"] == "missing_required_check" for alert in model["alerts"])
    assert model["rollback_trigger_candidate"] is None


def test_failed_p0_check_creates_rollback_trigger_candidate(tmp_path: Path) -> None:
    monitor = service(tmp_path)
    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-1",
    )

    assert any(alert["severity"] == "critical" for alert in model["alerts"])
    assert (
        model["rollback_trigger_candidate"]["recommended_action"]
        == "execute_step13_rollback"
    )


def test_false_positive_acknowledgement_removes_rollback_candidate(
    tmp_path: Path,
) -> None:
    monitor = service(tmp_path)
    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-1",
    )
    alert_id = next(
        alert["alert_id"] for alert in model["alerts"] if alert["severity"] == "critical"
    )

    model = monitor.acknowledge_alert(
        alert_id=alert_id,
        acknowledged_by="release_manager",
        disposition="false_positive",
        reason="Harness artifact was copied from a failed pre-release run.",
    )

    assert model["rollback_trigger_candidate"] is None


def test_successful_rollback_changes_monitoring_status(tmp_path: Path) -> None:
    model = service(tmp_path, exec_state=execution(rolled_back=True)).read_monitoring()

    assert model["status"] == "rolled_back"
    assert model["rollback_trigger_candidate"] is None
