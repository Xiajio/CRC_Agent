from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.api.services.release_monitoring_store import (
    ReleaseMonitoringIntegrityError,
    ReleaseMonitoringStore,
)
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
        "runs": [
            {"kind": "release_safety", "run_id": "release_safety_20260629_001"},
            {"kind": "literature_shadow_harness", "status": "shadow_only"},
        ],
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
    now: Any | None = None,
) -> ReleaseMonitoringService:
    return ReleaseMonitoringService(
        store=ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring"),
        execution_loader=lambda: exec_state if exec_state is not None else execution(),
        governance_loader=lambda: gov_state if gov_state is not None else governance(),
        dashboard_loader=lambda: dash_state if dash_state is not None else dashboard(),
        now=now if now is not None else lambda: "2026-07-03T11:00:00+08:00",
    )


def test_monitoring_idle_before_successful_release(tmp_path: Path) -> None:
    model = service(tmp_path, exec_state=execution(released=False)).read_monitoring()

    assert model["status"] == "idle"
    assert model["alerts"] == []
    assert model["rollback_trigger_candidate"] is None


def test_missing_required_checks_create_warning_alerts(tmp_path: Path) -> None:
    model = service(tmp_path).read_monitoring()

    assert model["status"] == "monitoring"
    assert model["latest_release"] == {
        "intent_id": INTENT_ID,
        "execution_id": EXECUTION_ID,
        "released_at": "2026-07-03T09:00:00+08:00",
        "flag_enabled": True,
        "rollback_plan_id": ROLLBACK_PLAN_ID,
    }
    assert any(
        item["check_type"] == "p0_harness_replay" and item["status"] == "missing"
        for item in model["required_checks"]
    )
    assert set(model["required_checks"][0]) == {
        "check_type",
        "status",
        "latest_check_id",
        "reason",
    }
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


def test_rollback_candidate_requires_matching_rollback_plan_intent(
    tmp_path: Path,
) -> None:
    gov_state = governance()
    gov_state["rollback_plan"] = {
        "rollback_plan_id": "rollback_plan_other_intent_1",
        "intent_id": "release_intent_other_20260629_001",
        "rollback_target": "agent_policy_20260624_0",
        "status": "accepted",
    }
    monitor = service(tmp_path, gov_state=gov_state)

    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-mismatched-rollback-plan",
    )

    assert any(
        alert["category"] == "post_release_check_failed"
        for alert in model["alerts"]
    )
    assert any(
        alert["category"] == "governance_drift"
        and "rollback plan intent" in alert["message"]
        for alert in model["alerts"]
    )
    assert model["rollback_trigger_candidate"] is None


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


def test_failed_check_alert_remains_visible_and_acknowledgeable_after_rollback(
    tmp_path: Path,
) -> None:
    exec_state = execution()
    monitor = service(tmp_path, exec_state=exec_state)
    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-before-rollback",
    )
    alert_id = next(
        alert["alert_id"]
        for alert in model["alerts"]
        if alert["category"] == "post_release_check_failed"
    )
    assert model["rollback_trigger_candidate"] is not None

    exec_state["results"].append(execution(rolled_back=True)["results"][-1])
    exec_state["feature_flag_state"] = exec_state["results"][-1]["new_flag_state"]
    model = monitor.read_monitoring()

    assert model["status"] == "rolled_back"
    assert any(alert["alert_id"] == alert_id for alert in model["alerts"])
    assert model["rollback_trigger_candidate"] is None

    model = monitor.acknowledge_alert(
        alert_id=alert_id,
        acknowledged_by="release_manager",
        disposition="rollback_started_elsewhere",
        reason="Rollback already completed through Step13.",
    )

    assert any(
        alert["alert_id"] == alert_id and alert["status"] == "acknowledged"
        for alert in model["alerts"]
    )


def test_idempotent_record_check_retry_ignores_advanced_now(
    tmp_path: Path,
) -> None:
    current_time = ["2026-07-03T11:00:00+08:00"]
    monitor = service(tmp_path, now=lambda: current_time[0])

    first = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-1",
    )
    current_time[0] = "2026-07-03T11:05:00+08:00"
    second = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-1",
    )

    matching_checks = [
        check
        for check in second["checks"]
        if check["idempotency_key"] == "agent-admin-smoke-1"
    ]
    assert len(matching_checks) == 1
    assert matching_checks[0]["observed_at"] == "2026-07-03T11:00:00+08:00"
    assert second["checks"] == first["checks"]


def test_idempotent_record_check_retry_rejects_changed_request_fields(
    tmp_path: Path,
) -> None:
    current_time = ["2026-07-03T11:00:00+08:00"]
    monitor = service(tmp_path, now=lambda: current_time[0])
    monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-1",
    )
    current_time[0] = "2026-07-03T11:05:00+08:00"

    with pytest.raises(
        ReleaseMonitoringIntegrityError,
        match="idempotency key payload mismatch",
    ):
        monitor.record_check(
            intent_id=INTENT_ID,
            execution_id=EXECUTION_ID,
            check_type="agent_admin_smoke",
            status="pass",
            observed_by="release_manager",
            summary="Changed smoke summary.",
            evidence_refs=["reports/smoke/agent_admin.json"],
            metrics={"passed": 1},
            idempotency_key="agent-admin-smoke-1",
        )


def test_enabled_flag_without_release_source_ids_creates_mismatch_alert(
    tmp_path: Path,
) -> None:
    exec_state = {
        "feature_flag_state": {"enabled": True},
        "results": [],
        "integrity": {"status": "verified", "warnings": []},
    }

    model = service(tmp_path, exec_state=exec_state).read_monitoring()

    assert model["status"] == "idle"
    assert any(
        alert["category"] == "feature_flag_state_mismatch"
        for alert in model["alerts"]
    )
    assert model["rollback_trigger_candidate"] is None


def test_read_monitoring_payload_is_stable_when_now_advances(tmp_path: Path) -> None:
    current_time = ["2026-07-03T11:00:00+08:00"]
    monitor = service(tmp_path, now=lambda: current_time[0])

    first = monitor.read_monitoring()
    current_time[0] = "2026-07-03T11:30:00+08:00"
    second = monitor.read_monitoring()

    assert second == first


def test_governance_integrity_failure_creates_critical_drift_alert(
    tmp_path: Path,
) -> None:
    gov_state = governance()
    gov_state["integrity"] = {"status": "failed", "warnings": ["tampered"]}

    model = service(tmp_path, gov_state=gov_state).read_monitoring()

    assert any(
        alert["category"] == "governance_drift"
        and alert["severity"] == "critical"
        and "governance integrity" in alert["message"]
        for alert in model["alerts"]
    )


def test_dashboard_release_report_drift_creates_governance_drift_alert(
    tmp_path: Path,
) -> None:
    dash_state = dashboard()
    dash_state["runs"] = [
        {"kind": "release_safety", "run_id": "release_safety_20260628_999"},
        {"kind": "literature_shadow_harness", "status": "shadow_only"},
    ]

    model = service(tmp_path, dash_state=dash_state).read_monitoring()

    assert any(
        alert["category"] == "governance_drift"
        and "release report" in alert["message"]
        for alert in model["alerts"]
    )


def test_idempotent_record_check_replay_after_rollback_returns_existing_model(
    tmp_path: Path,
) -> None:
    exec_state = execution()
    monitor = service(tmp_path, exec_state=exec_state)
    monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-rollback-replay",
    )
    exec_state["results"].append(execution(rolled_back=True)["results"][-1])
    exec_state["feature_flag_state"] = exec_state["results"][-1]["new_flag_state"]

    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-rollback-replay",
    )

    assert model["status"] == "rolled_back"
    assert [
        check["idempotency_key"]
        for check in model["checks"]
        if check["idempotency_key"] == "agent-admin-smoke-rollback-replay"
    ] == ["agent-admin-smoke-rollback-replay"]


def test_idempotent_record_check_replay_after_rollback_rejects_changed_fields(
    tmp_path: Path,
) -> None:
    exec_state = execution()
    monitor = service(tmp_path, exec_state=exec_state)
    monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="agent_admin_smoke",
        status="pass",
        observed_by="release_manager",
        summary="Agent admin smoke passed after release.",
        evidence_refs=["reports/smoke/agent_admin.json"],
        metrics={"passed": 1},
        idempotency_key="agent-admin-smoke-rollback-mismatch",
    )
    exec_state["results"].append(execution(rolled_back=True)["results"][-1])
    exec_state["feature_flag_state"] = exec_state["results"][-1]["new_flag_state"]

    with pytest.raises(
        ReleaseMonitoringIntegrityError,
        match="idempotency key payload mismatch",
    ):
        monitor.record_check(
            intent_id=INTENT_ID,
            execution_id=EXECUTION_ID,
            check_type="agent_admin_smoke",
            status="pass",
            observed_by="release_manager",
            summary="Changed after rollback.",
            evidence_refs=["reports/smoke/agent_admin.json"],
            metrics={"passed": 1},
            idempotency_key="agent-admin-smoke-rollback-mismatch",
        )
