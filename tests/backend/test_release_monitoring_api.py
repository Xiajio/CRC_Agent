from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes
from backend.api.services.release_monitoring_store import ReleaseMonitoringIntegrityError
from src.services.release_monitoring import (
    ReleaseMonitoringConflictError,
    ReleaseMonitoringValidationError,
)


READ_MODEL = {
    "status": "monitoring",
    "latest_release": {
        "intent_id": "intent-1",
        "execution_id": "execution-1",
        "released_at": "2026-07-03T09:00:00+08:00",
        "flag_enabled": True,
        "rollback_plan_id": "rollback-1",
    },
    "required_checks": [],
    "checks": [],
    "alerts": [],
    "rollback_trigger_candidate": None,
    "acknowledgements": [],
    "integrity": {"status": "verified", "warnings": []},
    "runtime": {
        "auth": "admin",
        "source": "reports/release_monitoring",
        "mode": "post_release_monitoring",
    },
}


class FakeReleaseMonitoringService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    def read_monitoring(self) -> dict[str, Any]:
        self.calls.append(("read_monitoring", {}))
        return READ_MODEL

    def record_check(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("record_check", payload))
        if self.error is not None:
            raise self.error
        return READ_MODEL

    def acknowledge_alert(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("acknowledge_alert", payload))
        if self.error is not None:
            raise self.error
        return READ_MODEL


def _client_with_fake_service(
    monkeypatch,
    service: FakeReleaseMonitoringService,
) -> TestClient:
    monkeypatch.setattr(
        admin_routes,
        "_release_monitoring_service",
        lambda: service,
        raising=False,
    )
    app = FastAPI()
    app.include_router(admin_routes.router)
    return TestClient(app)


def _check_payload() -> dict[str, Any]:
    return {
        "intent_id": "intent-1",
        "execution_id": "execution-1",
        "check_type": "agent_admin_smoke",
        "status": "pass",
        "observed_by": "release_manager",
        "summary": "Agent admin smoke passed after release.",
        "evidence_refs": ["reports/smoke/agent_admin.json"],
        "metrics": {"passed": 1},
        "idempotency_key": "agent-admin-smoke-1",
    }


def _acknowledgement_payload() -> dict[str, str]:
    return {
        "acknowledged_by": "release_manager",
        "disposition": "investigating",
        "reason": "Checking harness evidence before rollback execution.",
    }


def test_release_monitoring_get_returns_service_read_model(monkeypatch) -> None:
    service = FakeReleaseMonitoringService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.get("/api/admin/release-monitoring")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [("read_monitoring", {})]


def test_release_monitoring_checks_post_delegates_payload(monkeypatch) -> None:
    service = FakeReleaseMonitoringService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-monitoring/checks",
            json=_check_payload(),
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [("record_check", _check_payload())]


def test_release_monitoring_acknowledge_post_delegates_alert_id_and_payload(
    monkeypatch,
) -> None:
    service = FakeReleaseMonitoringService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-monitoring/alerts/alert-1/acknowledge",
            json=_acknowledgement_payload(),
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [
        (
            "acknowledge_alert",
            {"alert_id": "alert-1", **_acknowledgement_payload()},
        )
    ]


def test_release_monitoring_unknown_alert_validation_error_maps_to_404(
    monkeypatch,
) -> None:
    service = FakeReleaseMonitoringService()
    service.error = ReleaseMonitoringValidationError(
        "alert_id does not reference a current monitoring alert"
    )
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-monitoring/alerts/alert-404/acknowledge",
            json=_acknowledgement_payload(),
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert response.json()["detail"] == str(service.error)


def test_release_monitoring_unknown_alert_conflict_error_maps_to_404(
    monkeypatch,
) -> None:
    service = FakeReleaseMonitoringService()
    service.error = ReleaseMonitoringConflictError(
        "alert does not exist in current monitoring model"
    )
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-monitoring/alerts/alert-404/acknowledge",
            json=_acknowledgement_payload(),
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert response.json()["detail"] == str(service.error)


def test_release_monitoring_service_errors_map_to_conflict(monkeypatch) -> None:
    cases = [
        ReleaseMonitoringConflictError("check conflicts with latest release"),
        ReleaseMonitoringIntegrityError("release monitoring integrity failed"),
    ]

    for error in cases:
        service = FakeReleaseMonitoringService()
        service.error = error
        client = _client_with_fake_service(monkeypatch, service)

        try:
            response = client.post(
                "/api/admin/release-monitoring/checks",
                json=_check_payload(),
            )
        finally:
            client.close()

        assert response.status_code == 409
        assert response.json()["detail"] == str(error)


def test_release_monitoring_schema_rejects_extra_fields(monkeypatch) -> None:
    service = FakeReleaseMonitoringService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-monitoring/checks",
            json={**_check_payload(), "deployment_credentials": "secret"},
        )
    finally:
        client.close()

    assert response.status_code == 422
    assert service.calls == []
