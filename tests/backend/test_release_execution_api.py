from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes
from backend.api.services.release_execution_store import ReleaseExecutionIntegrityError
from src.services.release_execution import (
    ReleaseExecutionConflictError,
    ReleaseExecutionPreflightError,
)


READ_MODEL = {
    "governance": {
        "active_intent_id": None,
        "derived_status": None,
        "required_approvals_complete": False,
        "rollback_plan_id": None,
    },
    "preflight": {
        "release": {"allowed": False, "reasons": ["no active governance intent"]},
        "rollback": {
            "allowed": False,
            "reasons": ["no successful release execution exists"],
        },
    },
    "feature_flag_state": None,
    "requests": [],
    "results": [],
    "audit_events": [],
    "integrity": {"status": "verified", "warnings": []},
    "runtime": {
        "auth": "admin",
        "source": "reports/release_execution",
        "mode": "controlled_local_execution",
    },
}


class FakeReleaseExecutionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    def read_execution(self) -> dict[str, Any]:
        self.calls.append(("read_execution", {}))
        return READ_MODEL

    def execute_release(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("execute_release", payload))
        if self.error is not None:
            raise self.error
        return READ_MODEL

    def execute_rollback(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("execute_rollback", payload))
        if self.error is not None:
            raise self.error
        return READ_MODEL


def _client_with_fake_service(
    monkeypatch,
    service: FakeReleaseExecutionService,
) -> TestClient:
    monkeypatch.setattr(
        admin_routes,
        "_release_execution_service",
        lambda: service,
        raising=False,
    )
    app = FastAPI()
    app.include_router(admin_routes.router)
    return TestClient(app)


def _request_payload() -> dict[str, str]:
    return {
        "intent_id": "intent-1",
        "requested_by": "release_manager",
        "idempotency_key": "release-1",
        "reason": "Approved release.",
        "expected_rollback_plan_id": "rollback-1",
    }


def test_release_execution_get_returns_service_read_model(monkeypatch) -> None:
    service = FakeReleaseExecutionService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.get("/api/admin/release-execution")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [("read_execution", {})]


def test_release_execution_post_routes_delegate_to_service(monkeypatch) -> None:
    service = FakeReleaseExecutionService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        release_response = client.post(
            "/api/admin/release-execution/release",
            json=_request_payload(),
        )
        rollback_response = client.post(
            "/api/admin/release-execution/rollback",
            json={**_request_payload(), "idempotency_key": "rollback-1"},
        )
    finally:
        client.close()

    assert release_response.status_code == 200
    assert rollback_response.status_code == 200
    assert service.calls == [
        ("execute_release", _request_payload()),
        (
            "execute_rollback",
            {**_request_payload(), "idempotency_key": "rollback-1"},
        ),
    ]


def test_release_execution_schema_rejects_extra_fields(monkeypatch) -> None:
    service = FakeReleaseExecutionService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-execution/release",
            json={**_request_payload(), "deployment_credentials": "secret"},
        )
    finally:
        client.close()

    assert response.status_code == 422
    assert service.calls == []


def test_release_execution_service_errors_map_to_http_status(monkeypatch) -> None:
    cases = [
        (ReleaseExecutionPreflightError("required approvals are incomplete"), 409),
        (ReleaseExecutionConflictError("release already exists"), 409),
        (ReleaseExecutionIntegrityError("audit chain failed"), 409),
    ]

    for error, expected_status in cases:
        service = FakeReleaseExecutionService()
        service.error = error
        client = _client_with_fake_service(monkeypatch, service)

        try:
            response = client.post(
                "/api/admin/release-execution/release",
                json=_request_payload(),
            )
        finally:
            client.close()

        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)
