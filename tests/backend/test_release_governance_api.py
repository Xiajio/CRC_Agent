from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes
from backend.api.services.release_governance_store import GovernanceIntegrityError
from src.services.release_governance import (
    GovernanceConflictError,
    GovernanceValidationError,
)


READ_MODEL = {
    "dashboard_snapshot": {"release_decision": "feature_flag_or_pass"},
    "intents": [],
    "active_intent": None,
    "required_approvals": [],
    "rollback_plan": None,
    "audit_events": [],
    "integrity": {"status": "verified", "warnings": []},
    "disabled_execution_actions": [],
    "runtime": {
        "auth": "admin",
        "source": "reports/release_governance",
        "mode": "audit_only",
    },
}


class FakeReleaseGovernanceService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    def read_governance(self) -> dict[str, Any]:
        self.calls.append(("read_governance", {}))
        return READ_MODEL

    def create_intent(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("create_intent", payload))
        if self.error is not None:
            raise self.error
        return READ_MODEL

    def record_approval(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("record_approval", payload))
        return READ_MODEL

    def record_rollback_plan(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("record_rollback_plan", payload))
        return READ_MODEL

    def cancel_intent(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(("cancel_intent", payload))
        return READ_MODEL


def _client_with_fake_service(
    monkeypatch,
    service: FakeReleaseGovernanceService,
) -> TestClient:
    monkeypatch.setattr(
        admin_routes,
        "_release_governance_service",
        lambda: service,
        raising=False,
    )
    app = FastAPI()
    app.include_router(admin_routes.router)
    return TestClient(app)


def test_release_governance_get_returns_service_read_model(monkeypatch) -> None:
    service = FakeReleaseGovernanceService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.get("/api/admin/release-governance")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [("read_governance", {})]


def test_create_release_intent_route_delegates_to_service(monkeypatch) -> None:
    service = FakeReleaseGovernanceService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-governance/intents",
            json={
                "requested_by": "admin_operator",
                "target_scope": "shadow",
                "status": "pending_approval",
                "reason": "Prepare audited governance.",
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == READ_MODEL
    assert service.calls == [
        (
            "create_intent",
            {
                "requested_by": "admin_operator",
                "target_scope": "shadow",
                "status": "pending_approval",
                "reason": "Prepare audited governance.",
            },
        )
    ]


def test_nested_release_governance_post_routes_delegate_to_service(monkeypatch) -> None:
    service = FakeReleaseGovernanceService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        approval_response = client.post(
            "/api/admin/release-governance/intents/intent-1/approvals",
            json={
                "approver_role": "release_manager",
                "decision": "approve",
                "reason": "Release dashboard gates are clear.",
                "signed_by": "release_admin",
            },
        )
        rollback_response = client.post(
            "/api/admin/release-governance/intents/intent-1/rollback-plan",
            json={
                "owner": "release_manager",
                "status": "accepted",
                "verification_steps": [
                    "Confirm the active release report id.",
                    "Run P0 harness before rollback execution.",
                ],
            },
        )
        cancel_response = client.post(
            "/api/admin/release-governance/intents/intent-1/cancel",
            json={
                "actor": "release_manager",
                "reason": "Release window closed.",
            },
        )
    finally:
        client.close()

    assert approval_response.status_code == 200
    assert rollback_response.status_code == 200
    assert cancel_response.status_code == 200
    assert service.calls == [
        (
            "record_approval",
            {
                "intent_id": "intent-1",
                "approver_role": "release_manager",
                "decision": "approve",
                "reason": "Release dashboard gates are clear.",
                "signed_by": "release_admin",
            },
        ),
        (
            "record_rollback_plan",
            {
                "intent_id": "intent-1",
                "owner": "release_manager",
                "status": "accepted",
                "verification_steps": [
                    "Confirm the active release report id.",
                    "Run P0 harness before rollback execution.",
                ],
            },
        ),
        (
            "cancel_intent",
            {
                "intent_id": "intent-1",
                "actor": "release_manager",
                "reason": "Release window closed.",
            },
        ),
    ]


def test_release_governance_payload_validation_returns_422(monkeypatch) -> None:
    service = FakeReleaseGovernanceService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-governance/intents",
            json={
                "requested_by": "admin_operator",
                "target_scope": "shadow",
                "status": "pending_approval",
                "reason": "",
            },
        )
    finally:
        client.close()

    assert response.status_code == 422
    assert service.calls == []


def test_create_release_intent_rejects_terminal_status(monkeypatch) -> None:
    service = FakeReleaseGovernanceService()
    client = _client_with_fake_service(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/release-governance/intents",
            json={
                "requested_by": "admin_operator",
                "target_scope": "shadow",
                "status": "approved",
                "reason": "Bypass review.",
            },
        )
    finally:
        client.close()

    assert response.status_code == 422
    assert service.calls == []


def test_release_governance_service_errors_map_to_http_status(monkeypatch) -> None:
    cases = [
        (GovernanceValidationError("invalid governance request"), 422),
        (GovernanceConflictError("active intent already exists"), 409),
        (GovernanceIntegrityError("audit chain failed"), 409),
    ]

    for error, expected_status in cases:
        service = FakeReleaseGovernanceService()
        service.error = error
        client = _client_with_fake_service(monkeypatch, service)

        try:
            response = client.post(
                "/api/admin/release-governance/intents",
                json={
                    "requested_by": "admin_operator",
                    "target_scope": "shadow",
                    "status": "pending_approval",
                    "reason": "Prepare audited governance.",
                },
            )
        finally:
            client.close()

        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)
