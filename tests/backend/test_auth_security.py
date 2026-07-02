from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import BearerAuthMiddleware
from backend.api.services import settings as settings_module
from backend.api.services.settings import RuntimeSettings


def test_runtime_settings_default_to_bearer_and_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("API_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("API_ADMIN_BEARER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="API_BEARER_TOKEN"):
        settings_module.load_runtime_settings()


def test_runtime_settings_admin_token_falls_back_to_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.delenv("API_ADMIN_BEARER_TOKEN", raising=False)

    runtime_settings = settings_module.load_runtime_settings()

    assert runtime_settings.auth_mode == "bearer"
    assert runtime_settings.api_bearer_token == "user-token"
    assert runtime_settings.api_admin_bearer_token == "user-token"


def _auth_client(*, user_token: str = "user-token", admin_token: str | None = "admin-token") -> TestClient:
    app = FastAPI()

    @app.get("/api/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/database/cases/upsert")
    async def upsert_case() -> dict[str, bool]:
        return {"ok": True}

    @app.delete("/api/patient-registry/patients")
    async def clear_patients() -> dict[str, bool]:
        return {"ok": True}

    @app.delete("/api/patient-registry/patients/{patient_id}")
    async def delete_patient(patient_id: int) -> dict[str, bool | int]:
        return {"ok": True, "patient_id": patient_id}

    @app.get("/api/admin/tools")
    async def admin_tools() -> dict[str, object]:
        return {"tools": []}

    @app.get("/api/admin/release-dashboard")
    async def admin_release_dashboard() -> dict[str, object]:
        return {"runtime": {"auth": "admin"}}

    @app.get("/api/admin/release-governance")
    async def admin_release_governance() -> dict[str, object]:
        return {"runtime": {"auth": "admin", "mode": "audit_only"}}

    @app.post("/api/admin/release-governance/intents")
    async def create_release_intent() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/admin/release-governance/intents/{intent_id}/approvals")
    async def record_release_approval(intent_id: str) -> dict[str, bool | str]:
        return {"ok": True, "intent_id": intent_id}

    @app.post("/api/admin/release-governance/intents/{intent_id}/rollback-plan")
    async def record_release_rollback_plan(intent_id: str) -> dict[str, bool | str]:
        return {"ok": True, "intent_id": intent_id}

    @app.post("/api/admin/release-governance/intents/{intent_id}/cancel")
    async def cancel_release_intent(intent_id: str) -> dict[str, bool | str]:
        return {"ok": True, "intent_id": intent_id}

    app.add_middleware(
        BearerAuthMiddleware,
        settings=RuntimeSettings(
            auth_mode="bearer",
            api_bearer_token=user_token,
            api_admin_bearer_token=admin_token,
        ),
    )
    return TestClient(app)


def test_bearer_auth_required_for_api_routes() -> None:
    client = _auth_client()

    try:
        assert client.get("/api/ok").status_code == 401
        assert client.get("/api/ok", headers={"Authorization": "Bearer user-token"}).status_code == 200
    finally:
        client.close()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/database/cases/upsert"),
        ("delete", "/api/patient-registry/patients"),
        ("delete", "/api/patient-registry/patients/123"),
        ("get", "/api/admin/tools"),
        ("get", "/api/admin/release-dashboard"),
        ("get", "/api/admin/release-governance"),
        ("post", "/api/admin/release-governance/intents"),
        ("post", "/api/admin/release-governance/intents/intent-1/approvals"),
        ("post", "/api/admin/release-governance/intents/intent-1/rollback-plan"),
        ("post", "/api/admin/release-governance/intents/intent-1/cancel"),
    ],
)
def test_admin_endpoints_reject_user_token_when_admin_token_is_distinct(method: str, path: str) -> None:
    client = _auth_client()

    try:
        response = getattr(client, method)(path, headers={"Authorization": "Bearer user-token"})

        assert response.status_code == 403
        assert response.json()["detail"] == "Forbidden"
    finally:
        client.close()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/database/cases/upsert"),
        ("delete", "/api/patient-registry/patients"),
        ("delete", "/api/patient-registry/patients/123"),
        ("get", "/api/admin/tools"),
        ("get", "/api/admin/release-dashboard"),
        ("get", "/api/admin/release-governance"),
        ("post", "/api/admin/release-governance/intents"),
        ("post", "/api/admin/release-governance/intents/intent-1/approvals"),
        ("post", "/api/admin/release-governance/intents/intent-1/rollback-plan"),
        ("post", "/api/admin/release-governance/intents/intent-1/cancel"),
    ],
)
def test_admin_endpoints_accept_admin_token(method: str, path: str) -> None:
    client = _auth_client()

    try:
        response = getattr(client, method)(path, headers={"Authorization": "Bearer admin-token"})

        assert response.status_code == 200
    finally:
        client.close()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/database/cases/upsert"),
        ("delete", "/api/patient-registry/patients"),
        ("delete", "/api/patient-registry/patients/123"),
        ("get", "/api/admin/tools"),
        ("get", "/api/admin/release-dashboard"),
        ("get", "/api/admin/release-governance"),
        ("post", "/api/admin/release-governance/intents"),
        ("post", "/api/admin/release-governance/intents/intent-1/approvals"),
        ("post", "/api/admin/release-governance/intents/intent-1/rollback-plan"),
        ("post", "/api/admin/release-governance/intents/intent-1/cancel"),
    ],
)
def test_admin_endpoints_use_user_token_when_no_separate_admin_token(method: str, path: str) -> None:
    client = _auth_client(admin_token=None)

    try:
        response = getattr(client, method)(path, headers={"Authorization": "Bearer user-token"})

        assert response.status_code == 200
    finally:
        client.close()
