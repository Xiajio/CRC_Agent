from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes
from backend.api.services.settings import RuntimeSettings
from backend.app import BearerAuthMiddleware


def _admin_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(
        BearerAuthMiddleware,
        settings=RuntimeSettings(
            auth_mode="bearer",
            api_bearer_token="user-token",
            api_admin_bearer_token="admin-token",
            frontend_origins=[],
        ),
    )
    app.include_router(admin_routes.router)
    return TestClient(app)


def _collect_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _collect_keys(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _collect_keys(item)


def test_get_admin_rules_with_admin_token_returns_policy_projection() -> None:
    client = _admin_client()

    try:
        response = client.get(
            "/api/admin/rules",
            headers={"Authorization": "Bearer admin-token"},
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_id"] == "crc_safety_policy_v0"
    assert isinstance(payload["version"], str)
    assert payload["status"] == "draft"
    assert payload["applies_to"] == "patient_crc_triage"
    assert payload["severity_order"] == ["emergency", "urgent", "backfill", "routine"]
    assert payload["source_path"] == "config/safety_policy.yaml"
    assert payload["note"] == "read-only projection; not editable from admin UI"
    assert payload["rules"]
    assert payload["rules"][0] == {
        "id": "bowel_obstruction_red_flag",
        "priority": 100,
        "disposition": "emergency",
        "hard_fail_if_missed": True,
        "group": "safety",
        "condition_summary": "any: vomiting, obstipation; all: severe_abdominal_pain",
    }


def test_get_admin_rules_rejects_missing_token() -> None:
    client = _admin_client()

    try:
        response = client.get("/api/admin/rules")
    finally:
        client.close()

    assert response.status_code == 401


def test_get_admin_rules_rejects_non_admin_token() -> None:
    client = _admin_client()

    try:
        response = client.get(
            "/api/admin/rules",
            headers={"Authorization": "Bearer user-token"},
        )
    finally:
        client.close()

    assert response.status_code == 403


def test_get_admin_rules_does_not_expose_secrets_prompts_or_raw_policy_logic() -> None:
    client = _admin_client()

    try:
        payload = client.get(
            "/api/admin/rules",
            headers={"Authorization": "Bearer admin-token"},
        ).json()
    finally:
        client.close()

    forbidden_keys = {
        "condition",
        "fallback",
        "patient_message_key",
        "prompt",
        "system_prompt",
        "developer_prompt",
        "api_key",
        "token",
        "secret",
    }

    assert forbidden_keys.isdisjoint({key.lower() for key in _collect_keys(payload)})
