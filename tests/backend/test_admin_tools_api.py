from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes.admin import router


def _admin_client(*, web_search_enabled: bool) -> TestClient:
    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        settings=SimpleNamespace(
            web_search=SimpleNamespace(enabled=web_search_enabled),
        ),
    )
    app.include_router(router)
    return TestClient(app)


def test_admin_tools_returns_manifest_runtime_metadata() -> None:
    client = _admin_client(web_search_enabled=True)

    try:
        response = client.get("/api/admin/tools")
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["tools"], list)
    assert payload["tools"]
    assert isinstance(payload["groups"], list)
    assert payload["groups"]
    assert payload["runtime"] == {
        "web_search_enabled": True,
        "auth": "admin",
        "source": "src.tools.manifest",
    }


def test_admin_tools_response_does_not_expose_forbidden_fields() -> None:
    client = _admin_client(web_search_enabled=True)

    try:
        payload = client.get("/api/admin/tools").json()
    finally:
        client.close()

    forbidden_keys = {
        "factory_ref",
        "module",
        "module_path",
        "file_path",
        "path",
        "api_key",
        "token",
        "model_path",
        "notes",
    }

    assert payload["tools"]
    for tool in payload["tools"]:
        assert forbidden_keys.isdisjoint(tool)


def test_admin_tools_marks_web_required_tools_unavailable_when_web_search_disabled() -> None:
    client = _admin_client(web_search_enabled=False)

    try:
        payload = client.get("/api/admin/tools").json()
    finally:
        client.close()

    web_required_tools = [
        tool for tool in payload["tools"] if tool["requires_web"]
    ]

    assert payload["runtime"]["web_search_enabled"] is False
    assert web_required_tools
    assert all(tool["available"] is False for tool in web_required_tools)
