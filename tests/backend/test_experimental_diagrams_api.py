from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.api.routes import experimental_diagrams
from backend.api.services.settings import RuntimeSettings
from backend.app import BearerAuthMiddleware
from src.contracts.diagram import DiagramCompileRequest, DiagramCompileResult, DiagramSpec
from src.services.diagram_service import (
    DiagramGenerationError,
    DiagramOutputValidationError,
    DiagramService,
    DiagramServiceUnavailableError,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload = {
        "prompt": "采样后进入模型分析。",
        "requested_by": "admin_operator",
        "idempotency_key": "diagram-api-001",
        "diagram_type": "flowchart",
        "direction": "LR",
        "deidentified": True,
    }
    payload.update(overrides)
    return payload


def _spec(request: DiagramCompileRequest) -> DiagramSpec:
    return DiagramSpec.model_validate(
        {
            "metadata": {
                "title": "实验流程",
                "diagram_type": request.diagram_type,
            },
            "layout": {"direction": request.direction},
            "nodes": [
                {"id": "input", "label": "输入", "ports": ["out"]},
                {"id": "output", "label": "输出", "ports": ["in"]},
            ],
            "edges": [
                {
                    "id": "flow",
                    "source": "input.out",
                    "target": "output.in",
                    "type": "data_flow",
                }
            ],
        }
    )


class _Reasoner:
    def generate_spec(self, request: DiagramCompileRequest) -> DiagramSpec:
        return _spec(request)


def _result(request: DiagramCompileRequest) -> DiagramCompileResult:
    return DiagramService(reasoner=_Reasoner()).compile(request)


def _router_client() -> TestClient:
    app = FastAPI()
    app.include_router(experimental_diagrams.router)
    return TestClient(app)


def test_compile_diagram_api_returns_valid_non_persisted_shadow_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[DiagramCompileRequest] = []

    def compile_stub(request: DiagramCompileRequest) -> DiagramCompileResult:
        captured.append(request)
        return _result(request)

    monkeypatch.setattr(experimental_diagrams, "_compile_diagram", compile_stub)
    client = _router_client()

    try:
        response = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(),
        )
    finally:
        client.close()

    assert response.status_code == 200
    body = response.json()
    assert len(captured) == 1
    assert captured[0].deidentified is True
    assert body["validation"]["valid"] is True
    assert body["exports"]["mermaid"].startswith("flowchart LR")
    assert body["exports"]["dot"].startswith("digraph Diagram")
    assert body["runtime"] == {
        "mode": "shadow",
        "persisted": False,
        "renderer": "source_only",
        "applies_automatically": False,
        "clinical_state_mutated": False,
    }


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (DiagramServiceUnavailableError("model unavailable"), 503),
        (DiagramGenerationError("provider failed"), 502),
        (DiagramOutputValidationError("invalid model output"), 502),
    ],
)
def test_compile_diagram_api_maps_service_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    def fail(_request: DiagramCompileRequest) -> Any:
        raise error

    monkeypatch.setattr(experimental_diagrams, "_compile_diagram", fail)
    client = _router_client()

    try:
        response = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(),
        )
    finally:
        client.close()

    assert response.status_code == expected_status
    assert response.json()["detail"] == str(error)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": "x"},
        {"prompt": "x" * 12_001},
        {"prompt": "abc\x00"},
        {"prompt": "Patient ID: SYNTHETIC-MRN-123456 enters the graph."},
        {"deidentified": False},
        {"deidentified": 1},
        {"diagram_type": "mind_map"},
        {"direction": "diagonal"},
        {"unexpected": "field"},
    ],
)
def test_compile_diagram_api_rejects_invalid_request_payloads(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    calls = 0

    def must_not_compile(request: DiagramCompileRequest) -> DiagramCompileResult:
        nonlocal calls
        calls += 1
        return _result(request)

    monkeypatch.setattr(experimental_diagrams, "_compile_diagram", must_not_compile)
    client = _router_client()

    try:
        response = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(**overrides),
        )
    finally:
        client.close()

    assert response.status_code == 422
    assert calls == 0


def test_diagram_service_rejects_in_process_local_model_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experimental_diagrams,
        "load_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(mode="Local")),
    )

    with pytest.raises(
        DiagramServiceUnavailableError,
        match="requires LLM_MODE=API",
    ):
        experimental_diagrams._diagram_service()


def test_experimental_diagram_router_is_disabled_by_default_and_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "bearer")
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")
    monkeypatch.delenv("EXPERIMENTAL_DIAGRAMS_ENABLED", raising=False)

    disabled_paths = {route.path for route in app_module.create_app().routes}

    monkeypatch.setenv("EXPERIMENTAL_DIAGRAMS_ENABLED", "true")
    enabled_paths = {route.path for route in app_module.create_app().routes}

    path = "/api/admin/experimental/diagrams/compile"
    assert path not in disabled_paths
    assert path in enabled_paths


def test_experimental_diagram_router_requires_admin_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experimental_diagrams,
        "_compile_diagram",
        lambda request: _result(request),
    )
    app = FastAPI()
    app.include_router(experimental_diagrams.router)
    app.add_middleware(
        BearerAuthMiddleware,
        settings=RuntimeSettings(
            auth_mode="bearer",
            api_bearer_token="user-token",
            api_admin_bearer_token="admin-token",
        ),
    )
    client = TestClient(app)

    try:
        missing = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(),
        )
        user = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(),
            headers={"Authorization": "Bearer user-token"},
        )
        admin = client.post(
            "/api/admin/experimental/diagrams/compile",
            json=_payload(),
            headers={"Authorization": "Bearer admin-token"},
        )
    finally:
        client.close()

    assert missing.status_code == 401
    assert user.status_code == 403
    assert admin.status_code == 200
