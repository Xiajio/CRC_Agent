from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.api.services.patient_context_resolver import PatientContextResolver


class _NoopGraph:
    async def astream(
        self,
        payload: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ):
        del payload, config
        if False:
            yield {}


def test_create_app_lifespan_wires_patient_context_resolver(
    monkeypatch,
) -> None:
    real_path = Path
    runtime_root = real_path("output") / "test-app-runtime-wiring" / uuid4().hex

    def path_factory(*parts: str) -> Path:
        if parts == ("runtime",):
            return runtime_root
        return real_path(*parts)

    monkeypatch.setattr(app_module, "Path", path_factory)
    monkeypatch.setattr(app_module, "load_backend_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_module, "get_runner_mode", lambda: "fixture")
    monkeypatch.setattr(app_module, "should_warm_rag", lambda: False)
    monkeypatch.setattr(app_module, "get_patient_graph", lambda *_, **__: _NoopGraph())
    monkeypatch.setattr(app_module, "get_doctor_graph", lambda *_, **__: _NoopGraph())
    monkeypatch.setattr(
        app_module,
        "create_context_maintenance_service",
        lambda *_, **__: None,
    )

    app = app_module.create_app()

    with TestClient(app) as client:
        runtime = client.app.state.runtime
        assert isinstance(runtime.patient_context_resolver, PatientContextResolver)
        assert (
            runtime.patient_graph_service._patient_context_resolver
            is runtime.patient_context_resolver
        )
        assert (
            runtime.doctor_graph_service._patient_context_resolver
            is runtime.patient_context_resolver
        )
