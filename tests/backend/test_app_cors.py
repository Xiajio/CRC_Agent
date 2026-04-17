from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from backend import app as backend_app
from backend.api.services.settings import RuntimeSettings


def test_create_app_allows_delete_for_cors_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_app,
        "load_runtime_settings",
        lambda: RuntimeSettings(frontend_origins=["http://127.0.0.1:4173"]),
    )

    app = backend_app.create_app()
    cors = next(middleware for middleware in app.user_middleware if middleware.cls is CORSMiddleware)

    assert "DELETE" in cors.kwargs["allow_methods"]
