from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import sessions as session_routes
from backend.api.services.sqlite_session_store import SqliteSessionStore


def test_session_route_recovers_sqlite_session_after_store_rebuild(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SqliteSessionStore(db_path, ttl_days=None)
    monkeypatch.setattr(session_routes, "session_store", store)
    monkeypatch.setattr(session_routes, "patient_registry_service", None)
    monkeypatch.setattr(session_routes, "patient_command_service", None)
    monkeypatch.setattr(session_routes, "load_agent_state", lambda _session_id: None)
    monkeypatch.setattr(
        session_routes,
        "get_runtime_metadata",
        lambda: {"runner_mode": "fixture", "fixture_case": "database_case"},
    )
    app = FastAPI()
    app.include_router(session_routes.router)

    with TestClient(app) as client:
        create_response = client.post("/api/sessions", json={"scene": "doctor"})
        assert create_response.status_code == 200, create_response.text
        session_id = create_response.json()["session_id"]

        meta = store.get_session(session_id)
        assert meta is not None
        meta.uploaded_assets["asset-1"] = {
            "asset_id": "asset-1",
            "filename": "report.pdf",
        }
        store.touch(session_id)
        store.close()

        reopened = SqliteSessionStore(db_path, ttl_days=None)
        monkeypatch.setattr(session_routes, "session_store", reopened)
        get_response = client.get(f"/api/sessions/{session_id}")

        assert get_response.status_code == 200, get_response.text
        payload = get_response.json()
        assert payload["session_id"] == session_id
        assert payload["snapshot"]["uploaded_assets"]["asset-1"]["filename"] == "report.pdf"
        reopened.close()
