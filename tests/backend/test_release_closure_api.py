from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
import backend.api.routes.admin as admin_routes
from src.services.release_closure import ReleaseClosureConflictError


def test_get_release_closure_returns_status_for_admin_token(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")
    monkeypatch.setattr(
        admin_routes,
        "_release_closure_service",
        lambda: type("Svc", (), {"read_closure": lambda self: {"status": "idle"}})(),
    )
    client = TestClient(create_app())

    response = client.get(
        "/api/admin/release-closure",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "idle"


def test_get_release_closure_maps_store_os_error(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")

    class Service:
        def read_closure(self):
            raise OSError("closure store unavailable")

    monkeypatch.setattr(admin_routes, "_release_closure_service", lambda: Service())
    client = TestClient(create_app())

    response = client.get(
        "/api/admin/release-closure",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "closure store unavailable"


def test_record_closure_maps_gate_conflict(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")

    class Service:
        def record_closure(self, **kwargs):
            raise ReleaseClosureConflictError("active critical monitoring alerts exist")

    monkeypatch.setattr(admin_routes, "_release_closure_service", lambda: Service())
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/release-closure/closures",
        headers={"Authorization": "Bearer admin-token"},
        json={
            "intent_id": "intent-1",
            "release_execution_id": "release-exec-1",
            "closure_status": "accepted",
            "closed_by": "release_manager",
            "rationale": "Close release.",
            "idempotency_key": "close-1",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "active critical monitoring alerts exist"
