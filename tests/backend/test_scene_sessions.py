from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import sessions as session_routes
from backend.api.services.session_store import InMemorySessionStore


class _StubPatientRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._next_patient_id = 100

    def create_draft_patient(self, *, created_by_session_id: str) -> int:
        self.calls.append(created_by_session_id)
        patient_id = self._next_patient_id
        self._next_patient_id += 1
        return patient_id


def _build_client() -> tuple[TestClient, InMemorySessionStore, _StubPatientRegistry]:
    app = FastAPI()
    store = InMemorySessionStore()
    registry = _StubPatientRegistry()

    session_routes.session_store = store
    session_routes.patient_registry_service = registry
    session_routes.load_agent_state = lambda session_id: {}
    session_routes.get_runtime_metadata = lambda: {
        "runner_mode": "real",
        "fixture_case": None,
    }
    app.include_router(session_routes.router)
    return TestClient(app), store, registry


def test_create_patient_scene_returns_sqlite_backed_patient_id() -> None:
    client, _store, registry = _build_client()

    response = client.post("/api/sessions", json={"scene": "patient"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["scene"] == "patient"
    assert isinstance(payload["patient_id"], int)
    assert registry.calls == [payload["session_id"]]


def test_create_doctor_scene_returns_null_patient_id() -> None:
    client, _store, registry = _build_client()

    response = client.post("/api/sessions", json={"scene": "doctor"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["scene"] == "doctor"
    assert payload["patient_id"] is None
    assert registry.calls == []


def test_bind_patient_rejects_rebinding_without_reset() -> None:
    client, _store, _registry = _build_client()
    doctor = client.post("/api/sessions", json={"scene": "doctor"}).json()

    first = client.patch(
        f"/api/sessions/{doctor['session_id']}",
        json={"patient_id": 101},
    )
    second = client.patch(
        f"/api/sessions/{doctor['session_id']}",
        json={"patient_id": 102},
    )

    assert first.status_code == 200
    assert second.status_code == 409