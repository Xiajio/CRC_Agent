from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import sessions as session_routes
from backend.api.services.session_store import InMemorySessionStore


class _StubPatientRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_draft_patient(self, *, created_by_session_id: str) -> int:
        self.calls.append(created_by_session_id)
        return 1

    def get_patient_identity(self, patient_id: int):
        return None

    def set_patient_identity(
        self,
        patient_id: int,
        patient_name: str,
        patient_number: str,
    ) -> None:
        raise AssertionError("identity writes must use patient commands")


class _StubPatientCommands:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.identity_calls: list[dict[str, object]] = []
        self._next_patient_id = 100

    def create_patient(self, *, created_by_session_id: str):
        self.calls.append(created_by_session_id)
        patient_id = self._next_patient_id
        self._next_patient_id += 1
        return type(
            "Result",
            (),
            {
                "patient_id": patient_id,
                "patient_version": 1,
                "projection_version": 1,
                "event_ids": ["evt_created"],
            },
        )()

    def set_identity(
        self,
        *,
        patient_id: int,
        patient_name: str,
        patient_number: str,
        source_session_id: str,
    ):
        self.identity_calls.append(
            {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_number": patient_number,
                "source_session_id": source_session_id,
            }
        )


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, InMemorySessionStore, _StubPatientRegistry, _StubPatientCommands]:
    store = InMemorySessionStore()
    registry = _StubPatientRegistry()
    commands = _StubPatientCommands()
    monkeypatch.setattr(session_routes, "session_store", store)
    monkeypatch.setattr(session_routes, "patient_registry_service", registry)
    monkeypatch.setattr(session_routes, "patient_command_service", commands)
    monkeypatch.setattr(session_routes, "load_agent_state", lambda _session_id: None)
    monkeypatch.setattr(
        session_routes,
        "get_runtime_metadata",
        lambda: {"runner_mode": "real", "fixture_case": None},
    )

    app = FastAPI()
    app.include_router(session_routes.router)
    return TestClient(app), store, registry, commands


def test_create_patient_scene_returns_command_backed_patient_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store, registry, commands = _build_client(monkeypatch)

    response = client.post("/api/sessions", json={"scene": "patient"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["patient_id"] == 100
    assert commands.calls == [payload["session_id"]]
    assert registry.calls == []


def test_create_doctor_scene_returns_null_patient_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store, registry, commands = _build_client(monkeypatch)

    response = client.post("/api/sessions", json={"scene": "doctor"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["scene"] == "doctor"
    assert payload["patient_id"] is None
    assert commands.calls == []
    assert registry.calls == []


def test_bind_patient_rejects_rebinding_to_different_patient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store, _registry, _commands = _build_client(monkeypatch)
    create_response = client.post("/api/sessions", json={"scene": "doctor"})
    session_id = create_response.json()["session_id"]

    first_bind = client.patch(f"/api/sessions/{session_id}", json={"patient_id": 100})
    second_bind = client.patch(f"/api/sessions/{session_id}", json={"patient_id": 101})

    assert first_bind.status_code == 200, first_bind.text
    assert second_bind.status_code == 409
    assert second_bind.json()["detail"] == "Session already bound to a different patient"


def test_set_patient_identity_routes_through_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store, _registry, commands = _build_client(monkeypatch)
    create_response = client.post("/api/sessions", json={"scene": "patient"})
    payload = create_response.json()

    response = client.post(
        f"/api/sessions/{payload['session_id']}/identity",
        json={"patient_name": " Ada Lovelace ", "patient_number": " ab-123 "},
    )

    assert response.status_code == 200, response.text
    assert commands.identity_calls == [
        {
            "patient_id": 100,
            "patient_name": "Ada Lovelace",
            "patient_number": "ab-123",
            "source_session_id": payload["session_id"],
        }
    ]
