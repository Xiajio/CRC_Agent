from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import sessions
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def _make_scratch_root() -> Path:
    scratch_root = Path("tmp") / "backend-api" / "patient-identity" / uuid4().hex
    scratch_root.mkdir(parents=True, exist_ok=False)
    return scratch_root


@pytest.fixture()
def identity_app(monkeypatch: pytest.MonkeyPatch):
    scratch_root = _make_scratch_root()
    session_store = InMemorySessionStore()
    registry = PatientRegistryService(scratch_root / "patient_registry.sqlite3")
    commands = PatientCommandService(registry)

    monkeypatch.setattr(sessions, "session_store", session_store)
    monkeypatch.setattr(sessions, "patient_registry_service", registry)
    monkeypatch.setattr(sessions, "patient_command_service", commands)
    monkeypatch.setattr(sessions, "load_agent_state", lambda _session_id: None)
    monkeypatch.setattr(
        sessions,
        "get_runtime_metadata",
        lambda: {"runner_mode": "real", "fixture_case": None},
    )

    app = FastAPI()
    app.include_router(sessions.router)
    client = TestClient(app)
    try:
        yield client, session_store, registry
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def test_save_patient_identity_success(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, registry = identity_app
    session_meta = session_store.create_session(scene="patient")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)

    response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": " Ada Lovelace ", "patient_number": " ab-123 "},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["snapshot"]["patient_identity"] == {
        "patient_name": "Ada Lovelace",
        "patient_number": "ab-123",
        "identity_locked": True,
    }


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        ({"patient_name": "", "patient_number": "ABC"}, "patient_name"),
        ({"patient_name": "Ada", "patient_number": "   "}, "patient_number"),
    ],
)
def test_save_patient_identity_rejects_blank_fields(
    identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService],
    payload: dict[str, str],
    field_name: str,
) -> None:
    client, session_store, registry = identity_app
    session_meta = session_store.create_session(scene="patient")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)

    response = client.post(f"/api/sessions/{session_meta.session_id}/identity", json=payload)

    assert response.status_code == 422
    assert field_name in response.text


def test_save_patient_identity_detects_duplicate_patient_number(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, registry = identity_app
    first_session = session_store.create_session(scene="patient")
    second_session = session_store.create_session(scene="patient")
    first_patient_id = registry.create_draft_patient(created_by_session_id=first_session.session_id)
    second_patient_id = registry.create_draft_patient(created_by_session_id=second_session.session_id)
    session_store.set_patient_id(first_session.session_id, first_patient_id)
    session_store.set_patient_id(second_session.session_id, second_patient_id)

    first_response = client.post(
        f"/api/sessions/{first_session.session_id}/identity",
        json={"patient_name": "First", "patient_number": "ab123"},
    )
    assert first_response.status_code == 200, first_response.text

    conflict_response = client.post(
        f"/api/sessions/{second_session.session_id}/identity",
        json={"patient_name": "Second", "patient_number": "AB123"},
    )

    assert conflict_response.status_code == 409
    assert conflict_response.json()["detail"] == "PATIENT_NUMBER_ALREADY_EXISTS"


def test_save_patient_identity_maps_index_integrity_error_even_when_precheck_is_bypassed(
    identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_store, registry = identity_app
    first_session = session_store.create_session(scene="patient")
    second_session = session_store.create_session(scene="patient")
    first_patient_id = registry.create_draft_patient(created_by_session_id=first_session.session_id)
    second_patient_id = registry.create_draft_patient(created_by_session_id=second_session.session_id)
    session_store.set_patient_id(first_session.session_id, first_patient_id)
    session_store.set_patient_id(second_session.session_id, second_patient_id)

    first_response = client.post(
        f"/api/sessions/{first_session.session_id}/identity",
        json={"patient_name": "First", "patient_number": "ab123"},
    )
    assert first_response.status_code == 200, first_response.text

    monkeypatch.setattr(
        registry,
        "patient_number_exists",
        lambda normalized_number, exclude_patient_id=None: False,
    )

    conflict_response = client.post(
        f"/api/sessions/{second_session.session_id}/identity",
        json={"patient_name": "Second", "patient_number": "AB123"},
    )

    assert conflict_response.status_code == 409
    assert conflict_response.json()["detail"] == "PATIENT_NUMBER_ALREADY_EXISTS"


def test_save_patient_identity_rejects_locked_patient(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, registry = identity_app
    session_meta = session_store.create_session(scene="patient")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)

    first_response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "First", "patient_number": "A100"},
    )
    assert first_response.status_code == 200, first_response.text

    second_response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "Second", "patient_number": "B200"},
    )

    assert second_response.status_code == 409
    assert second_response.json()["detail"] == "PATIENT_IDENTITY_LOCKED"


def test_save_patient_identity_requires_patient_session(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, _registry = identity_app
    session_meta = session_store.create_session(scene="doctor")

    response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "Ada", "patient_number": "A100"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "NOT_PATIENT_SESSION"


def test_save_patient_identity_requires_bound_patient(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, _registry = identity_app
    session_meta = session_store.create_session(scene="patient")

    response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "Ada", "patient_number": "A100"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "PATIENT_IDENTITY_NOT_FOUND"


def test_save_patient_identity_rejects_missing_patient_row(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, _registry = identity_app
    session_meta = session_store.create_session(scene="patient", patient_id=99999)

    response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "Ada", "patient_number": "A100"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "PATIENT_IDENTITY_NOT_FOUND"


def test_get_session_includes_patient_identity_when_available(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, registry = identity_app
    session_meta = session_store.create_session(scene="patient")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)

    save_response = client.post(
        f"/api/sessions/{session_meta.session_id}/identity",
        json={"patient_name": "Ada", "patient_number": "A100"},
    )
    assert save_response.status_code == 200, save_response.text

    get_response = client.get(f"/api/sessions/{session_meta.session_id}")

    assert get_response.status_code == 200, get_response.text
    assert get_response.json()["snapshot"]["patient_identity"] == {
        "patient_name": "Ada",
        "patient_number": "A100",
        "identity_locked": True,
    }


def test_get_session_leaves_patient_identity_empty_when_patient_lookup_fails(identity_app: tuple[TestClient, InMemorySessionStore, PatientRegistryService]) -> None:
    client, session_store, _registry = identity_app
    session_meta = session_store.create_session(scene="patient", patient_id=424242)

    response = client.get(f"/api/sessions/{session_meta.session_id}")

    assert response.status_code == 200, response.text
    assert response.json()["snapshot"]["patient_identity"] is None
