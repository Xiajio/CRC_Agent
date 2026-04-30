from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_context_resolver import (
    PatientContextResolver,
    PatientContextStaleError,
)
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def test_resolver_builds_cache_when_missing(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=patient.patient_id)
    resolver = PatientContextResolver(registry, store)

    context = resolver.resolve(session.session_id)

    assert context["patient_id"] == patient.patient_id
    assert context["patient_version"] == 1
    assert context["projection_version"] == 1
    assert (
        store.get_session(session.session_id).context_state["patient_context_cache"][
            "patient_version"
        ]
        == 1
    )


def test_resolver_refreshes_stale_cache(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=patient.patient_id)
    store.merge_context_state(
        session.session_id,
        {
            "medical_card": {"legacy": True},
            "patient_context_cache": {
                "patient_id": patient.patient_id,
                "patient_version": 0,
                "projection_version": 0,
            },
        },
    )
    resolver = PatientContextResolver(registry, store)

    context = resolver.resolve(session.session_id)

    refreshed = store.get_session(session.session_id).context_state
    assert context["patient_version"] == 1
    assert refreshed["patient_context_cache"]["patient_version"] == 1
    assert "medical_card" not in refreshed


def test_resolver_fails_closed_when_projection_missing(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=404)
    resolver = PatientContextResolver(registry, store)

    with pytest.raises(PatientContextStaleError, match="PATIENT_CONTEXT_STALE") as exc_info:
        resolver.resolve(session.session_id)

    assert "projection unavailable" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_resolver_fails_closed_when_session_missing(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    store = InMemorySessionStore()
    resolver = PatientContextResolver(registry, store)

    with pytest.raises(PatientContextStaleError, match="PATIENT_CONTEXT_STALE"):
        resolver.resolve("sess_missing")


def test_resolver_returns_none_without_patient_id(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    store = InMemorySessionStore()
    session = store.create_session(scene="doctor")
    resolver = PatientContextResolver(registry, store)

    assert resolver.resolve(session.session_id) is None


def test_resolver_returns_copy_when_cache_is_fresh(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=patient.patient_id)
    fresh_cache = registry.get_patient_context_projection(patient.patient_id)
    fresh_cache["summary"]["local"] = "cached"
    store.set_patient_context_cache(session.session_id, fresh_cache)
    resolver = PatientContextResolver(registry, store)

    context = resolver.resolve(session.session_id)
    context["summary"]["local"] = "mutated"

    cached = store.get_session(session.session_id).context_state["patient_context_cache"]
    assert context is not cached
    assert cached["summary"]["local"] == "cached"
