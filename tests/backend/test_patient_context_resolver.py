from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from backend.api.services.payload_builder import build_graph_payload
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
            "unrelated": {"keep": True},
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
    assert refreshed["unrelated"] == {"keep": True}


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


def test_resolver_fails_closed_when_projection_json_is_corrupted(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    with registry._connect() as connection:
        connection.execute(
            "UPDATE patient_snapshots SET summary_json = ? WHERE patient_id = ?",
            ("{", patient.patient_id),
        )
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=patient.patient_id)
    resolver = PatientContextResolver(registry, store)

    with pytest.raises(PatientContextStaleError, match="PATIENT_CONTEXT_STALE") as exc_info:
        resolver.resolve(session.session_id)

    assert "projection unavailable" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_resolver_fails_closed_when_projection_json_has_wrong_shape(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    commands = PatientCommandService(registry)
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    with registry._connect() as connection:
        connection.execute(
            "UPDATE patient_snapshots SET active_alerts_json = ? WHERE patient_id = ?",
            ("{}", patient.patient_id),
        )
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=patient.patient_id)
    resolver = PatientContextResolver(registry, store)

    with pytest.raises(PatientContextStaleError, match="PATIENT_CONTEXT_STALE") as exc_info:
        resolver.resolve(session.session_id)

    assert "projection unavailable" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TypeError)


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
    store.merge_context_state(
        session.session_id,
        {"medical_card": {"legacy": True}, "unrelated": {"keep": True}},
    )
    resolver = PatientContextResolver(registry, store)

    context = resolver.resolve(session.session_id)
    context["summary"]["local"] = "mutated"

    context_state = store.get_session(session.session_id).context_state
    cached = context_state["patient_context_cache"]
    assert context is not cached
    assert cached["summary"]["local"] == "cached"
    assert "medical_card" not in context_state
    assert context_state["unrelated"] == {"keep": True}


def test_resolver_refreshes_version_matching_cache_with_malformed_snapshot(tmp_path: Path) -> None:
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
                "patient_version": 1,
                "projection_version": 1,
                "medical_card_snapshot": None,
            },
        },
    )
    resolver = PatientContextResolver(registry, store)

    context = resolver.resolve(session.session_id)

    context_state = store.get_session(session.session_id).context_state
    assert context["medical_card_snapshot"] == {}
    assert context_state["patient_context_cache"]["medical_card_snapshot"] == {}
    assert "medical_card" not in context_state


def test_payload_builder_uses_patient_context_cache_not_legacy_medical_card() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(
        session.session_id,
        {
            "medical_card": {"legacy": True},
            "patient_context_cache": {
                "patient_id": 1,
                "patient_version": 2,
                "projection_version": 2,
                "medical_card_snapshot": {"current": True},
            },
        },
    )

    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="hello")},
        session_meta=store.get_session(session.session_id),
        state_snapshot={},
    )

    assert prepared.payload["medical_card"] == {"current": True}
    assert prepared.payload["patient_context"]["patient_version"] == 2
    assert prepared.payload["patient_context"]["projection_version"] == 2
    assert prepared.payload["patient_context"] is not store.get_session(
        session.session_id
    ).context_state["patient_context_cache"]


def test_payload_builder_ignores_partial_patient_context_cache_without_snapshot() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(
        session.session_id,
        {
            "medical_card": {"legacy": True},
            "patient_context_cache": {
                "patient_id": 1,
                "patient_version": 2,
                "projection_version": 2,
            },
        },
    )

    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="hello")},
        session_meta=store.get_session(session.session_id),
        state_snapshot={},
    )

    assert prepared.payload["medical_card"] == {"legacy": True}
    assert prepared.payload["patient_context"] is None


def test_payload_builder_ignores_patient_context_cache_with_non_mapping_snapshot() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(
        session.session_id,
        {
            "medical_card": {"legacy": True},
            "patient_context_cache": {
                "patient_id": 1,
                "patient_version": 2,
                "projection_version": 2,
                "medical_card_snapshot": None,
            },
        },
    )

    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="hello")},
        session_meta=store.get_session(session.session_id),
        state_snapshot={},
    )

    assert prepared.payload["medical_card"] == {"legacy": True}
    assert prepared.payload["patient_context"] is None


def test_payload_builder_copies_medical_card_snapshot_separately_from_patient_context() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(
        session.session_id,
        {
            "patient_context_cache": {
                "patient_id": 1,
                "patient_version": 2,
                "projection_version": 2,
                "medical_card_snapshot": {"nested": {"current": True}},
            },
        },
    )

    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="hello")},
        session_meta=store.get_session(session.session_id),
        state_snapshot={},
    )
    prepared.payload["medical_card"]["nested"]["current"] = False

    assert prepared.payload["patient_context"]["medical_card_snapshot"] == {
        "nested": {"current": True}
    }
