from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from backend.api.services.graph_service import GraphService
from backend.api.services.patient_context_resolver import PatientContextStaleError
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from src.state import CRCAgentState


class SnapshottingSessionStore(InMemorySessionStore):
    def get_session(self, session_id: str) -> SessionMeta | None:
        meta = super().get_session(session_id)
        return deepcopy(meta) if meta is not None else None


class CaptureGraph:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def astream(
        self,
        payload: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ):
        del config
        self.payloads.append(dict(payload))
        if False:
            yield {}


class RefreshingResolver:
    def __init__(self, store: InMemorySessionStore) -> None:
        self._store = store
        self.calls: list[str] = []

    def resolve(self, session_id: str) -> dict[str, Any]:
        self.calls.append(session_id)
        cache = {
            "patient_id": 1,
            "patient_version": 3,
            "projection_version": 3,
            "medical_card_snapshot": {"current": True},
        }
        self._store.merge_context_state(
            session_id,
            {
                "medical_card": {"legacy": True},
                "patient_context_cache": cache,
            },
        )
        return dict(cache)


class FailingResolver:
    def resolve(self, session_id: str) -> None:
        del session_id
        raise PatientContextStaleError("PATIENT_CONTEXT_STALE: projection unavailable")


def _compile_patient_context_capture_graph(received_contexts: list[dict[str, Any] | None]):
    def capture_patient_context(state: CRCAgentState) -> dict[str, Any]:
        received_contexts.append(deepcopy(state.patient_context))
        return {}

    builder = StateGraph(CRCAgentState)
    builder.add_node("capture_patient_context", capture_patient_context)
    builder.set_entry_point("capture_patient_context")
    builder.add_edge("capture_patient_context", END)
    return builder.compile()


@pytest.mark.asyncio
async def test_stream_turn_resolves_patient_context_before_payload_build() -> None:
    store = SnapshottingSessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(session.session_id, {"medical_card": {"legacy": True}})
    graph = CaptureGraph()
    resolver = RefreshingResolver(store)
    service = GraphService(
        graph,
        store,
        patient_context_resolver=resolver,
        heartbeat_interval_seconds=0,
    )

    stream = service.stream_turn(
        session.session_id,
        {"message": HumanMessage(content="hello")},
    )
    chunks = [chunk async for chunk in stream]

    assert resolver.calls == [session.session_id]
    assert graph.payloads
    payload = graph.payloads[0]
    assert payload["medical_card"] == {"current": True}
    assert payload["patient_context"]["patient_version"] == 3
    assert payload["patient_context"]["projection_version"] == 3
    assert any("event: done" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_turn_state_graph_node_receives_patient_context() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    store.merge_context_state(
        session.session_id,
        {
            "patient_context_cache": {
                "patient_id": 1,
                "patient_version": 4,
                "projection_version": 4,
                "medical_card_snapshot": {"current": True},
            },
        },
    )
    received_contexts: list[dict[str, Any] | None] = []
    service = GraphService(
        _compile_patient_context_capture_graph(received_contexts),
        store,
        heartbeat_interval_seconds=0,
    )

    stream = service.stream_turn(
        session.session_id,
        {"message": HumanMessage(content="hello")},
    )
    chunks = [chunk async for chunk in stream]

    assert received_contexts == [
        {
            "patient_id": 1,
            "patient_version": 4,
            "projection_version": 4,
            "medical_card_snapshot": {"current": True},
        }
    ]
    assert not any("event: error" in chunk for chunk in chunks)


def test_stream_turn_surfaces_patient_context_resolver_failures() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    service = GraphService(
        CaptureGraph(),
        store,
        patient_context_resolver=FailingResolver(),
        heartbeat_interval_seconds=0,
    )

    with pytest.raises(PatientContextStaleError, match="PATIENT_CONTEXT_STALE"):
        service.stream_turn(
            session.session_id,
            {"message": HumanMessage(content="hello")},
        )
