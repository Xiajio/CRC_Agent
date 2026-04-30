from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from backend.api.services.graph_service import DoctorGraphService, GraphService
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


class FakeStreamingGraph:
    def __init__(self) -> None:
        self.last_payload: dict[str, Any] = {}

    async def astream(
        self,
        payload: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ):
        del config
        self.last_payload = dict(payload)
        if False:
            yield {}


class FakePatientRegistry:
    def __init__(
        self,
        *,
        summary_message: HumanMessage,
        patient_version: int,
    ) -> None:
        self._summary_message = summary_message
        self.patient_version = patient_version
        self.requested_patient_ids: list[int] = []
        self.requested_alert_patient_ids: list[int] = []
        self.fail_projection = False

    def get_patient_summary_message(self, patient_id: int) -> HumanMessage:
        self.requested_patient_ids.append(patient_id)
        return self._summary_message

    def list_patient_alerts(self, patient_id: int) -> list[dict[str, Any]]:
        self.requested_alert_patient_ids.append(patient_id)
        return []

    def get_patient_context_projection(self, patient_id: int) -> dict[str, Any]:
        if self.fail_projection:
            raise RuntimeError("projection unavailable")
        return {
            "patient_id": patient_id,
            "patient_version": self.patient_version,
            "projection_version": self.patient_version,
            "medical_card_snapshot": {},
        }


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


def make_chat_request(message: str) -> dict[str, Any]:
    return {"message": HumanMessage(content=message)}


async def collect_sse_events(stream) -> list[str]:
    return [chunk async for chunk in stream]


def parse_sse_payloads(chunks: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        data_line = next(
            (line for line in chunk.splitlines() if line.startswith("data: ")),
            None,
        )
        if data_line is not None:
            payloads.append(json.loads(data_line.removeprefix("data: ")))
    return payloads


def find_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    return next(event for event in events if event["type"] == event_type)


@pytest.mark.asyncio
async def test_done_event_includes_patient_version_used_for_bound_patient() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(
        summary_message=HumanMessage(content="patient v3"),
        patient_version=3,
    )
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    chunks = await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next")))
    done = find_event(parse_sse_payloads(chunks), "done")

    assert done["patient_version_used"] == 3
    assert done["patient_context_stale"] is False


@pytest.mark.asyncio
async def test_doctor_graph_service_reinjects_when_patient_version_changes() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(
        summary_message=HumanMessage(content="patient v1"),
        patient_version=1,
    )
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("first")))
    first_payload_messages = graph.last_payload["messages"]
    assert any(
        isinstance(message, HumanMessage) and "patient v1" in message.content
        for message in first_payload_messages
    )
    first_context_state = session_store.get_session(meta.session_id).context_state
    assert first_context_state["last_injected_patient_version"] == 1

    registry._summary_message = HumanMessage(content="patient v2")
    registry.patient_version = 2
    registry.requested_patient_ids.clear()

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("second")))

    payload_messages = graph.last_payload["messages"]
    assert registry.requested_patient_ids == [33]
    assert any(
        isinstance(message, HumanMessage) and "patient v2" in message.content
        for message in payload_messages
    )
    assert any(
        isinstance(message, HumanMessage) and "Patient version: 2." in message.content
        for message in payload_messages
    )
    context_state = session_store.get_session(meta.session_id).context_state
    assert context_state["last_injected_patient_version"] == 2


@pytest.mark.asyncio
async def test_doctor_graph_service_skips_reinjection_when_projection_fails_after_versioned_injection() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(
        summary_message=HumanMessage(content="patient v1"),
        patient_version=1,
    )
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("first")))
    registry.fail_projection = True
    registry._summary_message = HumanMessage(content="unexpected reinjection")
    registry.requested_patient_ids.clear()
    registry.requested_alert_patient_ids.clear()

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("second")))

    assert registry.requested_patient_ids == []
    assert registry.requested_alert_patient_ids == []
    payload_messages = graph.last_payload["messages"]
    assert not any(
        isinstance(message, HumanMessage) and "unexpected reinjection" in message.content
        for message in payload_messages
    )
    context_state = session_store.get_session(meta.session_id).context_state
    assert context_state["last_injected_patient_version"] == 1
    assert context_state["bound_patient_version"] == 1


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
    done = find_event(parse_sse_payloads(chunks), "done")
    assert done["patient_version_used"] == 3
    assert done["patient_context_stale"] is False


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


@pytest.mark.asyncio
async def test_stream_turn_surfaces_patient_context_resolver_failures() -> None:
    store = InMemorySessionStore()
    session = store.create_session(scene="patient", patient_id=1)
    graph = CaptureGraph()
    service = GraphService(
        graph,
        store,
        patient_context_resolver=FailingResolver(),
        heartbeat_interval_seconds=0,
    )

    chunks = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {"message": HumanMessage(content="hello")},
        )
    )
    events = parse_sse_payloads(chunks)
    error = find_event(events, "error")
    done = find_event(events, "done")

    assert error["code"] == "PATIENT_CONTEXT_STALE"
    assert "projection unavailable" in error["message"]
    assert done["patient_context_stale"] is True
    assert done["patient_version_used"] is None
    assert graph.payloads == []
    assert store.get_session(session.session_id).active_run_id is None
