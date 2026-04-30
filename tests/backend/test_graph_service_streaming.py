from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from backend.api.services.graph_service import (
    DoctorGraphService,
    GraphService,
    PatientGraphService,
    SceneGraphRouter,
)
from backend.api.services.patient_context_resolver import PatientContextStaleError
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from src.nodes.assessment_nodes import node_doctor_assessment, node_patient_assessment
from src.nodes.node_utils import _invoke_with_streaming
from src.state import CRCAgentState


def _decode_sse_event(payload: str) -> dict[str, object]:
    lines = payload.strip().splitlines()
    data_line = next(line for line in lines if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def _assert_iso8601(value: object) -> None:
    assert isinstance(value, str)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    datetime.fromisoformat(normalized)


class _StreamingChain:
    def __init__(self, parts: list[str], *, pause_seconds: float = 0.0) -> None:
        self._parts = parts
        self._pause_seconds = pause_seconds

    def stream(self, context: dict):
        for part in self._parts:
            if self._pause_seconds > 0:
                time.sleep(self._pause_seconds)
            yield AIMessage(content=part)


class FakeGraph:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        self.last_payload = payload
        yield {"general": {"messages": [AIMessage(content="ok")]}}


class SnapshottingSessionStore(InMemorySessionStore):
    def get_session(self, session_id: str) -> SessionMeta | None:
        meta = super().get_session(session_id)
        return deepcopy(meta) if meta is not None else None


class CaptureGraph:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

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

    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        self.last_payload = payload
        response = _invoke_with_streaming(
            _StreamingChain(["Hello ", "world"]),
            {},
            streaming=True,
            show_thinking=False,
        )
        yield {"general": {"messages": [response]}}


class _BrokenState:
    def __getattr__(self, name: str) -> object:
        raise ValueError(f"broken state access: {name}")


class _BrokenStateGraphService(GraphService):
    def load_agent_state(self, session_id: str) -> object:
        del session_id
        return _BrokenState()


class _ConcurrentCompiledGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        current_turn = payload["messages"][-1]
        assert isinstance(current_turn, HumanMessage)
        marker = current_turn.content

        response = await asyncio.to_thread(
            _invoke_with_streaming,
            _StreamingChain([f"{marker}-1", f"{marker}-2"], pause_seconds=0.02),
            {},
            True,
            False,
        )
        yield {"general": {"messages": [response]}}


class FakePatientRegistry:
    def __init__(
        self,
        summary_message: HumanMessage | None = None,
        alerts: list[dict[str, object]] | None = None,
        *,
        patient_version: int | None = None,
    ) -> None:
        self._summary_message = summary_message
        self._alerts = alerts or []
        self.patient_version = patient_version
        self.requested_patient_ids: list[int] = []
        self.requested_alert_patient_ids: list[int] = []
        self.fail_projection = False

    def get_patient_summary_message(self, patient_id: int) -> HumanMessage | None:
        self.requested_patient_ids.append(patient_id)
        return self._summary_message

    def list_patient_alerts(self, patient_id: int) -> list[dict[str, Any]]:
        self.requested_alert_patient_ids.append(patient_id)
        return list(self._alerts)

    def get_patient_context_projection(self, patient_id: int) -> dict[str, Any] | None:
        if self.fail_projection:
            raise RuntimeError("projection unavailable")
        if self.patient_version is None:
            return None
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


class _UnusedAssessmentModel:
    def with_structured_output(self, _schema):
        def _unexpected_invoke(_payload):
            raise AssertionError("Scene-aware assessment wrappers should not need model execution in this test.")

        return _unexpected_invoke



def make_chat_request(text: str, *, trace_id: str | None = None) -> dict[str, object]:
    request: dict[str, object] = {
        "message": HumanMessage(content=text),
    }
    if trace_id is not None:
        request["trace_id"] = trace_id
    return request


async def collect_sse_events(stream: AsyncIterator[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    async for chunk in stream:
        if chunk.startswith(": ping"):
            continue
        events.append(_decode_sse_event(chunk))
    return events


@pytest.mark.asyncio
async def test_stream_turn_emits_message_delta_before_matching_message_done() -> None:
    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=FakeStreamingGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(service.stream_turn(session.session_id, make_chat_request("hello")))

    delta_events = [event for event in events if event["type"] == "message.delta"]
    done_event = next(event for event in events if event["type"] == "message.done")

    assert [event["delta"] for event in delta_events] == ["Hello ", "world"]
    assert all(event["message_id"] == done_event["message_id"] for event in delta_events)
    assert events.index(delta_events[0]) < events.index(done_event)


@pytest.mark.asyncio
async def test_stream_turn_preserves_trace_id_in_graph_payload() -> None:
    session_store = InMemorySessionStore()
    session = session_store.create_session()
    graph = FakeGraph()
    service = GraphService(
        compiled_graph=graph,
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(
        service.stream_turn(
            session.session_id,
            make_chat_request("hello", trace_id="trace-123"),
        )
    )

    assert graph.last_payload is not None
    assert graph.last_payload["trace_id"] == "trace-123"


@pytest.mark.asyncio
async def test_stream_turn_restores_pending_context_messages_when_payload_build_fails() -> None:
    session_store = InMemorySessionStore()
    session = session_store.create_session()
    pending_message = HumanMessage(content="bound patient summary")
    session_store.enqueue_context_message(session.session_id, pending_message)
    service = _BrokenStateGraphService(
        compiled_graph=FakeGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with pytest.raises(ValueError, match="broken state access: patient_profile"):
        service.stream_turn(session.session_id, make_chat_request("hello"))

    assert session_store.get_session(session.session_id).pending_context_messages == [pending_message]


@pytest.mark.asyncio
async def test_stream_turn_emits_trace_start_before_business_events_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=FakeStreamingGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            make_chat_request("hello", trace_id="trace-123"),
        )
    )

    assert events[0]["type"] == "trace.start"
    assert events[0]["scene"] == "doctor"
    _assert_iso8601(events[0]["server_received_at"])
    _assert_iso8601(events[0]["graph_started_at"])
    assert events[0]["attrs"]["flush_controlled"] is False
    assert events[-1]["type"] == "trace.summary"
    _assert_iso8601(events[-1]["at"])
    business_types = [event["type"] for event in events if not str(event["type"]).startswith("trace.")]
    assert business_types.index("message.delta") < business_types.index("message.done") < business_types.index("done")


@pytest.mark.asyncio
async def test_stream_turn_keeps_request_scoped_stream_callbacks_isolated() -> None:
    session_store = InMemorySessionStore()
    first = session_store.create_session()
    second = session_store.create_session()
    service = GraphService(
        compiled_graph=_ConcurrentCompiledGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    first_events, second_events = await asyncio.gather(
        collect_sse_events(service.stream_turn(first.session_id, make_chat_request("alpha"))),
        collect_sse_events(service.stream_turn(second.session_id, make_chat_request("beta"))),
    )

    first_deltas = [event["delta"] for event in first_events if event["type"] == "message.delta"]
    second_deltas = [event["delta"] for event in second_events if event["type"] == "message.delta"]

    assert first_deltas == ["alpha-1", "alpha-2"]
    assert second_deltas == ["beta-1", "beta-2"]


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

    chunks = [
        chunk
        async for chunk in service.stream_turn(
            session.session_id,
            {"message": HumanMessage(content="hello")},
        )
    ]

    assert resolver.calls == [session.session_id]
    assert graph.payloads
    payload = graph.payloads[0]
    assert payload["medical_card"] == {"current": True}
    assert payload["patient_context"]["patient_version"] == 3
    assert payload["patient_context"]["projection_version"] == 3
    done_event = next(_decode_sse_event(chunk) for chunk in chunks if "event: done" in chunk)
    assert done_event["snapshot_version"] == 1


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


def test_scene_router_returns_patient_service_for_patient_session() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=10)
    router = SceneGraphRouter(
        patient_service=PatientGraphService(compiled_graph=FakeGraph(), session_store=session_store),
        doctor_service=DoctorGraphService(compiled_graph=FakeGraph(), session_store=session_store),
        session_store=session_store,
    )

    service = router.for_session(meta.session_id)

    assert isinstance(service, PatientGraphService)


@pytest.mark.asyncio
async def test_patient_graph_service_never_emits_context_maintenance_running() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=10)
    service = PatientGraphService(
        compiled_graph=FakeStreamingGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("hello")))

    assert all(event["type"] != "context.maintenance" for event in events)


@pytest.mark.asyncio
async def test_doctor_graph_service_injects_patient_summary_when_patient_is_newly_bound() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(summary_message=HumanMessage(content="patient_id=33 summary"))
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next")))

    assert registry.requested_patient_ids == [33]
    assert graph.last_payload is not None
    payload_messages = graph.last_payload["messages"]
    assert isinstance(payload_messages, list)
    assert any(isinstance(message, HumanMessage) and message.content == "patient_id=33 summary" for message in payload_messages)
    assert session_store.get_session(meta.session_id).context_state.get("bound_patient_id") == 33


@pytest.mark.asyncio
async def test_doctor_graph_service_does_not_reinject_patient_summary_when_already_bound() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(summary_message=HumanMessage(content="patient_id=33 summary"))
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("hello")))
    session_store.merge_context_state(meta.session_id, {"bound_patient_id": 33})
    registry.requested_patient_ids.clear()

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next")))

    assert registry.requested_patient_ids == []
    assert graph.last_payload is not None
    payload_messages = graph.last_payload["messages"]
    assert isinstance(payload_messages, list)
    assert not any(isinstance(message, HumanMessage) and message.content == "patient_id=33 summary" for message in payload_messages[:-1])


@pytest.mark.asyncio
async def test_doctor_graph_service_injects_registry_summary_with_alerts() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    graph = FakeStreamingGraph()
    registry = FakePatientRegistry(
        summary_message=HumanMessage(content="Bound patient summary: patient_id=33, clinical_stage=cT3N1M0."),
        alerts=[
            {
                "kind": "conflict_detected",
                "message": "Conflict detected on mmr_status.",
                "record_id": 9,
            }
        ],
    )
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=registry,
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next")))

    assert registry.requested_patient_ids == [33]
    assert registry.requested_alert_patient_ids == [33]
    assert graph.last_payload is not None
    payload_messages = graph.last_payload["messages"]
    assert isinstance(payload_messages, list)
    assert any(
        isinstance(message, HumanMessage) and "conflict_detected" in message.content
        for message in payload_messages
    )


def test_scene_specific_assessment_wrappers_diverge_on_the_same_symptom_input() -> None:
    patient_assessment = node_patient_assessment(
        model=_UnusedAssessmentModel(),
        tools=[],
        show_thinking=False,
    )
    doctor_assessment = node_doctor_assessment(
        model=_UnusedAssessmentModel(),
        tools=[],
        show_thinking=False,
    )

    patient_result = patient_assessment(
        CRCAgentState(
            messages=[HumanMessage(content="我最近有点腹痛")],
            findings={"user_intent": "clinical_assessment"},
        )
    )
    doctor_result = doctor_assessment(
        CRCAgentState(
            messages=[HumanMessage(content="我最近有点腹痛")],
            findings={"user_intent": "clinical_assessment"},
        )
    )

    assert patient_result["findings"]["inquiry_type"] == "symptom_inquiry"
    assert doctor_result["findings"]["inquiry_type"] == "pathology_required"
