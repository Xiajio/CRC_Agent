from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.api.services.graph_service import (
    DoctorGraphService,
    GraphService,
    PatientGraphService,
    SceneGraphRouter,
)
from backend.api.services.session_store import InMemorySessionStore
from src.nodes.assessment_nodes import node_doctor_assessment, node_patient_assessment
from src.nodes.node_utils import _invoke_with_streaming
from src.state import CRCAgentState


def _decode_sse_event(payload: str) -> dict[str, object]:
    lines = payload.strip().splitlines()
    data_line = next(line for line in lines if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


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


class FakeStreamingGraph:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

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
    ) -> None:
        self._summary_message = summary_message
        self._alerts = alerts or []
        self.requested_patient_ids: list[int] = []
        self.requested_alert_patient_ids: list[int] = []

    def get_patient_summary_message(self, patient_id: int) -> HumanMessage | None:
        self.requested_patient_ids.append(patient_id)
        return self._summary_message

    def list_patient_alerts(self, patient_id: int) -> list[dict[str, object]]:
        self.requested_alert_patient_ids.append(patient_id)
        return list(self._alerts)


class _UnusedAssessmentModel:
    def with_structured_output(self, _schema):
        def _unexpected_invoke(_payload):
            raise AssertionError("Scene-aware assessment wrappers should not need model execution in this test.")

        return _unexpected_invoke



def make_chat_request(text: str) -> dict[str, object]:
    return {
        "message": HumanMessage(content=text),
    }


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
