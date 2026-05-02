from __future__ import annotations

import asyncio
import json
from datetime import datetime
from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.api.services.graph_service import GraphService
from backend.api.services.session_store import InMemorySessionStore
from src.nodes.node_utils import _invoke_with_streaming


def _decode_sse_event(payload: str) -> dict[str, object]:
    lines = payload.strip().splitlines()
    data_line = next(line for line in lines if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def _assert_iso8601(value: object) -> None:
    assert isinstance(value, str)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    datetime.fromisoformat(normalized)


def _json_log_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]


def _assert_artifact_timestamp_fields(artifact: dict[str, object]) -> None:
    for field_name in (
        "server_received_at",
        "graph_started_at",
        "first_byte_at",
        "router_done_at",
        "retrieval_done_at",
        "llm_request_started_at",
        "llm_first_token_at",
        "message_done_at",
        "stream_done_at",
        "server_finished_at",
        "at",
    ):
        assert field_name in artifact


async def collect_sse_events(stream: AsyncIterator[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    async for chunk in stream:
        if chunk.startswith(": ping"):
            continue
        events.append(_decode_sse_event(chunk))
    return events


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _InternalClassifierGraph:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        self.last_payload = payload
        yield {
            "classifier": {
                "messages": [AIMessage(content="[Router] classify request")],
                "retrieval_ms": 12.0,
                "retrieved_references": [
                    {"source": "doc-1", "snippet": "alpha"},
                    {"source": "doc-2", "snippet": "beta"},
                ],
            }
        }
        response = _invoke_with_streaming(
            _StreamingChain(["final ", "visible answer"]),
            {},
            streaming=True,
            show_thinking=False,
            node_name="answer",
        )
        yield {
            "answer": {
                "messages": [response],
            }
        }


class _StreamingAnswerGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        yield {
            "answer": {
                "messages": [
                    AIMessage(content="final only answer"),
                ]
            }
        }


class _IntentRouterGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        del payload, config
        yield {
            "intent_router": {
                "findings": {"user_intent": "general_chat"},
            }
        }
        response = _invoke_with_streaming(
            _ReasoningStreamingChain(),
            {},
            streaming=True,
            show_thinking=False,
            node_name="general_chat",
        )
        yield {
            "general_chat": {
                "messages": [response],
            }
        }


class _BlockedGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        del payload, config
        await asyncio.Event().wait()
        if False:
            yield {}


class _ExplodingGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        del payload, config
        raise RuntimeError("boom")
        if False:
            yield {}


class _DualVisibleMessageGraph:
    def load_state(self, thread_id: str) -> dict[str, object]:
        return {}

    async def astream(self, payload: dict[str, object], config: dict[str, object]) -> AsyncIterator[dict[str, object]]:
        del payload, config
        yield {
            "analysis": {
                "messages": [AIMessage(content="draft answer")],
            }
        }
        yield {
            "answer": {
                "messages": [AIMessage(content="final visible answer")],
            }
        }


class _StreamingChain:
    def __init__(self, parts: list[str]) -> None:
        self._parts = parts

    def stream(self, context: dict[str, object]):
        del context
        for part in self._parts:
            yield AIMessage(content=part)


class _ReasoningStreamingChain:
    def stream(self, context: dict[str, object]):
        del context
        yield AIMessage(content="根据系统提示，我应该先分析。\n\n")
        yield AIMessage(content="最终回复：您好")


@pytest.mark.asyncio
async def test_phase0_logs_summary_for_final_visible_answer_only(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_PERF_LOG", "1")
    monkeypatch.setattr(
        "backend.api.services.chat_latency_trace.perf_counter",
        FakeClock([
            1000.01,
            1000.03,
            1000.05,
            1000.07,
            1000.10,
        ]),
    )

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    graph = _InternalClassifierGraph()
    service = GraphService(
        compiled_graph=graph,
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        events = await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-123",
                    "_latency_request_started_at": 1000.0,
                    "_latency_router_ms": 8.0,
                },
            )
        )

    assert any(event["type"] == "message.done" for event in events)
    summaries = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["trace_id"] == "trace-123"
    assert summary["session_id"] == session.session_id
    assert summary["run_id"].startswith("run_")
    assert summary["graph_path"] == ["classifier", "answer"]
    assert summary["router_ms"] == 8.0
    assert summary["retrieval_ms"] == 12.0
    assert summary["llm_startup_ms"] == 20.0
    assert summary["llm_generation_ms"] == 40.0
    assert summary["server_total_ms"] == 100.0
    assert summary["stream_flush_tail_ms"] == 30.0
    assert summary["server_unaccounted_ms"] == 0.0


@pytest.mark.asyncio
async def test_phase0_summary_is_suppressed_when_logging_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CHAT_PERF_LOG", raising=False)
    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_InternalClassifierGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-123",
                    "_latency_request_started_at": 1000.0,
                    "_latency_router_ms": 8.0,
                },
            )
        )

    assert not [record for record in caplog.records if record.message.startswith("{")]


@pytest.mark.asyncio
async def test_phase0_summary_handles_final_only_turn_without_delta(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_PERF_LOG", "1")
    monkeypatch.setattr(
        "backend.api.services.chat_latency_trace.perf_counter",
        FakeClock([
            2000.01,
            2000.02,
            2000.04,
        ]),
    )

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_StreamingAnswerGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        events = await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-final-only",
                    "_latency_request_started_at": 2000.0,
                    "_latency_router_ms": 5.0,
                },
            )
        )

    assert any(event["type"] == "message.done" for event in events)
    summaries = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["trace_id"] == "trace-final-only"
    assert summary["graph_path"] == ["answer"]
    assert summary["llm_startup_ms"] == 0.0
    assert summary["llm_generation_ms"] == 0.0
    assert summary["stream_flush_tail_ms"] == 10.0
    assert summary["server_unaccounted_ms"] == 5.0


@pytest.mark.asyncio
async def test_phase0_summary_marks_cancelled_turns_as_aborted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_PERF_LOG", "1")
    monkeypatch.setattr(
        "backend.api.services.chat_latency_trace.perf_counter",
        FakeClock([
            3000.01,
            3000.02,
            3000.03,
        ]),
    )

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_BlockedGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        task = asyncio.create_task(
            collect_sse_events(
                service.stream_turn(
                    session.session_id,
                    {
                        "message": HumanMessage(content="hello"),
                        "trace_id": "trace-abort",
                        "_latency_request_started_at": 3000.0,
                        "_latency_router_ms": 2.0,
                    },
                )
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    summaries = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["trace_id"] == "trace-abort"
    assert summary["status"] == "aborted"


@pytest.mark.asyncio
async def test_phase1_trace_summary_tracks_final_visible_answer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    graph = _InternalClassifierGraph()
    service = GraphService(
        compiled_graph=graph,
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {
                "message": HumanMessage(content="hello"),
                "trace_id": "trace-123",
                "model": "mock-model",
            },
        )
    )

    trace_events = [event for event in events if str(event["type"]).startswith("trace.")]
    assert trace_events[0]["type"] == "trace.start"
    assert trace_events[-1]["type"] == "trace.summary"
    start_event = trace_events[0]
    assert start_event["scene"] == "doctor"
    _assert_iso8601(start_event["server_received_at"])
    _assert_iso8601(start_event["graph_started_at"])
    assert start_event["attrs"]["flush_controlled"] is False

    business_types = [event["type"] for event in events if not str(event["type"]).startswith("trace.")]
    assert business_types.index("message.delta") < business_types.index("message.done") < business_types.index("done")

    step_names = [event["name"] for event in trace_events if event["type"] == "trace.step"]
    assert "stream.first_byte" in step_names
    assert "router.done" in step_names
    assert "retrieval.done" in step_names
    assert "llm.request.started" in step_names
    assert "llm.first_token" in step_names
    assert "message.done" in step_names
    assert "stream.done" in step_names
    for event in trace_events:
        if event["type"] == "trace.step":
            _assert_iso8601(event["at"])
            assert isinstance(event["attrs"], dict)

    summary = trace_events[-1]
    assert summary["trace_id"] == "trace-123"
    assert summary["session_id"] == session.session_id
    assert summary["run_id"].startswith("run_")
    _assert_iso8601(summary["at"])
    assert summary["scene"] == "doctor"
    assert summary["graph_path"] == ["classifier", "answer"]
    assert summary["model"] == "mock-model"
    assert summary["has_thinking"] is False
    assert summary["response_chars"] == len("final visible answer")
    assert summary["response_tokens"] is None
    assert summary["tool_calls"] == 0
    assert summary["retrieval_hit_count"] == 2
    assert summary["status"] == "completed"


@pytest.mark.asyncio
async def test_phase1_trace_emits_intent_done_and_llm_first_raw_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_IntentRouterGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {
                "message": HumanMessage(content="hello"),
                "trace_id": "trace-intent-router",
                "model": "mock-model",
            },
        )
    )

    trace_events = [event for event in events if str(event["type"]).startswith("trace.")]
    step_names = [event["name"] for event in trace_events if event["type"] == "trace.step"]
    assert "intent.done" in step_names
    assert "llm.request.started" in step_names
    assert "llm.first_raw_chunk" in step_names
    assert "llm.first_token" in step_names

    intent_done = next(
        event for event in trace_events if event["type"] == "trace.step" and event["name"] == "intent.done"
    )
    llm_request_started = next(
        event for event in trace_events if event["type"] == "trace.step" and event["name"] == "llm.request.started"
    )
    llm_first_raw_chunk = next(
        event for event in trace_events if event["type"] == "trace.step" and event["name"] == "llm.first_raw_chunk"
    )
    llm_first_token = next(
        event for event in trace_events if event["type"] == "trace.step" and event["name"] == "llm.first_token"
    )
    message_done = next(
        event for event in trace_events if event["type"] == "trace.step" and event["name"] == "message.done"
    )

    assert intent_done["node"] == "intent_router"
    assert llm_request_started["node"] == "general_chat"
    assert llm_first_raw_chunk["node"] == "general_chat"
    assert llm_first_token["node"] == "general_chat"
    assert trace_events.index(intent_done) < trace_events.index(llm_request_started)
    assert trace_events.index(llm_request_started) < trace_events.index(llm_first_raw_chunk)
    assert trace_events.index(llm_first_raw_chunk) < trace_events.index(llm_first_token)
    assert trace_events.index(llm_first_token) < trace_events.index(message_done)


@pytest.mark.asyncio
async def test_phase1_durable_artifact_emits_completed_json_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_InternalClassifierGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-artifact-completed",
                    "model": "mock-model",
                },
            )
        )

    artifacts = [entry for entry in _json_log_lines(caplog) if entry.get("event") == "chat_latency.phase1.artifact"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["trace_id"] == "trace-artifact-completed"
    assert artifact["session_id"] == session.session_id
    assert artifact["run_id"].startswith("run_")
    assert artifact["scene"] == "doctor"
    assert artifact["graph_path"] == ["classifier", "answer"]
    _assert_artifact_timestamp_fields(artifact)
    _assert_iso8601(artifact["server_received_at"])
    _assert_iso8601(artifact["graph_started_at"])
    _assert_iso8601(artifact["first_byte_at"])
    _assert_iso8601(artifact["router_done_at"])
    _assert_iso8601(artifact["retrieval_done_at"])
    _assert_iso8601(artifact["llm_request_started_at"])
    _assert_iso8601(artifact["llm_first_token_at"])
    _assert_iso8601(artifact["message_done_at"])
    _assert_iso8601(artifact["stream_done_at"])
    _assert_iso8601(artifact["server_finished_at"])
    _assert_iso8601(artifact["at"])
    assert artifact["response_chars"] == len("final visible answer")
    assert artifact["response_tokens"] is None
    assert artifact["status"] == "completed"
    assert artifact["attrs"]["flush_controlled"] is False


@pytest.mark.asyncio
async def test_phase1_durable_artifact_emits_error_json_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_ExplodingGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-artifact-error",
                    "model": "mock-model",
                },
            )
        )

    artifacts = [entry for entry in _json_log_lines(caplog) if entry.get("event") == "chat_latency.phase1.artifact"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["trace_id"] == "trace-artifact-error"
    assert artifact["session_id"] == session.session_id
    assert artifact["scene"] == "doctor"
    assert artifact["status"] == "error"
    _assert_artifact_timestamp_fields(artifact)
    _assert_iso8601(artifact["server_received_at"])
    _assert_iso8601(artifact["graph_started_at"])
    assert artifact["first_byte_at"] is None
    assert artifact["router_done_at"] is None
    assert artifact["retrieval_done_at"] is None
    assert artifact["llm_request_started_at"] is None
    assert artifact["llm_first_token_at"] is None
    assert artifact["message_done_at"] is None
    assert artifact["stream_done_at"] is None
    _assert_iso8601(artifact["server_finished_at"])
    assert artifact["response_chars"] == 0
    assert artifact["response_tokens"] is None


@pytest.mark.asyncio
async def test_phase1_durable_artifact_emits_aborted_json_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_BlockedGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        task = asyncio.create_task(
            collect_sse_events(
                service.stream_turn(
                    session.session_id,
                    {
                        "message": HumanMessage(content="hello"),
                        "trace_id": "trace-artifact-abort",
                        "model": "mock-model",
                    },
                )
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    artifacts = [entry for entry in _json_log_lines(caplog) if entry.get("event") == "chat_latency.phase1.artifact"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["trace_id"] == "trace-artifact-abort"
    assert artifact["session_id"] == session.session_id
    assert artifact["scene"] == "doctor"
    assert artifact["status"] == "aborted"
    _assert_artifact_timestamp_fields(artifact)
    _assert_iso8601(artifact["server_received_at"])
    _assert_iso8601(artifact["graph_started_at"])
    assert artifact["first_byte_at"] is None
    assert artifact["router_done_at"] is None
    assert artifact["retrieval_done_at"] is None
    assert artifact["llm_request_started_at"] is None
    assert artifact["llm_first_token_at"] is None
    assert artifact["message_done_at"] is None
    assert artifact["stream_done_at"] is None
    _assert_iso8601(artifact["server_finished_at"])


@pytest.mark.asyncio
async def test_phase1_trace_summary_handles_final_only_turn_without_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_StreamingAnswerGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {
                "message": HumanMessage(content="hello"),
                "trace_id": "trace-final-only",
                "model": "mock-model",
            },
        )
    )

    trace_events = [event for event in events if str(event["type"]).startswith("trace.")]
    assert trace_events[0]["type"] == "trace.start"
    assert trace_events[-1]["type"] == "trace.summary"
    _assert_iso8601(trace_events[0]["server_received_at"])
    _assert_iso8601(trace_events[0]["graph_started_at"])
    assert all(event["type"] != "message.delta" for event in events)

    summary = trace_events[-1]
    assert summary["trace_id"] == "trace-final-only"
    _assert_iso8601(summary["at"])
    assert summary["graph_path"] == ["answer"]
    assert summary["model"] == "mock-model"
    assert summary["response_chars"] == len("final only answer")
    assert summary["response_tokens"] is None
    assert summary["tool_calls"] == 0
    assert summary["status"] == "completed"


@pytest.mark.asyncio
async def test_phase1_trace_summary_uses_last_visible_message_when_multiple_are_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_DualVisibleMessageGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {
                "message": HumanMessage(content="hello"),
                "trace_id": "trace-multi-visible",
                "model": "mock-model",
            },
        )
    )

    trace_events = [event for event in events if str(event["type"]).startswith("trace.")]
    message_done_steps = [event for event in trace_events if event["type"] == "trace.step" and event["name"] == "message.done"]
    assert len(message_done_steps) == 2

    summary = trace_events[-1]
    assert summary["trace_id"] == "trace-multi-visible"
    assert summary["response_chars"] == len("final visible answer")
    assert summary["status"] == "completed"


@pytest.mark.asyncio
async def test_phase1_trace_emits_error_step_and_error_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_LATENCY_TRACE", "1")

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_ExplodingGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    events = await collect_sse_events(
        service.stream_turn(
            session.session_id,
            {
                "message": HumanMessage(content="hello"),
                "trace_id": "trace-error",
                "model": "mock-model",
            },
        )
    )

    trace_events = [event for event in events if str(event["type"]).startswith("trace.")]
    assert trace_events[0]["type"] == "trace.start"
    error_step = next(event for event in trace_events if event["type"] == "trace.step" and event["name"] == "error")
    _assert_iso8601(error_step["at"])
    assert error_step["attrs"]["model"] == "mock-model"
    assert trace_events[-1]["type"] == "trace.summary"
    assert trace_events[-1]["status"] == "error"
    _assert_iso8601(trace_events[-1]["at"])


@pytest.mark.asyncio
async def test_phase1_trace_is_suppressed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CHAT_LATENCY_TRACE", raising=False)

    session_store = InMemorySessionStore()
    session = session_store.create_session()
    service = GraphService(
        compiled_graph=_InternalClassifierGraph(),
        session_store=session_store,
        heartbeat_interval_seconds=0,
    )

    with caplog.at_level("INFO", logger="backend.api.services.chat_latency_trace"):
        events = await collect_sse_events(
            service.stream_turn(
                session.session_id,
                {
                    "message": HumanMessage(content="hello"),
                    "trace_id": "trace-off",
                    "model": "mock-model",
                },
            )
        )

    assert all(not str(event["type"]).startswith("trace.") for event in events)
    assert not [entry for entry in _json_log_lines(caplog) if entry.get("event") == "chat_latency.phase1.artifact"]
