from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage

from backend.api.schemas.events import TraceStartEvent, TraceStepEvent, TraceSummaryEvent


LOGGER = logging.getLogger(__name__)


def phase0_logging_enabled() -> bool:
    return os.getenv("CHAT_PERF_LOG") == "1"


def phase1_tracing_enabled() -> bool:
    return os.getenv("CHAT_LATENCY_TRACE") == "1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class _NodeTiming:
    started_at: float | None = None
    first_delta_at: float | None = None
    ended_at: float | None = None
    message_done_at: float | None = None


@dataclass(slots=True)
class Phase0ChatLatencyTrace:
    trace_id: str | None
    run_id: str
    session_id: str
    request_started_at: float
    router_ms: float = 0.0
    retrieval_ms: float = 0.0
    graph_path: list[str] = field(default_factory=list)
    _node_timings: dict[str, _NodeTiming] = field(default_factory=dict, repr=False)
    _visible_node_name: str | None = field(default=None, repr=False)

    def _node_timing(self, node_name: str) -> _NodeTiming:
        timing = self._node_timings.get(node_name)
        if timing is None:
            timing = _NodeTiming()
            self._node_timings[node_name] = timing
        return timing

    def record_router_ms(self, router_ms: float | int | None) -> None:
        if router_ms is None:
            return
        self.router_ms = max(float(router_ms), 0.0)

    def record_retrieval_ms(self, retrieval_ms: float | int | None) -> None:
        if retrieval_ms is None:
            return
        self.retrieval_ms += max(float(retrieval_ms), 0.0)

    def record_node_seen(self, node_name: str | None) -> None:
        if not node_name:
            return
        if not self.graph_path or self.graph_path[-1] != node_name:
            self.graph_path.append(node_name)

    def record_node_start(self, node_name: str | None, at: float | None = None) -> None:
        if not node_name:
            return
        timing = self._node_timing(node_name)
        if timing.started_at is None:
            timing.started_at = perf_counter() if at is None else at
        self.record_node_seen(node_name)

    def record_node_delta(self, node_name: str | None, at: float | None = None) -> None:
        if not node_name:
            return
        timing = self._node_timing(node_name)
        if timing.first_delta_at is None:
            timing.first_delta_at = perf_counter() if at is None else at
        self.record_node_seen(node_name)

    def record_node_end(self, node_name: str | None, at: float | None = None) -> None:
        if not node_name:
            return
        timing = self._node_timing(node_name)
        if timing.ended_at is None:
            timing.ended_at = perf_counter() if at is None else at
        self.record_node_seen(node_name)

    def record_visible_message_done(self, node_name: str | None, at: float | None = None) -> None:
        if not node_name:
            return
        timing = self._node_timing(node_name)
        current_at = perf_counter() if at is None else at
        if timing.started_at is None:
            timing.started_at = current_at
        if timing.message_done_at is None:
            timing.message_done_at = current_at
        self._visible_node_name = node_name
        self.record_node_seen(node_name)

    def _visible_breakdown(self) -> tuple[float, float]:
        if not self._visible_node_name:
            return 0.0, 0.0

        timing = self._node_timings.get(self._visible_node_name)
        if timing is None:
            return 0.0, 0.0

        started_at = timing.started_at
        first_delta_at = timing.first_delta_at
        finished_at = timing.message_done_at or timing.ended_at or started_at

        if started_at is None or finished_at is None:
            return 0.0, 0.0

        if first_delta_at is not None and first_delta_at >= started_at:
            llm_startup_ms = max((first_delta_at - started_at) * 1000.0, 0.0)
            llm_generation_ms = max((finished_at - first_delta_at) * 1000.0, 0.0)
            return round(llm_startup_ms, 3), round(llm_generation_ms, 3)

        llm_startup_ms = 0.0
        llm_generation_ms = max((finished_at - started_at) * 1000.0, 0.0)
        return round(llm_startup_ms, 3), round(llm_generation_ms, 3)

    def build_summary(self, *, status: str, finished_at: float | None = None) -> dict[str, Any]:
        end_at = perf_counter() if finished_at is None else finished_at
        llm_startup_ms, llm_generation_ms = self._visible_breakdown()
        server_total_ms = max((end_at - self.request_started_at) * 1000.0, 0.0)
        visible_done_at = None
        if self._visible_node_name:
            timing = self._node_timings.get(self._visible_node_name)
            if timing is not None:
                visible_done_at = timing.message_done_at or timing.ended_at

        stream_flush_tail_ms = 0.0
        if visible_done_at is not None:
            stream_flush_tail_ms = max((end_at - visible_done_at) * 1000.0, 0.0)

        accounted_ms = self.router_ms + self.retrieval_ms + llm_startup_ms + llm_generation_ms + stream_flush_tail_ms
        server_unaccounted_ms = max(server_total_ms - accounted_ms, 0.0)

        return {
            "event": "chat_latency.phase0.summary",
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": status,
            "graph_path": list(self.graph_path),
            "router_ms": round(self.router_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "llm_startup_ms": llm_startup_ms,
            "llm_generation_ms": llm_generation_ms,
            "stream_flush_tail_ms": round(stream_flush_tail_ms, 3),
            "server_unaccounted_ms": round(server_unaccounted_ms, 3),
            "server_total_ms": round(server_total_ms, 3),
        }

    def log_summary(self, *, status: str, finished_at: float | None = None) -> dict[str, Any] | None:
        if not phase0_logging_enabled():
            return None

        summary = self.build_summary(status=status, finished_at=finished_at)
        LOGGER.info(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        return summary


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, BaseMessage):
        return _message_content(value.content)
    return str(value or "")


def _message_tool_call_count(messages: Any) -> int:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return 0

    for item in reversed(messages):
        if isinstance(item, AIMessage):
            tool_calls = getattr(item, "tool_calls", None)
            if isinstance(tool_calls, list):
                return sum(1 for tool_call in tool_calls if isinstance(tool_call, Mapping))
    return 0


@dataclass(slots=True)
class Phase1ChatLatencyTrace:
    trace_id: str | None
    run_id: str
    session_id: str
    scene: str | None = None
    server_received_at: str = field(default_factory=_iso_now)
    graph_started_at: str | None = None
    model: str | None = None
    flush_controlled: bool = False
    first_byte_at: str | None = None
    router_done_at: str | None = None
    retrieval_done_at: str | None = None
    llm_request_started_at: str | None = None
    llm_first_raw_chunk_at: str | None = None
    llm_first_token_at: str | None = None
    message_done_at: str | None = None
    stream_done_at: str | None = None
    graph_path: list[str] = field(default_factory=list)
    response_chars: int = 0
    tool_calls: int = 0
    retrieval_hit_count: int = 0
    response_tokens: int | None = None
    has_thinking: bool = False
    _stream_first_byte_emitted: bool = field(default=False, repr=False)
    _intent_done_emitted: bool = field(default=False, repr=False)
    _llm_request_started_nodes: set[str] = field(default_factory=set, repr=False)
    _llm_first_raw_chunk_nodes: set[str] = field(default_factory=set, repr=False)
    _llm_first_token_nodes: set[str] = field(default_factory=set, repr=False)
    _router_done_emitted: bool = field(default=False, repr=False)
    _retrieval_done_emitted: bool = field(default=False, repr=False)
    _message_done_emitted: bool = field(default=False, repr=False)
    _stream_done_emitted: bool = field(default=False, repr=False)

    def record_node_seen(self, node_name: str | None) -> None:
        if not node_name:
            return
        if not self.graph_path or self.graph_path[-1] != node_name:
            self.graph_path.append(node_name)

    def _step_event(
        self,
        name: str,
        node_name: str | None = None,
        *,
        at: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> TraceStepEvent:
        event_attrs = dict(attrs or {})
        if self.model is not None and "model" not in event_attrs:
            event_attrs["model"] = self.model
        if node_name is not None and "node" not in event_attrs:
            event_attrs["node"] = node_name
        if "has_thinking" not in event_attrs:
            event_attrs["has_thinking"] = self.has_thinking
        return TraceStepEvent(
            trace_id=self.trace_id,
            run_id=self.run_id,
            session_id=self.session_id,
            name=name,
            at=at or _iso_now(),
            node=node_name,
            model=self.model,
            attrs=event_attrs,
        )

    def record_stream_first_byte(self, *, at: str | None = None) -> TraceStepEvent | None:
        if self._stream_first_byte_emitted:
            return None
        self._stream_first_byte_emitted = True
        self.first_byte_at = at or _iso_now()
        return self._step_event("stream.first_byte", at=self.first_byte_at)

    def record_llm_request_started(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if not node_name or node_name in self._llm_request_started_nodes:
            return None
        self._llm_request_started_nodes.add(node_name)
        self.llm_request_started_at = at or _iso_now()
        return self._step_event("llm.request.started", node_name=node_name, at=self.llm_request_started_at)

    def record_intent_done(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if self._intent_done_emitted:
            return None
        self._intent_done_emitted = True
        current_at = at or _iso_now()
        return self._step_event("intent.done", node_name=node_name, at=current_at)

    def record_llm_first_raw_chunk(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if not node_name or node_name in self._llm_first_raw_chunk_nodes:
            return None
        self._llm_first_raw_chunk_nodes.add(node_name)
        self.llm_first_raw_chunk_at = at or _iso_now()
        return self._step_event("llm.first_raw_chunk", node_name=node_name, at=self.llm_first_raw_chunk_at)

    def record_llm_first_token(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if not node_name or node_name in self._llm_first_token_nodes:
            return None
        self._llm_first_token_nodes.add(node_name)
        self.llm_first_token_at = at or _iso_now()
        return self._step_event("llm.first_token", node_name=node_name, at=self.llm_first_token_at)

    def record_router_done(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if self._router_done_emitted:
            return None
        self._router_done_emitted = True
        self.router_done_at = at or _iso_now()
        return self._step_event("router.done", node_name=node_name, at=self.router_done_at)

    def record_retrieval_done(self, node_name: str | None, *, at: str | None = None) -> TraceStepEvent | None:
        if self._retrieval_done_emitted:
            return None
        self._retrieval_done_emitted = True
        self.retrieval_done_at = at or _iso_now()
        return self._step_event("retrieval.done", node_name=node_name, at=self.retrieval_done_at)

    def record_message_done(
        self,
        *,
        node_name: str | None,
        message: Any,
        node_output: Mapping[str, Any] | None = None,
        at: str | None = None,
    ) -> TraceStepEvent | None:
        self._message_done_emitted = True
        self.record_node_seen(node_name)
        self.message_done_at = at or _iso_now()
        self.response_chars = len(_message_content(getattr(message, "content", "")))
        thinking = getattr(message, "thinking", None)
        self.has_thinking = bool(isinstance(thinking, str) and thinking.strip())
        self.tool_calls = _message_tool_call_count(node_output.get("messages") if isinstance(node_output, Mapping) else None)
        return self._step_event("message.done", node_name=node_name, at=self.message_done_at)

    def record_stream_done(self, *, at: str | None = None) -> TraceStepEvent | None:
        if self._stream_done_emitted:
            return None
        self._stream_done_emitted = True
        self.stream_done_at = at or _iso_now()
        return self._step_event("stream.done", at=self.stream_done_at)

    def record_error(self, *, at: str | None = None) -> TraceStepEvent:
        return self._step_event("error", at=at)

    def record_node_output(
        self,
        node_name: str | None,
        node_output: Mapping[str, Any],
        *,
        at: str | None = None,
    ) -> list[TraceStepEvent]:
        events: list[TraceStepEvent] = []
        if node_name:
            self.record_node_seen(node_name)

        current_at = at or _iso_now()
        if node_name == "intent_router":
            intent_done_event = self.record_intent_done(node_name, at=current_at)
            if intent_done_event is not None:
                events.append(intent_done_event)
        first_byte_event = self.record_stream_first_byte(at=current_at)
        if first_byte_event is not None:
            events.append(first_byte_event)

        retrieval_hit_count = node_output.get("retrieval_hit_count")
        if isinstance(retrieval_hit_count, (int, float)):
            self.retrieval_hit_count += max(int(retrieval_hit_count), 0)
        else:
            retrieved_references = node_output.get("retrieved_references")
            if isinstance(retrieved_references, Sequence) and not isinstance(retrieved_references, (str, bytes)):
                self.retrieval_hit_count += sum(1 for item in retrieved_references if isinstance(item, Mapping))
            else:
                references = node_output.get("references")
                if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
                    self.retrieval_hit_count += sum(1 for item in references if isinstance(item, Mapping))

        retrieval_ms = node_output.get("retrieval_ms")
        subagent_retrieval_ms = node_output.get("subagent_retrieval_ms")
        if isinstance(retrieval_ms, (int, float)) or isinstance(subagent_retrieval_ms, (int, float)):
            router_event = self.record_router_done(node_name, at=current_at)
            if router_event is not None:
                events.append(router_event)
            retrieval_event = self.record_retrieval_done(node_name, at=current_at)
            if retrieval_event is not None:
                events.append(retrieval_event)

        return events

    def record_stream_payload(self, payload: Mapping[str, Any]) -> list[TraceStepEvent]:
        event_type = str(payload.get("type") or "")
        node_name = payload.get("node") if isinstance(payload.get("node"), str) else None
        current_at = _iso_now()

        events: list[TraceStepEvent] = []
        if event_type == "start":
            first_byte_event = self.record_stream_first_byte(at=current_at)
            if first_byte_event is not None:
                events.append(first_byte_event)
            request_event = self.record_llm_request_started(node_name, at=current_at)
            if request_event is not None:
                events.append(request_event)
        elif event_type == "raw_first_chunk":
            raw_chunk_event = self.record_llm_first_raw_chunk(node_name, at=current_at)
            if raw_chunk_event is not None:
                events.append(raw_chunk_event)
        elif event_type == "delta":
            first_byte_event = self.record_stream_first_byte(at=current_at)
            if first_byte_event is not None:
                events.append(first_byte_event)
            token_event = self.record_llm_first_token(node_name, at=current_at)
            if token_event is not None:
                events.append(token_event)

        return events

    def build_start_event(self) -> TraceStartEvent:
        if self.graph_started_at is None:
            self.graph_started_at = _iso_now()
        attrs = {"flush_controlled": self.flush_controlled}
        return TraceStartEvent(
            trace_id=self.trace_id,
            run_id=self.run_id,
            session_id=self.session_id,
            scene=self.scene,
            server_received_at=self.server_received_at,
            graph_started_at=self.graph_started_at,
            model=self.model,
            graph_path=list(self.graph_path),
            attrs=attrs,
        )

    def build_summary(self, *, status: str) -> TraceSummaryEvent:
        current_at = _iso_now()
        return TraceSummaryEvent(
            trace_id=self.trace_id,
            run_id=self.run_id,
            session_id=self.session_id,
            status=status,
            at=current_at,
            scene=self.scene,
            graph_path=list(self.graph_path),
            model=self.model,
            has_thinking=self.has_thinking,
            response_chars=self.response_chars,
            response_tokens=self.response_tokens,
            tool_calls=self.tool_calls,
            retrieval_hit_count=self.retrieval_hit_count,
            attrs={"flush_controlled": self.flush_controlled},
        )

    def build_artifact(self, *, status: str, finished_at: str | None = None) -> dict[str, Any]:
        summary = self.build_summary(status=status).model_dump(mode="json")
        server_finished_at = finished_at or summary["at"]
        return {
            "event": "chat_latency.phase1.artifact",
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "scene": self.scene,
            "status": status,
            "graph_path": list(self.graph_path),
            "server_received_at": self.server_received_at,
            "graph_started_at": self.graph_started_at,
            "first_byte_at": self.first_byte_at,
            "router_done_at": self.router_done_at,
            "retrieval_done_at": self.retrieval_done_at,
            "llm_request_started_at": self.llm_request_started_at,
            "llm_first_raw_chunk_at": self.llm_first_raw_chunk_at,
            "llm_first_token_at": self.llm_first_token_at,
            "message_done_at": self.message_done_at,
            "stream_done_at": self.stream_done_at,
            "server_finished_at": server_finished_at,
            **summary,
        }

    def log_artifact(self, *, status: str, finished_at: str | None = None) -> dict[str, Any] | None:
        if not phase1_tracing_enabled():
            return None

        artifact = self.build_artifact(status=status, finished_at=finished_at)
        LOGGER.info(json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        return artifact
