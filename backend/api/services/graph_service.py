from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from backend.api.adapters.event_normalizer import normalize_tick
from backend.api.schemas.events import (
    ContextMaintenanceEvent,
    DoneEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageDoneEvent,
    TraceStepEvent,
)
from backend.api.services import chat_latency_trace
from backend.api.services.context_maintenance import (
    CONTEXT_MAINTENANCE_COMPLETED_MESSAGE,
    CONTEXT_MAINTENANCE_FAILED_MESSAGE,
    CONTEXT_MAINTENANCE_RUNNING_MESSAGE,
)
from backend.api.services.patient_context_resolver import PatientContextStaleError
from backend.api.services.payload_builder import build_graph_payload, restore_pending_context_messages
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from src.nodes.node_utils import clear_stream_callback, set_stream_callback
from contextlib import suppress


class SessionNotFoundError(RuntimeError):
    pass


class SessionBusyError(RuntimeError):
    pass


def _message_delta_from_callback(payload: Mapping[str, Any]) -> MessageDeltaEvent | None:
    if payload.get("type") != "delta":
        return None

    message_id = payload.get("message_id")
    delta = payload.get("delta")
    if not isinstance(message_id, str) or not message_id.strip():
        return None
    if not isinstance(delta, str) or not delta:
        return None

    node = payload.get("node")
    return MessageDeltaEvent(
        message_id=message_id,
        node=node if isinstance(node, str) and node else None,
        delta=delta,
    )


def _event_payload(event: Any) -> dict[str, Any]:
    if isinstance(event, BaseModel):
        return event.model_dump(mode="json")
    if isinstance(event, Mapping):
        return dict(event)
    raise TypeError(f"Unsupported stream event type: {type(event)!r}")


def encode_sse_event(event: Any) -> str:
    payload = _event_payload(event)
    event_type = str(payload["type"])
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


class GraphService:
    def __init__(
        self,
        compiled_graph: Any,
        session_store: InMemorySessionStore,
        *,
        patient_context_resolver: Any | None = None,
        context_finalizer: Any | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        self._compiled_graph = compiled_graph
        self._session_store = session_store
        self._patient_context_resolver = patient_context_resolver
        self._context_finalizer = context_finalizer
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._context_tasks: dict[str, asyncio.Task[Any]] = {}

    def _get_session_meta(self, session_id: str) -> SessionMeta:
        meta = self._session_store.get_session(session_id)
        if meta is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return meta

    def _int_patient_version(self, value: Any) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def _doctor_patient_version_used(self, meta: SessionMeta) -> int | None:
        context_state = meta.context_state if isinstance(meta.context_state, Mapping) else {}
        if meta.scene == "doctor" and meta.patient_id is not None:
            last_injected_version = self._int_patient_version(
                context_state.get("last_injected_patient_version")
            )
            if last_injected_version is not None:
                return last_injected_version
            return self._int_patient_version(context_state.get("bound_patient_version"))

        return None

    def _safe_session_patient_version(self, meta: SessionMeta) -> int | None:
        return self._doctor_patient_version_used(meta)

    def _patient_version_used(self, meta: SessionMeta, prepared_payload: Mapping[str, Any]) -> int | None:
        patient_context = prepared_payload.get("patient_context")
        if isinstance(patient_context, Mapping):
            patient_version = self._int_patient_version(patient_context.get("patient_version"))
            if patient_version is not None:
                return patient_version

        return self._doctor_patient_version_used(meta)

    def _stale_patient_context_stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        snapshot_version: int,
        patient_version_used: int | None,
        exc: PatientContextStaleError,
    ) -> AsyncIterator[str]:
        async def _generator() -> AsyncIterator[str]:
            yield encode_sse_event(
                ErrorEvent(
                    code="PATIENT_CONTEXT_STALE",
                    message=str(exc),
                    recoverable=True,
                )
            )
            yield encode_sse_event(
                DoneEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    snapshot_version=snapshot_version,
                    patient_version_used=patient_version_used,
                    patient_context_stale=True,
                )
            )

        return _generator()

    def _load_agent_state_for_thread(self, thread_id: str) -> dict[str, Any] | None:
        loader = getattr(self._compiled_graph, "load_state", None)
        if callable(loader):
            state = loader(thread_id)
            if isinstance(state, Mapping):
                return dict(state)

        getter = getattr(self._compiled_graph, "get_state", None)
        if callable(getter):
            state = None
            for candidate in (
                lambda: getter(thread_id),
                lambda: getter({"configurable": {"thread_id": thread_id}}),
                lambda: getter(config={"configurable": {"thread_id": thread_id}}),
            ):
                try:
                    state = candidate()
                    break
                except Exception:
                    continue

            if state is None:
                return None
            if isinstance(state, Mapping):
                return dict(state)

            values = getattr(state, "values", None)
            if isinstance(values, Mapping):
                return dict(values)

        return None

    def load_agent_state(self, session_id: str) -> dict[str, Any] | None:
        meta = self._session_store.get_session(session_id)
        if meta is None:
            return None
        return self._load_agent_state_for_thread(meta.thread_id) or {}

    def _prepare_session_meta(
        self,
        session_id: str,
        chat_request: Any,
        meta: SessionMeta,
    ) -> SessionMeta:
        del session_id, chat_request
        return meta

    def _cancel_context_maintenance(self, session_id: str) -> None:
        task = self._context_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
        if self._session_store.get_session(session_id) is not None:
            self._session_store.set_context_maintenance(session_id, None)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _phase0_trace_from_request(session_id: str, run_id: str, chat_request: Any) -> chat_latency_trace.Phase0ChatLatencyTrace:
        trace_id = None
        request_started_at: float | None = None
        router_ms = 0.0

        if isinstance(chat_request, Mapping):
            raw_trace_id = chat_request.get("trace_id")
            if isinstance(raw_trace_id, str) and raw_trace_id.strip():
                trace_id = raw_trace_id

            request_started = chat_request.get("_latency_request_started_at")
            if isinstance(request_started, (int, float)):
                request_started_at = float(request_started)

            router_value = chat_request.get("_latency_router_ms")
            if isinstance(router_value, (int, float)):
                router_ms = float(router_value)

        if request_started_at is None:
            request_started_at = chat_latency_trace.perf_counter()

        trace = chat_latency_trace.Phase0ChatLatencyTrace(
            trace_id=trace_id,
            run_id=run_id,
            session_id=session_id,
            request_started_at=request_started_at,
        )
        trace.record_router_ms(router_ms)
        return trace

    @staticmethod
    def _phase1_trace_from_request(
        session_id: str,
        run_id: str,
        chat_request: Any,
        *,
        scene: str | None,
    ) -> chat_latency_trace.Phase1ChatLatencyTrace | None:
        if not chat_latency_trace.phase1_tracing_enabled():
            return None

        trace_id = None
        model = None
        flush_controlled = False

        if isinstance(chat_request, Mapping):
            raw_trace_id = chat_request.get("trace_id")
            if isinstance(raw_trace_id, str) and raw_trace_id.strip():
                trace_id = raw_trace_id

            for key in ("model", "model_name"):
                raw_model = chat_request.get(key)
                if isinstance(raw_model, str) and raw_model.strip():
                    model = raw_model
                    break

            context = chat_request.get("context")
            if isinstance(context, Mapping):
                flush_controlled = bool(context.get("flush_controlled"))
            elif "flush_controlled" in chat_request:
                flush_controlled = bool(chat_request.get("flush_controlled"))

        return chat_latency_trace.Phase1ChatLatencyTrace(
            trace_id=trace_id,
            run_id=run_id,
            session_id=session_id,
            scene=scene,
            model=model,
            flush_controlled=flush_controlled,
        )

    @staticmethod
    def _record_phase0_stream_payload(
        phase0_trace: chat_latency_trace.Phase0ChatLatencyTrace | None,
        payload: Mapping[str, Any],
    ) -> None:
        if phase0_trace is None:
            return

        event_type = str(payload.get("type") or "")
        node_name = payload.get("node")
        if event_type == "start":
            phase0_trace.record_node_start(node_name if isinstance(node_name, str) else None)
        elif event_type == "delta":
            phase0_trace.record_node_delta(node_name if isinstance(node_name, str) else None)
        elif event_type == "end":
            phase0_trace.record_node_end(node_name if isinstance(node_name, str) else None)

    @staticmethod
    def _record_phase1_stream_payload(
        phase1_trace: chat_latency_trace.Phase1ChatLatencyTrace | None,
        payload: Mapping[str, Any],
    ) -> list[TraceStepEvent]:
        if phase1_trace is None:
            return []
        return phase1_trace.record_stream_payload(payload)

    @staticmethod
    def _record_phase0_node_output(
        phase0_trace: chat_latency_trace.Phase0ChatLatencyTrace | None,
        node_name: str,
        node_output: Mapping[str, Any],
    ) -> None:
        if phase0_trace is None:
            return

        phase0_trace.record_node_seen(node_name)
        retrieval_ms = GraphService._coerce_float(node_output.get("retrieval_ms"))
        if retrieval_ms is None:
            retrieval_ms = GraphService._coerce_float(node_output.get("subagent_retrieval_ms"))
        phase0_trace.record_retrieval_ms(retrieval_ms)

    @staticmethod
    def _record_phase1_node_output(
        phase1_trace: chat_latency_trace.Phase1ChatLatencyTrace | None,
        node_name: str,
        node_output: Mapping[str, Any],
    ) -> list[TraceStepEvent]:
        if phase1_trace is None:
            return []
        return phase1_trace.record_node_output(node_name, node_output)

    @staticmethod
    def _record_visible_message_done(
        phase0_trace: chat_latency_trace.Phase0ChatLatencyTrace | None,
        node_name: str,
    ) -> None:
        if phase0_trace is None:
            return
        phase0_trace.record_visible_message_done(node_name)

    @staticmethod
    def _record_phase1_message_done(
        phase1_trace: chat_latency_trace.Phase1ChatLatencyTrace | None,
        node_name: str,
        message_done_event: MessageDoneEvent,
        node_output: Mapping[str, Any] | None,
    ) -> TraceStepEvent | None:
        if phase1_trace is None:
            return None
        return phase1_trace.record_message_done(
            node_name=node_name,
            message=message_done_event,
            node_output=node_output,
        )

    async def _invoke_context_finalizer(
        self,
        *,
        agent_state: Mapping[str, Any] | None,
        existing_context_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if self._context_finalizer is None:
            return {}

        finalize = getattr(self._context_finalizer, "finalize", None)
        if not callable(finalize):
            return {}

        if inspect.iscoroutinefunction(finalize):
            result = await finalize(
                agent_state=agent_state,
                existing_context_state=existing_context_state,
            )
        else:
            result = await asyncio.to_thread(
                finalize,
                agent_state=agent_state,
                existing_context_state=existing_context_state,
            )

        return dict(result) if isinstance(result, Mapping) else {}

    def _schedule_context_maintenance(self, session_id: str, run_id: str) -> None:
        if self._context_finalizer is None:
            return

        async def _runner() -> None:
            try:
                meta = self._get_session_meta(session_id)
                updates = await self._invoke_context_finalizer(
                    agent_state=self.load_agent_state(session_id) or {},
                    existing_context_state=meta.context_state,
                )
                if updates:
                    self._session_store.merge_context_state(session_id, updates)
                self._session_store.set_context_maintenance(
                    session_id,
                    {
                        "status": "completed",
                        "message": CONTEXT_MAINTENANCE_COMPLETED_MESSAGE,
                        "source_run_id": run_id,
                    },
                )
                self._session_store.bump_snapshot_version(session_id)
            except asyncio.CancelledError:
                current_meta = self._session_store.get_session(session_id)
                if (
                    current_meta is not None
                    and isinstance(current_meta.context_maintenance, Mapping)
                    and current_meta.context_maintenance.get("source_run_id") == run_id
                ):
                    self._session_store.set_context_maintenance(session_id, None)
                raise
            except Exception as exc:
                self._session_store.set_context_maintenance(
                    session_id,
                    {
                        "status": "failed",
                        "message": CONTEXT_MAINTENANCE_FAILED_MESSAGE,
                        "error": str(exc),
                        "source_run_id": run_id,
                    },
                )
                self._session_store.bump_snapshot_version(session_id)
            finally:
                self._context_tasks.pop(session_id, None)

        self._context_tasks[session_id] = asyncio.create_task(_runner())

    async def _stream_runner_events(
        self,
        payload: Mapping[str, Any],
        thread_id: str,
        *,
        stream_event_queue: asyncio.Queue[dict[str, Any]] | None = None,
        phase0_trace: chat_latency_trace.Phase0ChatLatencyTrace | None = None,
        phase1_trace: chat_latency_trace.Phase1ChatLatencyTrace | None = None,
    ) -> AsyncIterator[dict[str, Any] | MessageDeltaEvent | TraceStepEvent]:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 200,
        }
        raw_stream = self._compiled_graph.astream(payload, config=config)
        next_event_task: asyncio.Task[Any] | None = asyncio.create_task(anext(raw_stream))
        next_stream_task: asyncio.Task[dict[str, Any]] | None = (
            asyncio.create_task(stream_event_queue.get()) if stream_event_queue is not None else None
        )

        def record_callback_payload(callback_payload: Mapping[str, Any]) -> None:
            self._record_phase0_stream_payload(phase0_trace, callback_payload)

        try:
            while next_event_task is not None or next_stream_task is not None:
                wait_targets = [task for task in (next_stream_task, next_event_task) if task is not None]
                if not wait_targets:
                    return

                if self._heartbeat_interval_seconds > 0:
                    done, _ = await asyncio.wait(
                        wait_targets,
                        timeout=self._heartbeat_interval_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        yield {":ping": True}
                        continue
                else:
                    done, _ = await asyncio.wait(wait_targets, return_when=asyncio.FIRST_COMPLETED)

                if next_stream_task is not None and next_stream_task in done:
                    callback_payload = next_stream_task.result()
                    record_callback_payload(callback_payload)
                    for trace_event in self._record_phase1_stream_payload(phase1_trace, callback_payload):
                        yield trace_event
                    next_stream_task = (
                        asyncio.create_task(stream_event_queue.get()) if stream_event_queue is not None else None
                    )
                    delta_event = _message_delta_from_callback(callback_payload)
                    if delta_event is not None:
                        yield delta_event

                if next_event_task is not None and next_event_task in done:
                    try:
                        event = next_event_task.result()
                    except StopAsyncIteration:
                        next_event_task = None
                        await asyncio.sleep(0)
                        if next_stream_task is not None and next_stream_task.done():
                            callback_payload = next_stream_task.result()
                            record_callback_payload(callback_payload)
                            for trace_event in self._record_phase1_stream_payload(phase1_trace, callback_payload):
                                yield trace_event
                            delta_event = _message_delta_from_callback(callback_payload)
                            if delta_event is not None:
                                yield delta_event
                        if stream_event_queue is not None:
                            while not stream_event_queue.empty():
                                callback_payload = stream_event_queue.get_nowait()
                                record_callback_payload(callback_payload)
                                for trace_event in self._record_phase1_stream_payload(phase1_trace, callback_payload):
                                    yield trace_event
                                delta_event = _message_delta_from_callback(callback_payload)
                                if delta_event is not None:
                                    yield delta_event
                        if next_stream_task is not None:
                            next_stream_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await next_stream_task
                            next_stream_task = None
                        return

                    if next_stream_task is not None and next_stream_task.done():
                        callback_payload = next_stream_task.result()
                        record_callback_payload(callback_payload)
                        for trace_event in self._record_phase1_stream_payload(phase1_trace, callback_payload):
                            yield trace_event
                        delta_event = _message_delta_from_callback(callback_payload)
                        if delta_event is not None:
                            yield delta_event
                        next_stream_task = (
                            asyncio.create_task(stream_event_queue.get()) if stream_event_queue is not None else None
                        )
                    if stream_event_queue is not None:
                        while not stream_event_queue.empty():
                            callback_payload = stream_event_queue.get_nowait()
                            record_callback_payload(callback_payload)
                            for trace_event in self._record_phase1_stream_payload(phase1_trace, callback_payload):
                                yield trace_event
                            delta_event = _message_delta_from_callback(callback_payload)
                            if delta_event is not None:
                                yield delta_event

                    if isinstance(event, Mapping):
                        yield dict(event)

                    next_event_task = asyncio.create_task(anext(raw_stream))
        finally:
            if next_event_task is not None and not next_event_task.done():
                next_event_task.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await next_event_task

            if next_stream_task is not None and not next_stream_task.done():
                next_stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await next_stream_task

            close_stream = getattr(raw_stream, "aclose", None)
            if callable(close_stream):
                with suppress(Exception):
                    await close_stream()

    def stream_turn(self, session_id: str, chat_request: Any) -> AsyncIterator[str]:
        meta = self._get_session_meta(session_id)
        meta = self._prepare_session_meta(session_id, chat_request, meta)
        patient_version_used: int | None = None
        if self._patient_context_resolver is not None:
            try:
                self._patient_context_resolver.resolve(session_id)
                meta = self._session_store.get_session(session_id) or meta
            except PatientContextStaleError as exc:
                meta = self._session_store.get_session(session_id) or meta
                patient_version_used = self._safe_session_patient_version(meta)
                return self._stale_patient_context_stream(
                    thread_id=meta.thread_id,
                    run_id=f"run_{uuid4().hex}",
                    snapshot_version=meta.snapshot_version,
                    patient_version_used=patient_version_used,
                    exc=exc,
                )
        self._cancel_context_maintenance(session_id)
        thread_id = meta.thread_id
        starting_snapshot_version = meta.snapshot_version
        run_id = f"run_{uuid4().hex}"
        phase0_trace = self._phase0_trace_from_request(session_id, run_id, chat_request)
        phase1_trace = self._phase1_trace_from_request(session_id, run_id, chat_request, scene=meta.scene)

        if not self._session_store.try_acquire_run_lock(session_id, run_id):
            raise SessionBusyError(f"Session is busy: {session_id}")

        prepared = None
        try:
            prepared = build_graph_payload(
                chat_request=chat_request,
                session_meta=meta,
                state_snapshot=self.load_agent_state(session_id) or {},
            )
            patient_version_used = self._patient_version_used(meta, prepared.payload)
            if isinstance(chat_request, Mapping):
                trace_id = chat_request.get("trace_id")
                if isinstance(trace_id, str) and trace_id.strip():
                    prepared.payload["trace_id"] = trace_id
        except Exception:
            self._session_store.release_run_lock(session_id, run_id)
            raise

        async def _generator() -> AsyncIterator[str]:
            success = False
            done_event: DoneEvent | None = None
            restored_pending_context = False
            run_lock_released = False
            stream_callback_token = None
            terminal_status = "error"

            def release_run_lock() -> None:
                nonlocal run_lock_released
                if run_lock_released:
                    return
                self._session_store.release_run_lock(session_id, run_id)
                run_lock_released = True

            try:
                stream_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def stream_callback(payload: dict[str, Any]) -> None:
                    event_payload = dict(payload)
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        current_loop = None

                    if current_loop is loop:
                        stream_event_queue.put_nowait(event_payload)
                    else:
                        loop.call_soon_threadsafe(stream_event_queue.put_nowait, event_payload)

                stream_callback_token = set_stream_callback(stream_callback)
                if phase1_trace is not None:
                    yield encode_sse_event(phase1_trace.build_start_event())

                async for raw_event in self._stream_runner_events(
                    prepared.payload,
                    thread_id,
                    stream_event_queue=stream_event_queue,
                    phase0_trace=phase0_trace,
                    phase1_trace=phase1_trace,
                ):
                    if raw_event == {":ping": True}:
                        yield ": ping\n\n"
                        continue
                    if isinstance(raw_event, TraceStepEvent):
                        yield encode_sse_event(raw_event)
                        continue
                    if isinstance(raw_event, MessageDeltaEvent):
                        yield encode_sse_event(raw_event)
                        continue

                    for node_name, node_output in raw_event.items():
                        if isinstance(node_output, Mapping):
                            self._record_phase0_node_output(phase0_trace, node_name, node_output)
                            for trace_event in self._record_phase1_node_output(phase1_trace, node_name, node_output):
                                yield encode_sse_event(trace_event)
                        for event in normalize_tick(node_name, node_output):
                            if isinstance(event, MessageDoneEvent):
                                self._record_visible_message_done(phase0_trace, node_name)
                                trace_event = self._record_phase1_message_done(
                                    phase1_trace,
                                    node_name,
                                    event,
                                    node_output if isinstance(node_output, Mapping) else None,
                                )
                                if trace_event is not None:
                                    yield encode_sse_event(trace_event)
                            yield encode_sse_event(event)

                success = True
                terminal_status = "completed"
                if self._context_finalizer is not None:
                    self._session_store.set_context_maintenance(
                        session_id,
                        {
                            "status": "running",
                            "message": CONTEXT_MAINTENANCE_RUNNING_MESSAGE,
                            "source_run_id": run_id,
                        },
                    )
                    yield encode_sse_event(
                        ContextMaintenanceEvent(
                            status="running",
                            message=CONTEXT_MAINTENANCE_RUNNING_MESSAGE,
                        )
                    )
                if phase1_trace is not None:
                    stream_done_event = phase1_trace.record_stream_done()
                    if stream_done_event is not None:
                        yield encode_sse_event(stream_done_event)
                snapshot_version = self._session_store.bump_snapshot_version(session_id)
                done_event = DoneEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    snapshot_version=snapshot_version,
                    patient_version_used=patient_version_used,
                    patient_context_stale=False,
                )
                self._schedule_context_maintenance(session_id, run_id)
                release_run_lock()
                yield encode_sse_event(done_event)
                if phase1_trace is not None:
                    yield encode_sse_event(phase1_trace.build_summary(status=terminal_status))
            except asyncio.CancelledError:
                restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                restored_pending_context = True
                terminal_status = "aborted"
                raise
            except Exception as exc:
                restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                restored_pending_context = True
                terminal_status = "error"
                if phase1_trace is not None:
                    yield encode_sse_event(phase1_trace.record_error())
                yield encode_sse_event(
                    ErrorEvent(
                        code="GRAPH_RUN_FAILED",
                        message=str(exc),
                        recoverable=True,
                    )
                )
                done_event = DoneEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    snapshot_version=starting_snapshot_version,
                    patient_version_used=patient_version_used,
                    patient_context_stale=False,
                )
                release_run_lock()
                yield encode_sse_event(done_event)
                if phase1_trace is not None:
                    yield encode_sse_event(phase1_trace.build_summary(status=terminal_status))
            finally:
                if stream_callback_token is not None:
                    clear_stream_callback(stream_callback_token)
                if not success and done_event is None and not restored_pending_context:
                    restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                release_run_lock()
                phase0_trace.log_summary(
                    status=terminal_status,
                    finished_at=chat_latency_trace.perf_counter(),
                )
                if phase1_trace is not None:
                    phase1_trace.log_artifact(status=terminal_status)

        return _generator()


class PatientGraphService(GraphService):
    def __init__(
        self,
        compiled_graph: Any,
        session_store: InMemorySessionStore,
        *,
        patient_context_resolver: Any | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        super().__init__(
            compiled_graph,
            session_store,
            patient_context_resolver=patient_context_resolver,
            context_finalizer=None,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )


class DoctorGraphService(GraphService):
    def __init__(
        self,
        compiled_graph: Any,
        session_store: InMemorySessionStore,
        *,
        patient_registry: Any | None = None,
        patient_context_resolver: Any | None = None,
        context_finalizer: Any | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        self._patient_registry = patient_registry
        super().__init__(
            compiled_graph,
            session_store,
            patient_context_resolver=patient_context_resolver,
            context_finalizer=context_finalizer,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )

    def _compose_registry_context_message(
        self,
        summary_message: Any,
        alerts: list[Mapping[str, Any]],
        patient_version: int | None = None,
    ) -> HumanMessage | Any:
        if isinstance(summary_message, HumanMessage):
            summary_text = summary_message.content
        else:
            summary_text = str(summary_message)

        if not alerts and patient_version is None:
            return summary_message if isinstance(summary_message, HumanMessage) else HumanMessage(content=summary_text)

        warning_parts: list[str] = []
        for alert in alerts:
            kind = alert.get("kind")
            message = alert.get("message")
            if isinstance(kind, str) and kind:
                if isinstance(message, str) and message:
                    warning_parts.append(f"{kind}: {message}")
                else:
                    warning_parts.append(kind)

        context_lines: list[str] = []
        if warning_parts:
            context_lines.append(f"Warnings: {'; '.join(warning_parts)}")
        if patient_version is not None:
            context_lines.append(f"Patient version: {patient_version}.")

        if context_lines:
            summary_text = summary_text.rstrip()
            if summary_text:
                summary_text = f"{summary_text}\n" + "\n".join(context_lines)
            else:
                summary_text = "\n".join(context_lines)

        return HumanMessage(content=summary_text)

    def _prepare_session_meta(
        self,
        session_id: str,
        chat_request: Any,
        meta: SessionMeta,
    ) -> SessionMeta:
        del chat_request
        patient_id = meta.patient_id
        if patient_id is None or self._patient_registry is None:
            return meta

        context_state = meta.context_state if isinstance(meta.context_state, Mapping) else {}
        current_patient_version = None
        patient_version_known = False
        get_patient_context_projection = getattr(self._patient_registry, "get_patient_context_projection", None)
        if callable(get_patient_context_projection):
            try:
                projection = get_patient_context_projection(patient_id)
            except Exception:
                projection = None
            if isinstance(projection, Mapping):
                version = projection.get("patient_version")
                if isinstance(version, int):
                    current_patient_version = version
                    patient_version_known = True

        bound_patient_id = context_state.get("bound_patient_id")
        if bound_patient_id == patient_id:
            if not patient_version_known:
                return meta
            if context_state.get("last_injected_patient_version") == current_patient_version:
                return meta

        get_summary_message = getattr(self._patient_registry, "get_patient_summary_message", None)
        list_patient_alerts = getattr(self._patient_registry, "list_patient_alerts", None)
        alerts: list[Mapping[str, Any]] = []
        if callable(list_patient_alerts):
            try:
                candidate_alerts = list_patient_alerts(patient_id)
            except Exception:
                candidate_alerts = []
            if isinstance(candidate_alerts, list):
                alerts = [alert for alert in candidate_alerts if isinstance(alert, Mapping)]

        if callable(get_summary_message):
            summary_message = get_summary_message(patient_id)
            if summary_message is not None:
                self._session_store.enqueue_context_message(
                    session_id,
                    self._compose_registry_context_message(
                        summary_message,
                        alerts,
                        patient_version=current_patient_version,
                    ),
                )

        bound_snapshot_version = None
        get_patient_detail = getattr(self._patient_registry, "get_patient_detail", None)
        if callable(get_patient_detail):
            try:
                detail = get_patient_detail(patient_id)
            except Exception:
                detail = None
            if isinstance(detail, Mapping):
                bound_snapshot_version = detail.get("updated_at")

        context_updates: dict[str, Any] = {
            "bound_patient_id": patient_id,
            "bound_patient_snapshot_version": bound_snapshot_version,
            "bound_patient_alert_count": len(alerts),
        }
        if patient_version_known:
            context_updates["bound_patient_version"] = current_patient_version
            context_updates["last_injected_patient_version"] = current_patient_version
        elif bound_patient_id != patient_id:
            context_updates["bound_patient_version"] = None
            context_updates["last_injected_patient_version"] = None

        self._session_store.merge_context_state(session_id, context_updates)
        refreshed_meta = self._session_store.get_session(session_id)
        return refreshed_meta or meta


class SceneGraphRouter:
    def __init__(
        self,
        *,
        patient_service: PatientGraphService,
        doctor_service: DoctorGraphService,
        session_store: InMemorySessionStore,
    ) -> None:
        self._patient_service = patient_service
        self._doctor_service = doctor_service
        self._session_store = session_store

    def for_session(self, session_id: str) -> GraphService:
        meta = self._session_store.get_session(session_id)
        if meta is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        if meta.scene == "patient":
            return self._patient_service
        return self._doctor_service

    def load_agent_state(self, session_id: str) -> dict[str, Any] | None:
        return self.for_session(session_id).load_agent_state(session_id)
