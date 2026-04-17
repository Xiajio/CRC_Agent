from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from backend.api.adapters.event_normalizer import normalize_tick
from backend.api.schemas.events import ContextMaintenanceEvent, DoneEvent, ErrorEvent, MessageDeltaEvent
from backend.api.services.context_maintenance import (
    CONTEXT_MAINTENANCE_COMPLETED_MESSAGE,
    CONTEXT_MAINTENANCE_FAILED_MESSAGE,
    CONTEXT_MAINTENANCE_RUNNING_MESSAGE,
)
from backend.api.services.payload_builder import build_graph_payload, restore_pending_context_messages
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from src.nodes.node_utils import clear_stream_callback, set_stream_callback


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
        context_finalizer: Any | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        self._compiled_graph = compiled_graph
        self._session_store = session_store
        self._context_finalizer = context_finalizer
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._context_tasks: dict[str, asyncio.Task[Any]] = {}

    def _get_session_meta(self, session_id: str) -> SessionMeta:
        meta = self._session_store.get_session(session_id)
        if meta is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return meta

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
    ) -> AsyncIterator[dict[str, Any] | MessageDeltaEvent]:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 200,
        }
        raw_stream = self._compiled_graph.astream(payload, config=config)
        next_event_task: asyncio.Task[Any] | None = asyncio.create_task(anext(raw_stream))
        next_stream_task: asyncio.Task[dict[str, Any]] | None = (
            asyncio.create_task(stream_event_queue.get()) if stream_event_queue is not None else None
        )

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
                            delta_event = _message_delta_from_callback(next_stream_task.result())
                            if delta_event is not None:
                                yield delta_event
                        if stream_event_queue is not None:
                            while not stream_event_queue.empty():
                                delta_event = _message_delta_from_callback(stream_event_queue.get_nowait())
                                if delta_event is not None:
                                    yield delta_event
                        if next_stream_task is not None:
                            next_stream_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await next_stream_task
                            next_stream_task = None
                        return

                    if next_stream_task is not None and next_stream_task.done():
                        delta_event = _message_delta_from_callback(next_stream_task.result())
                        if delta_event is not None:
                            yield delta_event
                        next_stream_task = (
                            asyncio.create_task(stream_event_queue.get()) if stream_event_queue is not None else None
                        )
                    if stream_event_queue is not None:
                        while not stream_event_queue.empty():
                            delta_event = _message_delta_from_callback(stream_event_queue.get_nowait())
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
        self._cancel_context_maintenance(session_id)
        thread_id = meta.thread_id
        starting_snapshot_version = meta.snapshot_version
        run_id = f"run_{uuid4().hex}"

        if not self._session_store.try_acquire_run_lock(session_id, run_id):
            raise SessionBusyError(f"Session is busy: {session_id}")

        prepared = None
        try:
            prepared = build_graph_payload(
                chat_request=chat_request,
                session_meta=meta,
                state_snapshot=self.load_agent_state(session_id) or {},
            )
        except Exception:
            self._session_store.release_run_lock(session_id, run_id)
            raise

        async def _generator() -> AsyncIterator[str]:
            success = False
            done_event: DoneEvent | None = None
            restored_pending_context = False
            run_lock_released = False
            stream_callback_token = None

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

                async for raw_event in self._stream_runner_events(
                    prepared.payload,
                    thread_id,
                    stream_event_queue=stream_event_queue,
                ):
                    if raw_event == {":ping": True}:
                        yield ": ping\n\n"
                        continue
                    if isinstance(raw_event, MessageDeltaEvent):
                        yield encode_sse_event(raw_event)
                        continue

                    for node_name, node_output in raw_event.items():
                        for event in normalize_tick(node_name, node_output):
                            yield encode_sse_event(event)

                success = True
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
                snapshot_version = self._session_store.bump_snapshot_version(session_id)
                done_event = DoneEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    snapshot_version=snapshot_version,
                )
                self._schedule_context_maintenance(session_id, run_id)
                release_run_lock()
                yield encode_sse_event(done_event)
            except asyncio.CancelledError:
                restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                restored_pending_context = True
                raise
            except Exception as exc:
                restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                restored_pending_context = True
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
                )
                release_run_lock()
                yield encode_sse_event(done_event)
            finally:
                if stream_callback_token is not None:
                    clear_stream_callback(stream_callback_token)
                if not success and done_event is None and not restored_pending_context:
                    restore_pending_context_messages(meta, prepared.drained_pending_context_messages)
                release_run_lock()

        return _generator()


class PatientGraphService(GraphService):
    def __init__(
        self,
        compiled_graph: Any,
        session_store: InMemorySessionStore,
        *,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        super().__init__(
            compiled_graph,
            session_store,
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
        context_finalizer: Any | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        self._patient_registry = patient_registry
        super().__init__(
            compiled_graph,
            session_store,
            context_finalizer=context_finalizer,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )

    def _compose_registry_context_message(
        self,
        summary_message: Any,
        alerts: list[Mapping[str, Any]],
    ) -> HumanMessage | Any:
        if isinstance(summary_message, HumanMessage):
            summary_text = summary_message.content
        else:
            summary_text = str(summary_message)

        if not alerts:
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

        if warning_parts:
            summary_text = summary_text.rstrip()
            summary_text = f"{summary_text}\nWarnings: {'; '.join(warning_parts)}"

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
        if context_state.get("bound_patient_id") == patient_id:
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
                    self._compose_registry_context_message(summary_message, alerts),
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

        self._session_store.merge_context_state(
            session_id,
            {
                "bound_patient_id": patient_id,
                "bound_patient_snapshot_version": bound_snapshot_version,
                "bound_patient_alert_count": len(alerts),
            },
        )
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
