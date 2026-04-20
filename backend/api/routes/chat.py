from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.services import chat_latency_trace
from backend.api.services.graph_service import GraphService, SessionBusyError, SessionNotFoundError

router = APIRouter(prefix="/api/sessions", tags=["chat"])


def _get_graph_service(request: Request, session_id: str) -> GraphService:
    runtime = getattr(request.app.state, "runtime", None)
    scene_router = getattr(runtime, "scene_router", None)
    if scene_router is None:
        raise HTTPException(status_code=503, detail="Graph runtime is not initialized")
    try:
        return scene_router.for_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/{session_id}/messages/stream")
async def stream_session_message(
    session_id: str,
    request: Request,
    body: dict[str, object],
):
    request_started_at = chat_latency_trace.perf_counter()
    graph_service = _get_graph_service(request, session_id)
    body["_latency_request_started_at"] = request_started_at
    body["_latency_router_ms"] = (chat_latency_trace.perf_counter() - request_started_at) * 1000.0
    try:
        event_iterator = graph_service.stream_turn(session_id, body)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail="Session is busy") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def _stream():
        async for event in event_iterator:
            yield event.encode("utf-8") if isinstance(event, str) else event

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
