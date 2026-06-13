# Production Readiness Guide

This guide describes the current production boundary for LangG's FastAPI BFF, React frontend, and POST SSE streaming path.

## Current Supported Deployment Shape

The current backend runtime is safe as a single worker process with asyncio concurrency. Use one Uvicorn worker for the current production boundary.

Do not deploy multiple backend workers for the same runtime unless a later implementation adds a shared run-lock backend. `SESSION_STORE_BACKEND=sqlite` persists BFF session metadata, but it does not provide a cross-process run lock. The active run lock is currently process-local.

## Session Store Boundary

`SESSION_STORE_BACKEND=memory` keeps BFF session metadata in the backend process.

`SESSION_STORE_BACKEND=sqlite` mirrors metadata such as `session_id`, `thread_id`, scene, patient binding, uploaded assets, pending context, context state, and `snapshot_version` into SQLite. It improves restart recovery for BFF metadata, but it does not make multi-worker runs safe.

Graph history and LangGraph state recovery still depend on `CHECKPOINT_KIND`. If `CHECKPOINT_KIND=memory`, a SQLite BFF session can be restored while the graph checkpoint is still missing after restart.

## POST SSE Boundary

The current chat stream uses `POST /api/sessions/{session_id}/messages/stream` and browser `fetch()` with `ReadableStream.getReader()`.

POST SSE currently has no mid-run resume. The server does not emit SSE `id:` lines, and the frontend does not use `Last-Event-ID` or an `after_seq` resume endpoint. A disconnect before `done` may surface as a recoverable stream error. If the graph already completed and checkpoint state is available, a page refresh can recover the latest snapshot; this is not equivalent to event replay.

## Reverse Proxy Requirements

Any reverse proxy, load balancer, or gateway in front of `/api/sessions/*/messages/stream` must avoid buffering SSE responses. For Nginx, the relevant settings are:

```nginx
location ~ ^/api/sessions/.*/messages/stream$ {
    proxy_pass http://langg_backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header X-Accel-Buffering no;
}
```

The exact timeout values should be larger than the expected maximum quiet period for a graph run. The backend sends heartbeat comments when no graph or token event is available, so proxy idle timeout must be greater than the heartbeat interval.

## Browser Token Boundary

VITE_API_BEARER_TOKEN is a browser-visible token. It is acceptable for local development and controlled internal tools, but it is not a secure long-term multi-user production credential.

Never expose a separate admin token to the browser. If the local UI needs admin-like operations for demo purposes, use the single-token local mode described in the README and keep that deployment private.

## Capacity Estimate

Estimate active SSE pressure with:

```text
active_sse_connections ~= concurrent_users * active_turn_ratio
```

Capacity is bounded by:

- backend worker event-loop and open connection limits
- graph run duration
- LLM provider concurrency and rate limits
- checkpoint backend throughput
- CPU and memory consumed by active graph state

Before increasing worker count, implement a shared run-lock backend and choose a production checkpoint backend.

## Production Roadmap Links

The staged roadmap is tracked in `docs/superpowers/specs/2026-06-13-production-readiness-roadmap-design.md`.

The next implementation slices after this guide are:

- Redis distributed run lock
- SSE event sequencing and resume
- short-lived browser tokens and OAuth/OIDC
- health checks, metrics, and rollout automation
