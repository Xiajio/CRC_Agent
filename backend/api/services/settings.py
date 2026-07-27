from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


AuthMode = Literal["none", "bearer"]
SessionStoreBackend = Literal["memory", "sqlite"]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_auth_mode(value: str | None) -> AuthMode:
    if value is None:
        return "bearer"
    normalized = value.strip().lower()
    if not normalized:
        return "bearer"
    if normalized in {"none", "bearer"}:
        return normalized
    raise RuntimeError("AUTH_MODE must be one of: none, bearer")


def _parse_session_store_backend(value: str | None) -> SessionStoreBackend:
    if value is None:
        return "memory"
    normalized = value.strip().lower()
    if not normalized:
        return "memory"
    if normalized in {"memory", "sqlite"}:
        return normalized
    raise RuntimeError("SESSION_STORE_BACKEND must be one of: memory, sqlite")


def _parse_optional_int(value: str | None, default: int | None) -> int | None:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise RuntimeError("SESSION_STORE_TTL_DAYS must be an integer, none, or null") from exc
    if parsed < 0:
        raise RuntimeError("SESSION_STORE_TTL_DAYS must be >= 0")
    return parsed


@dataclass(slots=True)
class RuntimeSettings:
    auth_mode: AuthMode = "bearer"
    api_bearer_token: str | None = None
    api_admin_bearer_token: str | None = None
    frontend_origins: list[str] | None = None
    graph_runner_mode: str = "real"
    rag_warmup: bool = True
    session_store_backend: SessionStoreBackend = "memory"
    session_store_sqlite_path: str | None = None
    session_store_ttl_days: int | None = 7
    experimental_diagrams_enabled: bool = False


def load_runtime_settings() -> RuntimeSettings:
    auth_mode = _parse_auth_mode(os.getenv("AUTH_MODE"))
    api_bearer_token = os.getenv("API_BEARER_TOKEN")
    api_admin_bearer_token = os.getenv("API_ADMIN_BEARER_TOKEN")
    frontend_origins_raw = os.getenv("FRONTEND_ORIGINS")
    frontend_origins = _parse_csv(frontend_origins_raw)
    if frontend_origins_raw is None:
        frontend_origins = ["http://localhost:5173"]

    settings = RuntimeSettings(
        auth_mode=auth_mode,
        api_bearer_token=api_bearer_token.strip() if isinstance(api_bearer_token, str) and api_bearer_token.strip() else None,
        api_admin_bearer_token=(
            api_admin_bearer_token.strip()
            if isinstance(api_admin_bearer_token, str) and api_admin_bearer_token.strip()
            else None
        ),
        frontend_origins=frontend_origins,
        graph_runner_mode=os.getenv("GRAPH_RUNNER_MODE", "real").strip().lower() or "real",
        rag_warmup=_parse_bool(os.getenv("RAG_WARMUP"), default=True),
        session_store_backend=_parse_session_store_backend(os.getenv("SESSION_STORE_BACKEND")),
        session_store_sqlite_path=(
            os.getenv("SESSION_STORE_SQLITE_PATH", "").strip() or None
        ),
        session_store_ttl_days=_parse_optional_int(os.getenv("SESSION_STORE_TTL_DAYS"), 7),
        experimental_diagrams_enabled=_parse_bool(
            os.getenv("EXPERIMENTAL_DIAGRAMS_ENABLED"),
            default=False,
        ),
    )
    if settings.auth_mode == "bearer" and not settings.api_bearer_token:
        raise RuntimeError("API_BEARER_TOKEN must be set when AUTH_MODE=bearer")
    if settings.auth_mode == "bearer" and not settings.api_admin_bearer_token:
        settings.api_admin_bearer_token = settings.api_bearer_token
    return settings
