from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


AuthMode = Literal["none", "bearer"]


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
        return "none"
    normalized = value.strip().lower()
    if not normalized:
        return "none"
    if normalized in {"none", "bearer"}:
        return normalized
    raise RuntimeError("AUTH_MODE must be one of: none, bearer")


@dataclass(slots=True)
class RuntimeSettings:
    auth_mode: AuthMode = "none"
    api_bearer_token: str | None = None
    frontend_origins: list[str] | None = None
    graph_runner_mode: str = "real"
    rag_warmup: bool = True


def load_runtime_settings() -> RuntimeSettings:
    auth_mode = _parse_auth_mode(os.getenv("AUTH_MODE"))
    api_bearer_token = os.getenv("API_BEARER_TOKEN")
    frontend_origins_raw = os.getenv("FRONTEND_ORIGINS")
    frontend_origins = _parse_csv(frontend_origins_raw)
    if frontend_origins_raw is None:
        frontend_origins = ["http://localhost:5173"]

    settings = RuntimeSettings(
        auth_mode=auth_mode,
        api_bearer_token=api_bearer_token.strip() if isinstance(api_bearer_token, str) and api_bearer_token.strip() else None,
        frontend_origins=frontend_origins,
        graph_runner_mode=os.getenv("GRAPH_RUNNER_MODE", "real").strip().lower() or "real",
        rag_warmup=_parse_bool(os.getenv("RAG_WARMUP"), default=True),
    )
    if settings.auth_mode == "bearer" and not settings.api_bearer_token:
        raise RuntimeError("API_BEARER_TOKEN must be set when AUTH_MODE=bearer")
    return settings
