from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .config import CheckpointSettings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover - optional dependency
    SqliteSaver = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception:  # pragma: no cover - optional dependency
    PostgresSaver = None

try:
    from langgraph.checkpoint.redis import RedisSaver
except Exception:  # pragma: no cover - optional dependency
    RedisSaver = None


def get_checkpointer(settings: CheckpointSettings) -> BaseCheckpointSaver:
    """Return a checkpoint backend based on configuration."""

    if settings.kind == "memory":
        return MemorySaver()

    if settings.kind == "sqlite" and settings.url and SqliteSaver:
        return SqliteSaver(settings.url)

    if settings.kind == "postgres" and settings.url and PostgresSaver:
        return PostgresSaver.from_uri(settings.url)

    if settings.kind == "redis" and settings.url and RedisSaver:
        return RedisSaver.from_uri(settings.url)

    if settings.kind == "langsmith" and settings.url:
        # LangSmith saver lives in the langsmith package; users can wire it here.
        raise NotImplementedError("Plug in LangSmith saver for LangGraph here.")

    msg = f"Unsupported checkpointer config: {settings.kind} / {settings.url}"
    raise ValueError(msg)

