import os

from .config import ObservabilitySettings


def init_observability(settings: ObservabilitySettings) -> None:
    """Configure LangSmith tracing env vars if enabled."""

    if settings.tracing:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if settings.api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", settings.api_key)
        if settings.project:
            os.environ.setdefault("LANGCHAIN_PROJECT", settings.project)

