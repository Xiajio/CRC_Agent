from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import Settings, load_settings

from backend.api.services.fixture_graph_runner import FixtureGraphRunner

_fixture_runner: FixtureGraphRunner | None = None
_patient_graph: Any | None = None
_doctor_graph: Any | None = None


def get_runner_mode() -> str:
    return os.getenv("GRAPH_RUNNER_MODE", "real").strip().lower() or "real"


def should_warm_rag() -> bool:
    value = os.getenv("RAG_WARMUP", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_backend_settings() -> Settings:
    return load_settings()


def get_patient_graph(
    settings: Settings | None = None,
    *,
    runner_mode: str | None = None,
    rag_warmup: bool | None = None,
) -> Any:
    mode = (runner_mode or get_runner_mode()).lower()
    if mode == "fixture":
        return get_fixture_runner()

    global _patient_graph
    if _patient_graph is not None:
        return _patient_graph

    backend_settings = settings or load_backend_settings()
    warmup = should_warm_rag() if rag_warmup is None else rag_warmup

    from src import graph_builder as graph_builder_module

    original_warmup = graph_builder_module.warmup_retriever
    if not warmup:
        graph_builder_module.warmup_retriever = lambda: None
    try:
        _patient_graph = graph_builder_module.build_patient_graph(backend_settings)
    finally:
        graph_builder_module.warmup_retriever = original_warmup

    return _patient_graph


def get_doctor_graph(
    settings: Settings | None = None,
    *,
    runner_mode: str | None = None,
    rag_warmup: bool | None = None,
) -> Any:
    mode = (runner_mode or get_runner_mode()).lower()
    if mode == "fixture":
        return get_fixture_runner()

    global _doctor_graph
    if _doctor_graph is not None:
        return _doctor_graph

    backend_settings = settings or load_backend_settings()
    warmup = should_warm_rag() if rag_warmup is None else rag_warmup

    from src import graph_builder as graph_builder_module

    original_warmup = graph_builder_module.warmup_retriever
    if not warmup:
        graph_builder_module.warmup_retriever = lambda: None
    try:
        _doctor_graph = graph_builder_module.build_doctor_graph(backend_settings)
    finally:
        graph_builder_module.warmup_retriever = original_warmup

    return _doctor_graph


def get_compiled_graph(
    settings: Settings | None = None,
    *,
    runner_mode: str | None = None,
    rag_warmup: bool | None = None,
) -> Any:
    return get_doctor_graph(
        settings=settings,
        runner_mode=runner_mode,
        rag_warmup=rag_warmup,
    )


def get_fixture_runner(fixtures_dir: Path | None = None) -> FixtureGraphRunner:
    global _fixture_runner
    if _fixture_runner is None or fixtures_dir is not None:
        default_case = os.getenv("GRAPH_FIXTURE_CASE", "database_case").strip() or "database_case"
        _fixture_runner = FixtureGraphRunner(fixtures_dir=fixtures_dir, default_case=default_case)
    return _fixture_runner


def create_graph_runner(settings: Settings | None = None) -> Any:
    return get_doctor_graph(settings=settings)
