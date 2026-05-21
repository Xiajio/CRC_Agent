from pathlib import Path

from backend.api.services.runtime_paths import resolve_runtime_root


def test_runtime_root_defaults_to_existing_runtime_directory(monkeypatch):
    monkeypatch.delenv("LANGG_RUNTIME_ROOT", raising=False)

    assert resolve_runtime_root() == Path("runtime")


def test_runtime_root_can_be_overridden_for_demo_mode(monkeypatch):
    monkeypatch.setenv("LANGG_RUNTIME_ROOT", "runtime/demo")

    assert resolve_runtime_root() == Path("runtime/demo")
