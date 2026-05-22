from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PATH_SECURITY_FILE = Path(__file__).resolve().parents[2] / "src" / "tools" / "path_security.py"
spec = importlib.util.spec_from_file_location("tool_path_security", PATH_SECURITY_FILE)
assert spec is not None and spec.loader is not None
path_security = importlib.util.module_from_spec(spec)
sys.modules["tool_path_security"] = path_security
spec.loader.exec_module(path_security)


def test_validate_model_path_accepts_default_model_root() -> None:
    default_model = (
        path_security.PROJECT_ROOT
        / "src"
        / "tools"
        / "tool"
        / "Tumor_Detection"
        / "best.pt"
    )

    assert Path(path_security.validate_model_path(None, default_path=default_model)) == default_model.resolve()


def test_validate_model_path_rejects_disallowed_existing_path(tmp_path: Path) -> None:
    model_path = tmp_path / "outside.pt"
    model_path.write_bytes(b"checkpoint")

    with pytest.raises(path_security.UnsafeToolPathError):
        path_security.validate_model_path(str(model_path))


def test_validate_input_path_accepts_extra_allowlist_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "case.png"
    image_path.write_bytes(b"image")
    monkeypatch.setenv("TOOL_ALLOWED_INPUT_ROOTS", str(tmp_path))

    assert Path(path_security.validate_tool_input_path(str(image_path))) == image_path.resolve()


def test_safe_torch_load_prefers_weights_only() -> None:
    calls: list[dict[str, object]] = []

    class FakeTorch:
        def load(self, model_path: str, **kwargs: object) -> str:
            calls.append(kwargs)
            return f"loaded:{model_path}"

    assert path_security.safe_torch_load(FakeTorch(), "model.pt", map_location="cpu") == "loaded:model.pt"
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_safe_torch_load_falls_back_for_old_torch() -> None:
    calls: list[dict[str, object]] = []

    class OldTorch:
        def load(self, model_path: str, **kwargs: object) -> str:
            calls.append(kwargs)
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return f"loaded:{model_path}"

    assert path_security.safe_torch_load(OldTorch(), "model.pt", map_location="cpu") == "loaded:model.pt"
    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]
