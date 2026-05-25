from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UnsafeToolPathError(ValueError):
    """Raised when a tool input or model path is outside configured roots."""


def _split_env_paths(name: str) -> list[Path]:
    value = os.environ.get(name, "")
    roots: list[Path] = []
    for raw_part in value.split(os.pathsep):
        part = raw_part.strip()
        if part:
            roots.append(_project_relative_path(part))
    return roots


def _project_relative_path(path: str | os.PathLike[str]) -> Path:
    if str(path).strip() == "":
        raise UnsafeToolPathError("path is empty")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _default_model_roots() -> list[Path]:
    return [
        PROJECT_ROOT / "src" / "tools" / "tool" / "Tumor_Detection",
        PROJECT_ROOT / "src" / "tools" / "tool" / "Tumor_Localization",
        PROJECT_ROOT
        / "src"
        / "tools"
        / "tool"
        / "Pathological_Slide_Classification"
        / "CLAM_Tool",
    ]


def _default_input_roots() -> list[Path]:
    return [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "runtime",
        PROJECT_ROOT / "tests" / "fixtures",
    ]


def _resolve_roots(roots: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for root in roots:
        resolved.append(root.resolve(strict=False))
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_allowed_path(
    path: str | os.PathLike[str],
    *,
    allowed_roots: Iterable[Path],
    label: str,
    must_exist: bool = True,
) -> Path:
    candidate = _project_relative_path(path)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise UnsafeToolPathError(f"{label} does not exist: {candidate}") from exc

    resolved_roots = _resolve_roots(allowed_roots)
    if not any(_is_relative_to(resolved, root) for root in resolved_roots):
        roots_text = ", ".join(str(root) for root in resolved_roots)
        raise UnsafeToolPathError(f"{label} is outside allowed roots: {resolved}. Allowed roots: {roots_text}")

    return resolved


def validate_model_path(
    model_path: str | os.PathLike[str] | None,
    *,
    default_path: str | os.PathLike[str] | None = None,
) -> str:
    if model_path is None:
        if default_path is None:
            raise UnsafeToolPathError("model_path is required")
        model_path = default_path

    roots = [*_default_model_roots(), *_split_env_paths("TOOL_ALLOWED_MODEL_ROOTS")]
    return str(
        resolve_allowed_path(
            model_path,
            allowed_roots=roots,
            label="model_path",
            must_exist=True,
        )
    )


def validate_tool_input_path(
    input_path: str | os.PathLike[str],
    *,
    label: str = "input_path",
    must_exist: bool = True,
) -> str:
    roots = [*_default_input_roots(), *_split_env_paths("TOOL_ALLOWED_INPUT_ROOTS")]
    return str(
        resolve_allowed_path(
            input_path,
            allowed_roots=roots,
            label=label,
            must_exist=must_exist,
        )
    )


def safe_torch_load(torch_module: object, model_path: str, **kwargs: object) -> object:
    try:
        return torch_module.load(model_path, **kwargs, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch_module.load(model_path, **kwargs)
