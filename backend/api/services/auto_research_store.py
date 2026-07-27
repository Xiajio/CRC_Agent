from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from src.contracts.auto_research import AutoResearchRun


class AutoResearchStoreIntegrityError(RuntimeError):
    """Raised when the auto-research store cannot be trusted."""


class _AutoResearchArtifactIntegrityError(AutoResearchStoreIntegrityError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        persisted_run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.persisted_run_id = persisted_run_id


class AutoResearchRunNotFoundError(LookupError):
    """Raised when a requested auto-research run does not exist."""


@dataclass(frozen=True)
class AutoResearchRunState:
    runs: list[AutoResearchRun]
    integrity: dict[str, Any]


_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_RECOVERY_ACTIONS: tuple[dict[str, Any], ...] = (
    {
        "code": "rerun_with_new_idempotency_key",
        "label": "Rerun as a new append-only Run",
        "instruction": (
            "Submit the research request again with a new idempotency key. "
            "Do not overwrite or rename the affected artifact."
        ),
        "overwrites_existing_artifact": False,
        "clinical_data_mutated": False,
    },
    {
        "code": "manual_quarantine",
        "label": "Quarantine the artifact manually",
        "instruction": (
            "Preserve the original bytes and checksum, record the operator, "
            "reason, and timestamp, then move the file outside the runs directory."
        ),
        "overwrites_existing_artifact": False,
        "clinical_data_mutated": False,
    },
)


class AutoResearchRunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    def read_state(self) -> AutoResearchRunState:
        warnings: list[str] = []
        affected_artifacts: list[dict[str, Any]] = []
        layout_warning = self._layout_warning()
        if layout_warning is not None:
            return AutoResearchRunState(
                runs=[],
                integrity=self._warning_integrity([layout_warning], []),
            )
        if not self.runs_dir.exists():
            return AutoResearchRunState(
                runs=[],
                integrity={"status": "verified", "warnings": []},
            )

        runs: list[AutoResearchRun] = []
        seen_ids: set[str] = set()
        try:
            paths = sorted(self.runs_dir.glob("*.json"))
        except OSError as exc:
            return AutoResearchRunState(
                runs=[],
                integrity=self._warning_integrity(
                    [f"run files could not be listed: {exc}"],
                    [],
                ),
            )

        for path in paths:
            try:
                run = self._read_path(path)
            except (OSError, ValueError, TypeError, AutoResearchStoreIntegrityError) as exc:
                artifact_path = f"runs/{path.name}"
                message = f"{artifact_path} auto-research run is invalid: {exc}"
                warnings.append(message)
                affected_artifacts.append(
                    {
                        "code": getattr(exc, "code", "invalid_artifact"),
                        "artifact_path": artifact_path,
                        "filename_run_id": path.stem,
                        "persisted_run_id": getattr(exc, "persisted_run_id", None),
                        "message": message,
                        "excluded_from_runs": True,
                    }
                )
                continue
            if run.run_id in seen_ids:
                message = f"duplicate auto-research run id: {run.run_id}"
                warnings.append(message)
                affected_artifacts.append(
                    {
                        "code": "duplicate_run_id",
                        "artifact_path": f"runs/{path.name}",
                        "filename_run_id": path.stem,
                        "persisted_run_id": run.run_id,
                        "message": message,
                        "excluded_from_runs": True,
                    }
                )
                continue
            seen_ids.add(run.run_id)
            runs.append(run)

        return AutoResearchRunState(
            runs=sorted(runs, key=lambda item: item.created_at, reverse=True),
            integrity=(
                {"status": "verified", "warnings": []}
                if not warnings
                else self._warning_integrity(warnings, affected_artifacts)
            ),
        )

    def find_run(self, run_id: str) -> AutoResearchRun | None:
        try:
            return self.get_run(run_id)
        except AutoResearchRunNotFoundError:
            return None

    def get_run(self, run_id: str) -> AutoResearchRun:
        path = self._artifact_path(run_id)
        if not path.exists():
            raise AutoResearchRunNotFoundError(
                f"auto-research run not found: {run_id}"
            )
        return self._read_path(path)

    def write_run(self, run: AutoResearchRun) -> None:
        if not isinstance(run, AutoResearchRun):
            raise TypeError("run must be an AutoResearchRun")
        # Frozen dataclasses still contain mutable lists. Rebuild the complete
        # contract immediately before publication so post-construction mutation
        # cannot bypass the persisted integrity checks.
        run = AutoResearchRun.from_dict(run.to_dict())
        self._raise_if_write_layout_unsafe()
        path = self._artifact_path(run.run_id)
        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{run.run_id}.",
                suffix=".tmp",
                dir=self.runs_dir,
            )
            temporary_path = Path(temporary_name)
            self._validate_path(temporary_path)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = None
                json.dump(
                    run.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_path, path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _read_path(self, path: Path) -> AutoResearchRun:
        self._validate_path(path)
        if path.is_symlink():
            raise _AutoResearchArtifactIntegrityError(
                "run artifact must not be a symlink",
                code="unsafe_artifact_type",
            )
        if not path.is_file():
            raise _AutoResearchArtifactIntegrityError(
                "run artifact must be a regular file",
                code="unsafe_artifact_type",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise _AutoResearchArtifactIntegrityError(
                "run artifact is not UTF-8",
                code="invalid_encoding",
            ) from exc
        except json.JSONDecodeError as exc:
            raise _AutoResearchArtifactIntegrityError(
                f"run artifact is not valid JSON: {exc.msg}",
                code="invalid_json",
            ) from exc
        if not isinstance(payload, dict):
            raise _AutoResearchArtifactIntegrityError(
                "run artifact must be a JSON object",
                code="invalid_contract",
            )
        try:
            run = AutoResearchRun.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            persisted_run_id = payload.get("run_id")
            raise _AutoResearchArtifactIntegrityError(
                "run artifact does not match the auto-research contract",
                code="invalid_contract",
                persisted_run_id=(
                    persisted_run_id if isinstance(persisted_run_id, str) else None
                ),
            ) from exc
        if path.stem != run.run_id:
            raise _AutoResearchArtifactIntegrityError(
                "run filename must match the persisted run_id",
                code="filename_run_id_mismatch",
                persisted_run_id=run.run_id,
            )
        return run

    @staticmethod
    def _warning_integrity(
        warnings: list[str],
        affected_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "status": "warning",
            "warnings": list(warnings),
            "affected_artifacts": [dict(item) for item in affected_artifacts],
            "recovery_actions": [dict(item) for item in _RECOVERY_ACTIONS],
        }

    def _layout_warning(self) -> str | None:
        for path, label in ((self.root, "root"), (self.runs_dir, "runs directory")):
            if path.is_symlink():
                return f"auto-research {label} must not be a symlink"
            if path.exists() and not path.is_dir():
                return f"auto-research {label} must be a directory"
        try:
            self._validate_path(self.runs_dir)
        except AutoResearchStoreIntegrityError as exc:
            return str(exc)
        return None

    def _raise_if_write_layout_unsafe(self) -> None:
        warning = self._layout_warning()
        if warning is not None:
            raise AutoResearchStoreIntegrityError(warning)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise AutoResearchStoreIntegrityError(
                "auto-research root must not be a symlink"
            )
        self.runs_dir.mkdir(parents=False, exist_ok=True)
        if self.runs_dir.is_symlink():
            raise AutoResearchStoreIntegrityError(
                "auto-research runs directory must not be a symlink"
            )
        self._validate_path(self.runs_dir)

    def _artifact_path(self, run_id: str) -> Path:
        self._validate_artifact_id(run_id)
        path = self.runs_dir / f"{run_id}.json"
        self._validate_path(path)
        return path

    def _validate_path(self, path: Path) -> None:
        root_resolved = self.root.resolve(strict=False)
        path_resolved = path.resolve(strict=False)
        try:
            path_resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise AutoResearchStoreIntegrityError(
                "auto-research artifact path must stay under its store root"
            ) from exc

    @staticmethod
    def _validate_artifact_id(value: str) -> None:
        if not isinstance(value, str) or not _ARTIFACT_ID_RE.fullmatch(value):
            raise AutoResearchStoreIntegrityError(
                "auto-research run id must be a safe filename id"
            )
        stem = value.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_DEVICE_NAMES:
            raise AutoResearchStoreIntegrityError(
                "auto-research run id must not be a Windows device name"
            )


__all__ = [
    "AutoResearchRunNotFoundError",
    "AutoResearchRunState",
    "AutoResearchRunStore",
    "AutoResearchStoreIntegrityError",
]
