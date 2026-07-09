from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from src.contracts.learning_job import CandidatePatch, LearningJob


class LearningJobIntegrityError(RuntimeError):
    """Raised when the learning job store is unsafe to append to."""


@dataclass(frozen=True)
class LearningJobState:
    jobs: list[LearningJob]
    candidates: list[CandidatePatch]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class _ReadResult:
    artifacts: list[Any]
    warnings: list[str]


_Artifact = TypeVar("_Artifact")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class LearningJobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.candidates_dir = self.root / "candidates"

    def read_state(self) -> LearningJobState:
        root_warning = self._root_layout_warning()
        if root_warning is not None:
            return LearningJobState(
                jobs=[],
                candidates=[],
                integrity={"status": "warning", "warnings": [root_warning]},
            )

        job_result = self._read_json_dir(
            self.jobs_dir,
            LearningJob,
            artifact_name="job",
            id_field="job_id",
        )
        candidate_result = self._read_json_dir(
            self.candidates_dir,
            CandidatePatch,
            artifact_name="candidate",
            id_field="patch_id",
        )
        warnings = (
            job_result.warnings
            + candidate_result.warnings
            + self._candidate_consistency_warnings(
                jobs=job_result.artifacts,
                candidates=candidate_result.artifacts,
            )
        )

        return LearningJobState(
            jobs=sorted(job_result.artifacts, key=lambda job: job.created_at),
            candidates=sorted(
                candidate_result.artifacts,
                key=lambda candidate: candidate.patch_id,
            ),
            integrity=(
                {"status": "verified", "warnings": []}
                if not warnings
                else {"status": "warning", "warnings": warnings}
            ),
        )

    def write_job(self, job: LearningJob, candidates: list[CandidatePatch]) -> None:
        self._raise_if_write_layout_unsafe()
        self._validate_candidate_consistency(job, candidates)
        job_path = self._artifact_path(self.jobs_dir, job.job_id)
        candidate_paths = [
            self._artifact_path(self.candidates_dir, candidate.patch_id)
            for candidate in candidates
        ]

        written_paths: list[Path] = []
        try:
            for candidate, candidate_path in zip(candidates, candidate_paths):
                self._write_json_once(candidate_path, candidate.to_dict())
                written_paths.append(candidate_path)
            self._write_json_once(job_path, job.to_dict())
            written_paths.append(job_path)
        except Exception:
            for path in reversed(written_paths):
                path.unlink(missing_ok=True)
            raise

    def _read_json_dir(
        self,
        directory: Path,
        factory: Callable[..., _Artifact],
        *,
        artifact_name: str,
        id_field: str,
    ) -> _ReadResult:
        layout_warning = self._directory_layout_warning(directory, artifact_name)
        if layout_warning is not None:
            return _ReadResult(artifacts=[], warnings=[layout_warning])
        if not directory.exists():
            return _ReadResult(artifacts=[], warnings=[])
        try:
            paths = sorted(directory.glob("*.json"))
        except OSError as exc:
            return _ReadResult(
                artifacts=[],
                warnings=[f"{directory} {artifact_name} files could not be listed: {exc}"],
            )

        artifacts: list[Any] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for path in paths:
            file_warning = self._file_layout_warning(path, f"{artifact_name} artifact")
            if file_warning is not None:
                warnings.append(file_warning)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError as exc:
                warnings.append(
                    f"{path} {artifact_name} artifact is not valid UTF-8: {exc}"
                )
                continue
            except OSError as exc:
                warnings.append(f"{path} {artifact_name} artifact could not be read: {exc}")
                continue
            except json.JSONDecodeError as exc:
                warnings.append(
                    f"{path} {artifact_name} artifact is not valid JSON: {exc.msg}"
                )
                continue
            if not isinstance(payload, dict):
                warnings.append(f"{path} {artifact_name} artifact must contain a JSON object")
                continue
            try:
                artifact = factory(**payload)
            except (TypeError, ValueError) as exc:
                warnings.append(f"{path} {artifact_name} artifact is invalid: {exc}")
                continue
            artifact_id = getattr(artifact, id_field)
            try:
                self._validate_artifact_id(artifact_id)
            except (TypeError, ValueError, LearningJobIntegrityError) as exc:
                warnings.append(f"{path} {artifact_name} artifact has unsafe id: {exc}")
                continue
            if path.stem != artifact_id:
                warnings.append(
                    f"{path} {artifact_name} artifact filename does not match {id_field}"
                )
                continue
            if artifact_id in seen_ids:
                warnings.append(f"{path} duplicate {artifact_name} id: {artifact_id}")
                continue
            seen_ids.add(artifact_id)
            artifacts.append(artifact)
        return _ReadResult(artifacts=artifacts, warnings=warnings)

    def _validate_candidate_consistency(
        self,
        job: LearningJob,
        candidates: list[CandidatePatch],
    ) -> None:
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")
        candidate_ids = [candidate.patch_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise FileExistsError("candidate_patch_ids must be unique")
        if len(job.candidate_patch_ids) != len(set(job.candidate_patch_ids)):
            raise ValueError("duplicate candidate_patch_ids are not allowed")
        if set(candidate_ids) != set(job.candidate_patch_ids):
            raise ValueError("candidate_patch_ids must match candidates")

    def _candidate_consistency_warnings(
        self,
        *,
        jobs: list[LearningJob],
        candidates: list[CandidatePatch],
    ) -> list[str]:
        candidate_ids = {candidate.patch_id for candidate in candidates}
        warnings: list[str] = []
        for job in jobs:
            if len(job.candidate_patch_ids) != len(set(job.candidate_patch_ids)):
                warnings.append(
                    f"{job.job_id} contains duplicate candidate_patch_ids"
                )
            missing = sorted(set(job.candidate_patch_ids) - candidate_ids)
            if missing:
                warnings.append(
                    f"{job.job_id} references missing candidate_patch_ids: {missing}"
                )
        return warnings

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with path.open("x", encoding="utf-8") as handle:
                created = True
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.write("\n")
        except Exception:
            if created:
                path.unlink(missing_ok=True)
            raise

    def _artifact_path(self, directory: Path, artifact_id: str) -> Path:
        self._validate_artifact_id(artifact_id)
        path = directory / f"{artifact_id}.json"
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        return path

    def _root_layout_warning(self) -> str | None:
        if not self.root.exists():
            return None
        if self.root.is_symlink():
            return f"{self.root} must not be a symlink"
        if not self.root.is_dir():
            return f"{self.root} must be a directory"
        return None

    def _raise_if_write_layout_unsafe(self) -> None:
        root_warning = self._root_layout_warning()
        if root_warning is not None:
            raise LearningJobIntegrityError(root_warning)
        for directory, artifact_name in (
            (self.jobs_dir, "job"),
            (self.candidates_dir, "candidate"),
        ):
            directory_warning = self._directory_layout_warning(
                directory,
                artifact_name,
            )
            if directory_warning is not None:
                raise LearningJobIntegrityError(directory_warning)

    def _directory_layout_warning(
        self,
        directory: Path,
        artifact_name: str,
    ) -> str | None:
        if not directory.exists():
            return None
        if directory.is_symlink():
            return f"{directory} {artifact_name} directory must not be a symlink"
        if not directory.is_dir():
            return f"{directory} {artifact_name} directory must be a directory"
        return None

    def _file_layout_warning(self, path: Path, artifact_name: str) -> str | None:
        if path.is_symlink():
            return f"{path} {artifact_name} must not be a symlink"
        if not path.is_file():
            return f"{path} {artifact_name} must be a file"
        try:
            self._validate_artifact_id(path.stem)
        except (TypeError, ValueError, LearningJobIntegrityError) as exc:
            return f"{path} {artifact_name} filename is unsafe: {exc}"
        if path.suffix != ".json":
            return f"{path} {artifact_name} must be a .json file"
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        return None

    def _validate_artifact_id(self, artifact_id: str) -> None:
        if not isinstance(artifact_id, str):
            raise TypeError("artifact id must be a string")
        if _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
            raise LearningJobIntegrityError("artifact id must be a safe filename id")
        stem = artifact_id.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_DEVICE_NAMES:
            raise LearningJobIntegrityError("artifact id must not be a Windows device name")

    def _raise_if_parent_outside_root(self, path: Path) -> None:
        root = self.root.resolve(strict=False)
        parent = path.parent.resolve(strict=False)
        if parent != root and root not in parent.parents:
            raise LearningJobIntegrityError(f"{path} parent must stay under {self.root}")

    def _raise_if_existing_write_target_outside_root(self, path: Path) -> None:
        if not path.exists():
            return
        root = self.root.resolve(strict=False)
        resolved = path.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise LearningJobIntegrityError(f"{path} must stay under {self.root}")


__all__ = [
    "LearningJobIntegrityError",
    "LearningJobState",
    "LearningJobStore",
]
