from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from src.contracts.release_execution import (
    GENESIS_EXECUTION_EVENT_HASH,
    ExecutionAuditEventType,
    FeatureFlagState,
    ReleaseExecutionAuditEvent,
    ReleaseExecutionAuditEvent as _ExecutionAuditEvent,
    ReleaseExecutionRequest,
    ReleaseExecutionResult,
    build_execution_audit_event,
    canonical_execution_payload_hash,
    make_release_execution_event_id,
)


class ReleaseExecutionIntegrityError(RuntimeError):
    """Raised when the release execution store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseExecutionState:
    requests: list[ReleaseExecutionRequest]
    results: list[ReleaseExecutionResult]
    feature_flag_state: dict[str, Any] | None
    audit_events: list[ReleaseExecutionAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class ReleaseExecutionIdempotencyMatch:
    request: ReleaseExecutionRequest
    result: ReleaseExecutionResult | None


@dataclass(frozen=True)
class _ArtifactReadResult:
    artifacts: list[Any]
    warnings: list[str]


@dataclass(frozen=True)
class _AuditEventRecord:
    event: ReleaseExecutionAuditEvent
    audit_path: Path
    line_number: int


@dataclass(frozen=True)
class _AuditReadResult:
    records: list[_AuditEventRecord]
    warnings: list[str]


_Artifact = TypeVar("_Artifact")
_AUDIT_FILE_NAME_RE = re.compile(r"^release_execution_(\d{8})\.jsonl$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class ReleaseExecutionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.requests_dir = self.root / "requests"
        self.results_dir = self.root / "results"
        self.feature_flags_dir = self.root / "feature_flags"
        self.feature_flag_history_dir = self.feature_flags_dir / "history"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseExecutionState:
        root_warning = self._root_layout_warning()
        if root_warning is not None:
            return ReleaseExecutionState(
                requests=[],
                results=[],
                feature_flag_state=None,
                audit_events=[],
                integrity={"status": "failed", "warnings": [root_warning]},
            )

        request_result = self._read_json_dir(
            self.requests_dir,
            ReleaseExecutionRequest,
            artifact_name="request",
            id_field="execution_id",
        )
        result_result = self._read_json_dir(
            self.results_dir,
            ReleaseExecutionResult,
            artifact_name="result",
            id_field="result_id",
        )
        feature_flag_state, flag_warnings = self._read_current_feature_flag()
        audit_result = self._read_audit_events_with_integrity()
        warnings = (
            request_result.warnings
            + result_result.warnings
            + flag_warnings
            + audit_result.warnings
            + self._artifact_consistency_warnings(
                requests=request_result.artifacts,
                results=result_result.artifacts,
                audit_events=[record.event for record in audit_result.records],
            )
        )

        return ReleaseExecutionState(
            requests=sorted(
                request_result.artifacts,
                key=lambda item: item.requested_at,
            ),
            results=sorted(
                result_result.artifacts,
                key=lambda item: item.started_at,
            ),
            feature_flag_state=feature_flag_state,
            audit_events=[record.event for record in audit_result.records],
            integrity=(
                {"status": "verified", "warnings": []}
                if not warnings
                else {"status": "failed", "warnings": warnings}
            ),
        )

    def find_by_idempotency_key(
        self,
        action: str,
        key: str,
    ) -> ReleaseExecutionIdempotencyMatch | None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise ReleaseExecutionIntegrityError(
                "release execution integrity failed; refusing idempotency lookup"
            )
        for request in state.requests:
            if request.action == action and request.idempotency_key == key:
                result = next(
                    (
                        item
                        for item in state.results
                        if item.execution_id == request.execution_id
                    ),
                    None,
                )
                return ReleaseExecutionIdempotencyMatch(
                    request=request,
                    result=result,
                )
        return None

    def assert_idempotent_request_matches(
        self,
        request: ReleaseExecutionRequest,
    ) -> None:
        match = self.find_by_idempotency_key(
            request.action,
            request.idempotency_key,
        )
        if match is None:
            return
        existing_hash = canonical_execution_payload_hash(
            match.request.to_dict()
        )
        incoming_hash = canonical_execution_payload_hash(request.to_dict())
        if existing_hash != incoming_hash:
            raise ReleaseExecutionIntegrityError(
                "idempotency key payload mismatch"
            )

    def write_successful_execution(
        self,
        request: ReleaseExecutionRequest,
        result: ReleaseExecutionResult,
        feature_flag_state: FeatureFlagState,
        *,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed()
        self.assert_idempotent_request_matches(request)
        if result.execution_id != request.execution_id:
            raise ReleaseExecutionIntegrityError(
                "result execution_id must match request execution_id"
            )
        if result.intent_id != request.intent_id:
            raise ReleaseExecutionIntegrityError(
                "result intent_id must match request intent_id"
            )
        if feature_flag_state.source_execution_id != request.execution_id:
            raise ReleaseExecutionIntegrityError(
                "feature flag source_execution_id must match request execution_id"
            )

        request_path = self._artifact_path(
            self.requests_dir,
            request.execution_id,
        )
        result_path = self._artifact_path(self.results_dir, result.result_id)
        history_path = self._artifact_path(
            self.feature_flag_history_dir,
            request.execution_id,
        )
        requested_event, succeeded_event = self._build_success_events(
            request,
            result,
            timestamp=timestamp,
        )

        self._write_json_once(request_path, request.to_dict())
        self._write_json_once(result_path, result.to_dict())
        self._write_json_once(history_path, feature_flag_state.to_dict())
        self._write_current_feature_flag(feature_flag_state)
        self._append_event(requested_event, timestamp=timestamp)
        self._append_event(succeeded_event, timestamp=timestamp)

    def _build_success_events(
        self,
        request: ReleaseExecutionRequest,
        result: ReleaseExecutionResult,
        *,
        timestamp: str,
    ) -> tuple[_ExecutionAuditEvent, _ExecutionAuditEvent]:
        request_event_type: ExecutionAuditEventType = (
            "release_requested"
            if request.action == "release"
            else "rollback_requested"
        )
        success_event_type: ExecutionAuditEventType = (
            "release_succeeded"
            if request.action == "release"
            else "rollback_succeeded"
        )
        previous_hash = self._last_event_hash(request.execution_id)
        requested_event = build_execution_audit_event(
            event_id=make_release_execution_event_id(
                request.execution_id,
                request_event_type,
                timestamp,
            ),
            execution_id=request.execution_id,
            intent_id=request.intent_id,
            event_type=request_event_type,
            actor=request.requested_by,
            timestamp=timestamp,
            payload=request.to_dict(),
            previous_event_hash=previous_hash,
        )
        succeeded_event = build_execution_audit_event(
            event_id=make_release_execution_event_id(
                request.execution_id,
                success_event_type,
                timestamp,
            ),
            execution_id=request.execution_id,
            intent_id=request.intent_id,
            event_type=success_event_type,
            actor=request.requested_by,
            timestamp=timestamp,
            payload=result.to_dict(),
            previous_event_hash=requested_event.event_hash,
        )
        return requested_event, succeeded_event

    def _last_event_hash(self, execution_id: str) -> str:
        records = self._read_audit_events_with_integrity().records
        for record in reversed(records):
            if record.event.execution_id == execution_id:
                return record.event.event_hash
        return GENESIS_EXECUTION_EVENT_HASH

    def _raise_if_integrity_failed(self) -> None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise ReleaseExecutionIntegrityError(
                "release execution integrity failed; refusing write"
            )

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

    def _write_current_feature_flag(self, state: FeatureFlagState) -> None:
        path = self.feature_flags_dir / "current.json"
        temp_path = self.feature_flags_dir / f".{state.source_execution_id}.tmp"
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        self.feature_flags_dir.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, sort_keys=True, indent=2)
            handle.write("\n")
        temp_path.replace(path)

    def _append_event(
        self,
        event: ReleaseExecutionAuditEvent,
        *,
        timestamp: str,
    ) -> None:
        audit_path = self.audit_dir / f"release_execution_{_audit_date(timestamp)}.jsonl"
        self._raise_if_parent_outside_root(audit_path)
        self._raise_if_existing_write_target_outside_root(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")

    def _read_json_dir(
        self,
        directory: Path,
        factory: Callable[..., _Artifact],
        *,
        artifact_name: str,
        id_field: str,
    ) -> _ArtifactReadResult:
        layout_warning = self._directory_layout_warning(directory, artifact_name)
        if layout_warning is not None:
            return _ArtifactReadResult(artifacts=[], warnings=[layout_warning])
        if not directory.exists():
            return _ArtifactReadResult(artifacts=[], warnings=[])
        try:
            paths = sorted(directory.glob("*.json"))
        except OSError as exc:
            return _ArtifactReadResult(
                artifacts=[],
                warnings=[
                    f"{directory} {artifact_name} files could not be listed: {exc}"
                ],
            )

        artifacts: list[Any] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for path in paths:
            file_warning = self._file_layout_warning(
                path,
                f"{artifact_name} artifact",
            )
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
                warnings.append(
                    f"{path} {artifact_name} artifact could not be read: {exc}"
                )
                continue
            except json.JSONDecodeError as exc:
                warnings.append(
                    f"{path} {artifact_name} artifact is not valid JSON: {exc.msg}"
                )
                continue
            if not isinstance(payload, dict):
                warnings.append(
                    f"{path} {artifact_name} artifact must contain a JSON object"
                )
                continue
            try:
                artifact = factory(**payload)
            except (TypeError, ValueError) as exc:
                warnings.append(f"{path} {artifact_name} artifact is invalid: {exc}")
                continue
            primary_id = getattr(artifact, id_field)
            try:
                _validate_artifact_id(primary_id)
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"{path} {artifact_name} {id_field} is not file-safe: {exc}"
                )
                continue
            if path.stem != primary_id:
                warnings.append(
                    f"{path} filename does not match {id_field}: {primary_id}"
                )
                continue
            if primary_id in seen_ids:
                warnings.append(f"{path} duplicate {artifact_name} id: {primary_id}")
                continue
            seen_ids.add(primary_id)
            artifacts.append(artifact)
        return _ArtifactReadResult(artifacts=artifacts, warnings=warnings)

    def _read_current_feature_flag(self) -> tuple[dict[str, Any] | None, list[str]]:
        path = self.feature_flags_dir / "current.json"
        directory_warning = self._directory_layout_warning(
            self.feature_flags_dir,
            "feature_flags",
        )
        if directory_warning is not None:
            return None, [directory_warning]
        if not path.exists():
            return None, []
        file_warning = self._file_layout_warning(path, "current feature flag")
        if file_warning is not None:
            return None, [file_warning]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            return None, [f"{path} current feature flag is not valid UTF-8: {exc}"]
        except OSError as exc:
            return None, [f"{path} current feature flag could not be read: {exc}"]
        except json.JSONDecodeError as exc:
            return None, [f"{path} current feature flag is not valid JSON: {exc.msg}"]
        if not isinstance(payload, dict):
            return None, [f"{path} current feature flag must contain a JSON object"]
        try:
            return FeatureFlagState(**payload).to_dict(), []
        except (TypeError, ValueError) as exc:
            return None, [f"{path} current feature flag is invalid: {exc}"]

    def _read_audit_events_with_integrity(self) -> _AuditReadResult:
        layout_warning = self._directory_layout_warning(self.audit_dir, "audit")
        if layout_warning is not None:
            return _AuditReadResult(records=[], warnings=[layout_warning])
        if not self.audit_dir.exists():
            return _AuditReadResult(records=[], warnings=[])
        try:
            audit_paths = sorted(self.audit_dir.glob("*.jsonl"))
        except OSError as exc:
            return _AuditReadResult(
                records=[],
                warnings=[f"{self.audit_dir} audit files could not be listed: {exc}"],
            )

        records: list[_AuditEventRecord] = []
        warnings: list[str] = []
        for path in audit_paths:
            try:
                _audit_file_date(path)
            except ValueError as exc:
                warnings.append(f"{path} audit filename is invalid: {exc}")
                continue
            file_warning = self._file_layout_warning(path, "audit file")
            if file_warning is not None:
                warnings.append(file_warning)
                continue
            try:
                content = path.read_bytes()
            except OSError as exc:
                warnings.append(f"{path} could not be read: {exc}")
                continue
            if content and not content.endswith(b"\n"):
                warnings.append(f"{path} audit file must end with a final newline")
            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                warnings.append(f"{path} is not valid UTF-8: {exc}")
                continue
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(
                        f"{path}:{line_number} is not valid JSON: {exc.msg}"
                    )
                    continue
                if not isinstance(payload, dict):
                    warnings.append(f"{path}:{line_number} must contain a JSON object")
                    continue
                try:
                    event = ReleaseExecutionAuditEvent(**payload)
                except (TypeError, ValueError) as exc:
                    warnings.append(
                        f"{path}:{line_number} audit event is invalid: {exc}"
                    )
                    continue
                records.append(
                    _AuditEventRecord(
                        event=event,
                        audit_path=path,
                        line_number=line_number,
                    )
                )
        warnings.extend(self._audit_chain_warnings(records))
        return _AuditReadResult(records=records, warnings=warnings)

    def _audit_chain_warnings(self, records: list[_AuditEventRecord]) -> list[str]:
        warnings: list[str] = []
        previous_hash_by_execution: dict[str, str] = {}
        for record in records:
            event = record.event
            expected_previous_hash = previous_hash_by_execution.get(
                event.execution_id,
                GENESIS_EXECUTION_EVENT_HASH,
            )
            if event.previous_event_hash != expected_previous_hash:
                warnings.append(
                    f"{event.event_id} previous_event_hash mismatch: "
                    f"expected {expected_previous_hash}, got {event.previous_event_hash}"
                )
            previous_hash_by_execution[event.execution_id] = event.event_hash
        return warnings

    def _artifact_consistency_warnings(
        self,
        *,
        requests: list[ReleaseExecutionRequest],
        results: list[ReleaseExecutionResult],
        audit_events: list[ReleaseExecutionAuditEvent],
    ) -> list[str]:
        warnings: list[str] = []
        request_ids = {request.execution_id for request in requests}
        for result in results:
            if result.execution_id not in request_ids:
                warnings.append(
                    f"result artifact {result.result_id} references unknown request {result.execution_id}"
                )
        request_hashes = {
            (request.execution_id, canonical_execution_payload_hash(request.to_dict()))
            for request in requests
        }
        result_hashes = {
            (result.execution_id, canonical_execution_payload_hash(result.to_dict()))
            for result in results
        }
        request_event_types = {"release_requested", "rollback_requested"}
        result_event_types = {"release_succeeded", "rollback_succeeded"}
        audit_request_hashes = {
            (event.execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type in request_event_types
        }
        audit_result_hashes = {
            (event.execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type in result_event_types
        }
        for execution_id, payload_hash in request_hashes:
            if (execution_id, payload_hash) not in audit_request_hashes:
                warnings.append(
                    f"request artifact {execution_id} does not match an audit payload hash"
                )
        for execution_id, payload_hash in result_hashes:
            if (execution_id, payload_hash) not in audit_result_hashes:
                warnings.append(
                    f"result artifact {execution_id} does not match an audit payload hash"
                )
        return warnings

    def _artifact_path(self, directory: Path, artifact_id: str) -> Path:
        _validate_artifact_id(artifact_id)
        return directory / f"{artifact_id}.json"

    def _root_layout_warning(self) -> str | None:
        try:
            root_is_symlink = self.root.is_symlink()
            root_exists = self.root.exists()
        except OSError as exc:
            return f"{self.root} execution root could not be inspected: {exc}"
        if root_is_symlink:
            return f"{self.root} execution root is a symlink"
        if not root_exists:
            return None
        try:
            root_is_dir = self.root.is_dir()
        except OSError as exc:
            return f"{self.root} execution root could not be inspected: {exc}"
        if not root_is_dir:
            return f"{self.root} execution root must be a directory"
        return None

    def _directory_layout_warning(self, path: Path, label: str) -> str | None:
        try:
            path_exists = path.exists()
            path_is_symlink = path.is_symlink()
        except OSError as exc:
            return f"{path} {label} path could not be inspected: {exc}"
        if path_is_symlink:
            return f"{path} {label} path is a symlink"
        if not path_exists:
            return None
        try:
            path_is_dir = path.is_dir()
        except OSError as exc:
            return f"{path} {label} path could not be inspected: {exc}"
        if not path_is_dir:
            return f"{path} {label} path must be a directory"
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            return f"{path} {label} path could not be resolved: {exc}"
        resolved_root = self.root.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            return f"{path} {label} path resolves outside execution root {self.root}"
        return None

    def _file_layout_warning(self, path: Path, label: str) -> str | None:
        try:
            if path.is_symlink():
                return f"{path} {label} path is a symlink"
            if not path.is_file():
                return f"{path} {label} path must be a regular file"
        except OSError as exc:
            return f"{path} {label} path could not be inspected: {exc}"
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            return f"{path} {label} path could not be resolved: {exc}"
        resolved_root = self.root.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            return f"{path} {label} path resolves outside execution root {self.root}"
        return None

    def _raise_if_parent_outside_root(self, path: Path) -> None:
        resolved_root = self.root.resolve(strict=False)
        resolved_parent = path.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseExecutionIntegrityError(
                f"{path.parent} resolves outside execution root {self.root}"
            ) from exc

    def _raise_if_existing_write_target_outside_root(self, path: Path) -> None:
        if path.is_symlink():
            raise ReleaseExecutionIntegrityError(
                f"{path} is a symlink and cannot be used for execution writes"
            )
        if not path.exists():
            return
        resolved_root = self.root.resolve(strict=False)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseExecutionIntegrityError(
                f"{path} resolves outside execution root {self.root}"
            ) from exc


def _audit_date(timestamp: str) -> str:
    date_prefix = timestamp[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_prefix) is None:
        raise ValueError("timestamp must start with YYYY-MM-DD")
    try:
        date.fromisoformat(date_prefix)
    except ValueError as exc:
        raise ValueError("timestamp must start with a valid YYYY-MM-DD date") from exc
    return date_prefix.replace("-", "")


def _audit_file_date(path: Path) -> str:
    match = _AUDIT_FILE_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError("audit filename must match release_execution_YYYYMMDD.jsonl")
    audit_date = match.group(1)
    date_prefix = f"{audit_date[:4]}-{audit_date[4:6]}-{audit_date[6:]}"
    try:
        date.fromisoformat(date_prefix)
    except ValueError as exc:
        raise ValueError("audit filename must contain a valid YYYYMMDD date") from exc
    return audit_date


def _validate_artifact_id(artifact_id: str) -> None:
    if not isinstance(artifact_id, str):
        raise TypeError("artifact_id must be a string")
    if _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        raise ValueError(
            "artifact_id must be a file-safe identifier matching "
            "[A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    if artifact_id.endswith(".") or artifact_id.endswith(" "):
        raise ValueError("artifact_id must be a file-safe identifier")
    reserved_check = artifact_id.split(".", 1)[0].upper()
    if reserved_check in _WINDOWS_RESERVED_DEVICE_NAMES:
        raise ValueError("artifact_id must be a file-safe identifier")


__all__ = [
    "ReleaseExecutionIdempotencyMatch",
    "ReleaseExecutionIntegrityError",
    "ReleaseExecutionState",
    "ReleaseExecutionStore",
]
