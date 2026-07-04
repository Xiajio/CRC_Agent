from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from src.contracts.release_monitoring import (
    GENESIS_MONITORING_EVENT_HASH,
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringAuditEvent,
    ReleaseMonitoringCheck,
    build_monitoring_audit_event,
    canonical_monitoring_payload_hash,
    make_monitoring_event_id,
)


class ReleaseMonitoringIntegrityError(RuntimeError):
    """Raised when the release monitoring store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseMonitoringState:
    checks: list[ReleaseMonitoringCheck]
    acknowledgements: list[ReleaseMonitoringAcknowledgement]
    audit_events: list[ReleaseMonitoringAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class ReleaseMonitoringCheckIdempotencyMatch:
    check: ReleaseMonitoringCheck


@dataclass(frozen=True)
class _ArtifactReadResult:
    artifacts: list[Any]
    warnings: list[str]


@dataclass(frozen=True)
class _AuditEventRecord:
    event: ReleaseMonitoringAuditEvent
    audit_path: Path
    line_number: int


@dataclass(frozen=True)
class _AuditReadResult:
    records: list[_AuditEventRecord]
    warnings: list[str]


_Artifact = TypeVar("_Artifact")
_AUDIT_FILE_NAME_RE = re.compile(r"^release_monitoring_(\d{8})\.jsonl$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class ReleaseMonitoringStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.checks_dir = self.root / "checks"
        self.acknowledgements_dir = self.root / "acknowledgements"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseMonitoringState:
        return self._read_state_with_integrity()

    def find_check_by_idempotency_key(
        self,
        check_type: str,
        key: str,
    ) -> ReleaseMonitoringCheckIdempotencyMatch | None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise ReleaseMonitoringIntegrityError(
                "release monitoring integrity failed; refusing idempotency lookup"
            )
        for check in state.checks:
            if check.check_type == check_type and check.idempotency_key == key:
                return ReleaseMonitoringCheckIdempotencyMatch(check=check)
        return None

    def assert_idempotent_check_matches(
        self,
        check: ReleaseMonitoringCheck,
    ) -> None:
        match = self.find_check_by_idempotency_key(
            check.check_type,
            check.idempotency_key,
        )
        if match is None:
            return
        existing_hash = canonical_monitoring_payload_hash(match.check.to_dict())
        incoming_hash = canonical_monitoring_payload_hash(check.to_dict())
        if existing_hash != incoming_hash:
            raise ReleaseMonitoringIntegrityError(
                "idempotency key payload mismatch"
            )

    def write_check(
        self,
        check: ReleaseMonitoringCheck,
        *,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed()
        self.assert_idempotent_check_matches(check)
        check_path = self._artifact_path(self.checks_dir, check.check_id)
        self._audit_path(timestamp)
        event = build_monitoring_audit_event(
            event_id=make_monitoring_event_id(
                check.execution_id,
                "check_recorded",
                f"{timestamp}#{check.check_id}",
            ),
            intent_id=check.intent_id,
            execution_id=check.execution_id,
            event_type="check_recorded",
            actor=check.observed_by,
            timestamp=timestamp,
            payload=check.to_dict(),
            previous_event_hash=self._last_event_hash(check.execution_id),
        )

        self._write_json_once(check_path, check.to_dict())
        try:
            self._append_event(event, timestamp=timestamp)
        except Exception:
            check_path.unlink(missing_ok=True)
            raise

    def write_acknowledgement(
        self,
        acknowledgement: ReleaseMonitoringAcknowledgement,
        *,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed()
        acknowledgement_path = self._artifact_path(
            self.acknowledgements_dir,
            acknowledgement.acknowledgement_id,
        )
        self._audit_path(timestamp)
        event = build_monitoring_audit_event(
            event_id=make_monitoring_event_id(
                acknowledgement.execution_id,
                "alert_acknowledged",
                f"{timestamp}#{acknowledgement.acknowledgement_id}",
            ),
            intent_id=acknowledgement.intent_id,
            execution_id=acknowledgement.execution_id,
            event_type="alert_acknowledged",
            actor=acknowledgement.acknowledged_by,
            timestamp=timestamp,
            payload=acknowledgement.to_dict(),
            previous_event_hash=self._last_event_hash(acknowledgement.execution_id),
        )

        self._write_json_once(acknowledgement_path, acknowledgement.to_dict())
        try:
            self._append_event(event, timestamp=timestamp)
        except Exception:
            acknowledgement_path.unlink(missing_ok=True)
            raise

    def _read_state_with_integrity(self) -> ReleaseMonitoringState:
        root_warning = self._root_layout_warning()
        if root_warning is not None:
            return ReleaseMonitoringState(
                checks=[],
                acknowledgements=[],
                audit_events=[],
                integrity={"status": "failed", "warnings": [root_warning]},
            )

        check_result = self._read_json_dir(
            self.checks_dir,
            ReleaseMonitoringCheck,
            artifact_name="check",
            id_field="check_id",
        )
        acknowledgement_result = self._read_json_dir(
            self.acknowledgements_dir,
            ReleaseMonitoringAcknowledgement,
            artifact_name="acknowledgement",
            id_field="acknowledgement_id",
        )
        audit_result = self._read_audit_events_with_integrity()
        audit_events = [record.event for record in audit_result.records]
        warnings = (
            check_result.warnings
            + acknowledgement_result.warnings
            + audit_result.warnings
            + self._artifact_consistency_warnings(
                checks=check_result.artifacts,
                acknowledgements=acknowledgement_result.artifacts,
                audit_events=audit_events,
            )
        )

        return ReleaseMonitoringState(
            checks=sorted(check_result.artifacts, key=lambda item: item.observed_at),
            acknowledgements=sorted(
                acknowledgement_result.artifacts,
                key=lambda item: item.acknowledged_at,
            ),
            audit_events=audit_events,
            integrity=(
                {"status": "verified", "warnings": []}
                if not warnings
                else {"status": "failed", "warnings": warnings}
            ),
        )

    def _last_event_hash(self, execution_id: str) -> str:
        records = self._read_audit_events_with_integrity().records
        for record in reversed(records):
            if record.event.execution_id == execution_id:
                return record.event.event_hash
        return GENESIS_MONITORING_EVENT_HASH

    def _raise_if_integrity_failed(self) -> None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise ReleaseMonitoringIntegrityError(
                "release monitoring integrity failed; refusing write"
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

    def _append_event(
        self,
        event: ReleaseMonitoringAuditEvent,
        *,
        timestamp: str,
    ) -> None:
        audit_path = self._audit_path(timestamp)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")

    def _audit_path(self, timestamp: str) -> Path:
        audit_path = self.audit_dir / f"release_monitoring_{_audit_date(timestamp)}.jsonl"
        self._raise_if_parent_outside_root(audit_path)
        self._raise_if_existing_write_target_outside_root(audit_path)
        return audit_path

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
        seen_event_ids: set[str] = set()
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
                    event = ReleaseMonitoringAuditEvent(**payload)
                except (TypeError, ValueError) as exc:
                    warnings.append(
                        f"{path}:{line_number} audit event is invalid: {exc}"
                    )
                    continue
                if event.event_id in seen_event_ids:
                    warnings.append(
                        f"{path}:{line_number} duplicate audit event id: {event.event_id}"
                    )
                seen_event_ids.add(event.event_id)
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
                GENESIS_MONITORING_EVENT_HASH,
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
        checks: list[ReleaseMonitoringCheck],
        acknowledgements: list[ReleaseMonitoringAcknowledgement],
        audit_events: list[ReleaseMonitoringAuditEvent],
    ) -> list[str]:
        warnings: list[str] = []
        check_hashes = {
            (check.execution_id, canonical_monitoring_payload_hash(check.to_dict()))
            for check in checks
        }
        acknowledgement_hashes = {
            (
                acknowledgement.execution_id,
                canonical_monitoring_payload_hash(acknowledgement.to_dict()),
            )
            for acknowledgement in acknowledgements
        }
        audit_check_hashes = {
            (event.execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type == "check_recorded"
        }
        audit_acknowledgement_hashes = {
            (event.execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type == "alert_acknowledged"
        }
        for execution_id, payload_hash in check_hashes:
            if (execution_id, payload_hash) not in audit_check_hashes:
                warnings.append(
                    f"check artifact {execution_id} does not match an audit payload hash"
                )
        for event in audit_events:
            if event.event_type != "check_recorded":
                continue
            if (event.execution_id, event.payload_hash) not in check_hashes:
                warnings.append(
                    f"audit check_recorded event {event.event_id} references missing check artifact"
                )
        for execution_id, payload_hash in acknowledgement_hashes:
            if (execution_id, payload_hash) not in audit_acknowledgement_hashes:
                warnings.append(
                    f"acknowledgement artifact {execution_id} does not match an audit payload hash"
                )
        for event in audit_events:
            if event.event_type != "alert_acknowledged":
                continue
            if (event.execution_id, event.payload_hash) not in acknowledgement_hashes:
                warnings.append(
                    f"audit alert_acknowledged event {event.event_id} references missing acknowledgement artifact"
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
            return f"{self.root} monitoring root could not be inspected: {exc}"
        if root_is_symlink:
            return f"{self.root} monitoring root is a symlink"
        if not root_exists:
            return None
        try:
            root_is_dir = self.root.is_dir()
        except OSError as exc:
            return f"{self.root} monitoring root could not be inspected: {exc}"
        if not root_is_dir:
            return f"{self.root} monitoring root must be a directory"
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
            return f"{path} {label} path resolves outside monitoring root {self.root}"
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
            return f"{path} {label} path resolves outside monitoring root {self.root}"
        return None

    def _raise_if_parent_outside_root(self, path: Path) -> None:
        resolved_root = self.root.resolve(strict=False)
        resolved_parent = path.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseMonitoringIntegrityError(
                f"{path.parent} resolves outside monitoring root {self.root}"
            ) from exc

    def _raise_if_existing_write_target_outside_root(self, path: Path) -> None:
        if path.is_symlink():
            raise ReleaseMonitoringIntegrityError(
                f"{path} is a symlink and cannot be used for monitoring writes"
            )
        if not path.exists():
            return
        resolved_root = self.root.resolve(strict=False)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseMonitoringIntegrityError(
                f"{path} resolves outside monitoring root {self.root}"
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
        raise ValueError("audit filename must match release_monitoring_YYYYMMDD.jsonl")
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
    "ReleaseMonitoringCheckIdempotencyMatch",
    "ReleaseMonitoringIntegrityError",
    "ReleaseMonitoringState",
    "ReleaseMonitoringStore",
]
