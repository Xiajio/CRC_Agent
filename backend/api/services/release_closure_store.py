from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from src.contracts.release_closure import (
    GENESIS_CLOSURE_EVENT_HASH,
    ReleaseClosureAuditEvent,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    build_release_closure_audit_event,
    canonical_closure_payload_hash,
    make_release_closure_event_id,
)


class ReleaseClosureIntegrityError(RuntimeError):
    """Raised when the release closure store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseClosureState:
    closures: list[ReleaseClosureRecord]
    evidence_packages: list[ReleaseEvidencePackage]
    audit_events: list[ReleaseClosureAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class ReleaseClosureIdempotencyMatch:
    closure: ReleaseClosureRecord
    package: ReleaseEvidencePackage


@dataclass(frozen=True)
class _ArtifactReadResult:
    artifacts: list[Any]
    warnings: list[str]


@dataclass(frozen=True)
class _AuditEventRecord:
    event: ReleaseClosureAuditEvent
    audit_path: Path
    line_number: int


@dataclass(frozen=True)
class _AuditReadResult:
    records: list[_AuditEventRecord]
    warnings: list[str]


_Artifact = TypeVar("_Artifact")
_AUDIT_FILE_NAME_RE = re.compile(r"^release_closure_(\d{8})\.jsonl$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class ReleaseClosureStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.closures_dir = self.root / "closures"
        self.packages_dir = self.root / "packages"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseClosureState:
        return self._read_state_with_integrity()

    def find_closure_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> ReleaseClosureIdempotencyMatch | None:
        state = self.read_state()
        if state.integrity["status"] != "verified":
            raise ReleaseClosureIntegrityError(
                "release closure integrity failed; refusing idempotency lookup"
            )
        packages_by_id = {
            package.package_id: package for package in state.evidence_packages
        }
        for closure in state.closures:
            if closure.idempotency_key != idempotency_key:
                continue
            package = packages_by_id.get(closure.evidence_package_id)
            if package is None:
                return None
            return ReleaseClosureIdempotencyMatch(closure=closure, package=package)
        return None

    def assert_idempotent_closure_matches(
        self,
        closure: ReleaseClosureRecord,
        package: ReleaseEvidencePackage,
    ) -> None:
        match = self.find_closure_by_idempotency_key(closure.idempotency_key)
        if match is None:
            return
        existing_closure_hash = canonical_closure_payload_hash(match.closure.to_dict())
        incoming_closure_hash = canonical_closure_payload_hash(closure.to_dict())
        existing_package_hash = canonical_closure_payload_hash(match.package.to_dict())
        incoming_package_hash = canonical_closure_payload_hash(package.to_dict())
        if (
            existing_closure_hash != incoming_closure_hash
            or existing_package_hash != incoming_package_hash
        ):
            raise FileExistsError("idempotency payload mismatch")

    def write_closure_with_package(
        self,
        closure: ReleaseClosureRecord,
        package: ReleaseEvidencePackage,
        *,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed()
        self._validate_closure_package_pair(closure, package)
        self.assert_idempotent_closure_matches(closure, package)
        self._ensure_root()
        closure_path = self._artifact_path(self.closures_dir, closure.closure_id)
        package_path = self._artifact_path(self.packages_dir, package.package_id)
        self._audit_path(timestamp)
        closure_event = build_release_closure_audit_event(
            event_id=make_release_closure_event_id(
                closure.release_execution_id,
                "closure_recorded",
                f"{timestamp}#{closure.closure_id}",
            ),
            intent_id=closure.intent_id,
            release_execution_id=closure.release_execution_id,
            event_type="closure_recorded",
            actor=closure.closed_by,
            timestamp=timestamp,
            payload=closure.to_dict(),
            previous_event_hash=self._last_event_hash(closure.release_execution_id),
        )
        package_event = build_release_closure_audit_event(
            event_id=make_release_closure_event_id(
                package.release_execution_id,
                "evidence_package_generated",
                f"{timestamp}#{package.package_id}",
            ),
            intent_id=package.intent_id,
            release_execution_id=package.release_execution_id,
            event_type="evidence_package_generated",
            actor=package.generated_by,
            timestamp=timestamp,
            payload=package.to_dict(),
            previous_event_hash=closure_event.event_hash,
        )

        self._write_json_once(closure_path, closure.to_dict())
        try:
            self._write_json_once(package_path, package.to_dict())
        except Exception:
            closure_path.unlink(missing_ok=True)
            raise
        try:
            self._append_audit_events(
                (closure_event, package_event),
                timestamp=timestamp,
            )
        except Exception:
            package_path.unlink(missing_ok=True)
            closure_path.unlink(missing_ok=True)
            raise

    def _read_state_with_integrity(self) -> ReleaseClosureState:
        root_warning = self._root_layout_warning()
        if root_warning is not None:
            return ReleaseClosureState(
                closures=[],
                evidence_packages=[],
                audit_events=[],
                integrity={"status": "failed", "warnings": [root_warning]},
            )

        closure_result = self._read_json_dir(
            self.closures_dir,
            ReleaseClosureRecord,
            artifact_name="closure",
            id_field="closure_id",
        )
        package_result = self._read_json_dir(
            self.packages_dir,
            ReleaseEvidencePackage,
            artifact_name="package",
            id_field="package_id",
        )
        audit_result = self._read_audit_events_with_integrity()
        audit_events = [record.event for record in audit_result.records]
        warnings = (
            closure_result.warnings
            + package_result.warnings
            + audit_result.warnings
            + self._artifact_consistency_warnings(
                closures=closure_result.artifacts,
                packages=package_result.artifacts,
                audit_events=audit_events,
            )
        )
        return ReleaseClosureState(
            closures=sorted(closure_result.artifacts, key=lambda item: item.closed_at),
            evidence_packages=sorted(
                package_result.artifacts,
                key=lambda item: item.generated_at,
            ),
            audit_events=audit_events,
            integrity=(
                {"status": "verified", "warnings": []}
                if not warnings
                else {"status": "failed", "warnings": warnings}
            ),
        )

    def _last_event_hash(self, release_execution_id: str) -> str:
        records = self._read_audit_events_with_integrity().records
        for record in reversed(records):
            if record.event.release_execution_id == release_execution_id:
                return record.event.event_hash
        return GENESIS_CLOSURE_EVENT_HASH

    def _raise_if_integrity_failed(self) -> None:
        state = self.read_state()
        if state.integrity["status"] != "verified":
            raise ReleaseClosureIntegrityError(
                "release closure integrity failed; refusing write"
            )

    def _ensure_root(self) -> None:
        self._raise_if_parent_outside_root(self.closures_dir)
        self._raise_if_parent_outside_root(self.packages_dir)
        self._raise_if_parent_outside_root(self.audit_dir)

    def _artifact_path(self, directory: Path, artifact_id: str) -> Path:
        _validate_artifact_id(artifact_id)
        return directory / f"{artifact_id}.json"

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

    def _append_audit_event(
        self,
        event: ReleaseClosureAuditEvent,
        *,
        timestamp: str,
    ) -> None:
        self._append_audit_events((event,), timestamp=timestamp)

    def _append_audit_events(
        self,
        events: tuple[ReleaseClosureAuditEvent, ...] | list[ReleaseClosureAuditEvent],
        *,
        timestamp: str,
    ) -> None:
        audit_path = self._audit_path(timestamp)
        self._raise_if_parent_outside_root(audit_path)
        self._raise_if_existing_write_target_outside_root(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_preexisted = audit_path.exists()
        with audit_path.open("a+", encoding="utf-8") as handle:
            handle.seek(0, 2)
            initial_size = handle.tell()
            try:
                for event in events:
                    handle.write(json.dumps(event.to_dict(), sort_keys=True))
                    handle.write("\n")
                handle.flush()
            except Exception:
                handle.flush()
                handle.seek(initial_size)
                handle.truncate()
                raise
        if not audit_preexisted and audit_path.exists() and audit_path.stat().st_size == 0:
            audit_path.unlink(missing_ok=True)

    def _audit_path(self, timestamp: str) -> Path:
        audit_path = self.audit_dir / f"release_closure_{_audit_date(timestamp)}.jsonl"
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
                    event = ReleaseClosureAuditEvent(**payload)
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

    def _artifact_consistency_warnings(
        self,
        *,
        closures: list[ReleaseClosureRecord],
        packages: list[ReleaseEvidencePackage],
        audit_events: list[ReleaseClosureAuditEvent],
    ) -> list[str]:
        warnings: list[str] = []
        packages_by_id = {package.package_id: package for package in packages}
        closure_hashes = {
            (
                closure.release_execution_id,
                canonical_closure_payload_hash(closure.to_dict()),
            )
            for closure in closures
        }
        package_hashes = {
            (
                package.release_execution_id,
                canonical_closure_payload_hash(package.to_dict()),
            )
            for package in packages
        }
        audit_closure_hashes = {
            (event.release_execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type == "closure_recorded"
        }
        audit_package_hashes = {
            (event.release_execution_id, event.payload_hash)
            for event in audit_events
            if event.event_type == "evidence_package_generated"
        }

        for closure in closures:
            if closure.evidence_package_id not in packages_by_id:
                warnings.append(
                    f"closure artifact {closure.closure_id} references missing evidence package"
                )
        for package in packages:
            matching_closure = next(
                (closure for closure in closures if closure.closure_id == package.closure_id),
                None,
            )
            if matching_closure is None:
                warnings.append(
                    f"package artifact {package.package_id} references missing closure artifact"
                )
            elif matching_closure.evidence_package_id != package.package_id:
                warnings.append(
                    f"package artifact {package.package_id} does not match closure evidence_package_id"
                )
        for release_execution_id, payload_hash in closure_hashes:
            if (release_execution_id, payload_hash) not in audit_closure_hashes:
                warnings.append(
                    f"closure artifact {release_execution_id} does not match an audit payload hash"
                )
        for release_execution_id, payload_hash in package_hashes:
            if (release_execution_id, payload_hash) not in audit_package_hashes:
                warnings.append(
                    f"package artifact {release_execution_id} does not match an audit payload hash"
                )
        for event in audit_events:
            if event.event_type == "closure_recorded" and (
                event.release_execution_id,
                event.payload_hash,
            ) not in closure_hashes:
                warnings.append(
                    f"audit closure_recorded event {event.event_id} references missing closure artifact"
                )
            if event.event_type == "evidence_package_generated" and (
                event.release_execution_id,
                event.payload_hash,
            ) not in package_hashes:
                warnings.append(
                    f"audit evidence_package_generated event {event.event_id} references missing package artifact"
                )
        return warnings

    def _audit_chain_warnings(self, records: list[_AuditEventRecord]) -> list[str]:
        warnings: list[str] = []
        previous_hash_by_execution: dict[str, str] = {}
        for record in records:
            event = record.event
            expected_previous_hash = previous_hash_by_execution.get(
                event.release_execution_id,
                GENESIS_CLOSURE_EVENT_HASH,
            )
            if event.previous_event_hash != expected_previous_hash:
                warnings.append(
                    f"{event.event_id} previous_event_hash mismatch: "
                    f"expected {expected_previous_hash}, got {event.previous_event_hash}"
                )
            previous_hash_by_execution[event.release_execution_id] = event.event_hash
        return warnings

    def _root_layout_warning(self) -> str | None:
        try:
            root_is_symlink = self.root.is_symlink()
            root_exists = self.root.exists()
        except OSError as exc:
            return f"{self.root} closure root could not be inspected: {exc}"
        if root_is_symlink:
            return f"{self.root} closure root is a symlink"
        if not root_exists:
            return None
        try:
            root_is_dir = self.root.is_dir()
        except OSError as exc:
            return f"{self.root} closure root could not be inspected: {exc}"
        if not root_is_dir:
            return f"{self.root} closure root must be a directory"
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
            return f"{path} {label} path resolves outside closure root {self.root}"
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
            return f"{path} {label} path resolves outside closure root {self.root}"
        return None

    def _raise_if_parent_outside_root(self, path: Path) -> None:
        resolved_root = self.root.resolve(strict=False)
        resolved_parent = path.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseClosureIntegrityError(
                f"{path.parent} resolves outside closure root {self.root}"
            ) from exc

    def _raise_if_existing_write_target_outside_root(self, path: Path) -> None:
        if path.is_symlink():
            raise ReleaseClosureIntegrityError(
                f"{path} is a symlink and cannot be used for closure writes"
            )
        if not path.exists():
            return
        resolved_root = self.root.resolve(strict=False)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ReleaseClosureIntegrityError(
                f"{path} resolves outside closure root {self.root}"
            ) from exc

    def _validate_closure_package_pair(
        self,
        closure: ReleaseClosureRecord,
        package: ReleaseEvidencePackage,
    ) -> None:
        mismatches: list[str] = []
        if package.closure_id != closure.closure_id:
            mismatches.append("closure_id")
        if package.package_id != closure.evidence_package_id:
            mismatches.append("evidence_package_id")
        if package.intent_id != closure.intent_id:
            mismatches.append("intent_id")
        if package.release_execution_id != closure.release_execution_id:
            mismatches.append("release_execution_id")
        if package.rollback_execution_id != closure.rollback_execution_id:
            mismatches.append("rollback_execution_id")
        if package.closure_status != closure.closure_status:
            mismatches.append("closure_status")

        expected_closure_ref = f"reports/release_closure/closures/{closure.closure_id}.json"
        if expected_closure_ref not in package.artifact_refs:
            mismatches.append("artifact_refs")
        for artifact_ref in package.artifact_refs:
            if artifact_ref.startswith("reports/release_closure/closures/") and artifact_ref != expected_closure_ref:
                mismatches.append("artifact_refs")
                break

        if mismatches:
            mismatch_summary = ", ".join(sorted(set(mismatches)))
            raise ValueError(f"closure/package mismatch: {mismatch_summary}")


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
        raise ValueError("audit filename must match release_closure_YYYYMMDD.jsonl")
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
    "ReleaseClosureIdempotencyMatch",
    "ReleaseClosureIntegrityError",
    "ReleaseClosureState",
    "ReleaseClosureStore",
]
