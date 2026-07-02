from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from src.contracts.release_governance import (
    GENESIS_EVENT_HASH,
    ReleaseApproval,
    ReleaseAuditEvent,
    ReleaseAuditEventType,
    ReleaseIntent,
    ReleaseRollbackPlan,
    build_audit_event,
    canonical_payload_hash,
    make_release_audit_event_id,
    validate_audit_event_hash,
)


class GovernanceIntegrityError(RuntimeError):
    """Raised when an existing governance store is not safe to append to."""


@dataclass(frozen=True)
class ReleaseGovernanceState:
    intents: list[ReleaseIntent]
    approvals: list[ReleaseApproval]
    rollback_plans: list[ReleaseRollbackPlan]
    audit_events: list[ReleaseAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class _AuditEventRecord:
    event: ReleaseAuditEvent
    audit_path: Path
    line_number: int


@dataclass(frozen=True)
class _AuditReadResult:
    records: list[_AuditEventRecord]
    warnings: list[str]
    failed_intent_ids: frozenset[str]
    global_failure: bool


@dataclass(frozen=True)
class _ArtifactReadResult:
    artifacts: list[Any]
    warnings: list[str]
    affected_intent_ids: frozenset[str]


@dataclass(frozen=True)
class _ArtifactConsistencyResult:
    warnings: list[str]
    affected_intent_ids: frozenset[str]


@dataclass(frozen=True)
class _PreparedAuditEvent:
    event: ReleaseAuditEvent
    audit_path: Path


_Artifact = TypeVar("_Artifact")
_AUDIT_FILE_NAME_RE = re.compile(r"^release_audit_(\d{8})\.jsonl$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class ReleaseGovernanceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.intents_dir = self.root / "intents"
        self.approvals_dir = self.root / "approvals"
        self.rollback_plans_dir = self.root / "rollback_plans"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseGovernanceState:
        audit_result = self._read_audit_events_with_integrity()
        intent_result = self._read_json_dir(
            self.intents_dir,
            ReleaseIntent,
            artifact_name="intent",
            id_field="intent_id",
        )
        approval_result = self._read_json_dir(
            self.approvals_dir,
            ReleaseApproval,
            artifact_name="approval",
            id_field="approval_id",
        )
        rollback_plan_result = self._read_json_dir(
            self.rollback_plans_dir,
            ReleaseRollbackPlan,
            artifact_name="rollback plan",
            id_field="rollback_plan_id",
        )
        artifact_warnings = (
            intent_result.warnings
            + approval_result.warnings
            + rollback_plan_result.warnings
        )
        consistency_result = self._verify_artifact_audit_consistency(
            intents=intent_result.artifacts,
            approvals=approval_result.artifacts,
            rollback_plans=rollback_plan_result.artifacts,
            audit_result=audit_result,
        )

        return ReleaseGovernanceState(
            intents=sorted(
                intent_result.artifacts,
                key=lambda intent: intent.requested_at,
            ),
            approvals=sorted(
                approval_result.artifacts,
                key=lambda approval: approval.signed_at,
            ),
            rollback_plans=sorted(
                rollback_plan_result.artifacts,
                key=lambda plan: plan.created_at,
            ),
            audit_events=[record.event for record in audit_result.records],
            integrity=self._verify_integrity(
                audit_result,
                artifact_warnings=artifact_warnings,
                consistency_warnings=consistency_result.warnings,
                artifact_affected_intent_ids=(
                    consistency_result.affected_intent_ids
                    | intent_result.affected_intent_ids
                    | approval_result.affected_intent_ids
                    | rollback_plan_result.affected_intent_ids
                ),
            ),
        )

    def write_intent(
        self,
        intent: ReleaseIntent,
        *,
        actor: str,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed(intent.intent_id)
        artifact_path = self._artifact_path(self.intents_dir, intent.intent_id)
        prepared_event = self._prepare_event(
            intent_id=intent.intent_id,
            event_type="intent_created",
            actor=actor,
            timestamp=timestamp,
            payload=intent.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=artifact_path,
            payload=intent.to_dict(),
            prepared_event=prepared_event,
        )

    def write_approval(
        self,
        approval: ReleaseApproval,
        *,
        actor: str,
        timestamp: str,
    ) -> None:
        state = self._raise_if_integrity_failed(approval.intent_id)
        self._raise_if_intent_missing(approval.intent_id, state)
        artifact_path = self._artifact_path(self.approvals_dir, approval.approval_id)
        prepared_event = self._prepare_event(
            intent_id=approval.intent_id,
            event_type="approval_recorded",
            actor=actor,
            timestamp=timestamp,
            payload=approval.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=artifact_path,
            payload=approval.to_dict(),
            prepared_event=prepared_event,
        )

    def write_rollback_plan(
        self,
        plan: ReleaseRollbackPlan,
        *,
        actor: str,
        timestamp: str,
    ) -> None:
        state = self._raise_if_integrity_failed(plan.intent_id)
        self._raise_if_intent_missing(plan.intent_id, state)
        artifact_path = self._artifact_path(
            self.rollback_plans_dir,
            plan.rollback_plan_id,
        )
        prepared_event = self._prepare_event(
            intent_id=plan.intent_id,
            event_type="rollback_plan_recorded",
            actor=actor,
            timestamp=timestamp,
            payload=plan.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=artifact_path,
            payload=plan.to_dict(),
            prepared_event=prepared_event,
        )

    def append_cancel_event(
        self,
        *,
        intent_id: str,
        actor: str,
        reason: str,
        timestamp: str,
    ) -> None:
        state = self._raise_if_integrity_failed(intent_id)
        self._raise_if_intent_missing(intent_id, state)
        prepared_event = self._prepare_event(
            intent_id=intent_id,
            event_type="intent_cancelled",
            actor=actor,
            timestamp=timestamp,
            payload={
                "intent_id": intent_id,
                "actor": actor,
                "reason": reason,
            },
        )
        self._append_prepared_event(prepared_event)

    def _write_artifact_and_append_event(
        self,
        *,
        path: Path,
        payload: dict[str, Any],
        prepared_event: _PreparedAuditEvent,
    ) -> None:
        self._write_json_once(path, payload)
        try:
            self._append_prepared_event(prepared_event)
        except Exception:
            try:
                self._remove_new_artifact(path)
            except Exception as cleanup_error:
                raise GovernanceIntegrityError(
                    f"artifact cleanup failed after audit append failure: {path}"
                ) from cleanup_error
            raise

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        self._raise_if_parent_outside_root(path)
        self._raise_if_existing_write_target_outside_root(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

    def _remove_new_artifact(self, path: Path) -> None:
        path.unlink()

    def _prepare_event(
        self,
        *,
        intent_id: str,
        event_type: ReleaseAuditEventType,
        actor: str,
        timestamp: str,
        payload: dict[str, Any],
    ) -> _PreparedAuditEvent:
        audit_path = self.audit_dir / f"release_audit_{_audit_date(timestamp)}.jsonl"
        latest_audit_file_name = self._latest_audit_file_name()
        if (
            latest_audit_file_name is not None
            and audit_path.name < latest_audit_file_name
        ):
            raise GovernanceIntegrityError(
                "backdated audit event would be stored before existing audit log"
            )
        last_record = self._last_event_record(intent_id)
        if last_record is not None and audit_path.name < last_record.audit_path.name:
            raise GovernanceIntegrityError(
                "backdated audit event would be stored before existing audit chain"
            )
        previous_event_hash = (
            last_record.event.event_hash
            if last_record is not None
            else GENESIS_EVENT_HASH
        )
        event = build_audit_event(
            event_id=make_release_audit_event_id(
                intent_id,
                event_type,
                timestamp,
            ),
            intent_id=intent_id,
            event_type=event_type,
            actor=actor,
            timestamp=timestamp,
            payload=payload,
            previous_event_hash=previous_event_hash,
        )
        return _PreparedAuditEvent(event=event, audit_path=audit_path)

    def _append_prepared_event(self, prepared_event: _PreparedAuditEvent) -> None:
        audit_path = prepared_event.audit_path
        self._raise_if_parent_outside_root(audit_path)
        self._raise_if_existing_write_target_outside_root(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prepared_event.event.to_dict(), sort_keys=True))
            handle.write("\n")

    def _last_event_record(self, intent_id: str) -> _AuditEventRecord | None:
        for record in reversed(self._read_audit_events_with_integrity().records):
            if record.event.intent_id == intent_id:
                return record
        return None

    def _latest_audit_file_name(self) -> str | None:
        layout_warning = self._directory_layout_warning(self.audit_dir, "audit")
        if layout_warning is not None:
            raise GovernanceIntegrityError(layout_warning)
        if not self.audit_dir.exists():
            return None
        try:
            audit_paths = sorted(self.audit_dir.glob("*.jsonl"))
        except OSError as exc:
            raise GovernanceIntegrityError(
                f"{self.audit_dir} audit files could not be listed: {exc}"
            ) from exc
        audit_file_names = []
        for path in audit_paths:
            try:
                _audit_file_date(path)
            except ValueError as exc:
                raise GovernanceIntegrityError(
                    f"{path} audit filename is invalid: {exc}"
                ) from exc
            audit_file_names.append(path.name)
        if not audit_file_names:
            return None
        return max(audit_file_names)

    def _raise_if_parent_outside_root(self, path: Path) -> None:
        resolved_root = self.root.resolve(strict=False)
        resolved_parent = path.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise GovernanceIntegrityError(
                f"{path.parent} resolves outside governance root {self.root}"
            ) from exc

    def _raise_if_existing_write_target_outside_root(self, path: Path) -> None:
        if path.is_symlink():
            raise GovernanceIntegrityError(
                f"{path} is a symlink and cannot be used for governance writes"
            )
        if not path.exists():
            return
        resolved_root = self.root.resolve(strict=False)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise GovernanceIntegrityError(
                f"{path} resolves outside governance root {self.root}"
            ) from exc

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
            path_is_directory = path.is_dir()
        except OSError as exc:
            return f"{path} {label} path could not be inspected: {exc}"
        if not path_is_directory:
            return f"{path} {label} path must be a directory"

        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            return f"{path} {label} path could not be resolved: {exc}"
        resolved_root = self.root.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            return f"{path} {label} path resolves outside governance root {self.root}"
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
            return f"{path} {label} path resolves outside governance root {self.root}"
        return None

    def _raise_if_integrity_failed(self, intent_id: str) -> ReleaseGovernanceState:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise GovernanceIntegrityError(
                f"release governance integrity failed; refusing write for {intent_id}"
            )
        return state

    def _raise_if_intent_missing(
        self,
        intent_id: str,
        state: ReleaseGovernanceState,
    ) -> None:
        _validate_artifact_id(intent_id)
        if intent_id not in {intent.intent_id for intent in state.intents}:
            raise GovernanceIntegrityError(f"unknown release intent: {intent_id}")

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
            return _ArtifactReadResult(
                artifacts=[],
                warnings=[layout_warning],
                affected_intent_ids=frozenset(),
            )
        if not directory.exists():
            return _ArtifactReadResult(
                artifacts=[],
                warnings=[],
                affected_intent_ids=frozenset(),
            )

        artifacts: list[Any] = []
        warnings: list[str] = []
        affected_intent_ids: set[str] = set()
        seen_primary_ids: set[str] = set()
        try:
            paths = sorted(directory.glob("*.json"))
        except OSError as exc:
            return _ArtifactReadResult(
                artifacts=[],
                warnings=[
                    f"{directory} {artifact_name} files could not be listed: {exc}"
                ],
                affected_intent_ids=frozenset(),
            )

        for path in paths:
            file_layout_warning = self._file_layout_warning(
                path,
                f"{artifact_name} artifact",
            )
            if file_layout_warning is not None:
                warnings.append(file_layout_warning)
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
            intent_id = getattr(artifact, "intent_id", primary_id)
            try:
                _validate_artifact_id(primary_id)
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"{path} {artifact_name} artifact {id_field} is not file-safe: {exc}"
                )
                affected_intent_ids.add(intent_id)
                continue
            if path.stem != primary_id:
                warnings.append(
                    f"{path} {artifact_name} artifact filename does not match {id_field}: {primary_id}"
                )
                affected_intent_ids.add(intent_id)
                continue
            if primary_id in seen_primary_ids:
                warnings.append(
                    f"{path} duplicate {artifact_name} artifact id: {primary_id}"
                )
                affected_intent_ids.add(intent_id)
                continue

            seen_primary_ids.add(primary_id)
            artifacts.append(artifact)

        return _ArtifactReadResult(
            artifacts=artifacts,
            warnings=warnings,
            affected_intent_ids=frozenset(affected_intent_ids),
        )

    def _read_audit_events(self) -> list[ReleaseAuditEvent]:
        return [
            record.event
            for record in self._read_audit_events_with_integrity().records
        ]

    def _read_audit_events_with_integrity(self) -> _AuditReadResult:
        layout_warning = self._directory_layout_warning(self.audit_dir, "audit")
        if layout_warning is not None:
            return _AuditReadResult(
                records=[],
                warnings=[layout_warning],
                failed_intent_ids=frozenset(),
                global_failure=True,
            )
        if not self.audit_dir.exists():
            return _AuditReadResult(
                records=[],
                warnings=[],
                failed_intent_ids=frozenset(),
                global_failure=False,
            )

        records: list[_AuditEventRecord] = []
        warnings: list[str] = []
        failed_intent_ids: set[str] = set()
        global_failure = False

        try:
            audit_paths = sorted(self.audit_dir.glob("*.jsonl"))
        except OSError as exc:
            return _AuditReadResult(
                records=[],
                warnings=[f"{self.audit_dir} audit files could not be listed: {exc}"],
                failed_intent_ids=frozenset(),
                global_failure=True,
            )

        for path in audit_paths:
            try:
                audit_file_date = _audit_file_date(path)
            except ValueError as exc:
                warnings.append(f"{path} audit filename is invalid: {exc}")
                global_failure = True
                continue

            file_layout_warning = self._file_layout_warning(path, "audit file")
            if file_layout_warning is not None:
                warnings.append(file_layout_warning)
                global_failure = True
                continue

            try:
                content = path.read_bytes()
            except OSError as exc:
                warnings.append(f"{path} could not be read: {exc}")
                global_failure = True
                continue

            if content and not content.endswith(b"\n"):
                warnings.append(f"{path} audit file must end with a final newline")
                global_failure = True

            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                warnings.append(f"{path} is not valid UTF-8: {exc}")
                global_failure = True
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
                    global_failure = True
                    continue

                if not isinstance(payload, dict):
                    warnings.append(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                    global_failure = True
                    continue

                intent_id = payload.get("intent_id")
                try:
                    event = ReleaseAuditEvent(**payload)
                    try:
                        event_date = _audit_date(event.timestamp)
                    except ValueError as exc:
                        warnings.append(
                            f"{path}:{line_number} audit event timestamp date is invalid: {exc}"
                        )
                        if isinstance(intent_id, str) and intent_id.strip():
                            failed_intent_ids.add(intent_id)
                        else:
                            global_failure = True
                    else:
                        if event_date != audit_file_date:
                            warnings.append(
                                f"{path}:{line_number} audit filename date {audit_file_date} "
                                f"does not match event timestamp date {event_date}"
                            )
                            failed_intent_ids.add(event.intent_id)
                    records.append(
                        _AuditEventRecord(
                            event=event,
                            audit_path=path,
                            line_number=line_number,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    warnings.append(
                        f"{path}:{line_number} audit event is invalid: {exc}"
                    )
                    if isinstance(intent_id, str) and intent_id.strip():
                        failed_intent_ids.add(intent_id)
                    else:
                        global_failure = True

        return _AuditReadResult(
            records=records,
            warnings=warnings,
            failed_intent_ids=frozenset(failed_intent_ids),
            global_failure=global_failure,
        )

    def _verify_artifact_audit_consistency(
        self,
        *,
        intents: list[ReleaseIntent],
        approvals: list[ReleaseApproval],
        rollback_plans: list[ReleaseRollbackPlan],
        audit_result: _AuditReadResult,
    ) -> _ArtifactConsistencyResult:
        warnings: list[str] = []
        affected_intent_ids: set[str] = set()

        intent_hashes: set[tuple[str, str]] = set()
        for intent in intents:
            try:
                intent_hashes.add(
                    (intent.intent_id, canonical_payload_hash(intent.to_dict()))
                )
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"intent artifact {intent.intent_id} payload hash validation failed: {exc}"
                )
                affected_intent_ids.add(intent.intent_id)

        approval_hashes: set[tuple[str, str]] = set()
        for approval in approvals:
            try:
                approval_hashes.add(
                    (
                        approval.intent_id,
                        canonical_payload_hash(approval.to_dict()),
                    )
                )
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"approval artifact for {approval.intent_id} payload hash validation failed: {exc}"
                )
                affected_intent_ids.add(approval.intent_id)

        rollback_plan_hashes: set[tuple[str, str]] = set()
        for plan in rollback_plans:
            try:
                rollback_plan_hashes.add(
                    (plan.intent_id, canonical_payload_hash(plan.to_dict()))
                )
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"rollback plan artifact for {plan.intent_id} payload hash validation failed: {exc}"
                )
                affected_intent_ids.add(plan.intent_id)

        audit_hashes_by_type = {
            "intent_created": set(),
            "approval_recorded": set(),
            "rollback_plan_recorded": set(),
        }
        for record in audit_result.records:
            event = record.event
            if event.event_type in audit_hashes_by_type:
                audit_hashes_by_type[event.event_type].add(
                    (event.intent_id, event.payload_hash)
                )

        known_intent_ids = {intent.intent_id for intent in intents}
        known_intent_ids.update(
            record.event.intent_id
            for record in audit_result.records
            if record.event.event_type == "intent_created"
        )

        for approval in approvals:
            if approval.intent_id not in known_intent_ids:
                warnings.append(
                    f"approval artifact {approval.approval_id} references unknown intent {approval.intent_id}"
                )
                affected_intent_ids.add(approval.intent_id)

        for plan in rollback_plans:
            if plan.intent_id not in known_intent_ids:
                warnings.append(
                    f"rollback plan artifact {plan.rollback_plan_id} references unknown intent {plan.intent_id}"
                )
                affected_intent_ids.add(plan.intent_id)

        reference_event_types = {
            "approval_recorded",
            "rollback_plan_recorded",
            "intent_cancelled",
        }
        for record in audit_result.records:
            event = record.event
            if (
                event.event_type in reference_event_types
                and event.intent_id not in known_intent_ids
            ):
                warnings.append(
                    f"{event.event_type} audit event {event.event_id} references unknown intent {event.intent_id}"
                )
                affected_intent_ids.add(event.intent_id)

        for intent_id, payload_hash in intent_hashes:
            if (
                intent_id,
                payload_hash,
            ) not in audit_hashes_by_type["intent_created"]:
                warnings.append(
                    f"intent artifact {intent_id} does not match an audit payload hash"
                )
                affected_intent_ids.add(intent_id)

        for intent_id, payload_hash in approval_hashes:
            if (
                intent_id,
                payload_hash,
            ) not in audit_hashes_by_type["approval_recorded"]:
                warnings.append(
                    f"approval artifact for {intent_id} does not match an audit payload hash"
                )
                affected_intent_ids.add(intent_id)

        for intent_id, payload_hash in rollback_plan_hashes:
            if (
                intent_id,
                payload_hash,
            ) not in audit_hashes_by_type["rollback_plan_recorded"]:
                warnings.append(
                    f"rollback plan artifact for {intent_id} does not match an audit payload hash"
                )
                affected_intent_ids.add(intent_id)

        for record in audit_result.records:
            event = record.event
            audit_key = (event.intent_id, event.payload_hash)
            if event.event_type == "intent_created" and audit_key not in intent_hashes:
                warnings.append(
                    f"intent_created audit event {event.event_id} has no matching intent artifact"
                )
                affected_intent_ids.add(event.intent_id)
            elif (
                event.event_type == "approval_recorded"
                and audit_key not in approval_hashes
            ):
                warnings.append(
                    f"approval_recorded audit event {event.event_id} has no matching approval artifact"
                )
                affected_intent_ids.add(event.intent_id)
            elif (
                event.event_type == "rollback_plan_recorded"
                and audit_key not in rollback_plan_hashes
            ):
                warnings.append(
                    f"rollback_plan_recorded audit event {event.event_id} has no matching rollback plan artifact"
                )
                affected_intent_ids.add(event.intent_id)

        return _ArtifactConsistencyResult(
            warnings=warnings,
            affected_intent_ids=frozenset(affected_intent_ids),
        )

    def _verify_integrity(
        self,
        audit_result: _AuditReadResult,
        *,
        artifact_warnings: list[str] | None = None,
        consistency_warnings: list[str] | None = None,
        artifact_affected_intent_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        warnings = (
            list(artifact_warnings or [])
            + list(consistency_warnings or [])
            + list(audit_result.warnings)
        )
        failed_intent_ids = set(audit_result.failed_intent_ids)
        failed_intent_ids.update(artifact_affected_intent_ids)
        global_failure = audit_result.global_failure or bool(artifact_warnings)
        previous_hash_by_intent: dict[str, str] = {}

        for record in audit_result.records:
            event = record.event
            try:
                validate_audit_event_hash(event)
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"{event.event_id} event_hash validation failed: {exc}"
                )
                failed_intent_ids.add(event.intent_id)
                continue

            expected_previous_hash = previous_hash_by_intent.get(
                event.intent_id,
                GENESIS_EVENT_HASH,
            )
            if event.previous_event_hash != expected_previous_hash:
                warnings.append(
                    f"{event.event_id} previous_event_hash mismatch: "
                    f"expected {expected_previous_hash}, "
                    f"got {event.previous_event_hash}"
                )
                failed_intent_ids.add(event.intent_id)
            previous_hash_by_intent[event.intent_id] = event.event_hash

        if not warnings:
            return {"status": "verified", "warnings": []}

        integrity: dict[str, Any] = {
            "status": "failed",
            "warnings": warnings,
        }
        if failed_intent_ids:
            integrity["affected_intent_ids"] = sorted(failed_intent_ids)
        if global_failure:
            integrity["global_failure"] = True
        return integrity

    def _artifact_path(self, directory: Path, artifact_id: str) -> Path:
        _validate_artifact_id(artifact_id)
        return directory / f"{artifact_id}.json"


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
        raise ValueError("audit filename must match release_audit_YYYYMMDD.jsonl")
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
    "GovernanceIntegrityError",
    "ReleaseGovernanceState",
    "ReleaseGovernanceStore",
]
