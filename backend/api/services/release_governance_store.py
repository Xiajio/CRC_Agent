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
        prepared_event = self._prepare_event(
            intent_id=intent.intent_id,
            event_type="intent_created",
            actor=actor,
            timestamp=timestamp,
            payload=intent.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=self._artifact_path(self.intents_dir, intent.intent_id),
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
        self._raise_if_integrity_failed(approval.intent_id)
        prepared_event = self._prepare_event(
            intent_id=approval.intent_id,
            event_type="approval_recorded",
            actor=actor,
            timestamp=timestamp,
            payload=approval.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=self._artifact_path(self.approvals_dir, approval.approval_id),
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
        self._raise_if_integrity_failed(plan.intent_id)
        prepared_event = self._prepare_event(
            intent_id=plan.intent_id,
            event_type="rollback_plan_recorded",
            actor=actor,
            timestamp=timestamp,
            payload=plan.to_dict(),
        )
        self._write_artifact_and_append_event(
            path=self._artifact_path(self.rollback_plans_dir, plan.rollback_plan_id),
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
        self._raise_if_integrity_failed(intent_id)
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

    def _raise_if_integrity_failed(self, intent_id: str) -> None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise GovernanceIntegrityError(
                f"release governance integrity failed; refusing write for {intent_id}"
            )

    def _read_json_dir(
        self,
        directory: Path,
        factory: Callable[..., _Artifact],
        *,
        artifact_name: str,
        id_field: str,
    ) -> _ArtifactReadResult:
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
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as exc:
                warnings.append(f"{path} is not valid UTF-8: {exc}")
                global_failure = True
                continue
            except OSError as exc:
                warnings.append(f"{path} could not be read: {exc}")
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
        if (
            not artifact_id.strip()
            or "/" in artifact_id
            or "\\" in artifact_id
            or artifact_id in {".", ".."}
        ):
            raise ValueError("artifact_id must be a file-safe identifier")
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


__all__ = [
    "GovernanceIntegrityError",
    "ReleaseGovernanceState",
    "ReleaseGovernanceStore",
]
