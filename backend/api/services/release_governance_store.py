from __future__ import annotations

from dataclasses import dataclass
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
    make_release_audit_event_id,
    validate_audit_event_hash,
)


class GovernanceIntegrityError(RuntimeError):
    """Raised when an existing audit chain is not safe to append to."""


@dataclass(frozen=True)
class ReleaseGovernanceState:
    intents: list[ReleaseIntent]
    approvals: list[ReleaseApproval]
    rollback_plans: list[ReleaseRollbackPlan]
    audit_events: list[ReleaseAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class _AuditReadResult:
    events: list[ReleaseAuditEvent]
    warnings: list[str]
    failed_intent_ids: frozenset[str]
    global_failure: bool


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
        return ReleaseGovernanceState(
            intents=sorted(
                self._read_json_dir(self.intents_dir, ReleaseIntent),
                key=lambda intent: intent.requested_at,
            ),
            approvals=sorted(
                self._read_json_dir(self.approvals_dir, ReleaseApproval),
                key=lambda approval: approval.signed_at,
            ),
            rollback_plans=sorted(
                self._read_json_dir(
                    self.rollback_plans_dir,
                    ReleaseRollbackPlan,
                ),
                key=lambda plan: plan.created_at,
            ),
            audit_events=list(audit_result.events),
            integrity=self._verify_integrity(audit_result),
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
        self._write_json_once(
            self._artifact_path(self.intents_dir, intent.intent_id),
            intent.to_dict(),
        )
        self._append_prepared_event(prepared_event)

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
        self._write_json_once(
            self._artifact_path(self.approvals_dir, approval.approval_id),
            approval.to_dict(),
        )
        self._append_prepared_event(prepared_event)

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
        self._write_json_once(
            self._artifact_path(self.rollback_plans_dir, plan.rollback_plan_id),
            plan.to_dict(),
        )
        self._append_prepared_event(prepared_event)

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

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

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
            previous_event_hash=self._last_event_hash(intent_id),
        )
        return _PreparedAuditEvent(event=event, audit_path=audit_path)

    def _append_prepared_event(self, prepared_event: _PreparedAuditEvent) -> None:
        audit_path = prepared_event.audit_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prepared_event.event.to_dict(), sort_keys=True))
            handle.write("\n")

    def _last_event_hash(self, intent_id: str) -> str:
        for event in reversed(self._read_audit_events()):
            if event.intent_id == intent_id:
                return event.event_hash
        return GENESIS_EVENT_HASH

    def _raise_if_integrity_failed(self, intent_id: str) -> None:
        audit_result = self._read_audit_events_with_integrity()
        integrity = self._verify_integrity(audit_result)
        affected_intent_ids = set(integrity.get("affected_intent_ids", []))
        if (
            integrity["status"] == "failed"
            and (
                integrity.get("global_failure")
                or intent_id in affected_intent_ids
            )
        ):
            raise GovernanceIntegrityError(
                f"release governance audit integrity failed for {intent_id}"
            )

    def _read_json_dir(
        self,
        directory: Path,
        factory: Callable[..., _Artifact],
    ) -> list[_Artifact]:
        if not directory.exists():
            return []
        artifacts: list[_Artifact] = []
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} must contain a JSON object")
            artifacts.append(factory(**payload))
        return artifacts

    def _read_audit_events(self) -> list[ReleaseAuditEvent]:
        return self._read_audit_events_with_integrity().events

    def _read_audit_events_with_integrity(self) -> _AuditReadResult:
        if not self.audit_dir.exists():
            return _AuditReadResult(
                events=[],
                warnings=[],
                failed_intent_ids=frozenset(),
                global_failure=False,
            )

        events: list[ReleaseAuditEvent] = []
        warnings: list[str] = []
        failed_intent_ids: set[str] = set()
        global_failure = False

        try:
            audit_paths = sorted(self.audit_dir.glob("*.jsonl"))
        except OSError as exc:
            return _AuditReadResult(
                events=[],
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
                    events.append(ReleaseAuditEvent(**payload))
                except (TypeError, ValueError) as exc:
                    warnings.append(
                        f"{path}:{line_number} audit event is invalid: {exc}"
                    )
                    if isinstance(intent_id, str) and intent_id.strip():
                        failed_intent_ids.add(intent_id)
                    else:
                        global_failure = True

        return _AuditReadResult(
            events=events,
            warnings=warnings,
            failed_intent_ids=frozenset(failed_intent_ids),
            global_failure=global_failure,
        )

    def _verify_integrity(
        self,
        audit_result: _AuditReadResult,
    ) -> dict[str, Any]:
        warnings = list(audit_result.warnings)
        failed_intent_ids = set(audit_result.failed_intent_ids)
        global_failure = audit_result.global_failure
        previous_hash_by_intent: dict[str, str] = {}

        for event in audit_result.events:
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
    return date_prefix.replace("-", "")


__all__ = [
    "GovernanceIntegrityError",
    "ReleaseGovernanceState",
    "ReleaseGovernanceStore",
]
