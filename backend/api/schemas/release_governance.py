from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.release_governance import (
    ReleaseApprovalDecision,
    ReleaseApproverRole,
    ReleaseRollbackPlanStatus,
    ReleaseTargetScope,
)


NonEmptyString = Annotated[str, Field(min_length=1)]
ReleaseIntentCreateStatus = Literal["draft", "pending_approval"]


class ReleaseGovernanceCreateIntentRequest(BaseModel):
    requested_by: NonEmptyString
    target_scope: ReleaseTargetScope
    status: ReleaseIntentCreateStatus
    reason: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseGovernanceApprovalRequest(BaseModel):
    approver_role: ReleaseApproverRole
    decision: ReleaseApprovalDecision
    reason: NonEmptyString
    signed_by: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseGovernanceRollbackPlanRequest(BaseModel):
    owner: NonEmptyString
    status: ReleaseRollbackPlanStatus
    verification_steps: list[NonEmptyString] = Field(min_length=2)

    model_config = ConfigDict(extra="forbid")


class ReleaseGovernanceCancelRequest(BaseModel):
    actor: NonEmptyString
    reason: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseGovernanceResponse(BaseModel):
    dashboard_snapshot: dict[str, Any]
    intents: list[dict[str, Any]]
    active_intent: dict[str, Any] | None
    required_approvals: list[dict[str, Any]]
    rollback_plan: dict[str, Any] | None
    audit_events: list[dict[str, Any]]
    integrity: dict[str, Any]
    disabled_execution_actions: list[dict[str, Any]]
    runtime: dict[str, Any]
    approvals: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
