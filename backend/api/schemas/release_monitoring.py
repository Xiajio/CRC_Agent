from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]
ReleaseMonitoringCheckType = Literal[
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
    "manual_operator_note",
]
ReleaseMonitoringCheckStatus = Literal["pass", "warning", "fail"]
ReleaseMonitoringAcknowledgementDisposition = Literal[
    "investigating",
    "accepted_risk",
    "rollback_started_elsewhere",
    "false_positive",
]


class ReleaseMonitoringCheckRequest(BaseModel):
    intent_id: NonEmptyString
    execution_id: NonEmptyString
    check_type: ReleaseMonitoringCheckType
    status: ReleaseMonitoringCheckStatus
    observed_by: NonEmptyString
    summary: NonEmptyString
    evidence_refs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseMonitoringAcknowledgeAlertRequest(BaseModel):
    acknowledged_by: NonEmptyString
    disposition: ReleaseMonitoringAcknowledgementDisposition
    reason: NonEmptyString

    model_config = ConfigDict(extra="forbid")
