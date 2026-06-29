from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


DoctorActionType = Literal[
    "accept",
    "edit",
    "reject",
    "escalate",
    "request_evidence",
    "mark_unsafe",
]

DoctorActionReasonCode = Literal[
    "fact_wrong",
    "missing_red_flag",
    "unsupported_claim",
    "bad_tone",
    "workflow_mismatch",
    "citation_not_traceable",
    "missing_information",
    "unsafe_disposition",
    "evidence_conflict",
    "template_mismatch",
]


def new_trace_id() -> str:
    return f"doctor_trace_{uuid4().hex}"


class DoctorActionTargetRefs(BaseModel):
    draft_id: str | None = None
    assertion_id: str | None = None
    assessment_id: str | None = None
    record_id: str | None = None
    care_card_id: str | None = None
    citation_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    def has_any_ref(self) -> bool:
        return any(
            isinstance(value, str) and value.strip()
            for value in (
                self.draft_id,
                self.assertion_id,
                self.assessment_id,
                self.record_id,
                self.care_card_id,
                self.citation_id,
            )
        )


class DoctorActionBeforeAfter(BaseModel):
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class DoctorActionTraceRequest(BaseModel):
    action_type: DoctorActionType
    target_object: str | None = None
    target_refs: DoctorActionTargetRefs = Field(default_factory=DoctorActionTargetRefs)
    before_after: DoctorActionBeforeAfter | None = None
    reason_code: DoctorActionReasonCode
    reviewer_role: str = "physician_reviewer"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_action_trace(self) -> "DoctorActionTraceRequest":
        has_target_object = (
            isinstance(self.target_object, str) and bool(self.target_object.strip())
        )
        if not has_target_object and not self.target_refs.has_any_ref():
            raise ValueError("At least target_object or one target ref is required")
        if self.action_type == "edit" and self.before_after is None:
            raise ValueError("before_after is required for edit actions")
        if self.action_type == "request_evidence" and self.reason_code not in {
            "citation_not_traceable",
            "unsupported_claim",
            "missing_information",
            "evidence_conflict",
        }:
            raise ValueError("Invalid reason_code for request_evidence")
        if self.action_type == "mark_unsafe" and self.reason_code not in {
            "unsafe_disposition",
            "missing_red_flag",
            "unsupported_claim",
        }:
            raise ValueError("Invalid reason_code for mark_unsafe")
        return self


class DoctorActionTrace(BaseModel):
    trace_id: str
    patient_id: int
    session_id: str
    action_type: DoctorActionType
    target_object: str | None = None
    target_refs: DoctorActionTargetRefs = Field(default_factory=DoctorActionTargetRefs)
    before_after: DoctorActionBeforeAfter | None = None
    reason_code: DoctorActionReasonCode
    reviewer_role: str = "physician_reviewer"
    deidentified: bool = True
    timestamp: str

    model_config = ConfigDict(extra="forbid")


class DoctorActionTraceResponse(BaseModel):
    patient_id: int
    trace: DoctorActionTrace
    event_ids: list[str] = Field(default_factory=list)
    patient_version: int
    projection_version: int
    snapshot_changed: bool = False
