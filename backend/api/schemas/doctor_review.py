from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DoctorReviewTimelineItem(BaseModel):
    item_id: str
    kind: str
    title: str
    created_at: str
    assertion_refs: list[str] = Field(default_factory=list)


class DoctorReviewProvenanceRef(BaseModel):
    kind: str
    id: str
    field: str | None = None


class DoctorReviewDraftSection(BaseModel):
    section_id: str
    kind: str
    title: str
    body: str
    assertion_refs: list[str] = Field(default_factory=list)
    provenance_refs: list[DoctorReviewProvenanceRef] = Field(default_factory=list)


class DoctorReviewDraft(BaseModel):
    draft_id: str
    sections: list[DoctorReviewDraftSection] = Field(default_factory=list)


class DoctorReviewResponse(BaseModel):
    patient_id: int
    session_id: str
    feature_flag: str = "doctor_review_cockpit_v0"
    timeline: list[DoctorReviewTimelineItem] = Field(default_factory=list)
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    draft: DoctorReviewDraft
    available_actions: list[str] = Field(default_factory=list)
