from typing import Any

from pydantic import BaseModel, Field

from backend.api.schemas.events import CardUpsertEvent


class SessionMessage(BaseModel):
    cursor: str
    type: str
    content: Any
    id: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    status: str | None = None
    asset_refs: list[dict[str, Any]] = Field(default_factory=list)
    inline_cards: list[dict[str, Any]] = Field(default_factory=list)


class MessageHistoryPage(BaseModel):
    messages_total: int = 0
    next_before_cursor: str | None = None
    messages: list[SessionMessage] = Field(default_factory=list)


class RecoverySnapshot(BaseModel):
    snapshot_version: int = 0
    messages: list[SessionMessage] = Field(default_factory=list)
    messages_total: int = 0
    messages_next_before_cursor: str | None = None
    cards: list[CardUpsertEvent] = Field(default_factory=list)
    roadmap: list[dict[str, Any]] = Field(default_factory=list)
    findings: dict[str, Any] = Field(default_factory=dict)
    patient_profile: dict[str, Any] | None = None
    stage: str | None = None
    assessment_draft: Any = None
    current_patient_id: int | str | None = None
    references: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    critic: dict[str, Any] | None = None
    safety_alert: dict[str, Any] | None = None
    uploaded_assets: dict[str, Any] = Field(default_factory=dict)
    context_maintenance: dict[str, Any] | None = None
    context_state: dict[str, Any] | None = None


class RuntimeInfo(BaseModel):
    runner_mode: str = "real"
    fixture_case: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    thread_id: str
    scene: str
    patient_id: int | None = None
    snapshot_version: int = 0
    snapshot: RecoverySnapshot
    runtime: RuntimeInfo = Field(default_factory=RuntimeInfo)


class MessageHistoryResponse(BaseModel):
    session_id: str
    thread_id: str
    snapshot_version: int = 0
    messages_total: int = 0
    next_before_cursor: str | None = None
    messages: list[SessionMessage] = Field(default_factory=list)
