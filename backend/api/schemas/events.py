from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CardSourceChannel = Literal["state", "findings", "message_kwargs"]


class CardUpsertEvent(BaseModel):
    type: Literal["card.upsert"] = "card.upsert"
    card_type: str
    payload: dict[str, Any]
    source_channel: CardSourceChannel

    model_config = ConfigDict(extra="forbid")


class StatusNodeEvent(BaseModel):
    type: Literal["status.node"] = "status.node"
    node: str

    model_config = ConfigDict(extra="forbid")


class MessageDoneEvent(BaseModel):
    type: Literal["message.done"] = "message.done"
    role: Literal["assistant"] = "assistant"
    content: Any
    thinking: str | None = None
    message_id: str | None = None
    node: str | None = None
    inline_cards: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="forbid")


class MessageDeltaEvent(BaseModel):
    type: Literal["message.delta"] = "message.delta"
    message_id: str
    node: str | None = None
    delta: str

    model_config = ConfigDict(extra="forbid")


class TraceStartEvent(BaseModel):
    type: Literal["trace.start"] = "trace.start"
    trace_id: str | None = None
    run_id: str
    session_id: str
    scene: str | None = None
    server_received_at: str
    graph_started_at: str
    model: str | None = None
    graph_path: list[str] = Field(default_factory=list)
    attrs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class TraceStepEvent(BaseModel):
    type: Literal["trace.step"] = "trace.step"
    trace_id: str | None = None
    run_id: str
    session_id: str
    name: str
    at: str
    node: str | None = None
    model: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class TraceSummaryEvent(BaseModel):
    type: Literal["trace.summary"] = "trace.summary"
    trace_id: str | None = None
    run_id: str
    session_id: str
    status: Literal["completed", "aborted", "error"]
    at: str
    scene: str | None = None
    graph_path: list[str] = Field(default_factory=list)
    model: str | None = None
    has_thinking: bool = False
    response_chars: int = 0
    response_tokens: int | None = None
    tool_calls: int = 0
    retrieval_hit_count: int = 0
    attrs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class StageUpdateEvent(BaseModel):
    type: Literal["stage.update"] = "stage.update"
    stage: str

    model_config = ConfigDict(extra="forbid")


class PatientProfileUpdateEvent(BaseModel):
    type: Literal["patient_profile.update"] = "patient_profile.update"
    profile: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class CriticVerdictEvent(BaseModel):
    type: Literal["critic.verdict"] = "critic.verdict"
    verdict: str
    feedback: str | None = None
    iteration_count: int | None = None

    model_config = ConfigDict(extra="forbid")


class RoadmapUpdateEvent(BaseModel):
    type: Literal["roadmap.update"] = "roadmap.update"
    roadmap: list[dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class PlanUpdateEvent(BaseModel):
    type: Literal["plan.update"] = "plan.update"
    plan: list[dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class SafetyAlertEvent(BaseModel):
    type: Literal["safety.alert"] = "safety.alert"
    message: str
    blocking: Literal[True] = True

    model_config = ConfigDict(extra="forbid")


class FindingsPatchEvent(BaseModel):
    type: Literal["findings.patch"] = "findings.patch"
    patch: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class ReferencesAppendEvent(BaseModel):
    type: Literal["references.append"] = "references.append"
    items: list[dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool = True

    model_config = ConfigDict(extra="forbid")


class ContextMaintenanceEvent(BaseModel):
    type: Literal["context.maintenance"] = "context.maintenance"
    status: Literal["running", "completed", "failed"]
    message: str

    model_config = ConfigDict(extra="forbid")


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    thread_id: str
    run_id: str
    snapshot_version: int
    patient_version_used: int | None = None
    patient_context_stale: bool = False

    model_config = ConfigDict(extra="forbid")
