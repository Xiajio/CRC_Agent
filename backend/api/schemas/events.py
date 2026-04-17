from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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

    model_config = ConfigDict(extra="forbid")
