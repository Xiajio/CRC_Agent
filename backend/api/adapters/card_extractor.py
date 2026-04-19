from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.messages import BaseMessage

from backend.api.schemas.events import CardSourceChannel, CardUpsertEvent


CARD_REGISTRY: tuple[tuple[str, str], ...] = (
    ("decision_json", "decision_card"),
    ("medical_card", "medical_card"),
    ("triage_card", "triage_card"),
    ("triage_question_card", "triage_question_card"),
    ("pathology_card", "pathology_card"),
    ("pathology_slide_card", "pathology_slide_card"),
    ("radiomics_report_card", "radiomics_report_card"),
    ("patient_card", "patient_card"),
    ("imaging_card", "imaging_card"),
    ("tumor_detection_card", "tumor_detection_card"),
)


def _coerce_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iter_messages(node_output: Mapping[str, Any]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    raw_messages = node_output.get("messages")
    if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
        for message in raw_messages:
            if isinstance(message, BaseMessage):
                messages.append(message)
    raw_message = node_output.get("message")
    if isinstance(raw_message, BaseMessage):
        messages.append(raw_message)
    return messages


def _collect_candidates(
    node_output: Mapping[str, Any],
    messages: Sequence[BaseMessage] | None = None,
) -> list[tuple[str, str, dict[str, Any], str]]:
    candidates: list[tuple[str, str, dict[str, Any], str]] = []

    for key, card_type in CARD_REGISTRY:
        payload = _coerce_payload(node_output.get(key))
        if payload is not None:
            candidates.append((card_type, "state", payload, key))

    findings = node_output.get("findings")
    if isinstance(findings, Mapping):
        for key, card_type in CARD_REGISTRY:
            payload = _coerce_payload(findings.get(key))
            if payload is not None:
                candidates.append((card_type, "findings", payload, key))

    message_iter = list(messages) if messages is not None else _iter_messages(node_output)
    message_iter.reverse()
    for message in message_iter:
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if not isinstance(additional_kwargs, Mapping) or not additional_kwargs:
            continue
        for key, card_type in CARD_REGISTRY:
            payload = _coerce_payload(additional_kwargs.get(key))
            if payload is not None:
                candidates.append((card_type, "message_kwargs", payload, key))

    return candidates


def extract_card_upsert_events(
    node_output: Mapping[str, Any] | None,
    messages: Sequence[BaseMessage] | None = None,
) -> list[CardUpsertEvent]:
    if node_output is not None and not isinstance(node_output, Mapping):
        return []

    candidate_source: Mapping[str, Any] = node_output or {}
    events: list[CardUpsertEvent] = []
    seen_payloads: set[tuple[str, str]] = set()
    chosen_card_types: set[str] = set()

    for card_type, source_channel, payload, _ in _collect_candidates(candidate_source, messages=messages):
        if card_type in chosen_card_types:
            continue

        payload_hash = _payload_hash(payload)
        dedupe_key = (card_type, payload_hash)
        if dedupe_key in seen_payloads:
            continue

        seen_payloads.add(dedupe_key)
        chosen_card_types.add(card_type)
        events.append(
            CardUpsertEvent(
                card_type=card_type,
                payload=payload,
                source_channel=cast(CardSourceChannel, source_channel),
            )
        )

    return events


def extract_cards(
    node_name: str,
    node_output: Mapping[str, Any] | None,
    messages: Sequence[BaseMessage] | None = None,
) -> list[CardUpsertEvent]:
    del node_name
    return extract_card_upsert_events(node_output=node_output, messages=messages)
