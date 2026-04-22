from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from backend.api.adapters.card_extractor import extract_cards
from backend.api.adapters.event_normalizer import normalize_tick
from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.schemas.events import CardUpsertEvent, MessageDoneEvent
from backend.api.services.session_store import SessionMeta
from src.nodes.triage_nodes import node_outpatient_triage
from src.state import CRCAgentState


def _run_outpatient_triage(
    user_message: str,
    *,
    findings: dict | None = None,
    symptom_snapshot: dict | None = None,
    encounter_track: str | None = None,
) -> dict:
    triage = node_outpatient_triage(show_thinking=False)
    state_payload: dict = {
        "messages": [HumanMessage(content=user_message)],
        "findings": findings or {},
    }
    if symptom_snapshot is not None:
        state_payload["symptom_snapshot"] = symptom_snapshot
    if encounter_track is not None:
        state_payload["encounter_track"] = encounter_track

    state = CRCAgentState(**state_payload)
    return triage(state)


def test_outpatient_triage_emits_question_card_for_active_inquiry() -> None:
    result = _run_outpatient_triage("最近腹痛腹泻")

    message = result["messages"][0]
    card = message.additional_kwargs.get("triage_question_card")

    assert result["findings"]["active_inquiry"] is True
    assert result["findings"]["triage_switch_prompt_active"] is False
    assert card is not None
    assert card["type"] == "triage_question_card"
    assert card["field_key"] == "duration"
    assert card["selection_mode"] == "single"
    assert card["question_id"] == "triage-q-duration-0"
    assert card["options"]


def test_outpatient_triage_suppresses_question_card_when_switch_prompt_is_active() -> None:
    triage = node_outpatient_triage(show_thinking=False)
    result = triage(CRCAgentState(messages=[HumanMessage(content="最近腹痛腹泻")], findings={}))
    conversation = ["最近腹痛腹泻"]

    for _ in range(3):
      conversation.append("我先不回答这个问题")
      result = triage(
          CRCAgentState(
              messages=[HumanMessage(content=item) for item in conversation],
              encounter_track="outpatient_triage",
              symptom_snapshot=result["symptom_snapshot"],
              findings=result["findings"],
          )
      )

    message = result["messages"][0]

    assert result["findings"]["active_inquiry"] is True
    assert result["findings"]["triage_switch_prompt_active"] is True
    assert "triage_question_card" not in message.additional_kwargs


def test_completed_triage_keeps_summary_card_and_omits_question_card() -> None:
    result = _run_outpatient_triage(
        "补充：没有新的情况",
        encounter_track="outpatient_triage",
        symptom_snapshot={
            "duration": "3天",
            "bleeding": "鲜红色便血",
            "bowel_change": "大便变细",
            "weight_loss": True,
            "fever": False,
        },
        findings={
            "encounter_track": "outpatient_triage",
            "active_inquiry": True,
            "triage_current_field": "fever",
            "inquiry_type": "outpatient_triage",
        },
    )

    message = result["messages"][0]

    assert result["findings"]["active_inquiry"] is False
    assert result["triage_card"]["type"] == "triage_card"
    assert "triage_question_card" not in message.additional_kwargs


def test_card_extraction_and_snapshot_restore_keep_triage_question_card_inline() -> None:
    triage_question_card = {
        "type": "triage_question_card",
        "version": 1,
        "question_id": "triage-q-duration-0",
        "field_key": "duration",
        "prompt": "腹痛/腹泻/便秘是从什么时候开始的？大概持续了多久？",
        "selection_mode": "single",
        "options": [{"id": "lt_1d", "label": "1天内", "submit_text": "持续时间少于1天"}],
        "allow_other": True,
    }
    triage_card = {
        "type": "triage_card",
        "risk_level": "medium",
        "disposition": "routine_gi_clinic",
    }
    message = AIMessage(
        content="为了继续门诊分诊，我先追问 1 个关键问题。",
        additional_kwargs={
            "triage_question_card": triage_question_card,
            "triage_card": triage_card,
        },
    )

    extracted = extract_cards("message", {}, messages=[message])
    normalized = normalize_tick("message", {}, messages=[message])
    session_meta = SessionMeta(session_id="sess_test", thread_id="thread_test", scene="patient")
    snapshot = build_recovery_snapshot(
        session_meta,
        {"messages": [message]},
        message_limit=10,
    )

    assert any(isinstance(event, CardUpsertEvent) and event.card_type == "triage_question_card" for event in extracted)
    assert any(isinstance(event, CardUpsertEvent) and event.card_type == "triage_card" for event in extracted)

    message_done = next(event for event in normalized if isinstance(event, MessageDoneEvent))
    assert message_done.inline_cards is not None
    assert any(card["card_type"] == "triage_question_card" for card in message_done.inline_cards)
    assert any(card["card_type"] == "triage_card" for card in message_done.inline_cards)

    assert snapshot.messages[0].inline_cards
    assert any(card["card_type"] == "triage_question_card" for card in snapshot.messages[0].inline_cards)
    assert any(card["card_type"] == "triage_card" for card in snapshot.messages[0].inline_cards)
    assert any(card.card_type == "triage_question_card" for card in snapshot.cards)
    assert any(card.card_type == "triage_card" for card in snapshot.cards)


def test_normalize_tick_deduplicates_ai_messages_with_same_nonempty_id() -> None:
    first = AIMessage(content="first answer", id="msg-123")
    second = AIMessage(content="second answer", id="msg-123")

    normalized = normalize_tick("message", {}, messages=[first, second])
    message_done_events = [event for event in normalized if isinstance(event, MessageDoneEvent)]

    assert len(message_done_events) == 1
    assert message_done_events[0].message_id == "msg-123"
    assert message_done_events[0].content == "first answer"
