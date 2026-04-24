import { describe, expect, it } from "vitest";

import type { CardUpsertEvent, MessageDeltaEvent, MessageDoneEvent } from "../api/types";
import { createInitialSessionState, reduceStreamEvent } from "./stream-reducer";

describe("reduceStreamEvent", () => {
  it("advances a scaffolded roadmap from status.node events", () => {
    const initialState = {
      ...createInitialSessionState(),
      roadmap: [
        { id: "intent", title: "intent", status: "waiting" },
        { id: "planner", title: "planner", status: "waiting" },
        { id: "assessment", title: "assessment", status: "waiting" },
      ],
    };

    const plannerState = reduceStreamEvent(initialState, {
      type: "status.node",
      node: "planner",
    });
    const assessmentState = reduceStreamEvent(plannerState, {
      type: "status.node",
      node: "assessment",
    });

    expect(plannerState.roadmap).toEqual([
      { id: "intent", title: "intent", status: "completed" },
      { id: "planner", title: "planner", status: "in_progress" },
      { id: "assessment", title: "assessment", status: "waiting" },
    ]);
    expect(assessmentState.roadmap).toEqual([
      { id: "intent", title: "intent", status: "completed" },
      { id: "planner", title: "planner", status: "completed" },
      { id: "assessment", title: "assessment", status: "in_progress" },
    ]);
  });

  it("does not create a roadmap from status.node when no workflow has been scaffolded", () => {
    const state = reduceStreamEvent(createInitialSessionState(), {
      type: "status.node",
      node: "database",
    });

    expect(state.statusNode).toBe("database");
    expect(state.roadmap).toEqual([]);
  });

  it("marks the active scaffolded plan step as blocked when the stream returns an error", () => {
    const initialState = {
      ...createInitialSessionState(),
      plan: [
        { id: "collect-context", title: "collect context", status: "completed" },
        { id: "generate-recommendation", title: "generate treatment recommendation", status: "in_progress" },
        { id: "finalize-report", title: "finalize report", status: "pending" },
      ],
    };

    const state = reduceStreamEvent(initialState, {
      type: "error",
      code: "BACKEND_ERROR",
      message: "decision failed",
      recoverable: true,
    });

    expect(state.plan).toEqual([
      { id: "collect-context", title: "collect context", status: "completed" },
      {
        id: "generate-recommendation",
        title: "generate treatment recommendation",
        status: "blocked",
        error_message: "decision failed",
      },
      { id: "finalize-report", title: "finalize report", status: "pending" },
    ]);
  });

  it("appends message.delta chunks into one assistant message and finalizes it on message.done", () => {
    const firstDelta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "msg-1",
      node: "general",
      delta: "Hello ",
    };
    const secondDelta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "msg-1",
      node: "general",
      delta: "world",
    };
    const done: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      node: "general",
      content: "Hello world",
      thinking: "reasoning",
    };

    const afterFirst = reduceStreamEvent(createInitialSessionState(), firstDelta);
    const afterSecond = reduceStreamEvent(afterFirst, secondDelta);
    const finalState = reduceStreamEvent(afterSecond, done);

    expect(afterFirst.messages).toHaveLength(1);
    expect(afterSecond.messages).toHaveLength(1);
    expect(afterSecond.messages[0]).toMatchObject({
      id: "msg-1",
      type: "ai",
      node: "general",
      content: "Hello world",
    });

    expect(finalState.messages).toHaveLength(1);
    expect(finalState.messages[0]).toMatchObject({
      id: "msg-1",
      type: "ai",
      node: "general",
      content: "Hello world",
      thinking: "reasoning",
    });
  });

  it("keeps inline cards attached when they arrive between message.delta and message.done", () => {
    const delta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "msg-1",
      node: "triage",
      delta: "Streaming answer",
    };
    const inlineCard: CardUpsertEvent = {
      type: "card.upsert",
      card_type: "triage_card",
      payload: { risk: "low" },
      source_channel: "state",
    };
    const done: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      node: "triage",
      content: "Streaming answer",
      inline_cards: [{ card_type: "triage_card", payload: { risk: "low" } }],
    };

    const withDelta = reduceStreamEvent(createInitialSessionState(), delta);
    const withCard = reduceStreamEvent(withDelta, inlineCard);
    const finalState = reduceStreamEvent(withCard, done);

    expect(withCard.messages[0]?.inlineCards).toEqual([
      {
        cardType: "triage_card",
        payload: { risk: "low" },
      },
    ]);
    expect(finalState.messages).toHaveLength(1);
    expect(finalState.messages[0]?.inlineCards).toEqual([
      {
        cardType: "triage_card",
        payload: { risk: "low" },
      },
    ]);
  });

  it("keeps triage_question_card inline with triage_card through attachment and finalization", () => {
    const delta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "msg-2",
      node: "triage",
      delta: "Follow-up question",
    };
    const triageCard: CardUpsertEvent = {
      type: "card.upsert",
      card_type: "triage_card",
      payload: { risk_level: "medium" },
      source_channel: "state",
    };
    const triageQuestionCard: CardUpsertEvent = {
      type: "card.upsert",
      card_type: "triage_question_card",
      payload: { question_id: "triage-q-fever-1" },
      source_channel: "message_kwargs",
    };
    const done: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "msg-2",
      node: "triage",
      content: "Follow-up question",
      inline_cards: [
        { card_type: "triage_card", payload: { risk_level: "medium" } },
        { card_type: "triage_question_card", payload: { question_id: "triage-q-fever-1" } },
      ],
    };

    const withDelta = reduceStreamEvent(createInitialSessionState(), delta);
    const withTriageCard = reduceStreamEvent(withDelta, triageCard);
    const withQuestionCard = reduceStreamEvent(withTriageCard, triageQuestionCard);
    const finalState = reduceStreamEvent(withQuestionCard, done);

    expect(withQuestionCard.messages[0]?.inlineCards).toEqual([
      {
        cardType: "triage_card",
        payload: { risk_level: "medium" },
      },
      {
        cardType: "triage_question_card",
        payload: { question_id: "triage-q-fever-1" },
      },
    ]);
    expect(finalState.messages[0]?.inlineCards).toEqual([
      {
        cardType: "triage_card",
        payload: { risk_level: "medium" },
      },
      {
        cardType: "triage_question_card",
        payload: { question_id: "triage-q-fever-1" },
      },
    ]);
  });

  it("still supports legacy final-only message.done events", () => {
    const done: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "msg-legacy",
      node: "general",
      content: "Final answer only",
    };

    const finalState = reduceStreamEvent(createInitialSessionState(), done);

    expect(finalState.messages).toHaveLength(1);
    expect(finalState.messages[0]).toMatchObject({
      id: "msg-legacy",
      content: "Final answer only",
      type: "ai",
    });
  });
});
