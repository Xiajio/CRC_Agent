import { describe, expect, it } from "vitest";

import type { CardUpsertEvent, MessageDeltaEvent, MessageDoneEvent, SessionState } from "../api/types";
import { createInitialSessionState, hydrateSessionState, reduceStreamEvent } from "./stream-reducer";

describe("hydrateSessionState", () => {
  it("hydrates split patient context fields from recovery snapshots", () => {
    const state = hydrateSessionState(createInitialSessionState(), {
      session_id: "sess",
      thread_id: "thread",
      scene: "doctor",
      patient_id: 7,
      snapshot_version: 1,
      runtime: { runner_mode: "real", fixture_case: null },
      snapshot: {
        snapshot_version: 1,
        messages: [],
        messages_total: 0,
        messages_next_before_cursor: null,
        cards: [],
        roadmap: [],
        findings: {},
        patient_profile: null,
        patient_identity: null,
        stage: null,
        assessment_draft: null,
        case_database_patient_id: "093",
        registry_patient_id: 7,
        current_patient_id: "093",
        references: [],
        plan: [],
        critic: null,
        safety_alert: null,
        uploaded_assets: {},
        context_maintenance: null,
        context_state: null,
      },
    });

    expect(state.caseDatabasePatientId).toBe("093");
    expect(state.registryPatientId).toBe(7);
    expect(state.currentPatientId).toBe("093");
  });

  it("clears runTrace when hydrating a reset session snapshot", () => {
    const state = hydrateSessionState(
      {
        ...createInitialSessionState(),
        runTrace: {
          traceId: "trace-1",
          runId: "run-1",
          scene: "doctor",
          status: "active",
          graphPath: ["planner"],
          steps: [{ name: "planner", at: "t1", attrs: {} }],
          summary: null,
          startedAt: "t0",
          finishedAt: null,
        },
      },
      {
        session_id: "sess",
        thread_id: "thread",
        scene: "doctor",
        patient_id: null,
        snapshot_version: 2,
        runtime: { runner_mode: "real", fixture_case: null },
        snapshot: {
          snapshot_version: 2,
          messages: [],
          messages_total: 0,
          messages_next_before_cursor: null,
          cards: [],
          roadmap: [],
          findings: {},
          patient_profile: null,
          patient_identity: null,
          stage: null,
          assessment_draft: null,
          references: [],
          plan: [],
          critic: null,
          safety_alert: null,
          uploaded_assets: {},
          context_maintenance: null,
          context_state: null,
        },
      },
    );

    expect(state.runTrace).toBeNull();
  });

  it("preserves runTrace when hydrating the same session after stream completion", () => {
    let state: SessionState = {
      ...createInitialSessionState(),
      sessionId: "sess",
    };

    state = reduceStreamEvent(state, {
      type: "trace.start",
      trace_id: "trace-1",
      scene: "doctor",
      session_id: "sess",
      run_id: "run-1",
      server_received_at: "t0",
      graph_started_at: "t0",
      graph_path: ["intent", "planner"],
      attrs: {},
    });
    state = reduceStreamEvent(state, {
      type: "trace.step",
      trace_id: "trace-1",
      name: "planner",
      at: "t1",
      session_id: "sess",
      run_id: "run-1",
      attrs: { duration_ms: 120 },
    });

    const hydrated = hydrateSessionState(state, {
      session_id: "sess",
      thread_id: "thread",
      scene: "doctor",
      patient_id: null,
      snapshot_version: 2,
      runtime: { runner_mode: "real", fixture_case: null },
      snapshot: {
        snapshot_version: 2,
        messages: [],
        messages_total: 0,
        messages_next_before_cursor: null,
        cards: [],
        roadmap: [],
        findings: {},
        patient_profile: null,
        patient_identity: null,
        stage: null,
        assessment_draft: null,
        references: [],
        plan: [],
        critic: null,
        safety_alert: null,
        uploaded_assets: {},
        context_maintenance: null,
        context_state: null,
      },
    });

    expect(hydrated.runTrace).toEqual(state.runTrace);
  });
});

describe("reduceStreamEvent", () => {
  it("records runTrace from trace SSE events", () => {
    let state = createInitialSessionState();

    state = reduceStreamEvent(state, {
      type: "trace.start",
      scene: "doctor",
      session_id: "s1",
      run_id: "r1",
      server_received_at: "t0",
      graph_started_at: "t0",
      graph_path: ["intent", "planner"],
      attrs: {},
    });
    state = reduceStreamEvent(state, {
      type: "trace.step",
      name: "planner",
      at: "t1",
      session_id: "s1",
      run_id: "r1",
      attrs: { duration_ms: 120 },
    });

    expect(state.runTrace?.runId).toBe("r1");
    expect(state.runTrace?.steps[0]).toMatchObject({ name: "planner" });
  });

  it("marks runTrace completed from trace.summary events", () => {
    let state = createInitialSessionState();

    state = reduceStreamEvent(state, {
      type: "trace.start",
      trace_id: "trace-1",
      scene: "doctor",
      session_id: "s1",
      run_id: "r1",
      server_received_at: "t0",
      graph_started_at: "t0",
      graph_path: ["intent", "planner"],
      attrs: {},
    });
    state = reduceStreamEvent(state, {
      type: "trace.summary",
      trace_id: "trace-1",
      scene: "doctor",
      session_id: "s1",
      run_id: "r1",
      at: "t2",
      status: "completed",
      graph_path: ["intent", "planner"],
      has_thinking: false,
      response_chars: 42,
      tool_calls: 1,
      retrieval_hit_count: 2,
      response_tokens: 12,
      attrs: { model_latency_ms: 300 },
    });

    expect(state.runTrace?.status).toBe("completed");
    expect(state.runTrace?.finishedAt).toBe("t2");
    expect(state.runTrace?.summary).toMatchObject({
      status: "completed",
      response_chars: 42,
    });
  });

  it("starts a fresh active runTrace when trace.step run_id changes", () => {
    let state = createInitialSessionState();

    state = reduceStreamEvent(state, {
      type: "trace.start",
      trace_id: "trace-1",
      scene: "doctor",
      session_id: "s1",
      run_id: "r1",
      server_received_at: "t0",
      graph_started_at: "t0",
      graph_path: ["intent"],
      attrs: {},
    });
    state = reduceStreamEvent(state, {
      type: "trace.step",
      trace_id: "trace-2",
      name: "planner",
      at: "t1",
      session_id: "s1",
      run_id: "r2",
      attrs: { duration_ms: 120 },
    });

    expect(state.runTrace?.runId).toBe("r2");
    expect(state.runTrace?.status).toBe("active");
    expect(state.runTrace?.graphPath).toEqual([]);
    expect(state.runTrace?.steps).toEqual([
      { name: "planner", at: "t1", attrs: { duration_ms: 120 } },
    ]);
  });

  it("records a bounded visible event log for clinical stream events", () => {
    let state = createInitialSessionState();

    state = reduceStreamEvent(state, { type: "status.node", node: "decision" });
    state = reduceStreamEvent(state, { type: "stage.update", stage: "Decision" });
    state = reduceStreamEvent(state, {
      type: "critic.verdict",
      verdict: "REJECTED",
      feedback: "missing references",
      iteration_count: 1,
      requires_human_review: true,
    } as any);
    state = reduceStreamEvent(state, {
      type: "plan.update",
      plan: [{ id: "plan-1", title: "Treatment sequence", status: "completed" }],
    });
    state = reduceStreamEvent(state, {
      type: "references.append",
      items: [{ id: "ref-1", title: "NCCN" }],
    });
    state = reduceStreamEvent(state, {
      type: "done",
      thread_id: "thread-1",
      run_id: "run-1",
      snapshot_version: 2,
    });

    expect((state as any).eventLog.map((entry: any) => entry.kind)).toEqual([
      "node",
      "stage",
      "critic",
      "plan",
      "references",
      "done",
    ]);
    expect((state as any).eventLog[2]).toMatchObject({
      kind: "critic",
      tone: "warning",
      title: "Critic REJECTED",
      detail: "missing references",
      requiresHumanReview: true,
    });
  });

  it("extracts readable critic feedback from raw thinking-wrapped JSON output", () => {
    const rawFeedback = [
      "<think>The critic considered the plan and quoted {'verdict': 'APPROVED'}.</think>",
      '{"verdict":"APPROVED","feedback":"需要补充 MMR/MSI 检测。"}',
    ].join("\n");

    const state = reduceStreamEvent(createInitialSessionState(), {
      type: "critic.verdict",
      verdict: "APPROVED",
      feedback: rawFeedback,
      requires_human_review: false,
    } as any);

    expect((state as any).critic.feedback).toBe("需要补充 MMR/MSI 检测。");
    expect((state as any).eventLog[0]).toMatchObject({
      kind: "critic",
      detail: "需要补充 MMR/MSI 检测。",
    });
    expect(JSON.stringify((state as any).eventLog)).not.toContain("<think>");
  });

  it("keeps only the latest clinical event log entries", () => {
    let state = createInitialSessionState();

    for (let index = 0; index < 35; index += 1) {
      state = reduceStreamEvent(state, { type: "status.node", node: `node-${index}` });
    }

    expect((state as any).eventLog).toHaveLength(25);
    expect((state as any).eventLog[0]).toMatchObject({ title: "node-10" });
    expect((state as any).eventLog[24]).toMatchObject({ title: "node-34" });
  });

  it("keeps clinical event log ids unique after the bounded log starts pruning", () => {
    let state = createInitialSessionState();

    for (let index = 0; index < 35; index += 1) {
      state = reduceStreamEvent(state, {
        type: "plan.update",
        plan: [{ id: "plan-1", title: "Treatment sequence", status: "completed" }],
      });
    }

    const ids = (state as any).eventLog.map((entry: any) => entry.id);

    expect(ids).toHaveLength(25);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("records roadmap updates in the visible event log", () => {
    const state = reduceStreamEvent(createInitialSessionState(), {
      type: "roadmap.update",
      roadmap: [
        { id: "assessment", title: "assessment", status: "completed" },
        { id: "citation", title: "citation", status: "blocked" },
      ],
    });

    expect(state.roadmap).toEqual([
      { id: "assessment", title: "assessment", status: "completed" },
      { id: "citation", title: "citation", status: "blocked" },
    ]);
    expect((state as any).eventLog).toEqual([
      expect.objectContaining({
        kind: "roadmap",
        title: "Roadmap updated",
        detail: "2 step(s)",
        tone: "neutral",
      }),
    ]);
  });

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

  it("appends a new assistant message when a later turn reuses a completed message id", () => {
    const firstDone: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "fixture-reused-id",
      node: "triage",
      content: "first fixture reply",
    };
    const secondDone: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "fixture-reused-id",
      node: "triage",
      content: "second fixture reply",
    };

    const afterFirst = reduceStreamEvent(createInitialSessionState(), firstDone);
    const finalState = reduceStreamEvent(afterFirst, secondDone);

    expect(finalState.messages).toHaveLength(2);
    expect(finalState.messages.map((message) => message.content)).toEqual([
      "first fixture reply",
      "second fixture reply",
    ]);
  });

  it("starts a fresh streaming message when a later turn reuses a completed delta id", () => {
    const firstDelta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "fixture-stream-id",
      node: "triage",
      delta: "first",
    };
    const firstDone: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "fixture-stream-id",
      node: "triage",
      content: "first",
    };
    const secondDelta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "fixture-stream-id",
      node: "triage",
      delta: "second",
    };

    const finalFirst = reduceStreamEvent(
      reduceStreamEvent(createInitialSessionState(), firstDelta),
      firstDone,
    );
    const afterSecondDelta = reduceStreamEvent(finalFirst, secondDelta);

    expect(afterSecondDelta.messages).toHaveLength(2);
    expect(afterSecondDelta.messages.map((message) => message.content)).toEqual(["first", "second"]);
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

  it("keeps tumor_screening_result as an inline card", () => {
    const delta: MessageDeltaEvent = {
      type: "message.delta",
      message_id: "msg-tumor",
      node: "rad_agent",
      delta: "Tumor screening complete",
    };
    const tumorCard: CardUpsertEvent = {
      type: "card.upsert",
      card_type: "tumor_screening_result",
      payload: { patient_id: "007", summary: "screening complete" },
      source_channel: "state",
    };
    const done: MessageDoneEvent = {
      type: "message.done",
      role: "assistant",
      message_id: "msg-tumor",
      node: "rad_agent",
      content: "Tumor screening complete",
      inline_cards: [
        { card_type: "tumor_screening_result", payload: { patient_id: "007", summary: "screening complete" } },
      ],
    };

    const withDelta = reduceStreamEvent(createInitialSessionState(), delta);
    const withCard = reduceStreamEvent(withDelta, tumorCard);
    const finalState = reduceStreamEvent(withCard, done);

    expect(withCard.messages[0]?.inlineCards).toEqual([
      {
        cardType: "tumor_screening_result",
        payload: { patient_id: "007", summary: "screening complete" },
      },
    ]);
    expect(finalState.messages[0]?.inlineCards).toEqual([
      {
        cardType: "tumor_screening_result",
        payload: { patient_id: "007", summary: "screening complete" },
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
