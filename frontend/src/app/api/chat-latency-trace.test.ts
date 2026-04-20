import { describe, expect, it } from "vitest";

import {
  buildChatLatencyTraceAnalysis,
  createChatLatencyTraceStore,
  isSimplePromptHeuristic,
  type ChatLatencyTraceRecord,
  type ChatLatencyTraceSummary,
} from "./chat-latency-trace";

function makeSummary(overrides: Partial<ChatLatencyTraceSummary> = {}): ChatLatencyTraceSummary {
  return {
    traceId: "trace-1",
    sessionId: "session-1",
    runId: "run-1",
    graphPath: ["graph.final"],
    status: "completed",
    serverTotalMs: 1000,
    routerMs: 100,
    retrievalMs: 350,
    llmStartupMs: 100,
    llmGenerationMs: 300,
    streamFlushTailMs: 50,
    serverOrchestrationMs: 600,
    firstServerEventAtMs: 20000,
    estimatedClientServerOffsetMs: 16000,
    flushControlled: true,
    model: "gpt-4.1",
    hasThinking: true,
    responseChars: 120,
    toolCalls: 2,
    retrievalHitCount: 4,
    promptText: "need help",
    uploadsCount: 0,
    contextKeys: [],
    ...overrides,
  };
}

function makeRecord(overrides: Partial<ChatLatencyTraceRecord> = {}): ChatLatencyTraceRecord {
  return {
    traceId: "trace-1",
    scene: "patient",
    promptText: "need help",
    clientWallClockAtSubmit: "2026-04-20T12:00:00.000Z",
    submitAt: 1000,
    firstEventReceivedAt: null,
    firstDeltaReceivedAt: null,
    messageDoneReceivedAt: 2400,
    uiCommittedAt: 2600,
    abortedAt: null,
    errorAt: null,
    status: "active",
    statusSource: null,
    submitOrder: 1,
    backendTraceStart: null,
    backendTraceSteps: [],
    backendTraceSummary: null,
    backendSummary: makeSummary(),
    uploadsCount: 0,
    contextKeys: [],
    sequence: 1,
    updatedAt: 1,
    ...overrides,
  };
}

describe("chat-latency trace analysis", () => {
  it("falls back ttftMs to messageDoneReceivedAt minus submitAt when no delta exists", () => {
    const analysis = buildChatLatencyTraceAnalysis(
      makeRecord({
        firstDeltaReceivedAt: null,
        messageDoneReceivedAt: 2400,
        submitAt: 1000,
      }),
    );

    expect(analysis.derived.ttftMs).toBe(1400);
  });

  it("uses overlap-free shares and keeps prompt-side heuristics independent of retrieval and tool calls", () => {
    const analysis = buildChatLatencyTraceAnalysis(
      makeRecord({
        promptText: "need help",
        backendSummary: makeSummary({
          retrievalMs: 900,
          serverOrchestrationMs: 1150,
          serverTotalMs: 2000,
          firstServerEventAtMs: null,
          estimatedClientServerOffsetMs: null,
          llmStartupMs: 50,
          llmGenerationMs: 40,
          toolCalls: 5,
          retrievalHitCount: 9,
          hasThinking: true,
          responseChars: 1200,
        }),
      }),
    );

    expect(analysis.derived.renderTailMs).toBe(200);
    expect(analysis.derived.uiCompleteMs).toBe(1600);
    expect(analysis.derived.serverOrchestrationOtherMs).toBe(250);
    expect(analysis.derived.accountedShare).toBeCloseTo(0.9);
    expect(analysis.derived.unaccountedShare).toBeCloseTo(0.1);
    expect(analysis.derived.networkOrFirstByteShare).toBeNull();
    expect(isSimplePromptHeuristic({
      promptText: "need help",
      uploadsCount: 0,
      contextKeys: [],
    })).toBe(true);
    expect(analysis.diagnosis.primary).toBe("retrieval");
    expect(analysis.diagnosis.candidates).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ key: "retrieval", ms: 900, share: 900 / 1600 }),
      ]),
    );
    expect(analysis.diagnosis.candidateKeys).toEqual(
      expect.arrayContaining([
        "frontend_render_tail",
        "retrieval",
      ]),
    );
    expect(analysis.diagnosis.secondaryFactors).toEqual(
      expect.arrayContaining(["simple_prompt", "thinking_enabled", "response_too_long"]),
    );
  });

  it("normalizes the first event clock instead of subtracting raw server wall time", () => {
    const analysis = buildChatLatencyTraceAnalysis(
      makeRecord({
        submitAt: 1000,
        backendSummary: makeSummary({
          firstServerEventAtMs: 20000,
          estimatedClientServerOffsetMs: 16000,
          serverOrchestrationMs: 400,
          retrievalMs: 80,
          serverTotalMs: 1000,
        }),
      }),
    );

    expect(analysis.derived.serverToClientFirstEventMs).toBe(3000);
  });

  it.each([
    ["server-to-client threshold not met", { firstServerEventAtMs: 1800, estimatedClientServerOffsetMs: 1000, serverOrchestrationMs: 400, flushControlled: true }, false],
    ["server orchestration threshold not met", { firstServerEventAtMs: 20000, estimatedClientServerOffsetMs: 16000, serverOrchestrationMs: 500, flushControlled: true }, false],
    ["flush controlled false", { firstServerEventAtMs: 20000, estimatedClientServerOffsetMs: 16000, serverOrchestrationMs: 400, flushControlled: false }, false],
    ["all three conditions met", { firstServerEventAtMs: 20000, estimatedClientServerOffsetMs: 16000, serverOrchestrationMs: 400, flushControlled: true }, true],
  ] as const)(
    "selects or rejects network_or_first_byte when %s",
    (_label, summaryPatch, shouldInclude) => {
      const analysis = buildChatLatencyTraceAnalysis(
        makeRecord({
          backendSummary: makeSummary({
            serverTotalMs: 1000,
            retrievalMs: 50,
            routerMs: 50,
            llmStartupMs: 50,
            llmGenerationMs: 100,
            streamFlushTailMs: 25,
            ...summaryPatch,
          }),
        }),
      );

      if (shouldInclude) {
        expect(analysis.diagnosis.candidateKeys).toContain("network_or_first_byte");
      } else {
        expect(analysis.diagnosis.candidateKeys).not.toContain("network_or_first_byte");
      }
    },
  );

  it("nulls transport-derived metrics when the offset is implausible", () => {
    const analysis = buildChatLatencyTraceAnalysis(
      makeRecord({
        backendSummary: makeSummary({
          firstServerEventAtMs: 20000,
          estimatedClientServerOffsetMs: 9999999,
          serverOrchestrationMs: 400,
          flushControlled: true,
        }),
      }),
    );

    expect(analysis.derived.estimatedClientServerOffsetMs).toBeNull();
    expect(analysis.derived.serverToClientFirstEventMs).toBeNull();
    expect(analysis.diagnosis.candidateKeys).not.toContain("network_or_first_byte");
  });

  it("falls back to mixed when elevated candidates are close and no single phase dominates uiCompleteMs", () => {
    const analysis = buildChatLatencyTraceAnalysis(
      makeRecord({
        submitAt: 1000,
        messageDoneReceivedAt: 1700,
        uiCommittedAt: 2000,
        backendSummary: makeSummary({
          serverTotalMs: 2000,
          retrievalMs: 420,
          serverOrchestrationMs: 900,
          llmStartupMs: 100,
          llmGenerationMs: 120,
          firstServerEventAtMs: null,
          estimatedClientServerOffsetMs: null,
          responseChars: 350,
          hasThinking: false,
        }),
      }),
    );

    expect(analysis.derived.uiCompleteMs).toBe(1000);
    expect(analysis.diagnosis.candidateKeys).toEqual(
      expect.arrayContaining([
        "frontend_render_tail",
        "retrieval",
      ]),
    );
    expect(analysis.diagnosis.primary).toBe("mixed");
  });
});

describe("chat-latency trace store exports", () => {
  it("keeps frontend-completed traces sticky against delayed backend terminal updates", () => {
    const store = createChatLatencyTraceStore();

    store.recordClientSubmit({
      traceId: "trace-1",
      scene: "patient",
      promptText: "first turn",
      submitAt: 1000,
      uploadsCount: 0,
      contextKeys: [],
    });
    store.recordClientUiComplete("trace-1", 1500);
    store.recordBackendTraceSummary({
      type: "trace.summary",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.000Z",
      status: "aborted",
      graph_path: ["root", "answer"],
      model: null,
      has_thinking: false,
      response_chars: 0,
      tool_calls: 0,
      retrieval_hit_count: 0,
      response_tokens: null,
      attrs: {},
    });

    const trace = store.getTrace("trace-1");
    expect(trace?.status).toBe("completed");
    expect(trace?.statusSource).toBe("frontend");
  });

  it("lets frontend completion recover from an earlier backend aborted state", () => {
    const store = createChatLatencyTraceStore();

    store.recordClientSubmit({
      traceId: "trace-1",
      scene: "patient",
      promptText: "first turn",
      submitAt: 1000,
      uploadsCount: 0,
      contextKeys: [],
    });
    store.recordBackendTraceSummary({
      type: "trace.summary",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.000Z",
      status: "aborted",
      graph_path: ["root", "answer"],
      model: null,
      has_thinking: false,
      response_chars: 0,
      tool_calls: 0,
      retrieval_hit_count: 0,
      response_tokens: null,
      attrs: {},
    });
    store.recordClientUiComplete("trace-1", 1500);

    const trace = store.getTrace("trace-1");
    expect(trace?.status).toBe("completed");
    expect(trace?.statusSource).toBe("frontend");
  });

  it("keeps a frontend abort sticky against a later backend error", () => {
    const store = createChatLatencyTraceStore();

    store.recordClientSubmit({
      traceId: "trace-1",
      scene: "patient",
      promptText: "first turn",
      submitAt: 1000,
      uploadsCount: 0,
      contextKeys: [],
    });
    store.recordClientAbort("trace-1", 1200);
    store.recordBackendTraceSummary({
      type: "trace.summary",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.000Z",
      status: "error",
      graph_path: ["root", "answer"],
      model: null,
      has_thinking: false,
      response_chars: 0,
      tool_calls: 0,
      retrieval_hit_count: 0,
      response_tokens: null,
      attrs: {},
    });

    const trace = store.getTrace("trace-1");
    expect(trace?.status).toBe("aborted");
    expect(trace?.statusSource).toBe("frontend");
  });

  it("ingests backend trace events and keeps latest export anchored to submit order", () => {
    const store = createChatLatencyTraceStore();

    store.recordClientSubmit({
      traceId: "trace-1",
      scene: "patient",
      promptText: "first turn",
      submitAt: 1000,
      uploadsCount: 0,
      contextKeys: [],
    });
    store.recordClientSubmit({
      traceId: "trace-2",
      scene: "patient",
      promptText: "second turn",
      submitAt: 2000,
      uploadsCount: 0,
      contextKeys: [],
    });

    store.recordBackendTraceStart({
      type: "trace.start",
      trace_id: "trace-1",
      scene: "patient",
      session_id: "session-1",
      run_id: "run-1",
      server_received_at: "2026-04-20T12:00:00.000Z",
      graph_started_at: "2026-04-20T12:00:00.100Z",
      graph_path: ["root", "answer"],
      attrs: { channel: "backend" },
    });
    store.recordBackendTraceStep({
      type: "trace.step",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      name: "router.done",
      at: "2026-04-20T12:00:00.200Z",
      attrs: { node: "router" },
    });
    store.recordBackendTraceSummary({
      type: "trace.summary",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.000Z",
      status: "completed",
      graph_path: ["root", "answer"],
      model: "gpt-4.1",
      has_thinking: true,
      response_chars: 123,
      tool_calls: 2,
      retrieval_hit_count: 1,
      response_tokens: null,
      attrs: { visible_answer: true },
    });
    store.markSuperseded("trace-1", 1400);
    store.recordBackendTraceStep({
      type: "trace.step",
      trace_id: "trace-1",
      session_id: "session-1",
      run_id: "run-1",
      name: "late.backend.step",
      at: "2026-04-20T12:00:02.000Z",
      attrs: { late: true },
    });

    const latest = JSON.parse(store.toLatestTraceJson()) as { traceId: string; status: string };
    const all = JSON.parse(store.toAllTracesJson()) as {
      traces: Array<{
        traceId: string;
        status: string;
        backendTraceStart?: { scene: string; graph_path: string[] };
        backendTraceSteps?: Array<{ name: string }>;
        backendTraceSummary?: { status: string };
      }>;
    };

    expect(latest.traceId).toBe("trace-2");
    expect(store.getTrace("trace-1")?.backendTraceStart?.graph_path).toEqual(["root", "answer"]);
    expect(store.getTrace("trace-1")?.backendTraceSteps).toHaveLength(2);
    expect(store.getTrace("trace-1")?.backendTraceSummary?.response_tokens).toBeNull();
    expect(all.traces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ traceId: "trace-1", status: "superseded" }),
        expect.objectContaining({ traceId: "trace-2" }),
      ]),
    );
  });
});
