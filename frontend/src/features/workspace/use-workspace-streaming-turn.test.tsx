import { act, renderHook } from "@testing-library/react";
import { useRef, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../../app/api/client";
import { createChatLatencyTraceStore } from "../../app/api/chat-latency-trace";
import { generateTraceId } from "../../app/api/generate-trace-id";
import { StreamRequestError } from "../../app/api/stream";
import type { ApiClient } from "../../app/api/client";
import type { SessionResponse, SessionState, StreamEvent } from "../../app/api/types";
import { buildApiClientStub, makeSessionResponse } from "../../test/test-utils";
import { createInitialSessionState, hydrateSessionState } from "../../app/store/stream-reducer";
import { useTurnLatencyProbe } from "./use-turn-latency-probe";
import { useWorkspaceStreamingTurn } from "./use-workspace-streaming-turn";
import { STREAM_EMPTY_RESPONSE_MESSAGE } from "./workspace-flow-utils";

vi.mock("../../app/api/generate-trace-id");

function makeSessionState(overrides: Partial<SessionState> = {}): SessionState {
  return {
    ...createInitialSessionState(),
    ...overrides,
  };
}

function createTurnHarness({
  scene = "patient",
  sessionState,
  applySessionResponse,
  apiClient,
  onSessionExpired,
}: {
  scene?: "patient" | "doctor";
  sessionState: SessionState;
  applySessionResponse?: (response: SessionResponse) => void;
  apiClient: ApiClient;
  onSessionExpired?: (scene: "patient" | "doctor") => Promise<SessionResponse | null>;
}) {
  return renderHook(() => {
    const [state, setState] = useState(sessionState);
    const traceStoreRef = useRef(createChatLatencyTraceStore());
    const latencyProbe = useTurnLatencyProbe();

    const turn = useWorkspaceStreamingTurn({
      scene,
      apiClient,
      sessionState: state,
      setSessionState: setState,
      applySessionResponse: applySessionResponse ?? (() => undefined),
      traceStoreRef,
      latencyProbe: {
        activeProbeRef: latencyProbe.activeProbeRef,
        beginTurn: latencyProbe.beginTurn,
        clearScene: latencyProbe.clearScene,
        markAborted: latencyProbe.markAborted,
        markError: latencyProbe.markError,
        clearActiveProbe: latencyProbe.clearActiveProbe,
        markMessageDone: latencyProbe.markMessageDone,
        markUiComplete: latencyProbe.markUiComplete,
      },
      onSessionExpired,
    });

    return {
      turn,
      state,
      traceStoreRef,
      latencyProbe,
    };
  });
}

describe("useWorkspaceStreamingTurn", () => {
  beforeEach(() => {
    vi.mocked(generateTraceId).mockReset().mockImplementation(() => "trace-1");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("submits and reduces stream events", async () => {
    const streamTurn = vi.fn(
      async (_sessionId: string, request: any, onEvent: (event: StreamEvent) => void) => {
        expect(request).toMatchObject({
          message: { role: "user", content: "hello" },
          trace_id: "trace-1",
        });

        onEvent({
          type: "message.delta",
          message_id: "assistant-1",
          delta: "hi ",
        });
        onEvent({
          type: "message.done",
          role: "assistant",
          message_id: "assistant-1",
          content: "hi there",
        });
      },
    );
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("  hello  ");
    });

    expect(view.result.current.state.messages).toHaveLength(2);
    expect(view.result.current.state.messages[0]).toMatchObject({ type: "human", content: "hello" });
    expect(view.result.current.state.messages[1]).toMatchObject({
      type: "ai",
      content: "hi there",
      id: "assistant-1",
    });
    expect(view.result.current.turn.isStreaming).toBe(false);
    expect(view.result.current.turn.errorMessage).toBeNull();
    expect(view.result.current.traceStoreRef.current.getTrace("trace-1")?.status).toBe("active");
  });

  it("does not mark card-only turns as empty when the stream ends after card upserts", async () => {
    const streamTurn = vi.fn(
      async (_sessionId: string, _request: unknown, onEvent: (event: StreamEvent) => void) => {
        onEvent({
          type: "card.upsert",
          card_type: "triage_summary",
          payload: { summary: "needs follow-up" },
          source_channel: "state",
        });
        onEvent({
          type: "done",
          thread_id: "thread-patient",
          run_id: "run-1",
          snapshot_version: 2,
        });
      },
    );
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("hello");
    });

    expect(view.result.current.state.lastError).toBeNull();
    expect(view.result.current.turn.errorMessage).toBeNull();
    expect(view.result.current.turn.isStreaming).toBe(false);
  });

  it("surfaces backend stream error details as a recoverable turn error", async () => {
    const streamTurn = vi.fn(async () => {
      throw new Error("backend fixture unavailable");
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("hello");
    });

    expect(view.result.current.state.lastError).toMatchObject({
      code: "STREAM_REQUEST_FAILED",
      message: "backend fixture unavailable",
      recoverable: true,
    });
    expect(view.result.current.turn.errorMessage).toBe("backend fixture unavailable");
    expect(view.result.current.latencyProbe.activeProbe?.status).toBe("error");
    expect(view.result.current.turn.isStreaming).toBe(false);
  });

  it("retries once when the stream request is briefly busy", async () => {
    vi.useFakeTimers();
    const streamTurn = vi
      .fn()
      .mockRejectedValueOnce(new StreamRequestError(409, "Session is busy"))
      .mockImplementationOnce(async (_sessionId: string, _request: unknown, onEvent: (event: StreamEvent) => void) => {
        onEvent({
          type: "message.done",
          role: "assistant",
          message_id: "assistant-1",
          content: "retry succeeded",
        });
      });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    let submit: Promise<void> = Promise.resolve();
    act(() => {
      submit = view.result.current.turn.submitPrompt("hello");
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(149);
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
      await submit;
    });

    expect(streamTurn).toHaveBeenCalledTimes(2);
    expect(view.result.current.state.messages).toHaveLength(2);
    expect(view.result.current.state.messages[1]).toMatchObject({
      type: "ai",
      content: "retry succeeded",
    });
    expect(view.result.current.state.lastError).toBeNull();
    expect(view.result.current.turn.errorMessage).toBeNull();
  });

  it("surfaces a recoverable stream error when the busy retry also fails", async () => {
    vi.useFakeTimers();
    const streamTurn = vi
      .fn()
      .mockRejectedValueOnce(new StreamRequestError(409, "Session is busy"))
      .mockRejectedValueOnce(new StreamRequestError(409, "Session is busy"));
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    let submit: Promise<void> = Promise.resolve();
    act(() => {
      submit = view.result.current.turn.submitPrompt("hello");
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(149);
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
      await submit;
    });

    expect(streamTurn).toHaveBeenCalledTimes(2);
    expect(view.result.current.state.lastError).toMatchObject({
      code: "STREAM_REQUEST_FAILED",
      message: "Session is busy",
      recoverable: true,
    });
    expect(view.result.current.turn.errorMessage).toBe("Session is busy");
  });

  it("does not retry non-busy stream failures", async () => {
    const streamTurn = vi.fn(async () => {
      throw new StreamRequestError(503, "backend unavailable");
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("hello");
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(view.result.current.state.lastError).toMatchObject({
      code: "STREAM_REQUEST_FAILED",
      message: "backend unavailable",
      recoverable: true,
    });
  });

  it("marks the turn recoverable when the stream ends without visible assistant output", async () => {
    const streamTurn = vi.fn(
      async (_sessionId: string, _request: unknown, onEvent: (event: StreamEvent) => void) => {
        onEvent({
          type: "done",
          thread_id: "thread-patient",
          run_id: "run-1",
          snapshot_version: 2,
        });
      },
    );
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      scene: "patient",
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("hello");
    });

    expect(view.result.current.state.messages).toHaveLength(1);
    expect(view.result.current.state.messages[0]).toMatchObject({ type: "human", content: "hello" });
    expect(view.result.current.state.lastError).toMatchObject({
      code: "STREAM_EMPTY_RESPONSE",
      message: STREAM_EMPTY_RESPONSE_MESSAGE,
      recoverable: true,
    });
    expect(view.result.current.turn.errorMessage).toBe(STREAM_EMPTY_RESPONSE_MESSAGE);
    expect(view.result.current.latencyProbe.activeProbe?.status).toBe("error");
    expect(view.result.current.turn.isStreaming).toBe(false);
  });

  it("ignores blank prompt submissions", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("   ");
    });

    expect(streamTurn).not.toHaveBeenCalled();
    expect(view.result.current.turn.errorMessage).toBeNull();
    expect(view.result.current.state.messages).toHaveLength(0);
  });

  it("supersedes the previous probe when a newer prompt starts", async () => {
    let inFlight = 0;
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, _cb: (event: StreamEvent) => void, signal?: AbortSignal) => {
      inFlight += 1;
      await new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }
        signal?.addEventListener("abort", () => resolve());
      });
    });
    const apiClient = buildApiClientStub({ streamTurn });

    vi.mocked(generateTraceId).mockImplementationOnce(() => "trace-1").mockImplementationOnce(() => "trace-2");

    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });
    let firstSubmit: Promise<void> = Promise.resolve();
    let secondSubmit: Promise<void> = Promise.resolve();

    act(() => {
      firstSubmit = view.result.current.turn.submitPrompt("first");
      secondSubmit = view.result.current.turn.submitPrompt("second");
    });

    expect(inFlight).toBe(2);
    expect(streamTurn).toHaveBeenCalledTimes(2);
    expect(view.result.current.traceStoreRef.current.getTrace("trace-1")?.status).toBe("superseded");
    expect(view.result.current.state.messages).toHaveLength(2);
    expect(view.result.current.state.messages[0].content).toBe("first");
    expect(view.result.current.state.messages[1].content).toBe("second");

    act(() => {
      view.result.current.turn.abortActiveTurn("scene_switch");
    });
    await act(async () => {
      await Promise.all([firstSubmit, secondSubmit]);
    });
    expect(view.result.current.turn.isStreaming).toBe(false);
    expect(view.result.current.latencyProbe.activeProbe).toBeNull();
  });

  it("ignores stale events from a superseded stream", async () => {
    const callbacks: Array<(event: StreamEvent) => void> = [];
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, callback: (event: StreamEvent) => void, signal?: AbortSignal) => {
      callbacks.push(callback);
      await new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
    });
    const apiClient = buildApiClientStub({ streamTurn });

    vi.mocked(generateTraceId).mockImplementationOnce(() => "trace-1").mockImplementationOnce(() => "trace-2");

    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });
    let firstSubmit: Promise<void> = Promise.resolve();
    let secondSubmit: Promise<void> = Promise.resolve();

    act(() => {
      firstSubmit = view.result.current.turn.submitPrompt("first");
      secondSubmit = view.result.current.turn.submitPrompt("second");
    });

    expect(callbacks).toHaveLength(2);

    act(() => {
      callbacks[0]({
        type: "message.delta",
        message_id: "old-assistant",
        delta: "stale ",
      });
      callbacks[0]({
        type: "message.done",
        role: "assistant",
        message_id: "old-assistant",
        content: "stale answer",
      });
      callbacks[1]({
        type: "message.delta",
        message_id: "fresh-assistant",
        delta: "fresh answer",
      });
    });

    expect(view.result.current.state.messages.map((message) => message.content)).toEqual([
      "first",
      "second",
      "fresh answer",
    ]);
    expect(view.result.current.state.messages.some((message) => String(message.content).includes("stale"))).toBe(false);

    act(() => {
      view.result.current.turn.abortActiveTurn("reset");
    });
    await act(async () => {
      await Promise.all([firstSubmit, secondSubmit]);
    });
  });

  it("ignores stale stream errors after superseding a turn", async () => {
    let callIndex = 0;
    let firstSubmit: Promise<void> = Promise.resolve();
    let secondSubmit: Promise<void> = Promise.resolve();
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, _callback: (event: StreamEvent) => void, signal?: AbortSignal) => {
      callIndex += 1;
      const currentCall = callIndex;
      await new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      if (currentCall === 1) {
        throw new Error("late stale failure");
      }
    });
    const apiClient = buildApiClientStub({ streamTurn });

    vi.mocked(generateTraceId).mockImplementationOnce(() => "trace-1").mockImplementationOnce(() => "trace-2");

    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    act(() => {
      firstSubmit = view.result.current.turn.submitPrompt("first");
      secondSubmit = view.result.current.turn.submitPrompt("second");
    });

    await act(async () => {
      await firstSubmit;
    });

    expect(view.result.current.turn.errorMessage).toBeNull();
    expect(view.result.current.state.lastError).toBeNull();
    expect(view.result.current.traceStoreRef.current.getTrace("trace-1")?.status).toBe("superseded");

    act(() => {
      view.result.current.turn.abortActiveTurn("reset");
    });
    await act(async () => {
      await secondSubmit;
    });
  });

  it("loads older messages from history and prepends them", async () => {
    const apiClient = buildApiClientStub({
      getMessages: vi.fn(async () => ({
        session_id: "patient-session",
        thread_id: "thread-patient",
        snapshot_version: 0,
        messages_total: 2,
        next_before_cursor: null,
        messages: [
          {
            cursor: "1",
            type: "ai",
            content: "old message",
            asset_refs: [],
          },
        ],
      })),
    });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
        messages: [
          {
            cursor: "2",
            type: "human",
            content: "new message",
            assetRefs: [],
          },
        ],
        messagesNextBeforeCursor: "2",
      }),
    });

    await act(async () => {
      await view.result.current.turn.loadMessageHistory();
    });

    expect(apiClient.getMessages).toHaveBeenCalledWith("patient-session", "2", 20);
    expect(view.result.current.state.messages).toHaveLength(2);
    expect(view.result.current.state.messages[0]).toMatchObject({ cursor: "1", content: "old message" });
    expect(view.result.current.state.messages[1]).toMatchObject({ cursor: "2", content: "new message" });
    expect(view.result.current.turn.isLoadingHistory).toBe(false);
  });

  it("clears the clinical event log after reset", async () => {
    const resetSession = vi.fn(async () =>
      makeSessionResponse({
        session_id: "doctor-session",
        scene: "doctor",
        snapshot: {
          messages: [],
          cards: [],
        },
      }),
    );
    const apiClient = buildApiClientStub({ resetSession });
    const view = renderHook(() => {
      const [state, setState] = useState(
        makeSessionState({
          sessionId: "doctor-session",
          eventLog: [
            {
              id: "node-1-decision",
              kind: "node",
              title: "decision",
              tone: "neutral",
            },
          ],
        }),
      );
      const traceStoreRef = useRef(createChatLatencyTraceStore());
      const latencyProbe = useTurnLatencyProbe();

      const turn = useWorkspaceStreamingTurn({
        scene: "doctor",
        apiClient,
        sessionState: state,
        setSessionState: setState,
        applySessionResponse: (response) => {
          setState((current) => hydrateSessionState(current, response));
        },
        traceStoreRef,
        latencyProbe: {
          activeProbeRef: latencyProbe.activeProbeRef,
          beginTurn: latencyProbe.beginTurn,
          clearScene: latencyProbe.clearScene,
          markAborted: latencyProbe.markAborted,
          markError: latencyProbe.markError,
          clearActiveProbe: latencyProbe.clearActiveProbe,
          markMessageDone: latencyProbe.markMessageDone,
          markUiComplete: latencyProbe.markUiComplete,
        },
      });

      return { turn, state };
    });

    await act(async () => {
      await view.result.current.turn.resetScene();
    });

    expect(resetSession).toHaveBeenCalledWith("doctor-session");
    expect(view.result.current.state.eventLog).toEqual([]);
  });

  it("resets by applying reset session response", async () => {
    const resetSession = vi.fn(async () =>
      makeSessionResponse({
        session_id: "patient-session",
        scene: "patient",
        snapshot: {
          messages: [],
          cards: [],
        },
      }),
    );
    const applySessionResponse = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ resetSession });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
      applySessionResponse,
    });

    let didReset = false;
    await act(async () => {
      didReset = await view.result.current.turn.resetScene();
    });

    expect(resetSession).toHaveBeenCalledWith("patient-session");
    expect(didReset).toBe(true);
    expect(applySessionResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        session_id: "patient-session",
      }),
    );
    expect(view.result.current.turn.errorMessage).toBeNull();
  });

  it("returns false and keeps an error message when reset fails", async () => {
    const resetSession = vi.fn(async () => {
      throw new Error("reset failed");
    });
    const applySessionResponse = vi.fn();
    const apiClient = buildApiClientStub({ resetSession });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
      applySessionResponse,
    });

    let didReset = true;
    await act(async () => {
      didReset = await view.result.current.turn.resetScene();
    });

    expect(resetSession).toHaveBeenCalledWith("patient-session");
    expect(didReset).toBe(false);
    expect(applySessionResponse).not.toHaveBeenCalled();
    expect(view.result.current.turn.errorMessage).toBe("reset failed");
  });

  it("recreates scene session on 404 reset response", async () => {
    const replacement = makeSessionResponse({
      session_id: "patient-session-new",
      scene: "patient",
      snapshot: {
        cards: [],
        messages: [],
      },
    });
    const resetSession = vi.fn(async () => {
      throw new ApiClientError(404, "gone", { detail: "gone" });
    });
    const createSession = vi.fn(async () => replacement);
    const applySessionResponse = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ resetSession, createSession });

    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
      applySessionResponse,
    });

    let didReset = false;
    await act(async () => {
      didReset = await view.result.current.turn.resetScene();
    });

    expect(resetSession).toHaveBeenCalledWith("patient-session");
    expect(createSession).toHaveBeenCalledWith("patient");
    expect(didReset).toBe(true);
    expect(applySessionResponse).toHaveBeenCalledWith(replacement);
    expect(view.result.current.turn.errorMessage).toBeNull();
  });

  it("notifies onSessionExpired and surfaces a recovery message when streamTurn returns 404", async () => {
    const streamTurn = vi.fn(async () => {
      const error = new Error("Session not found") as Error & { status: number };
      error.status = 404;
      throw error;
    });
    const onSessionExpired = vi.fn(async () => makeSessionResponse({ session_id: "recovered" }));
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      onSessionExpired,
      sessionState: makeSessionState({ sessionId: "patient-session" }),
    });

    await act(async () => {
      await view.result.current.turn.submitPrompt("hello");
    });

    expect(onSessionExpired).toHaveBeenCalledWith("patient");
    expect(view.result.current.turn.errorMessage).toContain("会话已失效");
    expect(view.result.current.state.lastError?.code).toBe("SESSION_NOT_FOUND");
  });

  it("notifies onSessionExpired when loadMessageHistory hits 404", async () => {
    const getMessages = vi.fn(async () => {
      throw new ApiClientError(404, "Session not found", { detail: "missing" });
    });
    const onSessionExpired = vi.fn(async () => makeSessionResponse({ session_id: "recovered" }));
    const apiClient = buildApiClientStub({ getMessages });
    const view = createTurnHarness({
      apiClient,
      onSessionExpired,
      sessionState: makeSessionState({
        sessionId: "patient-session",
        messagesNextBeforeCursor: "10",
      }),
    });

    await act(async () => {
      await view.result.current.turn.loadMessageHistory();
    });

    expect(onSessionExpired).toHaveBeenCalledWith("patient");
    expect(view.result.current.turn.errorMessage).toContain("会话已失效");
  });

  it("aborts active stream and records client abort", async () => {
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, _cb: (event: StreamEvent) => void, signal?: AbortSignal) => {
      await new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }
        signal?.addEventListener("abort", () => resolve());
      });
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });
    let submit: Promise<void> = Promise.resolve();

    act(() => {
      submit = view.result.current.turn.submitPrompt("hello");
    });

    act(() => {
      view.result.current.turn.abortActiveTurn("reset");
    });
    await act(async () => {
      await submit;
    });

    expect(view.result.current.turn.isStreaming).toBe(false);
    expect(view.result.current.traceStoreRef.current.getTrace("trace-1")?.status).toBe("aborted");
    expect(view.result.current.latencyProbe.activeProbe).toBeNull();
  });

  it("aborts an in-flight stream on hook unmount", () => {
    let capturedSignal: AbortSignal | undefined;
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, _cb: (event: StreamEvent) => void, signal?: AbortSignal) => {
      capturedSignal = signal;
      await new Promise<void>((resolve) => {
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = createTurnHarness({
      apiClient,
      sessionState: makeSessionState({
        sessionId: "patient-session",
      }),
    });

    act(() => {
      void view.result.current.turn.submitPrompt("hello");
    });

    expect(capturedSignal?.aborted).toBe(false);

    act(() => {
      view.unmount();
    });

    expect(capturedSignal?.aborted).toBe(true);
  });
});
