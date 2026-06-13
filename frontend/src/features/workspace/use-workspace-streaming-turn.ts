import { useCallback, useEffect, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from "react";

import { generateTraceId } from "../../app/api/generate-trace-id";
import type { ChatLatencyTraceStore } from "../../app/api/chat-latency-trace";
import type { ApiClient } from "../../app/api/client";
import { StreamRequestError, type StreamTraceTap } from "../../app/api/stream";
import type { ChatTurnRequest, Scene, SessionResponse, SessionState, StreamEvent } from "../../app/api/types";
import { mergeMessageHistory, reduceStreamEvent } from "../../app/store/stream-reducer";
import {
  appendOptimisticUserMessage,
  isAbortError,
  isNotFoundApiError,
  readWorkspaceErrorMessage,
  STREAM_EMPTY_RESPONSE_MESSAGE,
} from "./workspace-flow-utils";
import type {
  BeginTurnInput,
  MessageDoneInput,
  TurnLatencyAbortReason,
  TurnLatencyErrorInput,
  TurnLatencyProbe,
  UiCompleteInput,
} from "./use-turn-latency-probe";

type TurnLatencyApi = {
  activeProbeRef: MutableRefObject<TurnLatencyProbe | null>;
  beginTurn: (input: BeginTurnInput) => void;
  clearScene: (scene: Scene) => void;
  markAborted: (reason: TurnLatencyAbortReason) => void;
  markError: (input: TurnLatencyErrorInput) => void;
  clearActiveProbe: () => void;
  markMessageDone: (input: MessageDoneInput) => void;
  markUiComplete: (input: UiCompleteInput) => void;
};

function isProbeIncomplete(probe: TurnLatencyProbe | null): probe is TurnLatencyProbe {
  return probe !== null && probe.status !== "ui_complete" && probe.status !== "aborted" && probe.status !== "error";
}

function hasVisibleAssistantOutput(event: StreamEvent): boolean {
  if (event.type === "card.upsert") {
    return true;
  }

  if (event.type === "message.delta") {
    return event.delta.trim().length > 0;
  }

  if (event.type !== "message.done") {
    return false;
  }

  if (event.inline_cards?.length) {
    return true;
  }

  return typeof event.content === "string"
    ? event.content.trim().length > 0
    : event.content !== null && event.content !== undefined;
}

const STREAM_BUSY_RETRY_DELAY_MS = 150;

function isBusyStreamError(error: unknown): boolean {
  return error instanceof StreamRequestError && error.status === 409;
}

function createAbortError(): Error {
  if (typeof DOMException !== "undefined") {
    return new DOMException("Stream retry aborted", "AbortError");
  }
  const error = new Error("Stream retry aborted");
  error.name = "AbortError";
  return error;
}

function waitForBusyRetry(signal: AbortSignal, delayMs = STREAM_BUSY_RETRY_DELAY_MS): Promise<void> {
  if (signal.aborted) {
    return Promise.reject(createAbortError());
  }

  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      signal.removeEventListener("abort", handleAbort);
      resolve();
    }, delayMs);

    function handleAbort() {
      window.clearTimeout(timeoutId);
      signal.removeEventListener("abort", handleAbort);
      reject(createAbortError());
    }

    signal.addEventListener("abort", handleAbort, { once: true });
  });
}

export type UseWorkspaceStreamingTurnOptions = {
  scene: Scene;
  apiClient: ApiClient;
  sessionState: SessionState;
  setSessionState: Dispatch<SetStateAction<SessionState>>;
  applySessionResponse: (response: SessionResponse) => void;
  traceStoreRef: MutableRefObject<ChatLatencyTraceStore>;
  latencyProbe: TurnLatencyApi;
  primeInitialState?: (state: SessionState, prompt: string) => SessionState;
  /**
   * Invoked when an HTTP 404 surfaces from the backend, signalling the
   * persisted session no longer exists. Implementations should create a fresh
   * scene session and hydrate state; this hook only handles surfacing the
   * recovery message to the user.
   */
  onSessionExpired?: (scene: Scene) => Promise<SessionResponse | null>;
};

const SESSION_EXPIRED_TURN_MESSAGE =
  "后端会话已失效，本轮消息未发送成功。系统已为您创建新会话，请重新发送。";

export type UseWorkspaceStreamingTurn = {
  isStreaming: boolean;
  isLoadingHistory: boolean;
  errorMessage: string | null;
  submitPrompt: (prompt: string, context?: Record<string, unknown>) => Promise<void>;
  loadMessageHistory: () => Promise<void>;
  resetScene: () => Promise<boolean>;
  abortActiveTurn: (reason: TurnLatencyAbortReason) => void;
};

export function useWorkspaceStreamingTurn({
  scene,
  apiClient,
  sessionState,
  setSessionState,
  applySessionResponse,
  traceStoreRef,
  latencyProbe,
  primeInitialState,
  onSessionExpired,
}: UseWorkspaceStreamingTurnOptions): UseWorkspaceStreamingTurn {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const activeStreamRef = useRef<AbortController | null>(null);
  const streamSequenceRef = useRef(0);

  const abortActiveStream = useCallback(() => {
    activeStreamRef.current?.abort();
    activeStreamRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      abortActiveStream();
      streamSequenceRef.current += 1;
    };
  }, [abortActiveStream]);

  const submitPrompt = useCallback(
    async (prompt: string, context?: Record<string, unknown>) => {
      const normalizedPrompt = prompt.trim();
      const sessionId = sessionState.sessionId;

      if (!sessionId || !normalizedPrompt) {
        return;
      }

      const traceId = generateTraceId();
      const clientWallClockAtSubmit = new Date().toISOString();
      const startedAt = performance.now();
      const currentProbe = latencyProbe.activeProbeRef.current;
      const sequence = streamSequenceRef.current + 1;
      const uploadsCount = Object.keys(sessionState.uploadedAssets ?? {}).length;
      const contextKeys = context ? Object.keys(context).sort() : [];

      abortActiveStream();
      if (isProbeIncomplete(currentProbe)) {
        latencyProbe.markAborted("superseded");
        traceStoreRef.current.markSuperseded(currentProbe.traceId, startedAt);
      }
      latencyProbe.clearScene(scene);

      streamSequenceRef.current = sequence;
      latencyProbe.beginTurn({
        sequence,
        scene,
        traceId,
        prompt: normalizedPrompt,
        clientWallClockAtSubmit,
        uploadsCount,
        contextKeys,
        startedAt,
      });
      traceStoreRef.current.recordClientSubmit({
        traceId,
        scene,
        promptText: normalizedPrompt,
        clientWallClockAtSubmit,
        submitAt: startedAt,
        uploadsCount,
        contextKeys,
      });

      const controller = new AbortController();
      activeStreamRef.current = controller;
      setIsStreaming(true);
      setErrorMessage(null);

      setSessionState((current) => {
        const withOptimistic = appendOptimisticUserMessage(current, normalizedPrompt);
        return primeInitialState ? primeInitialState(withOptimistic, normalizedPrompt) : withOptimistic;
      });

      const request: ChatTurnRequest = {
        message: {
          role: "user",
          content: normalizedPrompt,
        },
        trace_id: traceId,
        ...(context ? { context } : {}),
      };

      const traceTap: StreamTraceTap = (event, receivedAt) => {
        traceStoreRef.current.recordStreamObservation(traceId, event, receivedAt);
      };
      let receivedVisibleAssistantOutput = false;
      let receivedRecoverableError = false;

      const runStreamTurn = () =>
        apiClient.streamTurn(
          sessionId,
          request,
          (event) => {
            if (activeStreamRef.current !== controller || streamSequenceRef.current !== sequence) {
              return;
            }

            if (hasVisibleAssistantOutput(event)) {
              receivedVisibleAssistantOutput = true;
            }
            if (event.type === "error") {
              receivedRecoverableError = true;
            }

            if (event.type === "message.done") {
              const messageDoneAt = performance.now();
              latencyProbe.markMessageDone({
                sequence,
                scene,
                at: messageDoneAt,
                assistantMessageId: event.message_id ?? null,
                finalContentText: typeof event.content === "string" ? event.content : null,
              });
              traceStoreRef.current.recordClientMessageDone(traceId, messageDoneAt);
            }
            setSessionState((current) => reduceStreamEvent(current, event));
          },
          controller.signal,
          traceTap,
        );

      try {
        try {
          await runStreamTurn();
        } catch (streamError) {
          if (
            isBusyStreamError(streamError)
            && activeStreamRef.current === controller
            && streamSequenceRef.current === sequence
            && !controller.signal.aborted
          ) {
            await waitForBusyRetry(controller.signal);
            await runStreamTurn();
          } else {
            throw streamError;
          }
        }

        if (
          !receivedVisibleAssistantOutput
          && !receivedRecoverableError
          && activeStreamRef.current === controller
          && streamSequenceRef.current === sequence
        ) {
          const errorAt = performance.now();
          latencyProbe.markError({
            sequence,
            scene,
            at: errorAt,
            message: STREAM_EMPTY_RESPONSE_MESSAGE,
          });
          traceStoreRef.current.recordClientError(traceId, errorAt);
          setSessionState((current) =>
            reduceStreamEvent(current, {
              type: "error",
              code: "STREAM_EMPTY_RESPONSE",
              message: STREAM_EMPTY_RESPONSE_MESSAGE,
              recoverable: true,
            }),
          );
          setErrorMessage(STREAM_EMPTY_RESPONSE_MESSAGE);
        }
      } catch (error) {
        if (
          !isAbortError(error)
          && activeStreamRef.current === controller
          && streamSequenceRef.current === sequence
        ) {
          const expired = isNotFoundApiError(error);
          const message = expired ? SESSION_EXPIRED_TURN_MESSAGE : readWorkspaceErrorMessage(error);
          const errorAt = performance.now();
          latencyProbe.markError({
            sequence,
            scene,
            at: errorAt,
            message,
          });
          if (streamSequenceRef.current === sequence) {
            traceStoreRef.current.recordClientError(traceId, errorAt);
          }
          setSessionState((current) =>
            reduceStreamEvent(current, {
              type: "error",
              code: expired ? "SESSION_NOT_FOUND" : "STREAM_REQUEST_FAILED",
              message,
              recoverable: true,
            }),
          );
          setErrorMessage(message);
          if (expired && onSessionExpired) {
            void onSessionExpired(scene);
          }
        }
      } finally {
        if (streamSequenceRef.current === sequence) {
          activeStreamRef.current = null;
          setIsStreaming(false);
        }
      }
    },
    [apiClient, abortActiveStream, latencyProbe, primeInitialState, scene, sessionState, setSessionState, traceStoreRef],
  );

  const loadMessageHistory = useCallback(async () => {
    const sessionId = sessionState.sessionId;
    const before = sessionState.messagesNextBeforeCursor;

    if (!sessionId || !before) {
      return;
    }

    setIsLoadingHistory(true);
    setErrorMessage(null);

    try {
      const history = await apiClient.getMessages(sessionId, before, 20);
      setSessionState((current) => mergeMessageHistory(current, history, "prepend"));
    } catch (error) {
      if (isNotFoundApiError(error) && onSessionExpired) {
        void onSessionExpired(scene);
        setErrorMessage(SESSION_EXPIRED_TURN_MESSAGE);
      } else {
        setErrorMessage(readWorkspaceErrorMessage(error));
      }
    } finally {
      setIsLoadingHistory(false);
    }
  }, [
    apiClient,
    onSessionExpired,
    scene,
    sessionState.sessionId,
    sessionState.messagesNextBeforeCursor,
    setSessionState,
  ]);

  const resetScene = useCallback(async (): Promise<boolean> => {
    const sessionId = sessionState.sessionId;

    if (!sessionId) {
      return false;
    }

    abortActiveStream();
    const currentProbe = latencyProbe.activeProbeRef.current;
    if (isProbeIncomplete(currentProbe)) {
      traceStoreRef.current.recordClientAbort(currentProbe.traceId, performance.now());
      latencyProbe.markAborted("reset");
    }
    latencyProbe.clearScene(scene);
    setIsStreaming(false);
    setErrorMessage(null);

    try {
      const response = await apiClient.resetSession(sessionId);
      applySessionResponse(response);
      setSessionState((current) => ({ ...current, eventLog: [] }));
      return true;
    } catch (error) {
      if (isNotFoundApiError(error)) {
        try {
          const replacement = await apiClient.createSession(scene);
          applySessionResponse(replacement);
          setSessionState((current) => ({ ...current, eventLog: [] }));
          return true;
        } catch (replacementError) {
          setErrorMessage(readWorkspaceErrorMessage(replacementError));
        }
        return false;
      }

      setErrorMessage(readWorkspaceErrorMessage(error));
      return false;
    }
  }, [
    abortActiveStream,
    apiClient,
    applySessionResponse,
    latencyProbe,
    scene,
    sessionState.sessionId,
    setSessionState,
    traceStoreRef,
  ]);

  const abortActiveTurn = useCallback(
    (reason: TurnLatencyAbortReason) => {
      const currentProbe = latencyProbe.activeProbeRef.current;
      if (isProbeIncomplete(currentProbe)) {
        traceStoreRef.current.recordClientAbort(currentProbe.traceId, performance.now());
      }
      abortActiveStream();
      latencyProbe.markAborted(reason);
      latencyProbe.clearActiveProbe();
      setIsStreaming(false);
    },
    [abortActiveStream, latencyProbe, traceStoreRef],
  );

  return {
    isStreaming,
    isLoadingHistory,
    errorMessage,
    submitPrompt,
    loadMessageHistory,
    resetScene,
    abortActiveTurn,
  };
}
