import { useCallback, useRef, useState } from "react";

import type { ConversationLatencyStatus } from "../chat/conversation-panel";
import type { Scene } from "../../app/api/types";

type TurnLatencyProbeStatus = "streaming" | "message_done" | "ui_complete" | "aborted" | "error";

export type TurnLatencyAbortReason = "scene_switch" | "reset" | "superseded";

export type TurnLatencyProbe = {
  sequence: number;
  scene: Scene;
  traceId: string;
  prompt: string;
  clientWallClockAtSubmit: string;
  uploadsCount: number;
  contextKeys: string[];
  status: TurnLatencyProbeStatus;
  startedAt: number;
  messageDoneAt: number | null;
  renderCommittedAt: number | null;
  assistantMessageId: string | null;
  assistantCursor: string | null;
  finalContentText: string | null;
  uiCompleteMs: number | null;
  errorMessage: string | null;
  abortReason: TurnLatencyAbortReason | null;
};

export type BeginTurnInput = {
  sequence: number;
  scene: Scene;
  traceId: string;
  prompt: string;
  clientWallClockAtSubmit: string;
  uploadsCount: number;
  contextKeys: string[];
  startedAt: number;
};

export type MessageDoneInput = {
  sequence: number;
  scene: Scene;
  at: number;
  assistantMessageId?: string | null;
  finalContentText?: string | null;
};

export type UiCompleteInput = {
  sequence: number;
  scene: Scene;
  at: number;
  assistantCursor?: string | null;
};

export type TurnLatencyErrorInput = {
  sequence: number;
  scene: Scene;
  at: number;
  message: string;
};

type RecentCompletedProbes = Record<Scene, TurnLatencyProbe | null>;

type TurnLatencyProbeState = {
  activeProbe: TurnLatencyProbe | null;
  recentCompletedProbes: RecentCompletedProbes;
};

function makeRecentCompletedProbes(): RecentCompletedProbes {
  return {
    patient: null,
    doctor: null,
  };
}

function isProbeIncomplete(probe: TurnLatencyProbe | null): probe is TurnLatencyProbe {
  return probe !== null && probe.status !== "ui_complete" && probe.status !== "aborted" && probe.status !== "error";
}

function isProbeMatch(probe: TurnLatencyProbe, input: { sequence: number; scene: Scene }): boolean {
  return probe.scene === input.scene && probe.sequence === input.sequence;
}

function computeLatencyStatus(
  scene: Scene,
  activeProbe: TurnLatencyProbe | null,
  recentCompletedProbes: RecentCompletedProbes,
): ConversationLatencyStatus | null {
  const completedProbe = recentCompletedProbes[scene];

  if (activeProbe && activeProbe.scene === scene && (activeProbe.status === "streaming" || activeProbe.status === "message_done")) {
    return { kind: "streaming" };
  }

  if (
    completedProbe
    && completedProbe.scene === scene
    && completedProbe.status === "ui_complete"
    && completedProbe.uiCompleteMs !== null
  ) {
    return {
      kind: "completed",
      uiCompleteMs: completedProbe.uiCompleteMs,
    };
  }

  return null;
}

export function useTurnLatencyProbe() {
  const activeProbeRef = useRef<TurnLatencyProbe | null>(null);
  const [state, setState] = useState<TurnLatencyProbeState>({
    activeProbe: null,
    recentCompletedProbes: makeRecentCompletedProbes(),
  });

  const activeProbe = state.activeProbe;

  const setProbe = useCallback((probe: TurnLatencyProbe | null): void => {
    activeProbeRef.current = probe;
    setState((current) => {
      if (current.activeProbe === probe) {
        return current;
      }

      return {
        ...current,
        activeProbe: probe,
      };
    });
  }, []);

  const beginTurn = useCallback((input: BeginTurnInput): void => {
    setProbe({
      sequence: input.sequence,
      scene: input.scene,
      traceId: input.traceId,
      prompt: input.prompt,
      clientWallClockAtSubmit: input.clientWallClockAtSubmit,
      uploadsCount: input.uploadsCount,
      contextKeys: input.contextKeys,
      status: "streaming",
      startedAt: input.startedAt,
      messageDoneAt: null,
      renderCommittedAt: null,
      assistantMessageId: null,
      assistantCursor: null,
      finalContentText: null,
      uiCompleteMs: null,
      errorMessage: null,
      abortReason: null,
    });
  }, [setProbe]);

  const clearActiveProbe = useCallback((): void => {
    setProbe(null);
  }, [setProbe]);

  const clearScene = useCallback(
    (scene: Scene): void => {
      const currentProbe = activeProbeRef.current;
      if (currentProbe && currentProbe.scene === scene) {
        setProbe(null);
      }

      setState((current) => ({
        ...current,
        recentCompletedProbes: {
          ...current.recentCompletedProbes,
          [scene]: null,
        },
      }));
    },
    [setProbe],
  );

  const markMessageDone = useCallback((input: MessageDoneInput): void => {
    const currentProbe = activeProbeRef.current;
    if (!currentProbe || !isProbeMatch(currentProbe, input) || currentProbe.status !== "streaming") {
      return;
    }

    setProbe({
      ...currentProbe,
      status: "message_done",
      messageDoneAt: input.at,
      assistantMessageId:
        input.assistantMessageId === undefined ? currentProbe.assistantMessageId : input.assistantMessageId,
      finalContentText:
        input.finalContentText === undefined ? currentProbe.finalContentText : input.finalContentText,
    });
  }, [setProbe]);

  const markUiComplete = useCallback((input: UiCompleteInput): void => {
    const currentProbe = activeProbeRef.current;
    if (!currentProbe || !isProbeMatch(currentProbe, input) || currentProbe.status !== "message_done") {
      return;
    }

    const completedProbe: TurnLatencyProbe = {
      ...currentProbe,
      status: "ui_complete",
      renderCommittedAt: input.at,
      assistantCursor: input.assistantCursor ?? currentProbe.assistantCursor,
      uiCompleteMs: Math.max(input.at - currentProbe.startedAt, 0),
    };

    setState((current) => ({
      ...current,
      recentCompletedProbes: {
        ...current.recentCompletedProbes,
        [currentProbe.scene]: completedProbe,
      },
    }));
    setProbe(completedProbe);
  }, [setProbe]);

  const markAborted = useCallback((reason: TurnLatencyAbortReason): void => {
    const currentProbe = activeProbeRef.current;
    if (!isProbeIncomplete(currentProbe)) {
      return;
    }

    setProbe({
      ...currentProbe,
      status: "aborted",
      abortReason: reason,
    });
  }, [setProbe]);

  const markError = useCallback((input: TurnLatencyErrorInput): void => {
    const currentProbe = activeProbeRef.current;
    if (!currentProbe || !isProbeMatch(currentProbe, input) || !isProbeIncomplete(currentProbe)) {
      return;
    }

    setProbe({
      ...currentProbe,
      status: "error",
      errorMessage: input.message,
    });
  }, [setProbe]);

  const latencyStatusForScene = useCallback(
    (scene: Scene): ConversationLatencyStatus | null =>
      computeLatencyStatus(scene, activeProbe, state.recentCompletedProbes),
    [activeProbe, state.recentCompletedProbes],
  );

  return {
    activeProbeRef,
    activeProbe,
    beginTurn,
    clearActiveProbe,
    clearScene,
    markMessageDone,
    markUiComplete,
    markAborted,
    markError,
    latencyStatusForScene,
  };
}
