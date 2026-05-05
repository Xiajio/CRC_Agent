import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  useTurnLatencyProbe,
  type MessageDoneInput,
  type UiCompleteInput,
} from "./use-turn-latency-probe";

describe("useTurnLatencyProbe", () => {
  it("moves from streaming to message_done to completed for a scene", () => {
    const { result } = renderHook(() => useTurnLatencyProbe());

    act(() => {
      result.current.beginTurn({
        sequence: 1,
        scene: "patient",
        traceId: "trace-patient",
        prompt: "hello",
        clientWallClockAtSubmit: "2026-01-01T00:00:00.000Z",
        uploadsCount: 0,
        contextKeys: [],
        startedAt: 1000,
      });
    });

    expect(result.current.latencyStatusForScene("patient")?.kind).toBe("streaming");
    expect(result.current.activeProbe?.status).toBe("streaming");

    const messageDone: MessageDoneInput = {
      sequence: 1,
      scene: "patient",
      at: 2500,
      assistantMessageId: "msg-1",
      finalContentText: "answer",
    };
    act(() => {
      result.current.markMessageDone(messageDone);
    });

    expect(result.current.activeProbe?.status).toBe("message_done");

    const uiComplete: UiCompleteInput = {
      sequence: 1,
      scene: "patient",
      at: 4200,
      assistantCursor: "1",
    };
    act(() => {
      result.current.markUiComplete(uiComplete);
    });

    const completedStatus = result.current.latencyStatusForScene("patient");
    expect(completedStatus?.kind).toBe("completed");
    if (completedStatus?.kind === "completed") {
      expect(completedStatus.uiCompleteMs).toBe(3200);
    }
  });

  it("keeps patient and doctor completed probes independent", () => {
    const { result } = renderHook(() => useTurnLatencyProbe());

    act(() => {
      result.current.beginTurn({
        sequence: 1,
        scene: "patient",
        traceId: "trace-patient",
        prompt: "patient prompt",
        clientWallClockAtSubmit: "2026-01-01T00:00:01.000Z",
        uploadsCount: 1,
        contextKeys: ["a", "b"],
        startedAt: 1000,
      });
      result.current.markMessageDone({
        sequence: 1,
        scene: "patient",
        at: 1500,
        assistantMessageId: "p-msg",
        finalContentText: "patient",
      });
      result.current.markUiComplete({
        sequence: 1,
        scene: "patient",
        at: 2600,
        assistantCursor: "1",
      });
    });

    act(() => {
      result.current.beginTurn({
        sequence: 2,
        scene: "doctor",
        traceId: "trace-doctor",
        prompt: "doctor prompt",
        clientWallClockAtSubmit: "2026-01-01T00:00:02.000Z",
        uploadsCount: 2,
        contextKeys: ["x"],
        startedAt: 3000,
      });
      result.current.markMessageDone({
        sequence: 2,
        scene: "doctor",
        at: 3500,
        assistantMessageId: "d-msg",
        finalContentText: "doctor",
      });
      result.current.markUiComplete({
        sequence: 2,
        scene: "doctor",
        at: 4600,
        assistantCursor: "1",
      });
    });

    const patientStatus = result.current.latencyStatusForScene("patient");
    expect(patientStatus?.kind).toBe("completed");
    if (patientStatus?.kind === "completed") {
      expect(patientStatus.uiCompleteMs).toBe(1600);
    }
    const doctorStatus = result.current.latencyStatusForScene("doctor");
    expect(doctorStatus?.kind).toBe("completed");
    if (doctorStatus?.kind === "completed") {
      expect(doctorStatus.uiCompleteMs).toBe(1600);
    }
  });

  it("keeps aborted and error probes from reporting completion", () => {
    const { result } = renderHook(() => useTurnLatencyProbe());

    act(() => {
      result.current.beginTurn({
        sequence: 1,
        scene: "patient",
        traceId: "trace-aborted",
        prompt: "bad",
        clientWallClockAtSubmit: "2026-01-01T00:00:03.000Z",
        uploadsCount: 0,
        contextKeys: [],
        startedAt: 500,
      });
      result.current.markAborted("scene_switch");
    });

    expect(result.current.latencyStatusForScene("patient")).toBeNull();
    expect(result.current.activeProbe?.status).toBe("aborted");
    expect(result.current.activeProbe?.abortReason).toBe("scene_switch");

    act(() => {
      result.current.markMessageDone({
        sequence: 1,
        scene: "patient",
        at: 900,
        assistantMessageId: "x",
        finalContentText: "late",
      });
      result.current.markUiComplete({
        sequence: 1,
        scene: "patient",
        at: 1300,
        assistantCursor: "1",
      });
    });

    expect(result.current.activeProbe?.status).toBe("aborted");
    expect(result.current.latencyStatusForScene("patient")).toBeNull();

    act(() => {
      result.current.clearActiveProbe();
      result.current.beginTurn({
        sequence: 2,
        scene: "doctor",
        traceId: "trace-error",
        prompt: "error",
        clientWallClockAtSubmit: "2026-01-01T00:00:04.000Z",
        uploadsCount: 0,
        contextKeys: [],
        startedAt: 1400,
      });
      result.current.markMessageDone({
        sequence: 2,
        scene: "doctor",
        at: 1600,
        assistantMessageId: "d",
      });
      result.current.markError({
        sequence: 2,
        scene: "doctor",
        at: 1700,
        message: "backend failed",
      });
      result.current.markUiComplete({
        sequence: 2,
        scene: "doctor",
        at: 2200,
        assistantCursor: "2",
      });
    });

    expect(result.current.activeProbe?.status).toBe("error");
    expect(result.current.activeProbe?.errorMessage).toBe("backend failed");
    expect(result.current.latencyStatusForScene("doctor")).toBeNull();
  });
});
