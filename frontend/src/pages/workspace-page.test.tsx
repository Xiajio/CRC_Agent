import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../app/providers";
import { ApiClientError } from "../app/api/client";
import type { SessionState, StreamEvent } from "../app/api/types";
import { createInitialSessionState, hydrateSessionState } from "../app/store/stream-reducer";
import { CRC_TRIAGE_START_PROMPT } from "../features/patient-crc-triage/crc-triage-context";
import { WorkspacePage } from "../pages/workspace-page";
import { buildApiClientStub, makeSessionResponse, renderWorkspaceWithSceneSessions } from "../test/test-utils";

let mockSceneSessions: any;
let lastPatientBackgroundProps: any;
let lastDoctorSceneProps: any;
let lastUploadsPanelProps: any;
let mockGenerateTraceId: ReturnType<typeof vi.fn>;
const mockUsePatientRegistry = vi.hoisted(() => vi.fn(() => ({ bindPatient: vi.fn() })));

vi.mock("../features/workspace/use-scene-sessions", () => ({
  useSceneSessions: () => mockSceneSessions,
}));

vi.mock("../app/api/generate-trace-id", () => ({
  generateTraceId: () => mockGenerateTraceId(),
}));

vi.mock("../features/patient-registry/use-patient-registry", () => ({
  usePatientRegistry: mockUsePatientRegistry,
}));

vi.mock("../features/patient-registry/use-registry-browser", () => ({
  useRegistryBrowser: () => ({}),
}));

vi.mock("../features/database/use-database-workbench", () => ({
  useDatabaseWorkbench: () => ({}),
}));

vi.mock("../components/layout/workspace-layout", () => ({
  WorkspaceLayout: ({
    toolbar,
    leftRail,
    centerWorkspace,
    rightInspector,
  }: {
    toolbar: ReactNode;
    leftRail: ReactNode;
    centerWorkspace: ReactNode;
    rightInspector: ReactNode;
  }) => (
    <main data-testid="workspace-layout">
      <div data-testid="workspace-toolbar">{toolbar}</div>
      <div data-testid="workspace-left-rail">{leftRail}</div>
      <div data-testid="workspace-center">{centerWorkspace}</div>
      <div data-testid="workspace-right">{rightInspector}</div>
    </main>
  ),
}));

vi.mock("../features/uploads/uploads-panel", () => ({
  UploadsPanel: (props: any) => {
    lastUploadsPanelProps = props;
    return (
      <div data-testid="uploads-panel" data-disabled={props.disabled ? "true" : "false"}>
        <button
          type="button"
          onClick={() =>
            props.onUpload?.(
              new File(["report"], "report.pdf", {
                type: "application/pdf",
              }),
            )
          }
        >
          trigger upload
        </button>
        <button
          type="button"
          onClick={() => {
            const file = new File(["oversized"], "too-large.pdf", {
              type: "application/pdf",
            });
            Object.defineProperty(file, "size", {
              value: 25 * 1024 * 1024 + 1,
            });
            props.onUpload?.(file);
          }}
        >
          trigger oversized upload
        </button>
      </div>
    );
  },
}));

vi.mock("../features/cards/patient-background-panel", () => ({
  PatientBackgroundPanel: (props: any) => {
    lastPatientBackgroundProps = props;
    return <div data-testid="patient-background-panel" />;
  },
}));

vi.mock("../features/doctor/doctor-scene-shell", () => ({
  DoctorSceneShell: (props: {
    toolbar?: ReactNode;
    onSwitchScene?: () => void;
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    surfaceSwitcher?: ReactNode;
    cards?: Record<string, unknown>;
    patientContext?: Record<string, unknown>;
    onSetCurrentCaseDatabasePatient?: (patientId: number) => void;
    onCardPromptRequest?: (prompt: string, context?: Record<string, unknown>) => void;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
  }) => {
    lastDoctorSceneProps = props;
    return (
      <section data-testid="doctor-scene-shell" data-latency-kind={props.latencyStatus?.kind ?? "idle"}>
        <div data-testid="doctor-toolbar">{props.toolbar}</div>
        <div data-testid="doctor-surface-switcher">{props.surfaceSwitcher}</div>
        <button type="button" aria-label="患者场景" onClick={props.onSwitchScene}>
          switch to patient
        </button>
        <output data-testid="doctor-draft">{props.draft}</output>
        <output data-testid="doctor-latency-ms">
          {props.latencyStatus && "uiCompleteMs" in props.latencyStatus ? props.latencyStatus.uiCompleteMs : ""}
        </output>
        <button type="button" onClick={() => props.onDraftChange("doctor draft")}>
          set doctor draft
        </button>
        <button type="button" onClick={() => props.onDraftChange("请查询患者093")}>
          set doctor explicit patient 093 draft
        </button>
        <button type="button" onClick={() => props.onDraftChange("查询患者093")}>
          set doctor query draft
        </button>
        <button type="button" onClick={() => props.onDraftChange("请基于当前患者信息，生成临床评估、证据依据和治疗建议。")}>
          set doctor clinical draft
        </button>
        <button type="button" onClick={() => props.onSubmit()}>
          submit doctor draft
        </button>
        <button type="button" onClick={() => props.onSetCurrentCaseDatabasePatient?.(93)}>
          set historical case patient 93
        </button>
        <button
          type="button"
          onClick={() => props.onCardPromptRequest?.("为病人 093 生成治疗方案", props.patientContext)}
        >
          submit doctor treatment card prompt
        </button>
        <button
          type="button"
          onClick={() => props.onCardPromptRequest?.("查询病人 #093 的影像资料", props.patientContext)}
        >
          submit doctor imaging card prompt
        </button>
        <button
          type="button"
          onClick={() => props.onCardPromptRequest?.("为病人 093 撰写当日病程记录", props.patientContext)}
        >
          submit doctor progress card prompt
        </button>
      </section>
    );
  },
}));

vi.mock("../features/execution-plan/execution-plan-panel", () => ({
  ExecutionPlanPanel: () => <div data-testid="execution-plan-panel" />,
}));

vi.mock("../features/roadmap/roadmap-panel", () => ({
  RoadmapPanel: () => <div data-testid="roadmap-panel" />,
}));

vi.mock("../features/chat/conversation-panel", () => ({
  ConversationPanel: ({
    draft,
    onDraftChange,
    onSubmit,
    onCardPromptRequest,
    patientContext,
    errorMessage,
    latencyStatus,
    emptyStateVariant,
    quickActions,
    onQuickActionSelect,
    onUploadRequest,
  }: {
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    onCardPromptRequest?: (prompt: string, context?: Record<string, unknown>) => void;
    patientContext?: Record<string, unknown>;
    errorMessage?: string | null;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
    emptyStateVariant?: "clinical" | "patient-assistant";
    quickActions?: { id: string; label: string; prompt: string }[];
    onQuickActionSelect?: (prompt: string) => void;
    onUploadRequest?: () => void;
  }) => (
    <section data-testid="mock-conversation-panel">
      <output data-testid="composer-draft">{draft}</output>
      <output data-testid="conversation-error">{errorMessage ?? ""}</output>
      <output data-testid="empty-state-variant">{emptyStateVariant ?? "clinical"}</output>
      <output data-testid="latency-kind">{latencyStatus?.kind ?? "idle"}</output>
      <output data-testid="latency-ms">
        {latencyStatus && "uiCompleteMs" in latencyStatus ? latencyStatus.uiCompleteMs : ""}
      </output>
      {quickActions?.map((action) => (
        <button
          key={action.id}
          type="button"
          onClick={() => onQuickActionSelect?.(action.prompt)}
        >
          {action.label}
        </button>
      ))}
      <button type="button" onClick={onUploadRequest}>
        upload from assistant home
      </button>
      <button type="button" onClick={() => onDraftChange("typed composer")}>
        set composer draft
      </button>
      <button type="button" onClick={() => onDraftChange("draft for card")}>
        set card draft
      </button>
      <button type="button" onClick={() => onSubmit()}>
        submit composer draft
      </button>
      <button
        type="button"
        onClick={() =>
          onCardPromptRequest?.("There has been fever for 3 days.", {
            triage_interaction: {
              question_id: "triage-q-fever-1",
              field_key: "fever",
              selection_mode: "single",
              selected_option_ids: ["fever"],
              other_text: null,
            },
          })
        }
      >
        submit triage answer
      </button>
      <button
        type="button"
        onClick={() => onCardPromptRequest?.("查询病人 #093 的影像资料", patientContext)}
      >
        submit patient imaging card prompt
      </button>
    </section>
  ),
}));

function makeSessionState(
  overrides: Partial<SessionState> & { patientIdentity?: SessionStateWithPatientIdentity["patientIdentity"] } = {},
): SessionStateWithPatientIdentity {
  return {
    ...createInitialSessionState(),
    ...overrides,
  } as SessionStateWithPatientIdentity;
}

type SessionStateWithPatientIdentity = SessionState & {
  patientIdentity?: {
    patient_name: string | null;
    patient_number: string | null;
    identity_locked: boolean;
  } | null;
};

function makeSceneController(state: SessionState) {
  const controller = {
    state,
    setState: vi.fn((update: SessionState | ((current: SessionState) => SessionState)) => {
      controller.state = typeof update === "function" ? update(controller.state) : update;
    }),
  };
  return controller;
}

function renderWorkspace(apiClient: ReturnType<typeof buildApiClientStub>) {
  const view = renderWorkspaceWithSceneSessions(apiClient);
  return {
    ...view,
    rerenderWorkspace() {
      act(() => {
        view.rerender(
          <AppProviders apiClient={apiClient}>
            <WorkspacePage />
          </AppProviders>,
        );
      });
    },
  };
}

function clickResetCurrentScene() {
  fireEvent.click(screen.getByRole("button", { name: "\u91cd\u7f6e\u5f53\u524d\u573a\u666f" }));
}

function installRequestAnimationFrameStub() {
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((callback: FrameRequestCallback) => window.setTimeout(() => callback(performance.now()), 0)),
  );
  vi.stubGlobal(
    "cancelAnimationFrame",
    vi.fn((id: number) => window.clearTimeout(id)),
  );
}

type ControlledStreamCall = {
  sessionId: string;
  request: unknown;
  callback: (event: StreamEvent) => void;
  signal?: AbortSignal;
  tap?: (event: StreamEvent, receivedAt: number) => void;
  resolve: () => void;
};

function createControlledStreamTurn(onCall?: (call: ControlledStreamCall) => void) {
  const calls: ControlledStreamCall[] = [];
  const streamTurn = vi.fn(async (
    sessionId: string,
    request: unknown,
    callback: (event: StreamEvent) => void,
    signal?: AbortSignal,
    tap?: (event: StreamEvent, receivedAt: number) => void,
  ) => {
    let resolve!: () => void;
    const done = new Promise<void>((doneResolve) => {
      resolve = doneResolve;
    });
    const call: ControlledStreamCall = {
      sessionId,
      request,
      callback,
      signal,
      tap,
      resolve,
    };
    calls.push(call);
    onCall?.(call);

    if (signal?.aborted) {
      resolve();
      return;
    }
    signal?.addEventListener("abort", () => resolve(), { once: true });
    await done;
  });

  return { calls, streamTurn };
}

function makeSceneSessions(overrides: Partial<typeof mockSceneSessions> = {}) {
  const patient = makeSceneController(
    makeSessionState({
      sessionId: "patient-session",
      currentPatientId: 101,
      cards: {
        triage_card: { type: "triage_card", title: "Triage" },
        triage_question_card: {
          type: "triage_question_card",
          question_id: "triage-q-fever-1",
        },
      },
      findings: {
        encounter_track: "outpatient_triage",
        active_inquiry: true,
        inquiry_type: "outpatient_triage",
      },
    }),
  );
  const doctor = makeSceneController(
    makeSessionState({
      sessionId: "doctor-session",
    }),
  );
  const applyResponseToScene = vi.fn((scene: "patient" | "doctor", response: any) => {
    const controller = scene === "patient" ? patient : doctor;
    controller.state = hydrateSessionState(controller.state, response);
  });
  const sessions = {
    activeScene: "patient",
    setActiveScene: vi.fn((scene: "patient" | "doctor") => {
      sessions.activeScene = scene;
    }),
    bootstrapStatus: "ready",
    bootstrapError: null,
    patient,
    doctor,
    applyResponseToScene,
    ...overrides,
  };
  return sessions;
}

function setPatientIdentity(state: SessionState, patientIdentity: SessionStateWithPatientIdentity["patientIdentity"]) {
  return {
    ...state,
    patientIdentity,
  } as SessionStateWithPatientIdentity as SessionState;
}

describe("WorkspacePage patient triage submission wiring", () => {
  beforeEach(() => {
    lastPatientBackgroundProps = null;
    lastDoctorSceneProps = null;
    lastUploadsPanelProps = null;
    mockSceneSessions = makeSceneSessions();
    mockGenerateTraceId = vi.fn(() => "trace-123");
    window.localStorage.removeItem("chatLatencyDebug");
    vi.restoreAllMocks();
    mockUsePatientRegistry.mockClear();
    mockUsePatientRegistry.mockImplementation(() => ({ bindPatient: vi.fn() }));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("keeps normal composer submissions text-only and clears the draft", async () => {
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "typed composer",
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("");
  });

  it("ignores late stream events from a superseded patient turn", async () => {
    mockGenerateTraceId
      .mockImplementationOnce(() => "trace-1")
      .mockImplementationOnce(() => "trace-2");
    const { calls, streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    await waitFor(() => expect(calls).toHaveLength(2));

    act(() => {
      calls[0]?.callback({
        type: "message.delta",
        message_id: "old-assistant",
        delta: "stale answer",
      });
      calls[1]?.callback({
        type: "message.delta",
        message_id: "fresh-assistant",
        delta: "fresh answer",
      });
    });

    const contents = mockSceneSessions.patient.state.messages.map((message: { content: string }) => message.content);
    expect(contents).toContain("fresh answer");
    expect(contents).not.toContain("stale answer");

    calls[1]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("refreshes the patient session after upload and propagates refreshed cards", async () => {
    const refreshedResponse = makeSessionResponse({
      session_id: "patient-session",
      snapshot: {
        cards: [
          {
            card_type: "patient_card",
            payload: {
              type: "patient_card",
              title: "Refreshed patient card",
            },
          },
        ] as any,
        uploaded_assets: {
          "1": {
            asset_url: "/api/sessions/patient-session/assets/1",
            filename: "report.pdf",
            derived: { record_id: 1 },
          },
        },
      },
    });
    const uploadFile = vi.fn(async () => ({
        asset_id: "1",
        asset_url: "/api/sessions/patient-session/assets/1",
        filename: "report.pdf",
        content_type: "application/pdf",
        size: 7,
        sha256: "sha",
        reused: false,
        derived: { record_id: 1 },
      }));
    const getSession = vi.fn(async () => refreshedResponse);
    const apiClient = buildApiClientStub({
      uploadFile,
      getSession,
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "上传报告" }));
    const uploadTrigger = await screen.findByRole("button", { name: /trigger upload/i });
    await act(async () => {
      fireEvent.click(uploadTrigger);
      await Promise.resolve();
    });

    await waitFor(() => expect(apiClient.uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(apiClient.getSession).toHaveBeenCalledWith("patient-session"));
    await waitFor(() => expect(mockSceneSessions.applyResponseToScene).toHaveBeenCalledWith("patient", refreshedResponse));

    expect(uploadFile.mock.invocationCallOrder[0]).toBeLessThan(getSession.mock.invocationCallOrder[0]);
    expect(getSession.mock.invocationCallOrder[0]).toBeLessThan(
      mockSceneSessions.applyResponseToScene.mock.invocationCallOrder[0],
    );
    expect(mockSceneSessions.patient.setState).toHaveBeenCalled();
    expect(mockSceneSessions.patient.state.uploadedAssets).toEqual({
      "1": {
        asset_url: "/api/sessions/patient-session/assets/1",
        filename: "report.pdf",
        derived: { record_id: 1 },
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "我的资料" }));
    await waitFor(() => expect(apiClient.getSessionPatientRecords).toHaveBeenCalledWith("patient-session"));
    expect(apiClient.getSessionCareCards).toHaveBeenCalledWith("patient-session");
    expect(lastPatientBackgroundProps?.cards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Refreshed patient card",
      },
    });
  });

  it("rejects oversized files before entering upload state", async () => {
    const uploadFile = vi.fn(async () => new Promise<never>(() => undefined));
    const apiClient = buildApiClientStub({ uploadFile });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "上传报告" }));
    fireEvent.click(await screen.findByRole("button", { name: /trigger oversized upload/i }));

    expect(uploadFile).not.toHaveBeenCalled();
    expect(screen.getByTestId("uploads-panel")).toHaveAttribute("data-disabled", "false");
    expect(screen.getByTestId("patient-active-error")).toHaveTextContent(
      "文件过大，最大上传大小为 25 MB。",
    );
  });

  it("maps backend 413 upload errors to a friendly size message", async () => {
    const uploadFile = vi.fn(async () => {
      throw new ApiClientError(
        413,
        "UPLOAD_TOO_LARGE: maximum size is 26214400 bytes",
        { detail: "UPLOAD_TOO_LARGE: maximum size is 26214400 bytes" },
      );
    });
    const apiClient = buildApiClientStub({ uploadFile });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "上传报告" }));
    fireEvent.click(await screen.findByRole("button", { name: /^trigger upload$/i }));

    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getByTestId("patient-active-error")).toHaveTextContent(
        "文件过大，最大上传大小为 25 MB。",
      ),
    );
    expect(lastUploadsPanelProps?.statusMessage).toBeNull();
  });

  it("emits a debug summary with the trace id when the UI probe completes", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    window.localStorage.setItem("chatLatencyDebug", "1");

    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    const consoleDebugSpy = vi.spyOn(console, "debug").mockImplementation(() => undefined);

    const callOrder: string[] = [];
    mockGenerateTraceId.mockImplementation(() => {
      callOrder.push("generate");
      return "trace-123";
    });
    const { calls, streamTurn } = createControlledStreamTurn((call) => {
      callOrder.push("stream");
      expect((call.request as { trace_id?: string }).trace_id).toBe("trace-123");
    });
    const apiClient = buildApiClientStub({ streamTurn });

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(callOrder).toEqual(["generate", "stream"]);

    now = 2500;
    act(() => {
      calls[0]?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "msg-1",
        content: "answer",
      });
    });
    view.rerenderWorkspace();

    now = 4200;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(consoleDebugSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        traceId: "trace-123",
        uiCompleteMs: 3200,
      }),
    );
    calls[0]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("bridges stream observations and probe milestones into window.__chatLatency", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    window.localStorage.setItem("chatLatencyDebug", "1");

    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    vi.spyOn(console, "debug").mockImplementation(() => undefined);

    const { calls, streamTurn } = createControlledStreamTurn((call) => {
      expect(window.__chatLatency?.latestTrace).toEqual(
        expect.objectContaining({
          traceId: "trace-123",
          submitAt: 1000,
          promptText: "typed composer",
        }),
      );
    });
    const apiClient = buildApiClientStub({ streamTurn });

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    const traceTap = calls[0]?.tap;
    expect(typeof traceTap).toBe("function");

    traceTap?.({
      type: "trace.start",
      trace_id: "trace-123",
      scene: "patient",
      session_id: "patient-session",
      run_id: "run-1",
      server_received_at: "2026-04-20T12:00:00.100Z",
      graph_started_at: "2026-04-20T12:00:00.200Z",
      graph_path: ["intent_router", "answer"],
      attrs: { flush_controlled: true },
    }, 1200);
    traceTap?.({
      type: "message.delta",
      message_id: "msg-1",
      delta: "a",
    }, 1500);
    traceTap?.({
      type: "trace.step",
      trace_id: "trace-123",
      session_id: "patient-session",
      run_id: "run-1",
      name: "llm.request.started",
      at: "2026-04-20T12:00:00.300Z",
      attrs: {},
    }, 1600);
    traceTap?.({
      type: "trace.step",
      trace_id: "trace-123",
      session_id: "patient-session",
      run_id: "run-1",
      name: "llm.first_token",
      at: "2026-04-20T12:00:00.500Z",
      attrs: {},
    }, 1700);
    traceTap?.({
      type: "trace.step",
      trace_id: "trace-123",
      session_id: "patient-session",
      run_id: "run-1",
      name: "message.done",
      at: "2026-04-20T12:00:01.000Z",
      attrs: {},
    }, 2300);
    traceTap?.({
      type: "trace.step",
      trace_id: "trace-123",
      session_id: "patient-session",
      run_id: "run-1",
      name: "stream.done",
      at: "2026-04-20T12:00:01.100Z",
      attrs: {},
    }, 2350);
    traceTap?.({
      type: "trace.summary",
      trace_id: "trace-123",
      session_id: "patient-session",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.100Z",
      status: "completed",
      graph_path: ["intent_router", "answer"],
      model: "mock-model",
      has_thinking: false,
      response_chars: 42,
      tool_calls: 0,
      retrieval_hit_count: 0,
      response_tokens: null,
      attrs: {},
    }, 2360);

    now = 2500;
    act(() => {
      calls[0]?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "msg-1",
        content: "answer",
      });
    });
    view.rerenderWorkspace();

    now = 4200;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");

    expect(window.__chatLatency?.latestTrace).toEqual(
      expect.objectContaining({
        traceId: "trace-123",
        firstEventReceivedAt: 1200,
        firstDeltaReceivedAt: 1500,
        messageDoneReceivedAt: 2500,
        uiCommittedAt: 4200,
        status: "completed",
        backendSummary: expect.objectContaining({
          serverTotalMs: 1000,
          llmStartupMs: 200,
          llmGenerationMs: 500,
          streamFlushTailMs: 100,
        }),
      }),
    );
    expect(window.__chatLatency?.latestDiagnosis).toEqual(
      expect.objectContaining({
        traceId: "trace-123",
      }),
    );
    calls[0]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("submits triage card prompts with context and keeps them out of the inspector", async () => {
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    expect(screen.getByTestId("workspace-left-rail")).toHaveClass("clinical-patient-left-column-collapsed");
    expect(screen.queryByTestId("patient-background-panel")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /set card draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /submit triage answer/i }));
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "There has been fever for 3 days.",
        },
        context: {
          triage_interaction: {
            question_id: "triage-q-fever-1",
            field_key: "fever",
            selection_mode: "single",
            selected_option_ids: ["fever"],
            other_text: null,
          },
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");
  });

  it("submits patient card prompts with split patient identity from the patient session", async () => {
    mockSceneSessions.patient.state = makeSessionState({
      sessionId: "patient-session",
      registryPatientId: 7,
      caseDatabasePatientId: "093",
    });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /submit patient imaging card prompt/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "查询病人 #093 的影像资料",
        },
        context: {
          registry_patient_id: 7,
          case_database_patient_id: "093",
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
  });

  it("renders patient chrome with the collapsible workspace surface switcher", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(screen.getByRole("navigation", { name: "患者工作台" })).toBeInTheDocument();
    const profileSwitch = screen.getByRole("button", { name: "切换工作台，当前为患者" });
    expect(profileSwitch).toHaveClass("clinical-surface-trigger");
    expect(profileSwitch).toHaveAttribute("aria-haspopup", "menu");
    fireEvent.click(profileSwitch);
    expect(screen.getByRole("menu", { name: "工作台切换" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /患者/ })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("menuitem", { name: /医生/ })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /后台/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "\u91cd\u7f6e\u5f53\u524d\u573a\u666f" })).toBeInTheDocument();
  });

  it("opens the agent admin surface without changing the active graph scene", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    fireEvent.click(screen.getByRole("button", { name: "切换工作台，当前为患者" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /后台/ }));

    expect(screen.getByTestId("agent-admin-console")).toBeInTheDocument();
    expect(screen.getByRole("banner")).toHaveTextContent("智能体后台");
    expect(mockSceneSessions.setActiveScene).not.toHaveBeenCalled();
    expect(document.documentElement).toHaveAttribute("data-theme", "agent-admin");
  });

  it("keeps doctor selection as a graph scene switch from the surface menu", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    fireEvent.click(screen.getByRole("button", { name: "切换工作台，当前为患者" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /医生/ }));

    expect(mockSceneSessions.setActiveScene).toHaveBeenCalledWith("doctor");
  });

  it("keeps patient workspace shell copy in UTF-8 Chinese", async () => {
    const apiClient = buildApiClientStub();
    renderWorkspaceWithSceneSessions(apiClient);

    const navigation = screen.getByRole("navigation", { name: "患者工作台" });
    const navButtons = within(navigation).getAllByRole("button");

    expect(screen.getByText("临床助手")).toBeInTheDocument();
    expect(navButtons.map((navButton) => navButton.textContent)).toEqual([
      "问助手",
      "专项问诊",
      "我的资料",
      "上传报告",
    ]);
    expect(screen.getByRole("button", { name: "问助手" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("empty-state-variant")).toHaveTextContent("patient-assistant");
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-assistant");
    expect(screen.queryByRole("button", { name: "症状" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "照护计划" })).not.toBeInTheDocument();
    expect(screen.getByText("安全会话")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "我的资料" }));
    await waitFor(() => expect(apiClient.getSessionPatientRecords).toHaveBeenCalledWith("patient-session"));
    expect(apiClient.getSessionCareCards).toHaveBeenCalledWith("patient-session");
    expect(lastPatientBackgroundProps).toMatchObject({
      title: "患者背景信息",
      emptyMessage: "当前暂无患者背景信息",
    });
  });

  it("omits placeholder-only patient top nav items in production", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    expect(navButtons).toHaveLength(4);
    for (const navButton of navButtons) {
      expect(navButton).not.toBeDisabled();
    }
  });

  it("starts crc triage from the dedicated patient tab with subflow context", async () => {
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "专项问诊" }));
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-crc_triage");
    expect(screen.getByTestId("crc-triage-panel")).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByTestId("crc-triage-start"));
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: CRC_TRIAGE_START_PROMPT,
        },
        context: {
          patient_subflow: "crc_triage",
          crc_triage: {
            action: "start",
            interaction_source: "patient_crc_triage_tab",
          },
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
  });

  it("submits crc triage follow-up answers with subflow context", async () => {
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "专项问诊" }));
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
      await Promise.resolve();
    });

    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "typed composer",
        },
        context: {
          patient_subflow: "crc_triage",
          crc_triage: {
            action: "answer",
            interaction_source: "patient_crc_triage_tab",
          },
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
  });

  it("saves completed crc triage assessments through the patient session", async () => {
    mockSceneSessions.patient.state = makeSessionState({
      sessionId: "patient-session",
      findings: {
        source_subflow: "crc_triage",
        active_inquiry: false,
        triage_summary: "患者近两周反复便血。",
        triage_risk_level: "medium",
        triage_disposition: "urgent_gi_clinic",
        triage_suggested_tests: ["肠镜"],
        symptom_snapshot: {
          chief_symptoms: "反复便血",
          symptom_focus: "便血",
        },
      },
    });
    const refreshedResponse = makeSessionResponse({
      session_id: "patient-session",
      snapshot: {
        findings: {
          source_subflow: "crc_triage",
          active_inquiry: false,
          triage_summary: "患者近两周反复便血。",
          triage_risk_level: "medium",
          triage_disposition: "urgent_gi_clinic",
        },
      },
    });
    const saveCrcTriageAssessment = vi.fn(async () => ({
      patient_id: 101,
      patient_version: 2,
      projection_version: 2,
      event_ids: ["event-2"],
      record_id: 9,
      reused: false,
    }));
    const getSession = vi.fn(async () => refreshedResponse);
    const getSessionPatientRecords = vi.fn(async () => ({ items: [] }));
    const getSessionCareCards = vi.fn(async () => ({
      focusMetrics: [],
      periodicChecks: [],
      dailyActions: [],
    }));
    const apiClient = buildApiClientStub({
      saveCrcTriageAssessment,
      getSession,
      getSessionPatientRecords,
      getSessionCareCards,
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "专项问诊" }));
    fireEvent.click(screen.getByTestId("crc-triage-save"));

    await waitFor(() => expect(saveCrcTriageAssessment).toHaveBeenCalledTimes(1));
    expect(saveCrcTriageAssessment).toHaveBeenCalledWith(
      "patient-session",
      {
        assessment: expect.objectContaining({
          record_type: "crc_triage_assessment",
          chief_complaint: "反复便血",
          source_session_id: "patient-session",
          source_subflow: "crc_triage",
        }),
      },
    );
    await waitFor(() => expect(getSession).toHaveBeenCalledWith("patient-session"));
    await waitFor(() => expect(getSessionPatientRecords).toHaveBeenCalledWith("patient-session"));
    expect(getSessionCareCards).toHaveBeenCalledWith("patient-session");
    expect(mockSceneSessions.applyResponseToScene).toHaveBeenCalledWith("patient", refreshedResponse);
  });

  it("keeps patient profile and upload reachable without making them the default layout", async () => {
    const uploadFile = vi.fn(async () => ({
      asset_id: "1",
      asset_url: "/api/sessions/patient-session/assets/1",
      filename: "report.pdf",
      content_type: "application/pdf",
      size: 7,
      sha256: "sha",
      reused: false,
      derived: { record_id: 1 },
    }));
    const getSession = vi.fn(async () =>
      makeSessionResponse({
        session_id: "patient-session",
        snapshot: {
          uploaded_assets: {
            "1": {
              asset_url: "/api/sessions/patient-session/assets/1",
              filename: "report.pdf",
              derived: { record_id: 1 },
            },
          },
        },
      }),
    );
    const getSessionPatientRecords = vi.fn(async () => ({
      items: [
        {
          record_id: 8,
          patient_id: 101,
          asset_id: 1,
          record_type: "crc_triage_assessment",
          document_type: "crc_triage_assessment",
          ingest_decision: "record_only",
          snapshot_contributed: false,
          conflict_detected: false,
          summary_text: "建议尽快消化专科评估。",
          source: "patient_generated",
          created_at: "2026-06-25T08:00:00Z",
        },
      ],
    }));
    const getSessionCareCards = vi.fn(async () => ({
      focusMetrics: ["留意便血或黑便是否加重"],
      periodicChecks: ["尽快预约消化专科门诊"],
      dailyActions: ["记录便血颜色、次数和伴随症状"],
    }));
    const apiClient = buildApiClientStub({
      uploadFile,
      getSession,
      getSessionPatientRecords,
      getSessionCareCards,
    });

    renderWorkspaceWithSceneSessions(apiClient);

    const assistantTab = screen.getByRole("button", { name: "问助手" });
    const profileTab = screen.getByRole("button", { name: "我的资料" });
    const uploadTab = screen.getByRole("button", { name: "上传报告" });
    expect(assistantTab).toHaveAttribute("aria-current", "page");
    expect(uploadTab).not.toBeDisabled();
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-assistant");
    expect(screen.getByTestId("workspace-left-rail")).toHaveClass("clinical-patient-left-column-collapsed");
    expect(screen.getByTestId("workspace-right")).toHaveClass("clinical-patient-right-column-collapsed");

    fireEvent.click(profileTab);

    expect(profileTab).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-profile");
    expect(screen.getByTestId("patient-identity-panel")).toBeInTheDocument();
    expect(screen.getByTestId("patient-background-panel")).toBeInTheDocument();
    await waitFor(() => expect(getSessionPatientRecords).toHaveBeenCalledWith("patient-session"));
    expect(getSessionCareCards).toHaveBeenCalledWith("patient-session");
    expect(await screen.findByText("个人随访提醒")).toBeInTheDocument();
    expect(screen.getByText("建议尽快消化专科评估。")).toBeInTheDocument();
    expect(screen.getByText("留意便血或黑便是否加重")).toBeInTheDocument();

    fireEvent.click(uploadTab);

    expect(uploadTab).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-upload");
    expect(screen.getByTestId("workspace-right")).toHaveClass("clinical-patient-right-column-collapsed");
    expect(screen.getByTestId("workspace-right")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByTestId("workspace-center")).toContainElement(screen.getByTestId("uploads-panel"));
    fireEvent.click(screen.getByRole("button", { name: /^trigger upload$/i }));

    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(getSession).toHaveBeenCalledWith("patient-session"));
  });

  it("prefills patient assistant quick actions and can switch to upload", () => {
    const apiClient = buildApiClientStub();
    makeSceneSessions();
    renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "解释检查报告" }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("请帮我解释检查报告");

    fireEvent.click(screen.getByRole("button", { name: "upload from assistant home" }));
    expect(screen.getByRole("button", { name: "上传报告" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("workspace-layout")).toHaveClass("clinical-patient-dashboard-upload");
  });

  it("keeps optional patient record loading failures as empty profile state", async () => {
    const getSessionPatientRecords = vi.fn(async () => {
      throw new ApiClientError(409, "PATIENT_IDENTITY_NOT_FOUND", { detail: "PATIENT_IDENTITY_NOT_FOUND" });
    });
    const getSessionCareCards = vi.fn(async () => {
      throw new ApiClientError(409, "PATIENT_IDENTITY_NOT_FOUND", { detail: "PATIENT_IDENTITY_NOT_FOUND" });
    });
    const apiClient = buildApiClientStub({
      getSessionPatientRecords,
      getSessionCareCards,
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "\u6211\u7684\u8d44\u6599" }));

    await waitFor(() => expect(getSessionPatientRecords).toHaveBeenCalledWith("patient-session"));
    expect(getSessionCareCards).toHaveBeenCalledWith("patient-session");
    expect(screen.queryByTestId("patient-active-error")).not.toBeInTheDocument();
    expect(screen.getByText("\u5f53\u524d\u6682\u65e0\u5386\u53f2\u95ee\u8bca\u8bb0\u5f55")).toBeInTheDocument();
  });

  it("keeps upload progress and success status copy in UTF-8 Chinese", async () => {
    let finishUpload!: (value: {
      asset_id: string;
      asset_url: string;
      filename: string;
      content_type: string;
      size: number;
      sha256: string;
      reused: boolean;
      derived: { record_id: number };
    }) => void;
    const uploadFile = vi.fn(() =>
      new Promise<{
        asset_id: string;
        asset_url: string;
        filename: string;
        content_type: string;
        size: number;
        sha256: string;
        reused: boolean;
        derived: { record_id: number };
      }>((resolve) => {
        finishUpload = resolve;
      }),
    );
    const getSession = vi.fn(async () =>
      makeSessionResponse({
        session_id: "patient-session",
        snapshot: {
          uploaded_assets: {
            "1": {
              asset_url: "/api/sessions/patient-session/assets/1",
              filename: "report.pdf",
              derived: { record_id: 1 },
            },
          },
        },
      }),
    );
    const apiClient = buildApiClientStub({ uploadFile, getSession });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "上传报告" }));
    fireEvent.click(screen.getByRole("button", { name: /^trigger upload$/i }));

    await waitFor(() => expect(lastUploadsPanelProps?.statusMessage).toBe("正在上传 report.pdf..."));

    await act(async () => {
      finishUpload({
        asset_id: "1",
        asset_url: "/api/sessions/patient-session/assets/1",
        filename: "report.pdf",
        content_type: "application/pdf",
        size: 7,
        sha256: "sha",
        reused: false,
        derived: { record_id: 1 },
      });
    });

    await waitFor(() => expect(lastUploadsPanelProps?.statusMessage).toBe("已上传 report.pdf"));
  });

  it("uses inline cards from doctor messages as visible doctor cards", async () => {
    const patientCard = {
      type: "patient_card",
      patient_id: 93,
      data: {
        patient_info: {
          gender: "male",
          age: 31,
        },
      },
    };
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      messages: [
        {
          cursor: "1",
          type: "ai",
          content: "patient details",
          assetRefs: [],
          inlineCards: [
            {
              cardType: "patient_card",
              payload: patientCard,
            },
          ],
        },
      ],
    });

    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument();
    expect(lastDoctorSceneProps?.cards).toEqual({
      patient_card: patientCard,
    });
  });

  it("passes the active doctor session with the review cockpit disabled by default", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session-review",
    });

    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(lastDoctorSceneProps).toEqual(
      expect.objectContaining({
        sessionId: "doctor-session-review",
        doctorReviewCockpitEnabled: false,
      }),
    );
  });

  it("enables the review cockpit shell prop from the explicit Vite env flag", async () => {
    vi.stubEnv("VITE_DOCTOR_REVIEW_COCKPIT", "true");
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session-review",
    });

    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(lastDoctorSceneProps).toEqual(
      expect.objectContaining({
        sessionId: "doctor-session-review",
        doctorReviewCockpitEnabled: true,
      }),
    );
  });

  it("does not bind patient registry from case database sample context", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      caseDatabasePatientId: "093",
      registryPatientId: null,
      currentPatientId: "093",
    });

    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(mockUsePatientRegistry).toHaveBeenCalledWith(
      expect.objectContaining({ registryPatientId: null }),
    );
  });

  it("does not request registry detail for legacy currentPatientId when registryPatientId is null", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      caseDatabasePatientId: null,
      registryPatientId: null,
      currentPatientId: "093",
    });

    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(mockUsePatientRegistry).toHaveBeenCalledWith(
      expect.objectContaining({ registryPatientId: null }),
    );
  });

  it("primes doctor workflow panels for clinical planning prompts before stream events arrive", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor clinical draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(mockSceneSessions.doctor.state.roadmap).toEqual([
      { id: "intent", title: "intent", status: "completed" },
      { id: "planner", title: "planner", status: "in_progress" },
      { id: "assessment", title: "assessment", status: "waiting" },
      { id: "decision", title: "decision", status: "waiting" },
      { id: "citation", title: "citation", status: "waiting" },
      { id: "evaluator", title: "evaluator", status: "waiting" },
      { id: "finalize", title: "finalize", status: "waiting" },
    ]);
    expect(mockSceneSessions.doctor.state.plan).toEqual([
      { id: "collect-context", title: "collect context", status: "completed" },
      { id: "retrieve-guidelines", title: "retrieve guidelines", status: "in_progress" },
      { id: "query-case-database", title: "query case database", status: "pending" },
      { id: "generate-assessment", title: "generate clinical assessment", status: "pending" },
      { id: "generate-recommendation", title: "generate treatment recommendation", status: "pending" },
      { id: "finalize-report", title: "finalize report", status: "pending" },
    ]);
  });

  it("marks the active doctor plan step blocked when a primed clinical request fails", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    const streamTurn = vi.fn(async () => {
      throw new Error("network failed");
    });
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor clinical draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    await waitFor(() => {
      expect(mockSceneSessions.doctor.state.plan[1]).toMatchObject({
        id: "retrieve-guidelines",
        status: "blocked",
        error_message: "network failed",
      });
    });
    expect(mockSceneSessions.doctor.state.lastError).toEqual({
      code: "STREAM_REQUEST_FAILED",
      message: "network failed",
      recoverable: true,
    });
  });

  it("does not prime doctor workflow panels for simple patient lookup prompts", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor query draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(mockSceneSessions.doctor.state.roadmap).toEqual([]);
    expect(mockSceneSessions.doctor.state.plan).toEqual([]);
  });

  it("submits doctor prompts with the active case database patient context", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      caseDatabasePatientId: "093",
    });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "doctor-session",
      expect.objectContaining({
        context: {
          case_database_patient_id: "093",
        },
      }),
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
  });

  it("derives case database context from an explicit doctor patient lookup prompt", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor explicit patient 093 draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "doctor-session",
      expect.objectContaining({
        context: {
          case_database_patient_id: "093",
        },
      }),
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
  });

  it("stores a selected historical case as doctor case context", () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    fireEvent.click(screen.getByRole("button", { name: /set historical case patient 93/i }));

    expect(mockSceneSessions.doctor.state.caseDatabasePatientId).toBe("093");
    expect(mockSceneSessions.doctor.state.currentPatientId).toBe("093");
  });

  it("submits doctor treatment card prompts with split patient identity and primes the workflow scaffold", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      registryPatientId: 7,
      caseDatabasePatientId: "093",
    });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /submit doctor treatment card prompt/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "doctor-session",
      {
        message: {
          role: "user",
          content: "为病人 093 生成治疗方案",
        },
        context: {
          registry_patient_id: 7,
          case_database_patient_id: "093",
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(mockSceneSessions.doctor.state.roadmap).toEqual([
      { id: "intent", title: "intent", status: "completed" },
      { id: "planner", title: "planner", status: "in_progress" },
      { id: "assessment", title: "assessment", status: "waiting" },
      { id: "decision", title: "decision", status: "waiting" },
      { id: "citation", title: "citation", status: "waiting" },
      { id: "evaluator", title: "evaluator", status: "waiting" },
      { id: "finalize", title: "finalize", status: "waiting" },
    ]);
    expect(mockSceneSessions.doctor.state.plan).toEqual([
      { id: "collect-context", title: "collect context", status: "completed" },
      { id: "retrieve-guidelines", title: "retrieve guidelines", status: "in_progress" },
      { id: "query-case-database", title: "query case database", status: "pending" },
      { id: "generate-assessment", title: "generate clinical assessment", status: "pending" },
      { id: "generate-recommendation", title: "generate treatment recommendation", status: "pending" },
      { id: "finalize-report", title: "finalize report", status: "pending" },
    ]);
  });

  it.each([
    ["submit doctor imaging card prompt", "查询病人 #093 的影像资料"],
    ["submit doctor progress card prompt", "为病人 093 撰写当日病程记录"],
  ])("submits doctor %s with split patient identity without priming treatment scaffolds", async (buttonName, prompt) => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      registryPatientId: 7,
      caseDatabasePatientId: "093",
    });
    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: new RegExp(buttonName, "i") }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "doctor-session",
      {
        message: {
          role: "user",
          content: prompt,
        },
        context: {
          registry_patient_id: 7,
          case_database_patient_id: "093",
        },
        trace_id: "trace-123",
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(mockSceneSessions.doctor.state.roadmap).toEqual([]);
    expect(mockSceneSessions.doctor.state.plan).toEqual([]);
  });

  it("measures patient chat from submit to committed assistant render", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const { calls, streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 2500;
    act(() => {
      calls[0]?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "msg-1",
        content: "answer",
      });
    });
    view.rerenderWorkspace();

    now = 4200;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("3200");
    calls[0]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("measures patient chat when message.done omits the message id", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const { calls, streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 1800;
    act(() => {
      calls[0]?.callback({
        type: "message.done",
        role: "assistant",
        message_id: null,
        content: "fallback answer",
      });
    });
    view.rerenderWorkspace();

    now = 2600;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("1600");
    calls[0]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("preserves completed latency per scene after another scene finishes a turn", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const callsBySession = new Map<string, ControlledStreamCall>();
    const { streamTurn } = createControlledStreamTurn((call) => {
      callsBySession.set(call.sessionId, call);
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    now = 2000;
    act(() => {
      callsBySession.get("patient-session")?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "patient-msg",
        content: "patient answer",
      });
    });
    view.rerenderWorkspace();

    now = 3100;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("2100");

    fireEvent.click(screen.getByRole("button", { name: "切换工作台，当前为患者" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /医生/ }));
    view.rerenderWorkspace();

    expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument();
    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "idle");

    fireEvent.click(screen.getByRole("button", { name: /set doctor draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(2);
    now = 3600;
    act(() => {
      callsBySession.get("doctor-session")?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "doctor-msg",
        content: "doctor answer",
      });
    });
    view.rerenderWorkspace();

    now = 4800;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "completed");
    expect(screen.getByTestId("doctor-latency-ms")).toHaveTextContent("1700");

    mockSceneSessions.setActiveScene("patient");
    view.rerenderWorkspace();

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("2100");
    callsBySession.get("patient-session")?.resolve();
    callsBySession.get("doctor-session")?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("keeps a superseded trace even if a late backend completion arrives for the older turn", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    window.localStorage.setItem("chatLatencyDebug", "1");
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    mockGenerateTraceId
      .mockImplementationOnce(() => "trace-1")
      .mockImplementationOnce(() => "trace-2");

    const { calls, streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    expect(streamTurn).toHaveBeenCalledTimes(1);

    now = 1500;
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    expect(streamTurn).toHaveBeenCalledTimes(2);

    calls[0]?.tap?.({
      type: "trace.summary",
      trace_id: "trace-1",
      session_id: "patient-session",
      run_id: "run-1",
      scene: "patient",
      at: "2026-04-20T12:00:01.000Z",
      status: "completed",
      graph_path: ["intent_router", "answer"],
      model: "mock-model",
      has_thinking: false,
      response_chars: 42,
      tool_calls: 0,
      retrieval_hit_count: 0,
      response_tokens: null,
      attrs: {},
    }, 1800);

    const all = JSON.parse(window.__chatLatency?.toAllTracesJson() ?? "{\"traces\":[]}") as {
      traces: Array<{ traceId: string; status: string }>;
    };

    expect(all.traces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ traceId: "trace-1", status: "superseded" }),
        expect.objectContaining({ traceId: "trace-2" }),
      ]),
    );
    calls[1]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("clears completed latency after resetting the active scene", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const { calls, streamTurn } = createControlledStreamTurn();
    const resetSession = vi.fn(async () => makeSessionResponse({ session_id: "patient-session", scene: "patient" }));
    const apiClient = buildApiClientStub({ streamTurn, resetSession });
    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    now = 2000;
    act(() => {
      calls[0]?.callback({
        type: "message.done",
        role: "assistant",
        message_id: "msg-1",
        content: "answer",
      });
    });
    view.rerenderWorkspace();

    now = 3200;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");

    clickResetCurrentScene();

    await act(async () => {
      await Promise.resolve();
    });
    expect(resetSession).toHaveBeenCalledTimes(1);
    expect(mockSceneSessions.applyResponseToScene).toHaveBeenCalledWith(
      "patient",
      expect.objectContaining({
        snapshot: expect.objectContaining({
          messages: [],
          cards: [],
          roadmap: [],
          plan: [],
          references: [],
        }),
      }),
    );
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("idle");
  });

  it("keeps the current draft when active scene reset fails", async () => {
    const resetSession = vi.fn(async () => {
      throw new Error("reset failed");
    });
    const apiClient = buildApiClientStub({ resetSession });

    renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");

    clickResetCurrentScene();

    await waitFor(() => expect(resetSession).toHaveBeenCalledWith("patient-session"));
    expect(mockSceneSessions.applyResponseToScene).not.toHaveBeenCalled();
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");
    expect(screen.getByTestId("conversation-error")).toHaveTextContent("reset failed");
  });

  it("recreates the active scene when reset finds a stale backend session", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    mockSceneSessions.doctor.state = makeSessionState({
      sessionId: "doctor-session",
      currentPatientId: 1024,
      cards: {
        patient_card: {
          type: "patient_card",
          patient_id: "1024",
          data: { patient_info: { age: 58 } },
        },
      },
    });

    const replacement = makeSessionResponse({
      session_id: "doctor-session-new",
      scene: "doctor",
      snapshot: {
        current_patient_id: null,
        cards: [],
        messages: [],
      },
    });
    const resetSession = vi.fn(async () => {
      throw Object.assign(new Error("Session not found"), { status: 404 });
    });
    const createSession = vi.fn(async () => replacement);
    const apiClient = buildApiClientStub({ resetSession, createSession });

    renderWorkspaceWithSceneSessions(apiClient);

    clickResetCurrentScene();

    await waitFor(() => expect(resetSession).toHaveBeenCalledWith("doctor-session"));
    await waitFor(() => expect(createSession).toHaveBeenCalledWith("doctor"));
    expect(mockSceneSessions.applyResponseToScene).toHaveBeenCalledWith("doctor", replacement);
    expect(mockSceneSessions.doctor.state.sessionId).toBe("doctor-session-new");
    expect(mockSceneSessions.doctor.state.currentPatientId).toBeNull();
    expect(mockSceneSessions.doctor.state.cards).toEqual({});
  });

  it("aborts an incomplete probe when switching scenes", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const { streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    fireEvent.click(screen.getByRole("button", { name: "切换工作台，当前为患者" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /医生/ }));
    view.rerenderWorkspace();

    expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument();
    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "idle");
  });

  it("suppresses successful latency when the scene receives a streaming error", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const { calls, streamTurn } = createControlledStreamTurn();
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    mockSceneSessions.patient.state = {
      ...mockSceneSessions.patient.state,
      lastError: {
        code: "GRAPH_RUN_FAILED",
        message: "backend failed",
        recoverable: true,
      },
    };
    view.rerenderWorkspace();
    view.rerenderWorkspace();

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("idle");
    calls[0]?.resolve();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("renders the patient identity panel in the patient scene and hydrates it from state", async () => {
    const apiClient = buildApiClientStub();

    mockSceneSessions = makeSceneSessions({
      patient: makeSceneController(
        setPatientIdentity(
          makeSessionState({
            sessionId: "patient-session",
            currentPatientId: 101,
          }),
          {
            patient_name: "王小明",
            patient_number: "P-2001",
            identity_locked: true,
          },
        ),
      ),
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "我的资料" }));
    await waitFor(() =>
      expect(screen.getByText("患者名称：王小明")).toBeInTheDocument(),
    );
    expect(screen.getByText("患者编号：P-2001")).toBeInTheDocument();
    expect(screen.getByText("如需修改，请在医生端数据库中处理")).toBeInTheDocument();
    expect(screen.getByTestId("workspace-right")).toBeInTheDocument();
  });
});
