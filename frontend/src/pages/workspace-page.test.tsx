import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../app/providers";
import type { SessionState, StreamEvent } from "../app/api/types";
import { createInitialSessionState, hydrateSessionState } from "../app/store/stream-reducer";
import { WorkspacePage } from "../pages/workspace-page";
import { buildApiClientStub, makeSessionResponse, renderWorkspaceWithSceneSessions } from "../test/test-utils";

let mockSceneSessions: any;
let lastClinicalCardsProps: any;
let lastUploadsPanelProps: any;
let mockGenerateTraceId: ReturnType<typeof vi.fn>;

vi.mock("../features/workspace/use-scene-sessions", () => ({
  useSceneSessions: () => mockSceneSessions,
}));

vi.mock("../app/api/generate-trace-id", () => ({
  generateTraceId: () => mockGenerateTraceId(),
}));

vi.mock("../features/patient-registry/use-patient-registry", () => ({
  usePatientRegistry: () => ({ bindPatient: vi.fn() }),
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
      </div>
    );
  },
}));

vi.mock("../features/cards/clinical-cards-panel", () => ({
  ClinicalCardsPanel: (props: any) => {
    lastClinicalCardsProps = props;
    return <div data-testid="clinical-cards-panel" />;
  },
}));

vi.mock("../features/doctor/doctor-scene-shell", () => ({
  DoctorSceneShell: ({
    draft,
    onDraftChange,
    onSubmit,
    latencyStatus,
  }: {
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
  }) => (
    <section data-testid="doctor-scene-shell" data-latency-kind={latencyStatus?.kind ?? "idle"}>
      <output data-testid="doctor-draft">{draft}</output>
      <output data-testid="doctor-latency-ms">
        {latencyStatus && "uiCompleteMs" in latencyStatus ? latencyStatus.uiCompleteMs : ""}
      </output>
      <button type="button" onClick={() => onDraftChange("doctor draft")}>
        set doctor draft
      </button>
      <button type="button" onClick={() => onSubmit()}>
        submit doctor draft
      </button>
    </section>
  ),
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
    latencyStatus,
  }: {
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    onCardPromptRequest?: (prompt: string, context?: Record<string, unknown>) => void;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
  }) => (
    <section data-testid="mock-conversation-panel">
      <output data-testid="composer-draft">{draft}</output>
      <output data-testid="latency-kind">{latencyStatus?.kind ?? "idle"}</output>
      <output data-testid="latency-ms">
        {latencyStatus && "uiCompleteMs" in latencyStatus ? latencyStatus.uiCompleteMs : ""}
      </output>
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
    </section>
  ),
}));

function makeSessionState(overrides: Partial<SessionState> = {}): SessionState {
  return {
    ...createInitialSessionState(),
    ...overrides,
  };
}

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
      view.rerender(
        <AppProviders apiClient={apiClient}>
          <WorkspacePage />
        </AppProviders>,
      );
    },
  };
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

describe("WorkspacePage patient triage submission wiring", () => {
  beforeEach(() => {
    lastClinicalCardsProps = null;
    lastUploadsPanelProps = null;
    mockSceneSessions = makeSceneSessions();
    mockGenerateTraceId = vi.fn(() => "trace-123");
    window.localStorage.removeItem("chatLatencyDebug");
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("keeps normal composer submissions text-only and clears the draft", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");

    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
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
            filename: "report.pdf",
            derived: { record_id: 1 },
          },
        },
      },
    });
    const uploadFile = vi.fn(async () => ({
        asset_id: "1",
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

    fireEvent.click(await screen.findByRole("button", { name: /trigger upload/i }));

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
        filename: "report.pdf",
        derived: { record_id: 1 },
      },
    });
    expect(lastClinicalCardsProps?.cards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Refreshed patient card",
      },
    });
  });

  it("emits a debug summary with the trace id when the UI probe completes", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    window.localStorage.setItem("chatLatencyDebug", "1");

    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    const consoleDebugSpy = vi.spyOn(console, "debug").mockImplementation(() => undefined);

    let onEvent: ((event: StreamEvent) => void) | undefined;
    const callOrder: string[] = [];
    mockGenerateTraceId.mockImplementation(() => {
      callOrder.push("generate");
      return "trace-123";
    });
    const streamTurn = vi.fn(async (_sessionId: string, request: any, callback: (event: StreamEvent) => void) => {
      callOrder.push("stream");
      expect(request.trace_id).toBe("trace-123");
      onEvent = callback;
    });
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(callOrder).toEqual(["generate", "stream"]);

    now = 2500;
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      content: "answer",
    });

    now = 4200;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));
    expect(consoleDebugSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        traceId: "trace-123",
        uiCompleteMs: 3200,
      }),
    );
  });

  it("bridges stream observations and probe milestones into window.__chatLatency", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    window.localStorage.setItem("chatLatencyDebug", "1");

    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    let onEvent: ((event: StreamEvent) => void) | undefined;
    let traceTap: ((event: StreamEvent, receivedAt: number) => void) | undefined;
    const streamTurn = vi.fn(async (
      _sessionId: string,
      _request: unknown,
      callback: (event: StreamEvent) => void,
      _signal?: AbortSignal,
      tap?: (event: StreamEvent, receivedAt: number) => void,
    ) => {
      onEvent = callback;
      traceTap = tap;
      expect(window.__chatLatency?.latestTrace).toEqual(
        expect.objectContaining({
          traceId: "trace-123",
          submitAt: 1000,
          promptText: "typed composer",
        }),
      );
    });
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
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
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      content: "answer",
    });

    now = 4200;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));

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
  });

  it("submits triage card prompts with context and keeps them out of the inspector", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    await screen.findByTestId("clinical-cards-panel");
    expect(lastClinicalCardsProps?.cards).toEqual({});

    fireEvent.click(screen.getByRole("button", { name: /set card draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");

    fireEvent.click(screen.getByRole("button", { name: /submit triage answer/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
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
      },
      expect.any(Function),
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");
  });

  it("measures patient chat from submit to committed assistant render", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    let onEvent: ((event: StreamEvent) => void) | undefined;
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, callback: (event: StreamEvent) => void) => {
      onEvent = callback;
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 2500;
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      content: "answer",
    });
    view.rerenderWorkspace();

    now = 4200;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("3200");
  });

  it("measures patient chat when message.done omits the message id", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    let onEvent: ((event: StreamEvent) => void) | undefined;
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, callback: (event: StreamEvent) => void) => {
      onEvent = callback;
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 1800;
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: null,
      content: "fallback answer",
    });
    view.rerenderWorkspace();

    now = 2600;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("1600");
  });

  it("preserves completed latency per scene after another scene finishes a turn", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const callbacks = new Map<string, (event: StreamEvent) => void>();
    const streamTurn = vi.fn(async (sessionId: string, _request: unknown, callback: (event: StreamEvent) => void) => {
      callbacks.set(sessionId, callback);
    });
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    now = 2000;
    callbacks.get("patient-session")?.({
      type: "message.done",
      role: "assistant",
      message_id: "patient-msg",
      content: "patient answer",
    });
    view.rerenderWorkspace();

    now = 3100;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("2100");

    fireEvent.click(screen.getByRole("button", { name: /doctor scene/i }));
    view.rerenderWorkspace();

    await waitFor(() => expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument());
    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "idle");

    fireEvent.click(screen.getByRole("button", { name: /set doctor draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(2));
    now = 3600;
    callbacks.get("doctor-session")?.({
      type: "message.done",
      role: "assistant",
      message_id: "doctor-msg",
      content: "doctor answer",
    });
    view.rerenderWorkspace();

    now = 4800;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "completed"));
    expect(screen.getByTestId("doctor-latency-ms")).toHaveTextContent("1700");

    fireEvent.click(screen.getByRole("button", { name: /patient scene/i }));
    view.rerenderWorkspace();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("2100");
  });

  it("supersedes an incomplete probe when a newer submit starts", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 1500;
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");
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

    const taps: Array<((event: StreamEvent, receivedAt: number) => void) | undefined> = [];
    const streamTurn = vi.fn(async (
      _sessionId: string,
      _request: unknown,
      _callback: (event: StreamEvent) => void,
      _signal?: AbortSignal,
      tap?: (event: StreamEvent, receivedAt: number) => void,
    ) => {
      taps.push(tap);
    });
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));

    now = 1500;
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(2));

    taps[0]?.({
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
  });

  it("clears completed latency after resetting the active scene", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    let onEvent: ((event: StreamEvent) => void) | undefined;
    const streamTurn = vi.fn(async (_sessionId: string, _request: unknown, callback: (event: StreamEvent) => void) => {
      onEvent = callback;
    });
    const resetSession = vi.fn(async () => makeSessionResponse({ session_id: "patient-session", scene: "patient" }));
    const apiClient = buildApiClientStub({ streamTurn, resetSession });
    const view = renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    now = 2000;
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      content: "answer",
    });
    view.rerenderWorkspace();

    now = 3200;
    vi.runOnlyPendingTimers();

    await waitFor(() => expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed"));

    fireEvent.click(within(screen.getByTestId("workspace-toolbar")).getAllByRole("button")[2]);

    await waitFor(() => expect(resetSession).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("idle");
  });

  it("aborts an incomplete probe when switching scenes", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    fireEvent.click(screen.getByRole("button", { name: /doctor scene/i }));

    await waitFor(() => expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument());
    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "idle");
  });

  it("suppresses successful latency when the scene receives a streaming error", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });
    const view = renderWorkspace(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
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

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("idle");
  });
});
