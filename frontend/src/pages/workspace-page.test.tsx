import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../app/providers";
import { ApiClientError } from "../app/api/client";
import type { SessionState, StreamEvent } from "../app/api/types";
import { createInitialSessionState, hydrateSessionState } from "../app/store/stream-reducer";
import { WorkspacePage } from "../pages/workspace-page";
import { buildApiClientStub, makeSessionResponse, renderWorkspaceWithSceneSessions } from "../test/test-utils";

let mockSceneSessions: any;
let lastPatientBackgroundProps: any;
let lastDoctorSceneProps: any;
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
    cards?: Record<string, unknown>;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
  }) => {
    lastDoctorSceneProps = props;
    return (
      <section data-testid="doctor-scene-shell" data-latency-kind={props.latencyStatus?.kind ?? "idle"}>
        <div data-testid="doctor-toolbar">{props.toolbar}</div>
        <button type="button" aria-label="patient scene" onClick={props.onSwitchScene}>
          switch to patient
        </button>
        <output data-testid="doctor-draft">{props.draft}</output>
        <output data-testid="doctor-latency-ms">
          {props.latencyStatus && "uiCompleteMs" in props.latencyStatus ? props.latencyStatus.uiCompleteMs : ""}
        </output>
        <button type="button" onClick={() => props.onDraftChange("doctor draft")}>
          set doctor draft
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
    errorMessage,
    latencyStatus,
  }: {
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    onCardPromptRequest?: (prompt: string, context?: Record<string, unknown>) => void;
    errorMessage?: string | null;
    latencyStatus?: { kind: "streaming" } | { kind: "completed"; uiCompleteMs: number } | null;
  }) => (
    <section data-testid="mock-conversation-panel">
      <output data-testid="composer-draft">{draft}</output>
      <output data-testid="conversation-error">{errorMessage ?? ""}</output>
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
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("keeps normal composer submissions text-only and clears the draft", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");

    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

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

    fireEvent.click(await screen.findByRole("button", { name: /trigger oversized upload/i }));

    expect(uploadFile).not.toHaveBeenCalled();
    expect(screen.getByTestId("uploads-panel")).toHaveAttribute("data-disabled", "false");
    expect(screen.getByTestId("conversation-error")).toHaveTextContent(
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

    fireEvent.click(await screen.findByRole("button", { name: /^trigger upload$/i }));

    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getByTestId("conversation-error")).toHaveTextContent(
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

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(callOrder).toEqual(["generate", "stream"]);

    now = 2500;
    onEvent?.({
      type: "message.done",
      role: "assistant",
      message_id: "msg-1",
      content: "answer",
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

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
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
  });

  it("submits triage card prompts with context and keeps them out of the inspector", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    await screen.findByTestId("patient-background-panel");
    expect(lastPatientBackgroundProps?.cards).toEqual({});

    fireEvent.click(screen.getByRole("button", { name: /set card draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");

    fireEvent.click(screen.getByRole("button", { name: /submit triage answer/i }));

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

  it("renders patient chrome with profile scene switching instead of standalone scene buttons", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    expect(screen.getByRole("navigation", { name: "患者工作台" })).toBeInTheDocument();
    const profileSwitch = screen.getByRole("button", { name: /doctor scene/i });
    expect(profileSwitch).toHaveClass("clinical-profile-switch");
    expect(profileSwitch).toHaveTextContent("患者");
    expect(screen.queryByRole("button", { name: /patient scene/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重置当前场景" })).toBeInTheDocument();
  });

  it("omits placeholder-only patient top nav items in production", () => {
    renderWorkspaceWithSceneSessions(buildApiClientStub());

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    expect(navButtons).toHaveLength(2);
    for (const navButton of navButtons) {
      expect(navButton).not.toBeDisabled();
    }
  });

  it("switches the patient top nav between profile and upload workspaces while keeping upload usable", async () => {
    const uploadFile = vi.fn(async () => ({
      asset_id: "1",
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
              filename: "report.pdf",
              derived: { record_id: 1 },
            },
          },
        },
      }),
    );
    const apiClient = buildApiClientStub({ uploadFile, getSession });

    renderWorkspaceWithSceneSessions(apiClient);

    const profileTab = screen.getByRole("button", { name: "资料填写" });
    const uploadTab = screen.getByRole("button", { name: "上传" });
    expect(profileTab).toHaveAttribute("aria-current", "page");
    expect(uploadTab).not.toBeDisabled();
    expect(screen.getByTestId("workspace-right")).toContainElement(screen.getByTestId("uploads-panel"));

    fireEvent.click(uploadTab);

    expect(uploadTab).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("workspace-center")).toContainElement(screen.getByTestId("uploads-panel"));
    fireEvent.click(screen.getByRole("button", { name: /^trigger upload$/i }));

    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(getSession).toHaveBeenCalledWith("patient-session"));

    fireEvent.click(profileTab);

    expect(profileTab).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("patient-identity-panel")).toBeInTheDocument();
    expect(screen.getByTestId("patient-background-panel")).toBeInTheDocument();
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

  it("primes doctor workflow panels for clinical planning prompts before stream events arrive", async () => {
    mockSceneSessions = makeSceneSessions({ activeScene: "doctor" });
    const streamTurn = vi.fn(async () => undefined);
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
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set doctor query draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(mockSceneSessions.doctor.state.roadmap).toEqual([]);
    expect(mockSceneSessions.doctor.state.plan).toEqual([]);
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

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
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
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
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

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
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
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
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

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    now = 2000;
    callbacks.get("patient-session")?.({
      type: "message.done",
      role: "assistant",
      message_id: "patient-msg",
      content: "patient answer",
    });
    view.rerenderWorkspace();

    now = 3100;
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");
    expect(screen.getByTestId("latency-ms")).toHaveTextContent("2100");

    fireEvent.click(screen.getByRole("button", { name: /doctor scene/i }));
    view.rerenderWorkspace();

    expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument();
    expect(screen.getByTestId("doctor-scene-shell")).toHaveAttribute("data-latency-kind", "idle");

    fireEvent.click(screen.getByRole("button", { name: /set doctor draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit doctor draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(2);
    now = 3600;
    callbacks.get("doctor-session")?.({
      type: "message.done",
      role: "assistant",
      message_id: "doctor-msg",
      content: "doctor answer",
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
  });

  it("supersedes an incomplete probe when a newer submit starts", async () => {
    vi.useFakeTimers();
    installRequestAnimationFrameStub();
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    now = 1500;
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(2);
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

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    expect(streamTurn).toHaveBeenCalledTimes(1);

    now = 1500;
    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));
    expect(streamTurn).toHaveBeenCalledTimes(2);

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

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
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
    await act(async () => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByTestId("latency-kind")).toHaveTextContent("completed");

    fireEvent.click(screen.getByRole("button", { name: "重置当前场景" }));

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

    fireEvent.click(screen.getByRole("button", { name: "重置当前场景" }));

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

    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    const view = renderWorkspace(apiClient);

    fireEvent.click(screen.getByRole("button", { name: /set composer draft/i }));
    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    expect(streamTurn).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("latency-kind")).toHaveTextContent("streaming");

    fireEvent.click(screen.getByRole("button", { name: /doctor scene/i }));
    view.rerenderWorkspace();

    expect(screen.getByTestId("doctor-scene-shell")).toBeInTheDocument();
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

    await waitFor(() =>
      expect(screen.getByText("患者名称：王小明")).toBeInTheDocument(),
    );
    expect(screen.getByText("患者编号：P-2001")).toBeInTheDocument();
    expect(screen.getByText("如需修改，请在医生端数据库中处理")).toBeInTheDocument();
    expect(screen.getByTestId("workspace-right")).toBeInTheDocument();
  });
});
