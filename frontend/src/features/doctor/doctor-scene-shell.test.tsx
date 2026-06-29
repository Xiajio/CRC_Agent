import type { ReactNode, SetStateAction } from "react";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockConversationPanel = vi.hoisted(() => vi.fn());
const mockDoctorMultimodalView = vi.hoisted(() => vi.fn());
const mockDoctorReportDraftView = vi.hoisted(() => vi.fn());
const mockDoctorReviewCockpit = vi.hoisted(() => vi.fn());
const mockDoctorViewState = vi.hoisted(() => {
  const state = {
    activeDoctorTab: "consultation" as
      | "consultation"
      | "database"
      | "multimodal"
      | "reports"
      | "review",
    activeDatabaseSource: "patient_registry" as "historical_case_base" | "patient_registry",
  };

  const setActiveDoctorTab = vi.fn((next: SetStateAction<typeof state.activeDoctorTab>) => {
    state.activeDoctorTab = typeof next === "function" ? next(state.activeDoctorTab) : next;
  });
  const setActiveDatabaseSource = vi.fn((next: SetStateAction<typeof state.activeDatabaseSource>) => {
    state.activeDatabaseSource =
      typeof next === "function" ? next(state.activeDatabaseSource) : next;
  });

  return { state, setActiveDoctorTab, setActiveDatabaseSource };
});

vi.mock("../chat/conversation-panel", () => ({
  ConversationPanel: (props: Record<string, unknown>) => {
    mockConversationPanel(props);
    return null;
  },
}));

vi.mock("./use-doctor-view-state", () => ({
  useDoctorViewState: () => ({
    activeDoctorTab: mockDoctorViewState.state.activeDoctorTab,
    setActiveDoctorTab: mockDoctorViewState.setActiveDoctorTab,
    activeDatabaseSource: mockDoctorViewState.state.activeDatabaseSource,
    setActiveDatabaseSource: mockDoctorViewState.setActiveDatabaseSource,
  }),
}));

vi.mock("../../components/layout/workspace-layout", () => ({
  WorkspaceLayout: ({
    toolbar,
    centerWorkspace,
    rightInspector,
  }: {
    toolbar: ReactNode;
    centerWorkspace: ReactNode;
    rightInspector?: ReactNode;
  }) => (
    <div>
      <div data-testid="toolbar">{toolbar}</div>
      <div data-testid="center-workspace">{centerWorkspace}</div>
      <div data-testid="right-inspector">{rightInspector}</div>
    </div>
  ),
}));

vi.mock("../cards/clinical-cards-panel", () => ({
  ClinicalCardsPanel: () => <section>医疗卡片</section>,
}));

vi.mock("../roadmap/roadmap-panel", () => ({
  RoadmapPanel: () => <section>工作流路线图</section>,
}));

vi.mock("../execution-plan/execution-plan-panel", () => ({
  ExecutionPlanPanel: () => (
    <section>
      <h2>执行计划</h2>
      <h2>参考列表（前 2 条）</h2>
    </section>
  ),
}));

vi.mock("../patient-registry/patient-registry-alerts", () => ({
  PatientRegistryAlertsPanel: () => null,
}));

vi.mock("../patient-registry/patient-records-panel", () => ({
  PatientRecordsPanel: () => null,
}));

vi.mock("./doctor-database-view", () => ({
  DoctorDatabaseView: () => null,
}));

vi.mock("./doctor-multimodal-view", () => ({
  DoctorMultimodalView: (props: Record<string, unknown>) => {
    mockDoctorMultimodalView(props);
    return <div data-testid="doctor-multimodal-view" />;
  },
}));

vi.mock("./doctor-report-draft-view", () => ({
  DoctorReportDraftView: (props: Record<string, unknown>) => {
    mockDoctorReportDraftView(props);
    return <div data-testid="doctor-report-draft-view" />;
  },
}));

vi.mock("./doctor-review-cockpit", () => ({
  DoctorReviewCockpit: (props: Record<string, unknown>) => {
    mockDoctorReviewCockpit(props);
    return <div data-testid="doctor-review-cockpit" />;
  },
}));

import { DoctorSceneShell } from "./doctor-scene-shell";

describe("DoctorSceneShell", () => {
  beforeEach(() => {
    mockConversationPanel.mockClear();
    mockDoctorMultimodalView.mockClear();
    mockDoctorReportDraftView.mockClear();
    mockDoctorReviewCockpit.mockClear();
    mockDoctorViewState.state.activeDoctorTab = "consultation";
    mockDoctorViewState.state.activeDatabaseSource = "patient_registry";
    mockDoctorViewState.setActiveDoctorTab.mockClear();
    mockDoctorViewState.setActiveDatabaseSource.mockClear();
  });

  function getDoctorProfileSwitch() {
    const profileText = screen.getByText("医生");
    return profileText.closest("button");
  }

  function renderDoctorSceneShell(
    overrides: Partial<Parameters<typeof DoctorSceneShell>[0]> = {},
    viewStateOverrides: Partial<typeof mockDoctorViewState.state> = {},
  ) {
    Object.assign(mockDoctorViewState.state, viewStateOverrides);

    return render(
      <DoctorSceneShell
        toolbar={null}
        registryPatientId={null}
        caseDatabasePatientId={null}
        patientRegistry={
          {
            boundPatientDetail: null,
            boundPatientAlerts: [],
            boundPatientRecords: [],
            isLoadingBoundPatient: false,
            isBindingPatient: false,
          } as never
        }
        databaseWorkbench={{} as never}
        registryBrowser={{} as never}
        messages={[]}
        draft=""
        statusNode={null}
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        errorMessage={null}
        latencyStatus={null}
        roadmap={[]}
        stage={null}
        plan={[]}
        cards={{}}
        references={[]}
        eventLog={[]}
        critic={null}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={vi.fn()}
        onSetCurrentPatient={vi.fn(async () => true)}
        onSetCurrentCaseDatabasePatient={vi.fn()}
        {...overrides}
      />,
    );
  }

  it("hides the review nav item when the review cockpit flag is disabled", () => {
    renderDoctorSceneShell();

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    expect(navButtons.slice(0, 3).map((navButton) => navButton.textContent)).toEqual([
      "会诊",
      "患者数据库",
      "多模态",
    ]);
    expect(navButtons).toHaveLength(4);
    expect(navButtons[3]?.textContent).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
    for (const navButton of navButtons) {
      expect(navButton).not.toBeDisabled();
    }
  });

  it("shows the review nav item when the review cockpit flag is enabled", () => {
    renderDoctorSceneShell({ doctorReviewCockpitEnabled: true });

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    expect(navButtons).toHaveLength(5);
    expect(navButtons[4]).toHaveTextContent("Review");
    expect(navButtons[4]).not.toBeDisabled();
  });

  it("calls setActiveDoctorTab with multimodal when the multimodal nav item is clicked", () => {
    renderDoctorSceneShell();

    screen.getByRole("button", { name: "多模态" }).click();

    expect(mockDoctorViewState.setActiveDoctorTab).toHaveBeenCalledWith("multimodal");
  });

  it("calls setActiveDoctorTab with reports when the reports nav item is clicked", () => {
    renderDoctorSceneShell();

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    fireEvent.click(navButtons[3]);

    expect(mockDoctorViewState.setActiveDoctorTab).toHaveBeenCalledWith("reports");
  });

  it("calls setActiveDoctorTab with review when the review nav item is clicked", () => {
    renderDoctorSceneShell({ doctorReviewCockpitEnabled: true });

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    fireEvent.click(navButtons[4]);

    expect(mockDoctorViewState.setActiveDoctorTab).toHaveBeenCalledWith("review");
  });

  it("marks each doctor route with stable shell classes for theme effects", () => {
    renderDoctorSceneShell();
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-consultation",
    );

    cleanup();
    renderDoctorSceneShell({}, { activeDoctorTab: "database" });
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-database",
    );

    cleanup();
    renderDoctorSceneShell({}, { activeDoctorTab: "multimodal" });
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-multimodal",
    );

    cleanup();
    renderDoctorSceneShell({}, { activeDoctorTab: "reports" });
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-reports",
    );

    cleanup();
    renderDoctorSceneShell(
      { doctorReviewCockpitEnabled: true },
      { activeDoctorTab: "review" },
    );
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-review",
    );
  });

  it("falls back to the consultation shell when review is active but the flag is disabled", () => {
    renderDoctorSceneShell({}, { activeDoctorTab: "review" });

    expect(screen.queryByTestId("doctor-review-cockpit")).not.toBeInTheDocument();
    expect(screen.getByTestId("doctor-scene")).toHaveClass(
      "clinical-app-shell",
      "clinical-app-shell-doctor",
      "clinical-app-shell-consultation",
    );
    expect(screen.getByRole("button", { name: "会诊" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
  });

  it("renders the clinical assistant dashboard chrome for consultation mode", () => {
    render(
      <DoctorSceneShell
        toolbar={<button type="button">重置当前场景</button>}
        registryPatientId={1024}
        caseDatabasePatientId="093"
        patientRegistry={
          {
            boundPatientDetail: null,
            boundPatientAlerts: [],
            boundPatientRecords: [],
            isLoadingBoundPatient: false,
            isBindingPatient: false,
          } as never
        }
        databaseWorkbench={{} as never}
        registryBrowser={{} as never}
        messages={[]}
        draft=""
        statusNode="assessment"
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        errorMessage={null}
        latencyStatus={null}
        roadmap={[]}
        stage="Assessment"
        plan={[]}
        cards={{}}
        references={[]}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={vi.fn()}
        onSetCurrentPatient={vi.fn(async () => true)}
        onSetCurrentCaseDatabasePatient={vi.fn()}
      />,
    );

    expect(screen.getByText("临床助手")).toBeInTheDocument();
    expect(screen.getByTestId("doctor-scene")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "会诊" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("患者摘要")).toBeInTheDocument();
    expect(screen.getByText("医疗卡片")).toBeInTheDocument();
    expect(screen.getByText("工作流路线图")).toBeInTheDocument();
    expect(screen.getByText("执行计划")).toBeInTheDocument();
    expect(screen.getByText("参考列表（前 2 条）")).toBeInTheDocument();
    expect(screen.getByText("登记号:")).toBeInTheDocument();
    expect(screen.getByText("P-1024")).toBeInTheDocument();
    expect(screen.getByText("病例样本:")).toBeInTheDocument();
    expect(screen.getByText("093")).toBeInTheDocument();
    const profileSwitch = getDoctorProfileSwitch();
    expect(profileSwitch).toHaveClass("clinical-profile-switch");
    expect(profileSwitch).toHaveTextContent("医生");
  });

  it("renders the anatomy panel in the consultation sidebar and submits region prompts", () => {
    const onCardPromptRequest = vi.fn();

    renderDoctorSceneShell({
      registryPatientId: 7,
      caseDatabasePatientId: "093",
      patientRegistry: {
        boundPatientDetail: {
          patient_id: 7,
          status: "active",
          created_at: "2026-06-02T00:00:00Z",
          updated_at: "2026-06-02T00:00:00Z",
          tumor_location: "乙状结肠",
        },
        boundPatientAlerts: [],
        boundPatientRecords: [],
        isLoadingBoundPatient: false,
        isBindingPatient: false,
      } as never,
      onCardPromptRequest,
    });

    expect(screen.getByText("解剖定位")).toBeInTheDocument();
    const sigmoidRegion = screen.getByRole("button", { name: "乙状结肠" });
    expect(sigmoidRegion).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(sigmoidRegion);

    expect(onCardPromptRequest).toHaveBeenCalledWith(
      "请针对乙状结肠病灶给出分期与下一步检查建议。",
      expect.objectContaining({
        registry_patient_id: 7,
        case_database_patient_id: "093",
        anatomy_region_code: "sigmoid_colon",
      }),
    );
  });

  it("renders the multimodal shell route with the derived patient context", () => {
    renderDoctorSceneShell(
      {
        patientContext: {},
      },
      {
        activeDoctorTab: "multimodal",
      },
    );

    expect(screen.getByTestId("doctor-multimodal-view")).toBeInTheDocument();
    expect(mockDoctorMultimodalView).toHaveBeenCalledWith(
      expect.objectContaining({
        registryPatientId: null,
        caseDatabasePatientId: null,
        patientContext: undefined,
      }),
    );
  });

  it("passes shell-derived registry and case ids to the multimodal view when patientContext is empty", () => {
    renderDoctorSceneShell(
      {
        registryPatientId: 1024,
        caseDatabasePatientId: "093",
        patientContext: {},
      },
      {
        activeDoctorTab: "multimodal",
      },
    );

    expect(screen.getByTestId("doctor-multimodal-view")).toBeInTheDocument();
    expect(mockDoctorMultimodalView).toHaveBeenCalledWith(
      expect.objectContaining({
        registryPatientId: 1024,
        caseDatabasePatientId: "093",
        patientContext: {
          registry_patient_id: 1024,
          case_database_patient_id: "093",
        },
      }),
    );
  });

  it("renders the report draft route with shell-derived patient context", () => {
    const onCardPromptRequest = vi.fn();
    renderDoctorSceneShell(
      {
        registryPatientId: 1024,
        caseDatabasePatientId: "093",
        patientContext: {},
        onCardPromptRequest,
      },
      {
        activeDoctorTab: "reports",
      },
    );

    expect(screen.getByTestId("doctor-report-draft-view")).toBeInTheDocument();
    expect(mockDoctorReportDraftView).toHaveBeenCalledWith(
      expect.objectContaining({
        registryPatientId: 1024,
        caseDatabasePatientId: "093",
        patientContext: {
          registry_patient_id: 1024,
          case_database_patient_id: "093",
        },
        onReportPromptRequest: onCardPromptRequest,
      }),
    );
  });

  it("renders the review cockpit route with session and feature flag props", () => {
    renderDoctorSceneShell(
      {
        sessionId: "sess-review",
        doctorReviewCockpitEnabled: true,
      },
      {
        activeDoctorTab: "review",
      },
    );

    expect(screen.getByTestId("doctor-review-cockpit")).toBeInTheDocument();
    expect(mockDoctorReviewCockpit).toHaveBeenCalledWith({
      sessionId: "sess-review",
      enabled: true,
    });
  });

  it("renders a true initial state when no doctor session data is present", () => {
    render(
      <DoctorSceneShell
        toolbar={null}
        registryPatientId={null}
        caseDatabasePatientId={null}
        patientRegistry={
          {
            boundPatientDetail: null,
            boundPatientAlerts: [],
            boundPatientRecords: [],
            isLoadingBoundPatient: false,
            isBindingPatient: false,
          } as never
        }
        databaseWorkbench={{} as never}
        registryBrowser={{} as never}
        messages={[]}
        draft=""
        statusNode={null}
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        errorMessage={null}
        latencyStatus={null}
        roadmap={[]}
        stage={null}
        plan={[]}
        cards={{}}
        references={[]}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={vi.fn()}
        onSetCurrentPatient={vi.fn(async () => true)}
        onSetCurrentCaseDatabasePatient={vi.fn()}
      />,
    );

    expect(mockConversationPanel).toHaveBeenCalledWith(
      expect.objectContaining({ messages: [] }),
    );
    expect(screen.getByText("暂无患者摘要。")).toBeInTheDocument();
    expect(screen.getByText("暂无上传资料。")).toBeInTheDocument();
    expect(screen.getByText("暂无事件。")).toBeInTheDocument();
    expect(screen.queryByText("P-1024")).not.toBeInTheDocument();
    expect(screen.queryByText("CT 报告")).not.toBeInTheDocument();
  });

  it("derives the patient summary from a returned patient card", () => {
    render(
      <DoctorSceneShell
        toolbar={null}
        registryPatientId={null}
        caseDatabasePatientId={null}
        patientRegistry={
          {
            boundPatientDetail: null,
            boundPatientAlerts: [],
            boundPatientRecords: [],
            isLoadingBoundPatient: false,
            isBindingPatient: false,
          } as never
        }
        databaseWorkbench={{} as never}
        registryBrowser={{} as never}
        messages={[]}
        draft=""
        statusNode={null}
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        errorMessage={null}
        latencyStatus={null}
        roadmap={[]}
        stage={null}
        plan={[]}
        cards={{
          patient_card: {
            type: "patient_card",
            patient_id: "093",
            data: {
              patient_info: {
                age: 31,
                gender: "male",
              },
              diagnosis_block: {
                primary_site: "colon",
                mmr_status: "pMMR",
              },
              staging_block: {
                clinical_stage: "cT3N1M0",
              },
            },
          },
        }}
        references={[]}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={vi.fn()}
        onSetCurrentPatient={vi.fn(async () => true)}
        onSetCurrentCaseDatabasePatient={vi.fn()}
      />,
    );

    expect(screen.getByText("093")).toBeInTheDocument();
    expect(screen.getByText("病例样本:")).toBeInTheDocument();
    expect(screen.getByText("31")).toBeInTheDocument();
    expect(screen.getByText("male")).toBeInTheDocument();
    expect(screen.getByText("CRC")).toBeInTheDocument();
    expect(screen.getByText("cT3N1M0")).toBeInTheDocument();
    expect(screen.getByText("pMMR")).toBeInTheDocument();
  });

  it("passes latencyStatus into the consultation conversation panel", () => {
    render(
      <DoctorSceneShell
        toolbar={null}
        registryPatientId={null}
        caseDatabasePatientId={null}
        patientRegistry={
          {
            boundPatientDetail: null,
            boundPatientAlerts: [],
            boundPatientRecords: [],
            isLoadingBoundPatient: false,
            isBindingPatient: false,
          } as never
        }
        databaseWorkbench={{} as never}
        registryBrowser={{} as never}
        messages={[]}
        draft=""
        statusNode={null}
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        errorMessage={null}
        latencyStatus={{ kind: "streaming" }}
        roadmap={[]}
        stage={null}
        plan={[]}
        cards={{}}
        references={[]}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={vi.fn()}
        onSetCurrentPatient={vi.fn(async () => true)}
        onSetCurrentCaseDatabasePatient={vi.fn()}
      />,
    );

    expect(mockConversationPanel).toHaveBeenCalled();
    expect(mockConversationPanel).toHaveBeenCalledWith(
      expect.objectContaining({
        latencyStatus: { kind: "streaming" },
        showThinking: true,
      }),
    );
  });

  it("renders clinical stream events with localized labels", () => {
    const safetyReview = "\u5b89\u5168\u590d\u6838";
    const safetyReviewRejected = "\u5b89\u5168\u590d\u6838\u672a\u901a\u8fc7";
    const rejected = "\u672a\u901a\u8fc7";
    const roadmapKind = "\u8def\u7ebf\u56fe\u66f4\u65b0";
    const roadmapUpdated = "\u8def\u7ebf\u56fe\u5df2\u66f4\u65b0";
    const stepCount = "2 \u4e2a\u6b65\u9aa4";

    renderDoctorSceneShell({
      eventLog: [
        {
          id: "event-1",
          kind: "critic",
          title: "Critic REJECTED",
          detail: "missing references",
          tone: "warning",
          requiresHumanReview: true,
        },
        {
          id: "event-2",
          kind: "roadmap",
          title: "Roadmap updated",
          detail: "2 step(s)",
          tone: "neutral",
          requiresHumanReview: false,
        },
      ],
      critic: {
        verdict: "REJECTED",
        feedback: "missing references",
        requires_human_review: true,
      },
    } as any);

    expect(screen.getByText(safetyReviewRejected)).toBeInTheDocument();
    expect(screen.getAllByText(safetyReview).length).toBeGreaterThan(0);
    expect(screen.getByText(rejected)).toBeInTheDocument();
    expect(screen.getByText(roadmapUpdated)).toBeInTheDocument();
    expect(screen.getByText(roadmapKind)).toBeInTheDocument();
    expect(screen.getByText(stepCount)).toBeInTheDocument();
    expect(screen.getAllByText("missing references").length).toBeGreaterThan(0);
    expect(screen.queryByText(/Critic REJECTED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^critic$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^REJECTED$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Roadmap updated/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^roadmap$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/step\(s\)/)).not.toBeInTheDocument();
    expect(screen.getAllByText("需人工复核").length).toBeGreaterThan(0);
  });

  it("keeps raw critic reasoning out of the event stream", () => {
    const rawFeedback = [
      "<think>The critic considered the treatment plan.</think>",
      '{"verdict":"APPROVED","feedback":"需要补充 MMR/MSI 检测。"}',
    ].join("\n");

    renderDoctorSceneShell({
      eventLog: [
        {
          id: "event-1",
          kind: "critic",
          title: "Critic APPROVED",
          detail: rawFeedback,
          tone: "success",
          requiresHumanReview: false,
        },
      ],
      critic: {
        verdict: "APPROVED",
        feedback: rawFeedback,
        requires_human_review: true,
      },
    } as any);

    expect(screen.getAllByText("需要补充 MMR/MSI 检测。").length).toBeGreaterThan(0);
    expect(screen.queryByText(/The critic considered/)).not.toBeInTheDocument();
    expect(screen.queryByText(/<think>/)).not.toBeInTheDocument();
  });
});
