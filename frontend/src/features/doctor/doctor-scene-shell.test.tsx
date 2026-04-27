import type { ReactNode } from "react";
import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockConversationPanel = vi.hoisted(() => vi.fn());

vi.mock("../chat/conversation-panel", () => ({
  ConversationPanel: (props: Record<string, unknown>) => {
    mockConversationPanel(props);
    return null;
  },
}));

vi.mock("./use-doctor-view-state", () => ({
  useDoctorViewState: () => ({
    activeDoctorTab: "consultation",
    setActiveDoctorTab: vi.fn(),
    activeDatabaseSource: "patient_registry",
    setActiveDatabaseSource: vi.fn(),
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

import { DoctorSceneShell } from "./doctor-scene-shell";

describe("DoctorSceneShell", () => {
  beforeEach(() => {
    mockConversationPanel.mockClear();
  });

  function getDoctorProfileSwitch() {
    const profileText = screen.getByText("医生");
    return profileText.closest("button");
  }

  function renderDoctorSceneShell(overrides: Partial<Parameters<typeof DoctorSceneShell>[0]> = {}) {
    return render(
      <DoctorSceneShell
        toolbar={null}
        currentPatientId={null}
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
        {...overrides}
      />,
    );
  }

  it("omits placeholder-only doctor top nav items in production", () => {
    renderDoctorSceneShell();

    const navButtons = within(screen.getByRole("navigation")).getAllByRole("button");
    expect(navButtons.map((navButton) => navButton.textContent)).toEqual(["会诊", "患者数据库"]);
    expect(navButtons).toHaveLength(2);
    for (const navButton of navButtons) {
      expect(navButton).not.toBeDisabled();
    }
    expect(screen.queryByRole("button", { name: "多模态" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "报表" })).not.toBeInTheDocument();
  });

  it("renders the clinical assistant dashboard chrome for consultation mode", () => {
    render(
      <DoctorSceneShell
        toolbar={<button type="button">重置当前场景</button>}
        currentPatientId={1024}
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
      />,
    );

    expect(screen.getByText("临床助手")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "会诊" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("患者摘要")).toBeInTheDocument();
    expect(screen.getByText("医疗卡片")).toBeInTheDocument();
    expect(screen.getByText("工作流路线图")).toBeInTheDocument();
    expect(screen.getByText("执行计划")).toBeInTheDocument();
    expect(screen.getByText("参考列表（前 2 条）")).toBeInTheDocument();
    const profileSwitch = getDoctorProfileSwitch();
    expect(profileSwitch).toHaveClass("clinical-profile-switch");
    expect(profileSwitch).toHaveTextContent("医生");
  });

  it("renders a true initial state when no doctor session data is present", () => {
    render(
      <DoctorSceneShell
        toolbar={null}
        currentPatientId={null}
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
        currentPatientId={null}
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
      />,
    );

    expect(screen.getByText("P-93")).toBeInTheDocument();
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
        currentPatientId={null}
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
      />,
    );

    expect(mockConversationPanel).toHaveBeenCalled();
    expect(mockConversationPanel).toHaveBeenCalledWith(
      expect.objectContaining({
        latencyStatus: { kind: "streaming" },
      }),
    );
  });
});
