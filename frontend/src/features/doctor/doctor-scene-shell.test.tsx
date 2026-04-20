import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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
  ClinicalCardsPanel: () => null,
}));

vi.mock("../roadmap/roadmap-panel", () => ({
  RoadmapPanel: () => null,
}));

vi.mock("../execution-plan/execution-plan-panel", () => ({
  ExecutionPlanPanel: () => null,
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
