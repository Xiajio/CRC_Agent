import type { ReactNode } from "react";

import { type FrontendMessage, type JsonObject } from "../../app/api/types";
import { WorkspaceLayout } from "../../components/layout/workspace-layout";
import { ClinicalCardsPanel } from "../cards/clinical-cards-panel";
import type { CardPromptHandler } from "../cards/card-renderers-extended";
import { useDatabaseWorkbench } from "../database/use-database-workbench";
import { ExecutionPlanPanel } from "../execution-plan/execution-plan-panel";
import { usePatientRegistry } from "../patient-registry/use-patient-registry";
import { PatientRegistryAlertsPanel } from "../patient-registry/patient-registry-alerts";
import { PatientRecordsPanel } from "../patient-registry/patient-records-panel";
import { useRegistryBrowser } from "../patient-registry/use-registry-browser";
import { RoadmapPanel } from "../roadmap/roadmap-panel";
import {
  CurrentPatientSummary,
  DoctorConsultationView,
  type DoctorLatencyStatus,
  GeneralClinicalMode,
} from "./doctor-consultation-view";
import { DoctorDatabaseView } from "./doctor-database-view";
import { useDoctorViewState } from "./use-doctor-view-state";

type DoctorSceneShellProps = {
  toolbar: ReactNode;
  currentPatientId: number | null;
  patientRegistry: ReturnType<typeof usePatientRegistry>;
  databaseWorkbench: ReturnType<typeof useDatabaseWorkbench>;
  registryBrowser: ReturnType<typeof useRegistryBrowser>;
  messages: FrontendMessage[];
  draft: string;
  statusNode: string | null;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  canLoadHistory: boolean;
  disabled: boolean;
  errorMessage: string | null;
  latencyStatus?: DoctorLatencyStatus | null;
  roadmap: JsonObject[];
  stage: string | null;
  plan: JsonObject[];
  cards: Record<string, JsonObject>;
  references: JsonObject[];
  onLoadHistory: () => void;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onSetCurrentPatient: (patientId: number) => Promise<boolean>;
  onCardPromptRequest?: CardPromptHandler;
};

export function DoctorSceneShell({
  toolbar,
  currentPatientId,
  patientRegistry,
  databaseWorkbench,
  registryBrowser,
  messages,
  draft,
  statusNode,
  isStreaming,
  isLoadingHistory,
  canLoadHistory,
  disabled,
  errorMessage,
  latencyStatus,
  roadmap,
  stage,
  plan,
  cards,
  references,
  onLoadHistory,
  onDraftChange,
  onSubmit,
  onSetCurrentPatient,
  onCardPromptRequest,
}: DoctorSceneShellProps) {
  const {
    activeDoctorTab,
    setActiveDoctorTab,
    activeDatabaseSource,
    setActiveDatabaseSource,
  } = useDoctorViewState();

  async function handleSetCurrentPatient(patientId: number) {
    const didBind = await onSetCurrentPatient(patientId);
    if (didBind) {
      setActiveDoctorTab("consultation");
    }
  }

  function openDatabasePatientRegistry() {
    setActiveDoctorTab("database");
    setActiveDatabaseSource("patient_registry");
  }

  const combinedToolbar = (
    <>
      {toolbar}
      <div style={{ width: "1px", height: "24px", background: "rgba(165, 73, 83, 0.2)", margin: "0 4px" }} />
      <button
        type="button"
        className={activeDoctorTab === "consultation" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => setActiveDoctorTab("consultation")}
        aria-label="consultation workspace"
      >
        🧑‍⚕️ 问诊工作区
      </button>
      <button
        type="button"
        className={activeDoctorTab === "database" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => setActiveDoctorTab("database")}
        aria-label="database"
      >
        🗄️ 数据库
      </button>
    </>
  );

  const consultationRightInspector = (
    <div className="workspace-panel-stack">
      <ClinicalCardsPanel cards={cards} selectedCardType={null} onPromptRequest={onCardPromptRequest} />
      <RoadmapPanel roadmap={roadmap} stage={stage} />
      <ExecutionPlanPanel plan={plan} references={references} />
    </div>
  );

  const consultationToolbar = (
    <div
      className="workspace-toolbar-row"
      style={{ display: "flex", gap: "12px", flexWrap: "wrap", alignItems: "center" }}
    >
      {combinedToolbar}
    </div>
  );

  const selectedView = activeDoctorTab === "consultation"
    ? (
      currentPatientId === null ? (
        <WorkspaceLayout
          toolbar={consultationToolbar}
          leftRail={<div />}
          leftRailOpen={false}
          rightInspector={consultationRightInspector}
          rightInspectorOpen
          centerWorkspace={(
            <div className="workspace-panel-stack">
              <GeneralClinicalMode onOpenDatabasePatientRegistry={openDatabasePatientRegistry} />
              <DoctorConsultationView
                messages={messages}
                draft={draft}
                statusNode={statusNode}
                isStreaming={isStreaming}
                isLoadingHistory={isLoadingHistory}
                canLoadHistory={canLoadHistory}
                disabled={disabled}
                errorMessage={errorMessage}
                latencyStatus={latencyStatus}
                onLoadHistory={onLoadHistory}
                onDraftChange={onDraftChange}
                onSubmit={onSubmit}
                onCardPromptRequest={onCardPromptRequest}
              />
            </div>
          )}
        />
      ) : (
        <WorkspaceLayout
          toolbar={consultationToolbar}
          leftRail={(
            <div className="workspace-panel-stack">
              <CurrentPatientSummary
                currentPatientId={currentPatientId}
                detail={patientRegistry.boundPatientDetail}
                isLoading={patientRegistry.isLoadingBoundPatient}
              />
              <PatientRegistryAlertsPanel
                alerts={patientRegistry.boundPatientAlerts}
                isLoading={patientRegistry.isLoadingBoundPatient}
              />
              <PatientRecordsPanel
                records={patientRegistry.boundPatientRecords}
                isLoading={patientRegistry.isLoadingBoundPatient}
              />
            </div>
          )}
          centerWorkspace={(
            <div className="workspace-panel-stack">
              <DoctorConsultationView
                messages={messages}
                draft={draft}
                statusNode={statusNode}
                isStreaming={isStreaming}
                isLoadingHistory={isLoadingHistory}
                canLoadHistory={canLoadHistory}
                disabled={disabled}
                errorMessage={errorMessage}
                latencyStatus={latencyStatus}
                onLoadHistory={onLoadHistory}
                onDraftChange={onDraftChange}
                onSubmit={onSubmit}
                onCardPromptRequest={onCardPromptRequest}
              />
            </div>
          )}
          rightInspector={consultationRightInspector}
        />
      )
    )
    : (
      <DoctorDatabaseView
        parentToolbar={combinedToolbar}
        activeSource={activeDatabaseSource}
        onSourceChange={setActiveDatabaseSource}
        currentPatientId={currentPatientId}
        databaseWorkbench={databaseWorkbench}
        registryBrowser={registryBrowser}
        isBindingCurrentPatient={patientRegistry.isBindingPatient}
        onSetCurrentPatient={handleSetCurrentPatient}
      />
    );

  return selectedView;
}
