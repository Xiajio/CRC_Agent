import { WorkspaceLayout } from "../../components/layout/workspace-layout";
import { DatabaseDetailPanel } from "../database/database-detail-panel";
import { DatabaseEditForm } from "../database/database-edit-form";
import { DatabaseWorkbenchPanel } from "../database/database-workbench-panel";
import { useDatabaseWorkbench } from "../database/use-database-workbench";
import { RecentPatientsPanel } from "../patient-registry/recent-patients-panel";
import { RegistryBrowserPane } from "../patient-registry/registry-browser-pane";
import { useRegistryBrowser } from "../patient-registry/use-registry-browser";
import type { DoctorDatabaseSource } from "./use-doctor-view-state";

type DoctorDatabaseViewProps = {
  parentToolbar?: React.ReactNode;
  activeSource: DoctorDatabaseSource;
  onSourceChange: (value: DoctorDatabaseSource) => void;
  currentPatientId: number | null;
  databaseWorkbench: ReturnType<typeof useDatabaseWorkbench>;
  registryBrowser: ReturnType<typeof useRegistryBrowser>;
  isBindingCurrentPatient: boolean;
  onSetCurrentPatient: (patientId: number) => void | Promise<void>;
};

function DatabaseSourceToolbar({
  activeSource,
  onSourceChange,
}: {
  activeSource: DoctorDatabaseSource;
  onSourceChange: (value: DoctorDatabaseSource) => void;
}) {
  return (
    <>
      <div style={{ width: "1px", height: "24px", background: "rgba(165, 73, 83, 0.2)", margin: "0 4px" }} />
      <button
        type="button"
        className={activeSource === "historical_case_base" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => onSourceChange("historical_case_base")}
        aria-label="historical case base"
      >
        📚 历史病例
      </button>
      <button
        type="button"
        className={activeSource === "patient_registry" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => onSourceChange("patient_registry")}
        aria-label="patient registry"
      >
        🏥 患者库
      </button>
    </>
  );
}

export function DoctorDatabaseView({
  parentToolbar,
  activeSource,
  onSourceChange,
  currentPatientId,
  databaseWorkbench,
  registryBrowser,
  isBindingCurrentPatient,
  onSetCurrentPatient,
}: DoctorDatabaseViewProps) {
  const toolbar = (
    <div className="workspace-toolbar-row" style={{ display: "flex", gap: "12px", flexWrap: "wrap", alignItems: "center" }}>
      {parentToolbar}
      <DatabaseSourceToolbar activeSource={activeSource} onSourceChange={onSourceChange} />
    </div>
  );

  if (activeSource === "historical_case_base") {
    return (
      <WorkspaceLayout
        toolbar={toolbar}
        leftRail={<div />}
        leftRailOpen={false}
        centerWorkspace={(
          <DatabaseWorkbenchPanel
            title="Historical Case Base"
            mode={databaseWorkbench.detail ? "detail" : "search"}
            naturalQuery={databaseWorkbench.naturalQuery}
            stats={databaseWorkbench.stats}
            searchRequest={databaseWorkbench.searchRequest}
            searchResponse={databaseWorkbench.searchResponse}
            selectedPatientId={databaseWorkbench.selectedPatientId}
            isParsing={databaseWorkbench.isParsing}
            isSearching={databaseWorkbench.isSearching}
            isLoadingDetail={databaseWorkbench.isLoadingDetail}
            isBootstrapping={databaseWorkbench.isBootstrapping}
            warnings={databaseWorkbench.intentWarnings}
            unsupportedTerms={databaseWorkbench.unsupportedTerms}
            error={databaseWorkbench.pageError}
            onNaturalQueryChange={databaseWorkbench.setNaturalQuery}
            onNaturalQuerySubmit={() => void databaseWorkbench.handleNaturalQuerySubmit()}
            onSelectPatient={(patientId) => void databaseWorkbench.loadCaseDetail(patientId)}
            onSortChange={databaseWorkbench.handleSortChange}
            onPageChange={databaseWorkbench.handlePageChange}
          />
        )}
        rightInspector={(
          <div className="workspace-panel-stack">
            <DatabaseDetailPanel detail={databaseWorkbench.detail} />
            <DatabaseEditForm
              record={databaseWorkbench.editRecord}
              isSaving={databaseWorkbench.isSaving}
              onFieldChange={databaseWorkbench.setEditField}
              onSave={() => void databaseWorkbench.saveRecord()}
            />
          </div>
        )}
      />
    );
  }

  return (
    <WorkspaceLayout
      toolbar={toolbar}
      leftRail={(
        <RecentPatientsPanel
          items={registryBrowser.recentPatients}
          previewedPatientId={registryBrowser.previewPatientId}
          isLoading={registryBrowser.isLoadingRecent}
          isLoadingPreview={registryBrowser.isLoadingPreview}
          error={registryBrowser.error}
          onPreviewPatient={(patientId) => void registryBrowser.previewPatient(patientId)}
        />
      )}
      centerWorkspace={(
        <RegistryBrowserPane
          searchState={registryBrowser.searchState}
          searchResults={registryBrowser.searchResults}
          previewPatientId={registryBrowser.previewPatientId}
          previewDetail={registryBrowser.previewDetail}
          previewRecords={registryBrowser.previewRecords}
          previewAlerts={registryBrowser.previewAlerts}
          currentPatientId={currentPatientId}
          isSearching={registryBrowser.isSearching}
          isLoadingPreview={registryBrowser.isLoadingPreview}
          isBindingCurrentPatient={isBindingCurrentPatient}
          isDeletingPatient={registryBrowser.isDeletingPatient}
          isClearingRegistry={registryBrowser.isClearingRegistry}
          error={registryBrowser.error}
          onSearchFieldChange={registryBrowser.setSearchField}
          onSearchSubmit={() => void registryBrowser.runSearch()}
          onPreviewPatient={(patientId) => void registryBrowser.previewPatient(patientId)}
          onSetCurrentPatient={(patientId) => void onSetCurrentPatient(patientId)}
          onDeletePatient={(patientId) => void registryBrowser.deletePatient(patientId)}
          onClearRegistry={() => void registryBrowser.clearRegistry()}
        />
      )}
      rightInspector={<div />}
      rightInspectorOpen={false}
    />
  );
}
