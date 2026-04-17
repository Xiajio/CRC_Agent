import { WorkspaceLayout } from "../components/layout/workspace-layout";
import { DatabaseDetailPanel } from "../features/database/database-detail-panel";
import { DatabaseEditForm } from "../features/database/database-edit-form";
import { DatabaseFiltersPanel } from "../features/database/database-filters-panel";
import { DatabaseWorkbenchPanel } from "../features/database/database-workbench-panel";
import { useDatabaseWorkbench } from "../features/database/use-database-workbench";

function triStateLabel(value: boolean | null | undefined): string {
  if (value === true) {
    return "是";
  }
  if (value === false) {
    return "否";
  }
  return "-";
}

export function DatabasePage() {
  const {
    stats,
    searchRequest,
    searchResponse,
    selectedPatientId,
    detail,
    editRecord,
    naturalQuery,
    intentWarnings,
    unsupportedTerms,
    pageError,
    isBootstrapping,
    isSearching,
    isLoadingDetail,
    isParsing,
    isSaving,
    setNaturalQuery,
    setFilters,
    runSearch,
    loadCaseDetail,
    handleNaturalQuerySubmit,
    saveRecord,
    resetWorkbench,
    setEditField,
    handleSortChange,
    handlePageChange,
  } = useDatabaseWorkbench();

  const toolbar = (
    <header className="workspace-global-header">
      <div className="workspace-header-left">
        <span className="workspace-brand">{"亿铸科技 -- 虚拟数据库控制台"}</span>
      </div>
      <div className="workspace-header-center">
        <span className="workspace-stage-badge">{"Agentic UI"}</span>
      </div>
      <div className="workspace-header-right">
        <a className="workspace-header-link" href="/">
          {"对话工作台"}
        </a>
      </div>
    </header>
  );

  const leftRail = (
    <section className="workspace-panel-stack">
      <DatabaseFiltersPanel
        filters={searchRequest.filters}
        isSearching={isSearching}
        onFiltersChange={setFilters}
        onApply={() =>
          void runSearch({
            ...searchRequest,
            pagination: {
              ...searchRequest.pagination,
              page: 1,
            },
          })
        }
        onReset={() => {
          resetWorkbench();
        }}
      />
      <div className="workspace-card">
        <h2>{"当前筛选"}</h2>
        <dl className="workspace-definition-list workspace-definition-list-compact">
          <div>
            <dt>Patient ID</dt>
            <dd>{searchRequest.filters.patient_id ?? "-"}</dd>
          </div>
          <div>
            <dt>{"年龄"}</dt>
            <dd>{`${searchRequest.filters.age_min ?? "-"} ~ ${searchRequest.filters.age_max ?? "-"}`}</dd>
          </div>
          <div>
            <dt>{"部位"}</dt>
            <dd>{searchRequest.filters.tumor_location.join(", ") || "-"}</dd>
          </div>
          <div>
            <dt>MMR</dt>
            <dd>{searchRequest.filters.mmr_status.join(", ") || "-"}</dd>
          </div>
          <div>
            <dt>{"家族史"}</dt>
            <dd>{triStateLabel(searchRequest.filters.family_history)}</dd>
          </div>
          <div>
            <dt>{"活检确认"}</dt>
            <dd>{triStateLabel(searchRequest.filters.biopsy_confirmed)}</dd>
          </div>
          <div>
            <dt>ECOG</dt>
            <dd>{`${searchRequest.filters.ecog_min ?? "-"} ~ ${searchRequest.filters.ecog_max ?? "-"}`}</dd>
          </div>
        </dl>
      </div>
    </section>
  );

  const centerWorkspace = (
    <DatabaseWorkbenchPanel
      mode={selectedPatientId !== null ? "detail" : "search"}
      naturalQuery={naturalQuery}
      stats={stats}
      searchRequest={searchRequest}
      searchResponse={searchResponse}
      selectedPatientId={selectedPatientId}
      isParsing={isParsing}
      isSearching={isSearching}
      isBootstrapping={isBootstrapping}
      warnings={intentWarnings}
      unsupportedTerms={unsupportedTerms}
      error={pageError}
      onNaturalQueryChange={setNaturalQuery}
      onNaturalQuerySubmit={() => void handleNaturalQuerySubmit()}
      onSelectPatient={(patientId) => void loadCaseDetail(patientId)}
      onSortChange={handleSortChange}
      onPageChange={handlePageChange}
    />
  );

  const rightInspector = (
    <section className="workspace-panel-stack">
      {isLoadingDetail ? <div className="workspace-banner">{"正在加载患者详情..."}</div> : null}
      <DatabaseDetailPanel
        detail={detail}
        onPromptRequest={(prompt) => {
          setNaturalQuery(prompt);
        }}
      />
      <DatabaseEditForm
        record={editRecord}
        isSaving={isSaving}
        onFieldChange={setEditField}
        onSave={() => void saveRecord()}
      />
    </section>
  );

  return (
    <WorkspaceLayout
      toolbar={toolbar}
      leftRail={leftRail}
      centerWorkspace={centerWorkspace}
      rightInspector={rightInspector}
      leftRailOpen
      rightInspectorOpen
    />
  );
}
