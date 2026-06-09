import type {
  PatientRegistryAlert,
  PatientRegistryDetail,
  PatientRegistryItem,
  PatientRegistryRecord,
} from "../../app/api/types";
import { Button, Card, Input } from "../../components/ui";
import { PatientRegistryAlertsPanel } from "./patient-registry-alerts";
import { PatientRecordsPanel } from "./patient-records-panel";

type RegistryBrowserPaneProps = {
  searchState: {
    patientId: string;
    tumorLocation: string;
    mmrStatus: string;
    clinicalStage: string;
    limit: number;
  };
  searchResults: PatientRegistryItem[];
  previewPatientId: number | null;
  previewDetail: PatientRegistryDetail | null;
  previewRecords: PatientRegistryRecord[];
  previewAlerts: PatientRegistryAlert[];
  registryPatientId: number | null;
  isSearching: boolean;
  isLoadingPreview: boolean;
  isBindingCurrentPatient: boolean;
  isDeletingPatient: boolean;
  isClearingRegistry: boolean;
  error: string | null;
  onSearchFieldChange: (
    field: "patientId" | "tumorLocation" | "mmrStatus" | "clinicalStage" | "limit",
    value: string | number,
  ) => void;
  onSearchSubmit: () => void;
  onPreviewPatient: (patientId: number) => void;
  onSetCurrentPatient: (patientId: number) => void;
  onDeletePatient: (patientId: number) => void;
  onClearRegistry: () => void;
};

function patientSummary(item: PatientRegistryItem): string {
  const segments = [item.tumor_location, item.clinical_stage, item.mmr_status]
    .filter((value) => typeof value === "string" && value.trim().length > 0);
  return segments.length > 0 ? segments.join(" / ") : "暂无结构化摘要";
}

function previewSummary(detail: PatientRegistryDetail | null): string[] {
  if (!detail) {
    return [];
  }

  return [
    detail.chief_complaint ? `主诉: ${detail.chief_complaint}` : null,
    detail.age !== null && detail.age !== undefined ? `年龄: ${detail.age}` : null,
    detail.gender ? `性别: ${detail.gender}` : null,
    detail.tumor_location ? `肿瘤部位: ${detail.tumor_location}` : null,
    detail.clinical_stage ? `临床分期: ${detail.clinical_stage}` : null,
    detail.mmr_status ? `MMR状态: ${detail.mmr_status}` : null,
  ].filter((value): value is string => value !== null);
}

export function RegistryBrowserPane({
  searchState,
  searchResults,
  previewPatientId,
  previewDetail,
  previewRecords,
  previewAlerts,
  registryPatientId,
  isSearching,
  isLoadingPreview,
  isBindingCurrentPatient,
  isDeletingPatient,
  isClearingRegistry,
  error,
  onSearchFieldChange,
  onSearchSubmit,
  onPreviewPatient,
  onSetCurrentPatient,
  onDeletePatient,
  onClearRegistry,
}: RegistryBrowserPaneProps) {
  const summaryItems = previewSummary(previewDetail);
  const canBind = previewPatientId !== null && previewPatientId !== registryPatientId && !isLoadingPreview;
  const canDeletePreview = previewPatientId !== null && previewPatientId !== registryPatientId && !isLoadingPreview;
  const canClearRegistry = registryPatientId === null;

  return (
    <div className="clinical-panel-stack">
      <Card as="section" variant="clinical-panel" data-testid="registry-browser-pane">
        <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "1.2rem" }}>🗂️</span> 患者库检索
        </h2>
        <p className="clinical-copy clinical-copy-tight">
          在此处检索并预览患者库中的患者。删除工具仅用于开发环境清理。
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "12px",
            marginTop: "16px",
          }}
        >
          <label className="database-field">
            <span className="database-field-label">患者 ID</span>
            <Input
              aria-label="registry patient id"
              type="text"
              value={searchState.patientId}
              onChange={(event) => onSearchFieldChange("patientId", event.target.value)}
            />
          </label>
          <label className="database-field">
            <span className="database-field-label">肿瘤部位</span>
            <Input
              aria-label="registry tumor location"
              type="text"
              value={searchState.tumorLocation}
              onChange={(event) => onSearchFieldChange("tumorLocation", event.target.value)}
            />
          </label>
          <label className="database-field">
            <span className="database-field-label">MMR状态</span>
            <Input
              aria-label="registry mmr status"
              type="text"
              value={searchState.mmrStatus}
              onChange={(event) => onSearchFieldChange("mmrStatus", event.target.value)}
            />
          </label>
          <label className="database-field">
            <span className="database-field-label">临床分期</span>
            <Input
              aria-label="registry clinical stage"
              type="text"
              value={searchState.clinicalStage}
              onChange={(event) => onSearchFieldChange("clinicalStage", event.target.value)}
            />
          </label>
        </div>
        <div className="database-action-row">
          <Button
            type="button"
            variant="primary"
            disabled={isSearching}
            onClick={onSearchSubmit}
          >
            {isSearching ? "正在检索..." : "检索患者库"}
          </Button>
          <Button
            type="button"
            variant="secondary"
            aria-label="clear registry"
            disabled={!canClearRegistry || isClearingRegistry}
            onClick={() => {
              if (window.confirm("确定要清空患者库中的所有患者吗？此操作仅用于开发环境清理。")) {
                onClearRegistry();
              }
            }}
          >
            {isClearingRegistry ? "正在清空..." : "清空患者库"}
          </Button>
        </div>
        {!canClearRegistry ? (
          <p className="clinical-copy clinical-copy-tight">
            清空患者库前请先重置当前的医生会话场景。
          </p>
        ) : null}
        {error ? <p className="clinical-copy clinical-copy-alert">{error}</p> : null}
        {searchResults.length > 0 ? (
          <ul className="clinical-list" style={{ gap: "10px", marginTop: "16px" }}>
            {searchResults.map((item) => {
              const isPreviewed = item.patient_id === previewPatientId;
              const isCurrent = item.patient_id === registryPatientId;
              return (
                <li key={item.patient_id} className="clinical-list-item">
                  <strong>{`患者 #${item.patient_id}`}</strong>
                  <p className="clinical-copy clinical-copy-tight">{patientSummary(item)}</p>
                  <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    <Button
                      type="button"
                      variant={isPreviewed ? "primary" : "secondary"}
                      onClick={() => onPreviewPatient(item.patient_id)}
                      disabled={isLoadingPreview}
                      aria-label={isPreviewed ? `previewing ${item.patient_id}` : `preview patient ${item.patient_id}`}
                    >
                      {isPreviewed ? `✅ 正在预览 #${item.patient_id}` : `👀 预览 #${item.patient_id}`}
                    </Button>
                    {isCurrent ? <span className="clinical-stage-badge">当前患者</span> : null}
                  </div>
                </li>
              );
            })}
          </ul>
        ) : null}
        {!isSearching && searchResults.length === 0 ? (
          <p className="clinical-copy" style={{ marginTop: "16px" }}>
            没有找到匹配当前过滤条件的患者。
          </p>
        ) : null}
      </Card>

      <Card as="section" variant="clinical-panel" data-testid="registry-preview-panel">
        <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "1.2rem" }}>🔎</span>
          {previewPatientId !== null ? `患者库预览: 患者 #${previewPatientId}` : "患者库预览"}
        </h2>
        {isLoadingPreview ? <p className="clinical-copy" style={{ color: "var(--color-primary)" }}>正在加载预览...</p> : null}
        {!isLoadingPreview && previewPatientId === null ? (
          <p className="clinical-copy">
            请从最近患者列表或搜索结果中选择一名患者，在绑定前查看其详情。
          </p>
        ) : null}
        {!isLoadingPreview && previewPatientId !== null && summaryItems.length > 0 ? (
          <ul className="clinical-list" style={{ gap: "8px", marginTop: "12px" }}>
            {summaryItems.map((item) => (
              <li key={item} className="clinical-list-item">{item}</li>
            ))}
          </ul>
        ) : null}
        <div className="database-action-row">
          <Button
            type="button"
            variant={canBind ? "primary" : "secondary"}
            disabled={!canBind || isBindingCurrentPatient}
            onClick={() => {
              if (previewPatientId !== null) {
                onSetCurrentPatient(previewPatientId);
              }
            }}
            aria-label={previewPatientId !== null ? `set current patient ${previewPatientId}` : "set current patient"}
          >
            {previewPatientId !== null && previewPatientId === registryPatientId
              ? `✅ 当前患者 #${previewPatientId}`
              : isBindingCurrentPatient
                ? "正在绑定..."
                : "🔗 设为当前患者"}
          </Button>
          <Button
            type="button"
            variant="secondary"
            aria-label={previewPatientId !== null ? `delete patient ${previewPatientId}` : "delete patient"}
            disabled={!canDeletePreview || isDeletingPatient}
            onClick={() => {
              if (
                previewPatientId !== null
                && window.confirm(`确定要从患者库中删除患者 #${previewPatientId} 吗？`)
              ) {
                onDeletePatient(previewPatientId);
              }
            }}
          >
            {isDeletingPatient ? "正在删除..." : "🗑️ 删除患者"}
          </Button>
        </div>
        {!canDeletePreview && previewPatientId === registryPatientId && previewPatientId !== null ? (
          <p className="clinical-copy clinical-copy-tight">
            删除此患者记录前，请先重置或更改当前患者。
          </p>
        ) : null}
      </Card>

      <PatientRegistryAlertsPanel alerts={previewAlerts} isLoading={isLoadingPreview} />
      <PatientRecordsPanel records={previewRecords} isLoading={isLoadingPreview} />
    </div>
  );
}
