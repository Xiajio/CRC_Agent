import type { FrontendMessage, JsonObject } from "../../app/api/types";
import type {
  PatientRegistryAlert,
  PatientRegistryDetail,
  PatientRegistryRecord,
} from "../../app/api/types";
import type { CardPromptHandler } from "../cards/card-renderers-extended";
import { ConversationPanel } from "../chat/conversation-panel";

export type DoctorConsultationViewProps = {
  currentPatientId: number | null;
  patientDetail: PatientRegistryDetail | null;
  patientAlerts: PatientRegistryAlert[];
  patientRecords: PatientRegistryRecord[];
  messages: FrontendMessage[];
  draft: string;
  statusNode: string | null;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  canLoadHistory: boolean;
  disabled: boolean;
  errorMessage: string | null;
  roadmap: JsonObject[];
  stage: string | null;
  plan: JsonObject[];
  cards: Record<string, JsonObject>;
  references: JsonObject[];
  isLoadingPatient: boolean;
  onLoadHistory: () => void;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onOpenDatabasePatientRegistry: () => void;
  onCardPromptRequest?: CardPromptHandler;
};

function formatPatientSummary(detail: PatientRegistryDetail | null): string[] {
  if (!detail) {
    return [];
  }

  return [
    detail.chief_complaint ? `Chief complaint: ${detail.chief_complaint}` : null,
    detail.age !== null && detail.age !== undefined ? `Age: ${detail.age}` : null,
    detail.gender ? `Gender: ${detail.gender}` : null,
    detail.tumor_location ? `Tumor location: ${detail.tumor_location}` : null,
    detail.clinical_stage ? `Clinical stage: ${detail.clinical_stage}` : null,
    detail.mmr_status ? `MMR status: ${detail.mmr_status}` : null,
  ].filter((value): value is string => value !== null);
}

export function GeneralClinicalMode({
  onOpenDatabasePatientRegistry,
}: {
  onOpenDatabasePatientRegistry: () => void;
}) {
  return (
    <section className="workspace-card" aria-label="general clinical mode" data-testid="general-clinical-mode">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}>🧑‍⚕️</span> 通用临床模式
      </h2>
      <p className="workspace-copy">当前医生会话未绑定患者。</p>
      <p className="workspace-copy workspace-copy-tight">
        您可以在对话面板进行通用的临床问诊，或者前往患者库预览并绑定患者。
      </p>
      <button
        type="button"
        className="workspace-primary-button"
        aria-label="open database > patient registry"
        onClick={onOpenDatabasePatientRegistry}
      >
        打开 数据库 &gt; 患者库
      </button>
    </section>
  );
}

export function CurrentPatientSummary({
  currentPatientId,
  detail,
  isLoading,
}: {
  currentPatientId: number;
  detail: PatientRegistryDetail | null;
  isLoading: boolean;
}) {
  const summaryItems = formatPatientSummary(detail);

  return (
    <section className="workspace-card" aria-label="current patient summary">
      <h2>Current Patient Summary</h2>
      <p
        className="workspace-copy workspace-copy-tight"
        style={{ color: "#8e4a55", fontWeight: "600" }}
      >{`Patient #${currentPatientId}`}</p>
      {isLoading ? <p className="workspace-copy" style={{ color: "#8e4a55" }}>Loading patient summary...</p> : null}
      {!isLoading && summaryItems.length > 0 ? (
        <ul className="workspace-list" style={{ gap: "8px", marginTop: "12px" }}>
          {summaryItems.map((item) => (
            <li key={item} className="workspace-list-item">
              {item}
            </li>
          ))}
        </ul>
      ) : null}
      {!isLoading && summaryItems.length === 0 ? (
        <p className="workspace-copy">No patient summary is available yet.</p>
      ) : null}
    </section>
  );
}

export function DoctorConsultationView({
  messages,
  draft,
  statusNode,
  isStreaming,
  isLoadingHistory,
  canLoadHistory,
  disabled,
  errorMessage,
  onLoadHistory,
  onDraftChange,
  onSubmit,
  onCardPromptRequest,
}: Omit<
  DoctorConsultationViewProps,
  | "currentPatientId"
  | "patientDetail"
  | "patientAlerts"
  | "patientRecords"
  | "roadmap"
  | "stage"
  | "plan"
  | "cards"
  | "references"
  | "isLoadingPatient"
  | "onOpenDatabasePatientRegistry"
>) {
  return (
    <ConversationPanel
      messages={messages}
      draft={draft}
      statusNode={statusNode}
      isStreaming={isStreaming}
      isLoadingHistory={isLoadingHistory}
      canLoadHistory={canLoadHistory}
      disabled={disabled}
      errorMessage={errorMessage}
      onLoadHistory={onLoadHistory}
      onDraftChange={onDraftChange}
      onSubmit={onSubmit}
      onCardPromptRequest={onCardPromptRequest}
    />
  );
}
