import { useCallback, useRef, useState, type ReactNode } from "react";

import {
  type ClinicalEventLogEntry,
  type FrontendMessage,
  type JsonObject,
  type PatientRegistryDetail,
} from "../../app/api/types";
import {
  CLINICAL_HUMAN_REVIEW_LABEL,
  CLINICAL_PATIENT_SCENE_ARIA_LABEL,
} from "../../app/clinical/clinical-copy";
import {
  formatClinicalCriticVerdict,
  formatClinicalEventDetail,
  formatClinicalEventKind,
  formatClinicalEventTitle,
} from "../../app/clinical/clinical-event-labels";
import { compactClinicalEventDetail, formatCriticFeedback } from "../../app/clinical/critic-feedback";
import { ClinicalEmptyState } from "../../components/layout/clinical-empty-state";
import { ClinicalTopNav } from "../../components/layout/clinical-top-nav";
import { useViewTransition } from "../../components/motion/use-view-transition";
import { Card } from "../../components/ui";
import { AnatomyHighlightPanel } from "../anatomy/anatomy-highlight-panel";
import { ClinicalCardsPanel } from "../cards/clinical-cards-panel";
import type { CardPatientContext, CardPromptHandler } from "../cards/card-renderers-extended";
import { useDatabaseWorkbench } from "../database/use-database-workbench";
import { ExecutionPlanPanel } from "../execution-plan/execution-plan-panel";
import { usePatientRegistry } from "../patient-registry/use-patient-registry";
import { useRegistryBrowser } from "../patient-registry/use-registry-browser";
import { RoadmapPanel } from "../roadmap/roadmap-panel";
import {
  DoctorConsultationView,
  type DoctorLatencyStatus,
} from "./doctor-consultation-view";
import { DoctorDatabaseView } from "./doctor-database-view";
import { DoctorMultimodalView } from "./doctor-multimodal-view";
import { DoctorReportDraftView } from "./doctor-report-draft-view";
import { DoctorReviewCockpit } from "./doctor-review-cockpit";
import { useDoctorViewState, type DoctorTab } from "./use-doctor-view-state";

type DoctorSceneShellProps = {
  toolbar: ReactNode;
  surfaceSwitcher?: ReactNode;
  onSwitchScene?: () => void;
  registryPatientId: number | null;
  caseDatabasePatientId: string | null;
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
  critic?: JsonObject | null;
  eventLog?: ClinicalEventLogEntry[];
  onLoadHistory: () => void;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onSetCurrentPatient: (patientId: number) => Promise<boolean>;
  onSetCurrentCaseDatabasePatient: (patientId: number) => void;
  onCardPromptRequest?: CardPromptHandler;
  patientContext?: CardPatientContext | null;
  doctorReviewCockpitEnabled?: boolean;
  sessionId?: string | null;
};

type NavItem = {
  key: DoctorTab;
  label: string;
  disabled?: boolean;
};

const DOCTOR_NAV_ITEMS: NavItem[] = [
  { key: "consultation", label: "会诊" },
  { key: "database", label: "患者数据库" },
  { key: "multimodal", label: "多模态" },
  { key: "reports", label: "报表" },
  { key: "review", label: "Review" },
];

const PRODUCTION_DOCTOR_NAV_ITEMS = DOCTOR_NAV_ITEMS.filter((item) => !item.disabled);

function SmallIcon({
  name,
}: {
  name: "patient" | "paperclip" | "event" | "external" | "user" | "check" | "activity";
}) {
  if (name === "patient") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="7" r="4" />
        <path d="M4 21c1.4-4.2 4.1-6.2 8-6.2s6.6 2 8 6.2" />
      </svg>
    );
  }
  if (name === "paperclip") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m8 12 6.8-6.8a4 4 0 0 1 5.7 5.7l-8.3 8.3a5 5 0 0 1-7.1-7.1l8-8" />
      </svg>
    );
  }
  if (name === "event") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="3" />
        <path d="M5 12a7 7 0 0 1 7-7" />
        <path d="M19 12a7 7 0 0 0-7-7" />
        <path d="M5 12a7 7 0 0 0 7 7" />
        <path d="M19 12a7 7 0 0 1-7 7" />
      </svg>
    );
  }
  if (name === "external") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M14 4h6v6" />
        <path d="m10 14 10-10" />
        <path d="M20 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h5" />
      </svg>
    );
  }
  if (name === "user") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 22c1.8-4.8 4.5-7.2 8-7.2s6.2 2.4 8 7.2" />
      </svg>
    );
  }
  if (name === "check") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m5 12 4 4L19 6" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 12h4l2-7 4 14 2-7h4" />
    </svg>
  );
}


function tumorTypeFromDetail(value: string | null | undefined): string {
  if (!value) {
    return "暂未提供";
  }
  const normalized = value.toLowerCase();
  if (normalized.includes("colon") || normalized.includes("rect")) {
    return "CRC";
  }
  return value;
}

function tnmFromDetail(detail: DoctorSceneShellProps["patientRegistry"]["boundPatientDetail"]): string {
  if (!detail) {
    return "暂未提供";
  }
  if (detail.clinical_stage) {
    return detail.clinical_stage;
  }
  const parts = [detail.t_stage, detail.n_stage, detail.m_stage].filter(Boolean);
  return parts.length > 0 ? parts.join("") : "暂未提供";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readCardText(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return null;
}

function readCardNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function firstText(...values: unknown[]): string | null {
  for (const value of values) {
    const text = readCardText(value);
    if (text !== null) {
      return text;
    }
  }
  return null;
}

function patientCardData(cards: Record<string, JsonObject>): Record<string, unknown> | null {
  const patientCard = asRecord(cards.patient_card);
  return asRecord(patientCard?.data) ?? patientCard;
}

function patientCardSection(cards: Record<string, JsonObject>, section: string): Record<string, unknown> | null {
  const data = patientCardData(cards);
  return asRecord(data?.[section]);
}

function patientIdFromCards(cards: Record<string, JsonObject>): number | null {
  const patientCard = asRecord(cards.patient_card);
  const data = patientCardData(cards);
  return readCardNumber(patientCard?.patient_id ?? patientCard?.patientId ?? data?.patient_id ?? data?.patientId);
}

function patientDetailFromCards(cards: Record<string, JsonObject>): PatientRegistryDetail | null {
  if (!cards.patient_card) {
    return null;
  }

  const data = patientCardData(cards);
  const patientInfo =
    patientCardSection(cards, "patient_info") ??
    patientCardSection(cards, "patientInfo") ??
    patientCardSection(cards, "demographics");
  const diagnosis =
    patientCardSection(cards, "diagnosis_block") ??
    patientCardSection(cards, "diagnosisBlock") ??
    patientCardSection(cards, "diagnosis");
  const staging =
    patientCardSection(cards, "staging_block") ??
    patientCardSection(cards, "stagingBlock") ??
    patientCardSection(cards, "staging");

  const patientId = patientIdFromCards(cards);
  return {
    patient_id: patientId ?? 0,
    status: "derived_from_patient_card",
    created_at: "",
    updated_at: "",
    age: readCardNumber(patientInfo?.age ?? data?.age),
    gender: firstText(patientInfo?.gender, data?.gender),
    tumor_location: firstText(
      diagnosis?.tumor_location,
      diagnosis?.tumorLocation,
      diagnosis?.primary_site,
      diagnosis?.primarySite,
      diagnosis?.confirmed,
      diagnosis?.diagnosis,
      data?.tumor_location,
      data?.tumorLocation,
    ),
    mmr_status: firstText(
      diagnosis?.mmr_status,
      diagnosis?.mmrStatus,
      diagnosis?.mmr_msi_status,
      diagnosis?.mmrMsiStatus,
      data?.mmr_status,
      data?.mmrStatus,
    ),
    clinical_stage: firstText(staging?.clinical_stage, staging?.clinicalStage, staging?.tnm, data?.clinical_stage),
    t_stage: firstText(staging?.t_stage, staging?.tStage, staging?.ct_stage, staging?.ctStage),
    n_stage: firstText(staging?.n_stage, staging?.nStage, staging?.cn_stage, staging?.cnStage),
    m_stage: firstText(staging?.m_stage, staging?.mStage, staging?.cm_stage, staging?.cmStage),
  };
}

function ClinicalPanelHeader({
  icon,
  title,
}: {
  icon: ReactNode;
  title: string;
}) {
  return (
    <div className="clinical-panel-header">
      <span className="clinical-panel-icon">{icon}</span>
      <h2>{title}</h2>
    </div>
  );
}

function ClinicalPatientSummary({
  registryPatientId,
  caseDatabasePatientId,
  detail,
  isLoading,
}: {
  registryPatientId: number | null;
  caseDatabasePatientId: string | null;
  detail: DoctorSceneShellProps["patientRegistry"]["boundPatientDetail"];
  isLoading: boolean;
}) {
  const hasSummary = registryPatientId !== null || caseDatabasePatientId !== null || detail !== null;
  const rows = hasSummary
    ? [
        ["登记号:", registryPatientId !== null ? `P-${registryPatientId}` : "未绑定"],
        ["病例样本:", caseDatabasePatientId ?? "未绑定"],
        ["年龄:", detail?.age ?? "暂未提供"],
        ["性别:", detail?.gender ?? "暂未提供"],
        ["肿瘤部位:", detail?.tumor_location ? tumorTypeFromDetail(detail.tumor_location) : "暂未提供"],
        ["TNM:", detail ? tnmFromDetail(detail) : "暂未提供"],
        ["MMR/MSI:", detail?.mmr_status ?? "暂未提供"],
      ]
    : [];

  return (
    <Card as="section" padding="none" className="clinical-card clinical-summary-card" aria-label="患者摘要">
      <ClinicalPanelHeader icon={<SmallIcon name="patient" />} title="患者摘要" />
      {rows.length > 0 ? (
        <dl className="clinical-summary-list">
          {rows.map(([label, value]) => (
            <div key={label} className="clinical-summary-row">
              <dt>{label}</dt>
              <dd className={label === "MMR/MSI:" && ["pending", "unknown", "暂未提供"].includes(String(value).toLowerCase()) ? "clinical-warning-text" : undefined}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <ClinicalEmptyState
          compact
          icon="summary"
          title="患者摘要待绑定"
          message="暂无患者摘要。"
        />
      )}
      {isLoading ? <p className="clinical-loading-copy">正在加载患者摘要…</p> : null}
    </Card>
  );
}

function uploadItemsFromRecords(records: DoctorSceneShellProps["patientRegistry"]["boundPatientRecords"]) {
  return records.slice(0, 3).map((record, index) => ({
    title: record.document_type || record.record_type || `资料 #${record.record_id}`,
    tone: index === 0 ? "blue" : index === 1 ? "violet" : "cyan",
  }));
}

function ClinicalUploads({
  records,
  isLoading,
}: {
  records: DoctorSceneShellProps["patientRegistry"]["boundPatientRecords"];
  isLoading: boolean;
}) {
  const uploads = uploadItemsFromRecords(records);

  return (
    <Card as="section" padding="none" className="clinical-card clinical-uploads-card">
      <ClinicalPanelHeader icon={<SmallIcon name="paperclip" />} title="上传资料" />
      <div className="clinical-upload-list">
        {uploads.length > 0 ? (
          uploads.map((item) => (
            <div key={item.title} className="clinical-upload-item">
              <span className={`clinical-upload-file clinical-upload-file-${item.tone}`} aria-hidden="true" />
              <span>{item.title}</span>
              <span className="clinical-upload-check">
                <SmallIcon name="check" />
              </span>
            </div>
          ))
        ) : (
          <ClinicalEmptyState
            compact
            icon="uploads"
            title="资料待上传"
            message="暂无上传资料。"
          />
        )}
      </div>
      {isLoading ? <p className="clinical-loading-copy">正在加载上传资料…</p> : null}
    </Card>
  );
}

function criticRequiresHumanReview(critic: JsonObject | null | undefined): boolean {
  if (!critic) {
    return false;
  }
  if (typeof critic.requires_human_review === "boolean") {
    return critic.requires_human_review;
  }
  const verdict = typeof critic.verdict === "string" ? critic.verdict.trim().toUpperCase() : "";
  return Boolean(verdict && verdict !== "APPROVED");
}

function ClinicalEventStream({
  events,
  critic,
}: {
  events: ClinicalEventLogEntry[];
  critic?: JsonObject | null;
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const requiresHumanReview = criticRequiresHumanReview(critic);
  const reviewFeedback = formatCriticFeedback(critic?.feedback);
  const reviewVerdict = formatClinicalCriticVerdict(critic?.verdict);
  return (
    <Card
      as="section"
      padding="none"
      className="clinical-card clinical-event-stream"
      data-expanded={isExpanded}
    >
      <div className="clinical-panel-header clinical-event-console-header">
        <span className="clinical-panel-icon">
          <SmallIcon name="event" />
        </span>
        <h2>事件流</h2>
        <span className="clinical-event-console-meta">
          {events.length > 0 ? `${events.length} 条事件` : "暂无事件"}
        </span>
        <button
          type="button"
          className="clinical-event-console-toggle"
          aria-expanded={isExpanded}
          onClick={() => setIsExpanded((current) => !current)}
        >
          {isExpanded ? "收起事件流" : "展开事件流"}
        </button>
      </div>
      {isExpanded ? (
        <>
          {requiresHumanReview ? (
            <div className="clinical-review-warning" role="status">
              <div className="clinical-review-warning-head">
                <strong>{CLINICAL_HUMAN_REVIEW_LABEL}</strong>
                {reviewVerdict ? <span>{reviewVerdict}</span> : null}
              </div>
              <p className="clinical-review-feedback">{reviewFeedback}</p>
            </div>
          ) : null}
          {events.length > 0 ? (
            <div className="clinical-event-row">
              {events.map((event) => {
                const rawDetail = event.kind === "critic"
                  ? compactClinicalEventDetail(event.detail)
                  : event.detail;
                const detail = formatClinicalEventDetail(rawDetail);
                return (
                  <article
                    key={event.id}
                    className={`clinical-event-chip clinical-event-chip-${event.tone}`}
                  >
                    <div>
                      <strong>{formatClinicalEventTitle(event)}</strong>
                      <span>{formatClinicalEventKind(event.kind)}</span>
                    </div>
                    {detail ? <p>{detail}</p> : null}
                    {event.requiresHumanReview ? <p>{CLINICAL_HUMAN_REVIEW_LABEL}</p> : null}
                  </article>
                );
              })}
            </div>
          ) : (
            <ClinicalEmptyState
              compact
              icon="events"
              title="事件流待启动"
              message="暂无事件。"
            />
          )}
        </>
      ) : null}
    </Card>
  );
}

export function DoctorSceneShell({
  toolbar,
  surfaceSwitcher,
  registryPatientId,
  caseDatabasePatientId,
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
  critic = null,
  eventLog = [],
  onLoadHistory,
  onDraftChange,
  onSubmit,
  onSwitchScene,
  onSetCurrentPatient,
  onSetCurrentCaseDatabasePatient,
  onCardPromptRequest,
  patientContext: providedPatientContext,
  doctorReviewCockpitEnabled = false,
  sessionId = null,
}: DoctorSceneShellProps) {
  const {
    activeDoctorTab,
    setActiveDoctorTab,
    activeDatabaseSource,
    setActiveDatabaseSource,
  } = useDoctorViewState();
  const sceneRef = useRef<HTMLElement | null>(null);
  const setSceneElement = useCallback((element: HTMLElement | null) => {
    sceneRef.current = element;
  }, []);

  useViewTransition(sceneRef, activeDoctorTab);

  async function handleSetCurrentPatient(patientId: number) {
    const didBind = await onSetCurrentPatient(patientId);
    if (didBind) {
      setActiveDoctorTab("consultation");
    }
  }

  function handleSetCurrentCaseDatabasePatient(patientId: number) {
    onSetCurrentCaseDatabasePatient(patientId);
    setActiveDoctorTab("consultation");
  }

  function handleTabChange(tab: string) {
    if (
      tab !== "consultation" &&
      tab !== "database" &&
      tab !== "multimodal" &&
      tab !== "reports" &&
      tab !== "review"
    ) {
      return;
    }

    setActiveDoctorTab(tab);
    if (tab === "database") {
      setActiveDatabaseSource("patient_registry");
    }
  }


  const topNav = (
    <ClinicalTopNav
       brandLabel="临床助手"
       navLabel="临床导航"
      items={PRODUCTION_DOCTOR_NAV_ITEMS}
      activeKey={activeDoctorTab}
      onSelect={handleTabChange}
      actions={toolbar}
       statusLabel="SSE 连接正常"
      statusTone="connected"
      profileLabel="医生"
      profileAriaLabel={CLINICAL_PATIENT_SCENE_ARIA_LABEL}
      onProfileClick={onSwitchScene}
      profileControl={surfaceSwitcher}
      className="clinical-top-nav-doctor"
    />
  );

  if (activeDoctorTab === "database") {
    return (
      <div
        ref={setSceneElement}
        className="clinical-app-shell clinical-app-shell-doctor clinical-app-shell-database"
        data-testid="doctor-scene"
      >
        {topNav}
        <DoctorDatabaseView
          parentToolbar={null}
          activeSource={activeDatabaseSource}
          onSourceChange={setActiveDatabaseSource}
          registryPatientId={registryPatientId}
          databaseWorkbench={databaseWorkbench}
          registryBrowser={registryBrowser}
          isBindingCurrentPatient={patientRegistry.isBindingPatient}
          onSetCurrentPatient={handleSetCurrentPatient}
          onSetCurrentCaseDatabasePatient={handleSetCurrentCaseDatabasePatient}
        />
      </div>
    );
  }

  if (activeDoctorTab === "multimodal") {
    const patientContext: CardPatientContext = {};
    if (registryPatientId !== null) {
      patientContext.registry_patient_id = registryPatientId;
    }
    if (caseDatabasePatientId !== null) {
      patientContext.case_database_patient_id = caseDatabasePatientId;
    }
    const cardPatientContext =
      providedPatientContext && Object.keys(providedPatientContext).length > 0
        ? { ...patientContext, ...providedPatientContext }
        : Object.keys(patientContext).length > 0
          ? patientContext
          : undefined;

    return (
      <div
        ref={setSceneElement}
        className="clinical-app-shell clinical-app-shell-doctor clinical-app-shell-multimodal"
        data-testid="doctor-scene"
      >
        {topNav}
        <DoctorMultimodalView
          registryPatientId={registryPatientId}
          caseDatabasePatientId={caseDatabasePatientId}
          patientRegistry={patientRegistry}
          cards={cards}
          critic={critic}
          eventLog={eventLog}
          isStreaming={isStreaming}
          disabled={disabled}
          onCardPromptRequest={onCardPromptRequest}
          patientContext={cardPatientContext}
        />
      </div>
    );
  }

  if (activeDoctorTab === "reports") {
    const patientContext: CardPatientContext = {};
    if (registryPatientId !== null) {
      patientContext.registry_patient_id = registryPatientId;
    }
    if (caseDatabasePatientId !== null) {
      patientContext.case_database_patient_id = caseDatabasePatientId;
    }
    const cardPatientContext =
      providedPatientContext && Object.keys(providedPatientContext).length > 0
        ? { ...patientContext, ...providedPatientContext }
        : Object.keys(patientContext).length > 0
          ? patientContext
          : undefined;

    return (
      <div
        ref={setSceneElement}
        className="clinical-app-shell clinical-app-shell-doctor clinical-app-shell-reports"
        data-testid="doctor-scene"
      >
        {topNav}
        <DoctorReportDraftView
          registryPatientId={registryPatientId}
          caseDatabasePatientId={caseDatabasePatientId}
          messages={messages}
          isStreaming={isStreaming}
          disabled={disabled}
          onReportPromptRequest={onCardPromptRequest}
          patientContext={cardPatientContext}
        />
      </div>
    );
  }

  if (activeDoctorTab === "review") {
    return (
      <div
        ref={setSceneElement}
        className="clinical-app-shell clinical-app-shell-doctor clinical-app-shell-review"
        data-testid="doctor-scene"
      >
        {topNav}
        <DoctorReviewCockpit
          sessionId={sessionId ?? null}
          enabled={doctorReviewCockpitEnabled === true}
        />
      </div>
    );
  }

  const cardPatientId = patientIdFromCards(cards);
  const summaryCaseDatabasePatientId = caseDatabasePatientId ?? (
    cardPatientId !== null ? String(cardPatientId).padStart(3, "0") : null
  );
  const summaryPatientDetail = patientRegistry.boundPatientDetail ?? patientDetailFromCards(cards);
  const visibleMessages = messages;
  const patientContext: CardPatientContext = {};
  if (registryPatientId !== null) {
    patientContext.registry_patient_id = registryPatientId;
  }
  if (caseDatabasePatientId !== null) {
    patientContext.case_database_patient_id = caseDatabasePatientId;
  }
  const cardPatientContext =
    providedPatientContext && Object.keys(providedPatientContext).length > 0
      ? { ...patientContext, ...providedPatientContext }
      : Object.keys(patientContext).length > 0
        ? patientContext
        : undefined;

  return (
    <main
      ref={setSceneElement}
      className="clinical-app-shell clinical-app-shell-doctor clinical-app-shell-consultation"
      data-testid="doctor-scene"
    >
      {topNav}
      <div className="clinical-dashboard">
        <aside className="clinical-left-column">
          <ClinicalPatientSummary
            registryPatientId={registryPatientId}
            caseDatabasePatientId={summaryCaseDatabasePatientId}
            detail={summaryPatientDetail}
            isLoading={patientRegistry.isLoadingBoundPatient}
          />
          <AnatomyHighlightPanel
            detail={summaryPatientDetail}
            patientContext={cardPatientContext}
            onPromptRequest={onCardPromptRequest}
            disabled={disabled}
            isStreaming={isStreaming}
          />
          <ClinicalUploads
            records={patientRegistry.boundPatientRecords}
            isLoading={patientRegistry.isLoadingBoundPatient}
          />
        </aside>
        <section className="clinical-center-column">
          <DoctorConsultationView
            messages={visibleMessages}
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
            patientContext={cardPatientContext}
          />
          <ClinicalCardsPanel
            title="医疗卡片"
            emptyMessage="暂无医疗卡片。"
            cards={cards}
            selectedCardType={null}
            onPromptRequest={onCardPromptRequest}
            patientContext={cardPatientContext}
          />
        </section>
        <aside className="clinical-right-column">
          <RoadmapPanel roadmap={roadmap} stage={stage} />
          <ExecutionPlanPanel plan={plan} references={references} critic={critic} />
        </aside>
        <div className="clinical-event-column">
          <ClinicalEventStream events={eventLog} critic={critic} />
        </div>
      </div>
    </main>
  );
}
