import type { ReactNode } from "react";

import type {
  ClinicalEventLogEntry,
  JsonObject,
  PatientRegistryAlert,
  PatientRegistryDetail,
  PatientRegistryRecord,
} from "../../app/api/types";
import { CLINICAL_HUMAN_REVIEW_LABEL } from "../../app/clinical/clinical-copy";
import { compactClinicalEventDetail, formatCriticFeedback } from "../../app/clinical/critic-feedback";
import { Button, Card } from "../../components/ui";
import { ClinicalCardsPanel } from "../cards/clinical-cards-panel";
import type { CardPatientContext, CardPromptHandler } from "../cards/card-renderers-extended";
import {
  buildMultimodalActionState,
  buildMultimodalPrompt,
  buildMultimodalPromptContext,
  type MultimodalPromptContext,
  groupMultimodalCards,
  MULTIMODAL_ACTIONS,
} from "./doctor-multimodal-utils";

export type DoctorPatientRegistrySnapshot = {
  boundPatientDetail: PatientRegistryDetail | null;
  boundPatientRecords: PatientRegistryRecord[];
  boundPatientAlerts: PatientRegistryAlert[];
  isLoadingBoundPatient: boolean;
} & Record<string, unknown>;

export type DoctorMultimodalViewProps = {
  registryPatientId: number | null;
  caseDatabasePatientId: string | null;
  patientRegistry: DoctorPatientRegistrySnapshot;
  cards: Record<string, JsonObject>;
  critic?: JsonObject | null;
  eventLog?: ClinicalEventLogEntry[];
  isStreaming: boolean;
  disabled: boolean;
  patientContext?: CardPatientContext | null;
  onCardPromptRequest?: CardPromptHandler;
};

type MultimodalGroupKey = "imaging" | "pathology" | "radiomics";

const EMPTY_GROUP_MESSAGES: Record<MultimodalGroupKey, string> = {
  imaging: "暂无影像卡片",
  pathology: "暂无病理卡片",
  radiomics: "暂无放射组学卡片",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function textFromValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return null;
}

function formatRegistryPatientLabel(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `P-${value}` : "未绑定";
}

function formatCaseDatabasePatientLabel(value: string | number | null | undefined): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value).padStart(3, "0");
  }
  const text = textFromValue(value);
  if (!text) {
    return "未绑定";
  }
  const numeric = Number(text);
  return Number.isFinite(numeric) && text === String(numeric) ? String(numeric).padStart(3, "0") : text;
}

function displayCaseDatabasePatientId(context: MultimodalPromptContext | null): string {
  return context?.case_database_patient_id ?? "未绑定";
}

function formatStage(detail: PatientRegistryDetail | null): string {
  if (!detail) {
    return "暂无";
  }

  if (detail.clinical_stage) {
    return detail.clinical_stage;
  }

  const parts = [detail.t_stage, detail.n_stage, detail.m_stage].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join("") : "暂无";
}

function formatPatientDetailValue(value: unknown): string {
  return textFromValue(value) ?? "暂无";
}

function formatRecordLabel(record: PatientRegistryRecord): string {
  return [record.document_type, record.record_type, `#${record.record_id}`].filter(Boolean).join(" / ");
}

function alertSeverityLabel(alert: PatientRegistryAlert): string {
  const token = [alert.kind, alert.message, alert.document_type].filter(Boolean).join(" ").toLowerCase();
  if (/(critical|severe|high|urgent|danger)/.test(token)) {
    return "严重告警";
  }
  if (/(review|pending|unreviewed|pending_review|needs_review)/.test(token)) {
    return "待审阅";
  }
  return "告警";
}

function patientAlertMessage(alert: PatientRegistryAlert): string {
  const bits = [alertSeverityLabel(alert)];
  if (alert.document_type) {
    bits.push(alert.document_type);
  }
  if (alert.field_name) {
    bits.push(alert.field_name);
  } else if (alert.field_names && alert.field_names.length > 0) {
    bits.push(alert.field_names.join("、"));
  }
  bits.push(alert.message);
  return bits.filter(Boolean).join("：");
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

function resolveMultimodalPatientContext({
  registryPatientId,
  caseDatabasePatientId,
  patientRegistry,
  patientContext,
}: DoctorMultimodalViewProps): MultimodalPromptContext | null {
  const mergedContext: CardPatientContext = {
    ...(patientContext ?? {}),
  };

  const registryId = registryPatientId ?? patientRegistry.boundPatientDetail?.patient_id ?? null;
  if (registryId !== null) {
    mergedContext.registry_patient_id = registryId;
  }

  if (caseDatabasePatientId !== null) {
    mergedContext.case_database_patient_id = caseDatabasePatientId;
  }

  const resolvedContext = buildMultimodalPromptContext(mergedContext);
  return Object.keys(resolvedContext).length > 0 ? resolvedContext : null;
}

function renderDefinitionList(items: Array<{ label: string; value: ReactNode }>) {
  return (
    <dl className="clinical-multimodal-definition-list">
      {items.map((item) => (
        <div key={item.label} className="clinical-multimodal-definition-row">
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderPatientContextCard(
  resolvedContext: MultimodalPromptContext | null,
  patientDetail: PatientRegistryDetail | null,
  isLoadingBoundPatient: boolean,
) {
  const hasContext = Boolean(resolvedContext || patientDetail);

  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-card clinical-multimodal-patient-context">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>患者上下文</h2>
      </div>
      {hasContext ? (
        <div className="clinical-multimodal-card-body">
          {renderDefinitionList([
            { label: "登记号", value: formatRegistryPatientLabel(resolvedContext?.registry_patient_id ?? patientDetail?.patient_id ?? null) },
            {
              label: "病例样本",
              value: displayCaseDatabasePatientId(resolvedContext),
            },
            { label: "年龄", value: formatPatientDetailValue(patientDetail?.age) },
            { label: "性别", value: formatPatientDetailValue(patientDetail?.gender) },
            { label: "肿瘤位置", value: formatPatientDetailValue(patientDetail?.tumor_location) },
            { label: "分期", value: formatStage(patientDetail) },
            { label: "MMR", value: formatPatientDetailValue(patientDetail?.mmr_status) },
          ])}
          {isLoadingBoundPatient ? <p className="clinical-multimodal-status">正在加载患者资料...</p> : null}
        </div>
      ) : (
        <div className="clinical-multimodal-empty-state">
          <p>未绑定注册患者</p>
          <p>未绑定病例样本</p>
          <p>暂无患者资料</p>
        </div>
      )}
    </Card>
  );
}

function renderRecordsCard(records: PatientRegistryRecord[]) {
  const visibleRecords = records.slice(0, 4);

  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-card clinical-multimodal-records">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>资料卡片</h2>
      </div>
      {visibleRecords.length > 0 ? (
        <ul className="clinical-multimodal-list">
          {visibleRecords.map((record) => (
            <li key={record.record_id}>{formatRecordLabel(record)}</li>
          ))}
        </ul>
      ) : (
        <div className="clinical-multimodal-empty-state">
          <p>暂无资料卡片</p>
        </div>
      )}
    </Card>
  );
}

function renderAlertsCard(alerts: PatientRegistryAlert[]) {
  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-card clinical-multimodal-alerts">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>告警卡片</h2>
      </div>
      {alerts.length > 0 ? (
        <ul className="clinical-multimodal-list">
          {alerts.map((alert, index) => (
            <li key={`${alert.kind}-${alert.record_id ?? index}`}>{patientAlertMessage(alert)}</li>
          ))}
        </ul>
      ) : (
        <div className="clinical-multimodal-empty-state">
          <p>暂无告警</p>
        </div>
      )}
    </Card>
  );
}

function multimodalGroupCards(cards: Array<{ cardType: string; payload: JsonObject }>) {
  return cards.reduce<Record<string, JsonObject>>((accumulator, card) => {
    accumulator[card.cardType] = card.payload;
    return accumulator;
  }, {});
}

function renderEmptyMultimodalPanel(title: string, emptyMessage: string) {
  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-empty-group">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>{title}</h2>
      </div>
      <div className="clinical-multimodal-empty-state">
        <p>{emptyMessage}</p>
      </div>
    </Card>
  );
}

function renderMultimodalGroups(
  cards: Record<string, JsonObject>,
  patientContext: MultimodalPromptContext | null,
  onCardPromptRequest?: CardPromptHandler,
) {
  const grouped = groupMultimodalCards(cards);
  if (grouped.length === 0) {
    return (
      <>
        {renderEmptyMultimodalPanel("影像组", EMPTY_GROUP_MESSAGES.imaging)}
        {renderEmptyMultimodalPanel("病理组", EMPTY_GROUP_MESSAGES.pathology)}
        {renderEmptyMultimodalPanel("放射组学组", EMPTY_GROUP_MESSAGES.radiomics)}
      </>
    );
  }

  return (
    <>
      {grouped.map((group) => (
        <ClinicalCardsPanel
          key={group.key}
          title={group.title}
          emptyMessage={group.summary}
          cards={multimodalGroupCards(group.cards)}
          selectedCardType={null}
          onPromptRequest={onCardPromptRequest}
          patientContext={patientContext}
        />
      ))}
    </>
  );
}

function actionDisableReason(
  actionKey: string,
  context: MultimodalPromptContext | null,
): string | null {
  if (actionKey === "imaging_review" || actionKey === "pathology_review") {
    return context?.case_database_patient_id ? null : "需要病例样本编号";
  }

  return context?.registry_patient_id !== undefined ? null : "需要登记号";
}

function renderActionPanel(
  props: DoctorMultimodalViewProps,
  resolvedPatientContext: MultimodalPromptContext | null,
) {
  const promptContext = buildMultimodalPromptContext(resolvedPatientContext);
  const anyContext = Object.keys(promptContext).length > 0;
  const actionStates = MULTIMODAL_ACTIONS.map((action) => ({
    action,
    state: buildMultimodalActionState(action, promptContext),
  }));

  const canPrompt = Boolean(props.onCardPromptRequest) && !props.disabled && !props.isStreaming;
  const anyEnabledAction = actionStates.some(({ state }) => !state.disabled);
  const statusMessage = props.isStreaming
    ? "正在生成中，暂时不能发起多模态操作。"
    : props.disabled
      ? "当前页面已禁用。"
      : !props.onCardPromptRequest
        ? "当前没有可用的多模态回调。"
        : !anyContext
          ? "暂无患者上下文，部分操作不可用。"
          : !anyEnabledAction
          ? "影像和病理需要病例样本编号，摘要和交接需要登记号。"
          : null;

  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-card clinical-multimodal-actions">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>多模态操作</h2>
      </div>
      {statusMessage ? <p className="clinical-multimodal-status">{statusMessage}</p> : null}
      <div className="clinical-multimodal-action-grid">
        {actionStates.map(({ action, state }) => {
          const buttonDisabled = !canPrompt || state.disabled;
          const disabledReason = buttonDisabled ? actionDisableReason(action.key, resolvedPatientContext) : null;

          return (
            <Button
              key={action.key}
              type="button"
              variant={buttonDisabled ? "secondary" : "primary"}
              size="sm"
              disabled={buttonDisabled}
              title={disabledReason ?? action.summary}
              onClick={() => {
                if (buttonDisabled || !props.onCardPromptRequest) {
                  return;
                }
                props.onCardPromptRequest(buildMultimodalPrompt(action), promptContext);
              }}
            >
              {action.title}
            </Button>
          );
        })}
      </div>
    </Card>
  );
}

function renderReviewPanel(props: DoctorMultimodalViewProps) {
  const critic = props.critic ?? null;
  const requiresHumanReview = criticRequiresHumanReview(critic);
  const reviewFeedback = formatCriticFeedback(critic?.feedback);
  const recentEvents = (props.eventLog ?? []).slice(-4);

  return (
    <Card as="section" padding="none" className="clinical-card clinical-multimodal-card clinical-multimodal-review">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon" aria-hidden="true" />
        <h2>复核与事件</h2>
      </div>
      {requiresHumanReview ? (
        <div className="clinical-multimodal-review-warning" role="status">
          <strong>{CLINICAL_HUMAN_REVIEW_LABEL}</strong>
          {typeof critic?.verdict === "string" && critic.verdict.trim() ? <p>{critic.verdict.trim()}</p> : null}
          <p>{reviewFeedback}</p>
        </div>
      ) : null}
      {recentEvents.length > 0 ? (
        <div className="clinical-multimodal-review-list">
          {recentEvents.map((event) => {
            const detail = event.kind === "critic" ? compactClinicalEventDetail(event.detail) : event.detail;
            return (
              <article key={event.id} className={`clinical-multimodal-review-entry clinical-multimodal-review-entry-${event.tone}`}>
                <div className="clinical-multimodal-review-entry-head">
                  <strong>{event.title}</strong>
                  <span>{event.kind}</span>
                </div>
                {detail ? <p>{detail}</p> : null}
                {event.requiresHumanReview ? <p>{CLINICAL_HUMAN_REVIEW_LABEL}</p> : null}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="clinical-multimodal-empty-state">
          <p>暂无事件记录</p>
        </div>
      )}
    </Card>
  );
}

export function DoctorMultimodalView(props: DoctorMultimodalViewProps) {
  const resolvedPatientContext = resolveMultimodalPatientContext(props);

  return (
    <main data-testid="doctor-multimodal-view" className="clinical-multimodal-dashboard">
      <section className="clinical-multimodal-left-column">
        {renderPatientContextCard(
          resolvedPatientContext,
          props.patientRegistry.boundPatientDetail,
          props.patientRegistry.isLoadingBoundPatient,
        )}
        {renderRecordsCard(props.patientRegistry.boundPatientRecords)}
        {renderAlertsCard(props.patientRegistry.boundPatientAlerts)}
      </section>
      <section className="clinical-multimodal-center-column">
        {renderMultimodalGroups(props.cards, resolvedPatientContext, props.onCardPromptRequest)}
      </section>
      <aside className="clinical-multimodal-right-column">
        {renderActionPanel(props, resolvedPatientContext)}
        {renderReviewPanel(props)}
      </aside>
    </main>
  );
}
