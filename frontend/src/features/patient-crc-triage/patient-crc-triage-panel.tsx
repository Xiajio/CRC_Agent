import { useMemo } from "react";

import type {
  CrcTriageAssessmentPayload,
  CrcTriageNodeResult,
  CrcTriageQaItem,
  CrcTriageProtocolState,
  CrcTriageQuestion,
  SessionState,
} from "../../app/api/types";
import { Button, Card } from "../../components/ui";
import {
  CRC_TRIAGE_START_PROMPT,
  buildCrcTriageAssessmentDraft,
  buildCrcTriageContext,
} from "./crc-triage-context";

export type CrcTriageSaveStatus = "idle" | "saving" | "saved" | "error";

export interface PatientCrcTriagePanelProps {
  sessionState: SessionState;
  disabled: boolean;
  saveStatus: CrcTriageSaveStatus;
  saveErrorMessage?: string | null;
  onStart: (prompt: string, context: Record<string, unknown>) => void;
  onUploadRequest: () => void;
  onSaveAssessment: (assessment: CrcTriageAssessmentPayload) => void;
}

const COPY = {
  title: "\u7ed3\u76f4\u80a0\u764c\u4e13\u9879\u9884\u95ee\u8bca",
  intro:
    "\u56f4\u7ed5\u4fbf\u8840\u3001\u6392\u4fbf\u4e60\u60ef\u6539\u53d8\u3001\u8d2b\u8840\u3001\u4f53\u91cd\u4e0b\u964d\u548c\u5bb6\u65cf\u53f2\u7b49\u7ebf\u7d22\u8fdb\u884c\u7ed3\u6784\u5316\u9884\u95ee\u8bca\u3002",
  startButton: "\u5f00\u59cb\u4e13\u9879\u95ee\u8bca",
  uploadButton: "\u4e0a\u4f20\u62a5\u544a",
  completedTitle: "\u5df2\u751f\u6210\u9884\u95ee\u8bca\u6458\u8981",
  saveButton: "\u4fdd\u5b58\u5230\u60a3\u8005\u8bb0\u5f55",
  saving: "\u6b63\u5728\u4fdd\u5b58\u9884\u95ee\u8bca\u8bb0\u5f55...",
  saved: "\u5df2\u4fdd\u5b58\u5230\u60a3\u8005\u8bb0\u5f55\u3002",
  saveFailed: "\u4fdd\u5b58\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002",
  pending:
    "\u5b8c\u6210\u4e13\u9879\u95ee\u8bca\u540e\uff0c\u7cfb\u7edf\u4f1a\u5728\u8fd9\u91cc\u6c47\u603b\u98ce\u9669\u5206\u5c42\u3001\u5efa\u8bae\u53bb\u5411\u548c\u5f85\u8865\u5145\u8d44\u6599\u3002",
  resultLabel: "\u95ee\u8bca\u7ed3\u679c",
  suggestedTests: "\u5efa\u8bae\u68c0\u67e5",
  missingInfo: "\u5f85\u8865\u5145\u4fe1\u606f",
  risk: "\u98ce\u9669\u5206\u5c42",
  disposition: "\u5efa\u8bae\u53bb\u5411",
  currentQuestion: "\u5f53\u524d\u95ee\u9898",
  nodeResults: "\u8282\u70b9\u8bc4\u4f30\u6458\u8981",
  qaSummary: "\u95ee\u8bca\u8bb0\u5f55",
  patientAnswer: "\u60a3\u8005\u56de\u7b54",
  nextStep: "\u4e0b\u4e00\u6b65",
  none: "\u6682\u65e0",
};

const STAGE_TITLES: Record<string, string> = {
  vitals: "\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30",
  red_flags: "\u8282\u70b92\uff1a\u5168\u7cfb\u7edf\u5371\u9669\u4fe1\u53f7\u7b5b\u67e5",
  symptom_cluster: "\u8282\u70b93\uff1a\u4e3b\u8981\u75c7\u72b6\u805a\u7c7b",
  differential: "\u8282\u70b94\uff1a\u5bf9\u5e94\u75c7\u72b6\u7ec4\u8be6\u7ec6\u9274\u522b",
  tests: "\u8282\u70b95\uff1a\u8f85\u52a9\u68c0\u67e5\u7ed3\u679c\u89e3\u8bfb",
  final: "\u8282\u70b96\uff1a\u7ec8\u70b9\u8f93\u51fa",
};

const TRIAGE_RISK_LABELS: Record<string, string> = {
  low: "\u4f4e\u98ce\u9669",
  medium: "\u4e2d\u98ce\u9669",
  high: "\u9ad8\u98ce\u9669",
  unknown: "\u5f85\u8bc4\u4f30",
  pending: "\u5f85\u7ee7\u7eed\u95ee\u8bca",
};

const TRIAGE_DISPOSITION_LABELS: Record<string, string> = {
  observe: "\u89c2\u5bdf\u968f\u8bbf",
  observe_followup: "\u89c2\u5bdf\u968f\u8bbf",
  routine_gi_clinic: "\u5e38\u89c4\u6d88\u5316\u95e8\u8bca",
  urgent_gi_clinic: "\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca",
  emergency: "\u6025\u8bca\u5c31\u533b",
  enter_crc_flow: "\u8fdb\u5165 CRC \u4e34\u5e8a\u8bc4\u4f30",
  pending: "\u5f85\u7ee7\u7eed\u95ee\u8bca",
};

const STAGE_SEQUENCE = [
  "vitals",
  "red_flags",
  "symptom_cluster",
  "differential",
  "tests",
  "final",
] as const;

export function PatientCrcTriagePanel({
  sessionState,
  disabled,
  saveStatus,
  saveErrorMessage,
  onStart,
  onUploadRequest,
  onSaveAssessment,
}: PatientCrcTriagePanelProps) {
  const assessmentDraft = useMemo(() => buildCrcTriageAssessmentDraft(sessionState), [sessionState]);
  const protocolState = getCrcTriageProtocolState(sessionState);
  const currentQuestion = getCurrentQuestion(protocolState);
  const nodeResults = getNodeResults(protocolState);
  const isSaving = saveStatus === "saving";

  function handleStart() {
    onStart(CRC_TRIAGE_START_PROMPT, buildCrcTriageContext("start"));
  }

  function handleAnswer(option: string) {
    if (!currentQuestion) {
      return;
    }
    onStart(option, buildCrcTriageContext("answer", { question_id: currentQuestion.id }));
  }

  function handleSave() {
    if (!assessmentDraft) {
      return;
    }
    onSaveAssessment(assessmentDraft);
  }

  const statusMessage =
    saveStatus === "saving"
      ? COPY.saving
      : saveStatus === "saved"
        ? COPY.saved
        : saveStatus === "error"
          ? saveErrorMessage ?? COPY.saveFailed
          : null;

  return (
    <Card as="section" variant="clinical-panel" data-testid="crc-triage-panel">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon clinical-list-icon" aria-hidden="true" />
        <div>
          <h2>{COPY.title}</h2>
          <p className="clinical-copy clinical-copy-tight">{COPY.intro}</p>
        </div>
      </div>

      {protocolState && !assessmentDraft ? (
        <ProtocolStateSummary
          currentQuestion={currentQuestion}
          disabled={disabled}
          nodeResults={nodeResults}
          onAnswer={handleAnswer}
          protocolState={protocolState}
        />
      ) : null}

      {assessmentDraft ? (
        <div className="clinical-panel-stack">
          <AssessmentResultCard assessment={assessmentDraft} />
          <div className="patient-assistant-quick-actions">
            <Button
              data-testid="crc-triage-save"
              disabled={disabled || isSaving}
              onClick={handleSave}
              variant="primary"
            >
              {isSaving ? COPY.saving : COPY.saveButton}
            </Button>
            <Button data-testid="crc-triage-upload" disabled={disabled || isSaving} onClick={onUploadRequest}>
              {COPY.uploadButton}
            </Button>
          </div>
        </div>
      ) : currentQuestion ? (
        <div className="patient-assistant-quick-actions">
          <Button data-testid="crc-triage-upload" disabled={disabled} onClick={onUploadRequest}>
            {COPY.uploadButton}
          </Button>
        </div>
      ) : (
        <div className="clinical-panel-stack">
          <p className="clinical-copy" data-testid="crc-triage-pending">
            {COPY.pending}
          </p>
          <div className="patient-assistant-quick-actions">
            <Button
              data-testid="crc-triage-start"
              disabled={disabled}
              onClick={handleStart}
              variant="primary"
            >
              {COPY.startButton}
            </Button>
            <Button data-testid="crc-triage-upload" disabled={disabled} onClick={onUploadRequest}>
              {COPY.uploadButton}
            </Button>
          </div>
        </div>
      )}

      {statusMessage ? (
        <p
          className={[
            "clinical-copy",
            saveStatus === "error" ? "clinical-copy-alert" : "clinical-copy-tight",
          ].join(" ")}
          data-testid="crc-triage-save-status"
          role={saveStatus === "error" ? "alert" : "status"}
        >
          {statusMessage}
        </p>
      ) : null}
    </Card>
  );
}

interface AssessmentResultCardProps {
  assessment: CrcTriageAssessmentPayload;
}

function AssessmentResultCard({ assessment }: AssessmentResultCardProps) {
  const riskLabel = formatRiskLabel(assessment.risk_level);
  const dispositionLabel = formatDispositionLabel(assessment.disposition);
  const nextStepLabel = formatDispositionLabel(assessment.next_step);
  const nodeResults = assessment.node_results ?? [];

  return (
    <section
      aria-label={COPY.completedTitle}
      className="crc-triage-result-card"
      data-testid="crc-triage-result-card"
    >
      <div className="crc-triage-result-header">
        <span className="crc-triage-card-label">{COPY.resultLabel}</span>
        <span
          className="crc-triage-result-risk"
          data-risk-tone={riskTone(assessment.risk_level)}
          data-testid="crc-triage-result-risk"
        >
          {riskLabel}
        </span>
      </div>

      <div className="crc-triage-result-title">
        <h3>{COPY.completedTitle}</h3>
        <p data-testid="crc-triage-result-summary">{assessment.patient_summary}</p>
      </div>

      <dl className="crc-triage-result-metrics">
        <div className="crc-triage-result-metric">
          <dt>{COPY.risk}</dt>
          <dd>{riskLabel}</dd>
        </div>
        <div className="crc-triage-result-metric" data-testid="crc-triage-result-disposition">
          <dt>{COPY.disposition}</dt>
          <dd>{dispositionLabel}</dd>
        </div>
        <div className="crc-triage-result-metric">
          <dt>{COPY.nextStep}</dt>
          <dd>{nextStepLabel}</dd>
        </div>
      </dl>

      <div className="crc-triage-result-grid">
        <ResultList
          items={assessment.suggested_tests}
          label={COPY.suggestedTests}
          testId="crc-triage-result-tests"
        />
        <ResultList
          items={assessment.missing_information}
          label={COPY.missingInfo}
          testId="crc-triage-result-missing"
        />
      </div>

      <AssessmentQaSummary items={assessment.qa_summary} />

      {nodeResults.length > 0 ? (
        <section aria-label={COPY.nodeResults} className="crc-triage-result-nodes">
          <h3>{COPY.nodeResults}</h3>
          <div className="crc-triage-result-node-list">
            {nodeResults.map((result) => (
              <article
                className="crc-triage-result-node-card"
                data-testid="crc-triage-result-node-card"
                key={`${result.stage}-${result.title}`}
              >
                <div className="crc-triage-node-result-head">
                  <h4>{result.title}</h4>
                  <span>{formatRiskLabel(result.risk_level)}</span>
                </div>
                <p>{result.summary}</p>
                <div className="crc-triage-next-step">
                  <strong>{COPY.nextStep}</strong>
                  <span>{formatDispositionLabel(result.next_step)}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </section>
  );
}

interface AssessmentQaSummaryProps {
  items: Array<Record<string, unknown>>;
}

function AssessmentQaSummary({ items }: AssessmentQaSummaryProps) {
  const qaItems = normalizeQaSummary(items);

  if (qaItems.length === 0) {
    return null;
  }

  return (
    <section
      aria-label={COPY.qaSummary}
      className="crc-triage-result-qa"
      data-testid="crc-triage-result-qa"
    >
      <h3>{COPY.qaSummary}</h3>
      <div className="crc-triage-result-qa-list">
        {qaItems.map((item, index) => {
          const question = item.question && item.question.trim().length > 0
            ? item.question
            : item.question_id;

          return (
            <article
              className="crc-triage-result-qa-item"
              data-testid="crc-triage-result-qa-item"
              key={`${item.stage}-${item.question_id}-${index}`}
            >
              <span className="crc-triage-result-qa-stage">{formatStageTitle(item.stage)}</span>
              <p>{question}</p>
              <div className="crc-triage-result-qa-answer">
                <strong>{COPY.patientAnswer}</strong>
                <span>{item.answer}</span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

interface ResultListProps {
  label: string;
  items: string[];
  testId: string;
}

function ResultList({ label, items, testId }: ResultListProps) {
  return (
    <section className="crc-triage-result-list" data-testid={testId}>
      <h4>{label}</h4>
      {items.length > 0 ? (
        <ul>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p>{COPY.none}</p>
      )}
    </section>
  );
}

interface ProtocolStateSummaryProps {
  protocolState: CrcTriageProtocolState;
  currentQuestion: CrcTriageQuestion | null;
  nodeResults: CrcTriageNodeResult[];
  disabled: boolean;
  onAnswer: (option: string) => void;
}

function ProtocolStateSummary({
  protocolState,
  currentQuestion,
  nodeResults,
  disabled,
  onAnswer,
}: ProtocolStateSummaryProps) {
  const stage = typeof protocolState.stage === "string" ? protocolState.stage : currentQuestion?.stage;
  const stageTitle = stage ? STAGE_TITLES[stage] : null;
  const stageIndex = stage ? STAGE_SEQUENCE.indexOf(stage as (typeof STAGE_SEQUENCE)[number]) : -1;
  const currentStep = stageIndex >= 0 ? stageIndex + 1 : 1;
  const totalSteps = STAGE_SEQUENCE.length;
  const progressPercent = Math.round((currentStep / totalSteps) * 100);

  return (
    <section
      aria-label={"CRC \u4e13\u9879\u95ee\u8bca\u5f53\u524d\u8282\u70b9"}
      className="crc-triage-command-card"
      data-testid="crc-triage-command-card"
    >
      <div className="crc-triage-command-header">
        <span className="crc-triage-stage-pill" data-testid="crc-triage-stage-progress">
          {"\u8282\u70b9"} {currentStep}/{totalSteps}
        </span>
        <div className="crc-triage-command-heading">
          <h3>{stageTitle ?? COPY.currentQuestion}</h3>
          <p>{"\u6bcf\u6b21\u53ea\u56de\u7b54\u5f53\u524d\u95ee\u9898\uff0c\u7cfb\u7edf\u4f1a\u6309\u8282\u70b9\u63a8\u8fdb\u9884\u95ee\u8bca\u3002"}</p>
        </div>
      </div>
      <div
        aria-label={"\u4e13\u9879\u95ee\u8bca\u8fdb\u5ea6"}
        aria-valuemax={totalSteps}
        aria-valuemin={1}
        aria-valuenow={currentStep}
        className="crc-triage-progress-track"
        role="progressbar"
      >
        <span style={{ width: `${progressPercent}%` }} />
      </div>

      {currentQuestion ? (
        <section
          aria-label={COPY.currentQuestion}
          className="crc-triage-question-card"
          data-testid="crc-triage-current-question-card"
        >
          <span className="crc-triage-card-label">{COPY.currentQuestion}</span>
          <p>{currentQuestion.text}</p>
          {currentQuestion.options.length > 0 ? (
            <div className="crc-triage-option-grid">
              {currentQuestion.options.map((option) => (
                <Button
                  className="crc-triage-option-card"
                  key={option}
                  disabled={disabled}
                  onClick={() => onAnswer(option)}
                >
                  {option}
                </Button>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {nodeResults.length > 0 ? (
        <section aria-label={COPY.nodeResults} className="crc-triage-node-results">
          <h3>{COPY.nodeResults}</h3>
          <div className="clinical-panel-stack">
            {nodeResults.map((result) => (
              <article
                className="crc-triage-node-result-card"
                data-testid="crc-triage-node-result-card"
                key={`${result.stage}-${result.title}`}
              >
                <div className="crc-triage-node-result-head">
                  <h4>{result.title}</h4>
                  <span>{formatRiskLabel(result.risk_level)}</span>
                </div>
                <p>{result.summary}</p>
                <div className="crc-triage-next-step">
                  <strong>{COPY.nextStep}</strong>
                  <span>{formatDispositionLabel(result.next_step)}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </section>
  );
}

function getCrcTriageProtocolState(state: SessionState): CrcTriageProtocolState | null {
  const protocolState = state.findings.crc_triage_state;
  if (protocolState && typeof protocolState === "object" && !Array.isArray(protocolState)) {
    return protocolState as CrcTriageProtocolState;
  }
  return null;
}

function getCurrentQuestion(protocolState: CrcTriageProtocolState | null): CrcTriageQuestion | null {
  const question = protocolState?.current_question;
  if (!question || typeof question !== "object" || Array.isArray(question)) {
    return null;
  }

  const candidate = question as Record<string, unknown>;
  if (
    typeof candidate.id !== "string"
    || typeof candidate.stage !== "string"
    || typeof candidate.text !== "string"
  ) {
    return null;
  }

  return {
    id: candidate.id,
    stage: candidate.stage,
    text: candidate.text,
    options: Array.isArray(candidate.options)
      ? candidate.options.filter((option): option is string => typeof option === "string")
      : [],
  };
}

function getNodeResults(protocolState: CrcTriageProtocolState | null): CrcTriageNodeResult[] {
  if (!Array.isArray(protocolState?.node_results)) {
    return [];
  }

  return protocolState.node_results.filter((result): result is CrcTriageNodeResult => (
    typeof result.stage === "string"
    && typeof result.title === "string"
    && typeof result.risk_level === "string"
    && typeof result.summary === "string"
    && typeof result.next_step === "string"
  ));
}

function normalizeQaSummary(items: Array<Record<string, unknown>>): CrcTriageQaItem[] {
  if (!Array.isArray(items)) {
    return [];
  }

  return items.filter((item): item is CrcTriageQaItem => (
    item !== null
    && typeof item === "object"
    && !Array.isArray(item)
    && typeof item.stage === "string"
    && typeof item.question_id === "string"
    && (typeof item.question === "string" || item.question === null)
    && typeof item.answer === "string"
    && item.answer.trim().length > 0
  ));
}

function formatStageTitle(value: string): string {
  return STAGE_TITLES[value] ?? value;
}

function formatRiskLabel(value: string): string {
  return TRIAGE_RISK_LABELS[value] ?? value;
}

function formatDispositionLabel(value: string): string {
  return TRIAGE_DISPOSITION_LABELS[value] ?? value;
}

function riskTone(value: string): string {
  if (value === "high" || value.includes("\u9ad8") || value.includes("\u6025")) {
    return "high";
  }
  if (value === "medium" || value.includes("\u4e2d")) {
    return "medium";
  }
  if (value === "low" || value.includes("\u4f4e") || value.includes("\u5e73\u7a33")) {
    return "low";
  }
  return "neutral";
}
