import { useMemo } from "react";

import type { CrcTriageAssessmentPayload, SessionState } from "../../app/api/types";
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
  suggestedTests: "\u5efa\u8bae\u68c0\u67e5",
  missingInfo: "\u5f85\u8865\u5145\u4fe1\u606f",
  risk: "\u98ce\u9669\u5206\u5c42",
  disposition: "\u5efa\u8bae\u53bb\u5411",
  none: "\u6682\u65e0",
};

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
  const isSaving = saveStatus === "saving";

  function handleStart() {
    onStart(CRC_TRIAGE_START_PROMPT, buildCrcTriageContext("start"));
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

      {assessmentDraft ? (
        <div className="clinical-panel-stack">
          <section aria-label={COPY.completedTitle}>
            <h3>{COPY.completedTitle}</h3>
            <p className="clinical-copy clinical-copy-tight" data-testid="crc-triage-summary">
              {assessmentDraft.patient_summary}
            </p>
            <dl className="clinical-copy clinical-copy-tight" data-testid="crc-triage-draft-fields">
              <dt>{COPY.risk}</dt>
              <dd>{assessmentDraft.risk_level}</dd>
              <dt>{COPY.disposition}</dt>
              <dd>{assessmentDraft.disposition}</dd>
              <dt>{COPY.suggestedTests}</dt>
              <dd>{formatList(assessmentDraft.suggested_tests)}</dd>
              <dt>{COPY.missingInfo}</dt>
              <dd>{formatList(assessmentDraft.missing_information)}</dd>
            </dl>
          </section>
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

function formatList(items: string[]): string {
  return items.length > 0 ? items.join("\u3001") : COPY.none;
}
