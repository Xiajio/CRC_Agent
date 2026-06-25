import type { CrcTriageAssessmentPayload, SessionState } from "../../app/api/types";

export const CRC_TRIAGE_SUBFLOW = "crc_triage";
export const CRC_TRIAGE_START_PROMPT = "\u6211\u60f3\u8fdb\u884c CRC \u4e13\u9879\u9884\u95ee\u8bca\uff0c\u8bf7\u6309\u7ed3\u6784\u5316\u95ee\u9898\u5f15\u5bfc\u6211\u5b8c\u6210\u3002";

export type CrcTriageAction = "start" | "answer" | "save";

export function buildCrcTriageContext(
  action: CrcTriageAction,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    patient_subflow: CRC_TRIAGE_SUBFLOW,
    crc_triage: {
      action,
      interaction_source: "patient_crc_triage_tab",
      ...extra,
    },
  };
}

export function hasCompletedCrcTriage(state: SessionState): boolean {
  const findings = state.findings;

  return (
    findings.source_subflow === CRC_TRIAGE_SUBFLOW
    && findings.active_inquiry === false
    && hasText(findings.triage_summary)
    && hasText(findings.triage_risk_level)
    && hasText(findings.triage_disposition)
  );
}

export function buildCrcTriageAssessmentDraft(state: SessionState): CrcTriageAssessmentPayload | null {
  if (!hasCompletedCrcTriage(state)) {
    return null;
  }

  const findings = state.findings;
  const knownCrcSignals = objectRecord(findings.known_crc_signals);
  const symptomSnapshot = objectRecord(findings.symptom_snapshot);
  const disposition = stringOrFallback(findings.triage_disposition, "observe_followup");

  return {
    record_type: "crc_triage_assessment",
    chief_complaint: stringOrFallback(
      symptomSnapshot.chief_symptoms,
      "\u60a3\u8005\u5b8c\u6210 CRC \u4e13\u9879\u9884\u95ee\u8bca",
    ),
    symptom_group: stringOrFallback(
      symptomSnapshot.symptom_focus,
      "CRC\u76f8\u5173\u95e8\u8bca\u5206\u8bca",
    ),
    risk_level: stringOrFallback(findings.triage_risk_level, "unknown"),
    disposition,
    red_flags: Object.entries(knownCrcSignals)
      .filter(([, value]) => value === true)
      .map(([key]) => key)
      .sort(),
    known_crc_signals: knownCrcSignals,
    suggested_tests: stringArrayOrEmpty(findings.triage_suggested_tests),
    missing_information: stringArrayOrEmpty(findings.missing_critical_data),
    qa_summary: [],
    patient_summary: stringOrFallback(
      findings.triage_summary,
      "\u5df2\u5b8c\u6210 CRC \u4e13\u9879\u9884\u95ee\u8bca\u3002",
    ),
    next_step: disposition,
    source_session_id: state.sessionId ?? "",
    source_subflow: CRC_TRIAGE_SUBFLOW,
  };
}

function objectRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return { ...(value as Record<string, unknown>) };
  }
  return {};
}

function stringArrayOrEmpty(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function stringOrFallback(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : fallback;
}

function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
