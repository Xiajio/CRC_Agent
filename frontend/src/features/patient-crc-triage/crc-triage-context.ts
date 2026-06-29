import type {
  CrcTriageAssessmentPayload,
  CrcTriageNodeResult,
  CrcTriageQaItem,
  CrcTriageProtocolState,
  SessionState,
} from "../../app/api/types";

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
  const protocolState = getCrcTriageProtocolState(state);
  const protocolAssessment = objectRecord(protocolState?.assessment);

  if (protocolAssessment.record_type === "crc_triage_assessment") {
    const normalizedAssessment = normalizeProtocolAssessment(
      protocolAssessment,
      protocolState,
      state.sessionId ?? "",
    );
    if (normalizedAssessment) {
      return normalizedAssessment;
    }
  }

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
    qa_summary: protocolQaSummary(protocolState),
    node_results: protocolNodeResults(protocolState),
    protocol_state: sanitizedProtocolState(protocolState),
    patient_summary: stringOrFallback(
      findings.triage_summary,
      "\u5df2\u5b8c\u6210 CRC \u4e13\u9879\u9884\u95ee\u8bca\u3002",
    ),
    next_step: disposition,
    source_session_id: state.sessionId ?? "",
    source_subflow: CRC_TRIAGE_SUBFLOW,
  };
}

function getCrcTriageProtocolState(state: SessionState): CrcTriageProtocolState | null {
  const protocolState = state.findings.crc_triage_state;
  if (protocolState && typeof protocolState === "object" && !Array.isArray(protocolState)) {
    return protocolState as CrcTriageProtocolState;
  }
  return null;
}

function sanitizedProtocolState(protocolState: CrcTriageProtocolState | null): Record<string, unknown> {
  if (!protocolState) {
    return {};
  }
  const snapshot: Record<string, unknown> = { ...protocolState };
  delete snapshot.assessment;
  return snapshot;
}

function normalizeProtocolAssessment(
  assessment: Record<string, unknown>,
  protocolState: CrcTriageProtocolState | null,
  sourceSessionId: string,
): CrcTriageAssessmentPayload | null {
  const chiefComplaint = requiredString(assessment.chief_complaint);
  const symptomGroup = requiredString(assessment.symptom_group);
  const riskLevel = requiredString(assessment.risk_level);
  const disposition = requiredString(assessment.disposition);
  const patientSummary = requiredString(assessment.patient_summary);
  const nextStep = requiredString(assessment.next_step);

  if (
    !chiefComplaint
    || !symptomGroup
    || !riskLevel
    || !disposition
    || !patientSummary
    || !nextStep
    || assessment.source_subflow !== CRC_TRIAGE_SUBFLOW
  ) {
    return null;
  }

  return {
    record_type: "crc_triage_assessment",
    chief_complaint: chiefComplaint,
    symptom_group: symptomGroup,
    risk_level: riskLevel,
    disposition,
    red_flags: stringArrayOrEmpty(assessment.red_flags),
    known_crc_signals: objectRecord(assessment.known_crc_signals),
    suggested_tests: stringArrayOrEmpty(assessment.suggested_tests),
    missing_information: stringArrayOrEmpty(assessment.missing_information),
    qa_summary: protocolQaSummary(protocolState, assessment),
    node_results: protocolNodeResults(protocolState, assessment),
    protocol_state: sanitizedProtocolState(protocolState),
    patient_summary: patientSummary,
    next_step: nextStep,
    source_session_id: sourceSessionId,
    source_subflow: CRC_TRIAGE_SUBFLOW,
  };
}

function protocolQaSummary(
  protocolState: CrcTriageProtocolState | null,
  assessment: Record<string, unknown> = {},
): CrcTriageQaItem[] {
  if (Array.isArray(protocolState?.qa_summary)) {
    return normalizeQaItems(protocolState.qa_summary);
  }
  if (Array.isArray(assessment.qa_summary)) {
    return normalizeQaItems(assessment.qa_summary);
  }
  return [];
}

function protocolNodeResults(
  protocolState: CrcTriageProtocolState | null,
  assessment: Record<string, unknown> = {},
): CrcTriageNodeResult[] {
  if (Array.isArray(protocolState?.node_results)) {
    return normalizeNodeResults(protocolState.node_results);
  }
  if (Array.isArray(assessment.node_results)) {
    return normalizeNodeResults(assessment.node_results);
  }
  return [];
}

function normalizeQaItems(items: unknown[]): CrcTriageQaItem[] {
  return items.filter((item): item is CrcTriageQaItem => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return false;
    }
    const candidate = item as Record<string, unknown>;
    return (
      typeof candidate.stage === "string"
      && typeof candidate.question_id === "string"
      && (typeof candidate.question === "string" || candidate.question === null)
      && typeof candidate.answer === "string"
    );
  });
}

function normalizeNodeResults(items: unknown[]): CrcTriageNodeResult[] {
  return items.filter((item): item is CrcTriageNodeResult => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return false;
    }
    const candidate = item as Record<string, unknown>;
    return (
      typeof candidate.stage === "string"
      && typeof candidate.title === "string"
      && typeof candidate.risk_level === "string"
      && typeof candidate.summary === "string"
      && typeof candidate.next_step === "string"
    );
  });
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

function requiredString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
