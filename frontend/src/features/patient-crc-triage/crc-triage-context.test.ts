import type { SessionState } from "../../app/api/types";
import {
  CRC_TRIAGE_START_PROMPT,
  CRC_TRIAGE_SUBFLOW,
  buildCrcTriageAssessmentDraft,
  buildCrcTriageContext,
  hasCompletedCrcTriage,
} from "./crc-triage-context";

function stateWithFindings(findings: Record<string, unknown>): SessionState {
  return {
    sessionId: "session-123",
    threadId: "thread-123",
    snapshotVersion: 1,
    runtime: null,
    messages: [],
    messagesTotal: 0,
    messagesNextBeforeCursor: null,
    cards: {},
    roadmap: [],
    findings,
    patientProfile: null,
    stage: null,
    references: [],
    plan: [],
    critic: null,
    safetyAlert: null,
    assessmentDraft: null,
    caseDatabasePatientId: null,
    registryPatientId: null,
    currentPatientId: null,
    uploadedAssets: {},
    contextMaintenance: null,
    contextState: null,
    statusNode: null,
    lastError: null,
    activeRunId: null,
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
    streamingMessageCursors: {},
    eventLog: [],
  };
}

describe("crc triage context helpers", () => {
  it("builds start context for the crc triage subflow", () => {
    expect(CRC_TRIAGE_SUBFLOW).toBe("crc_triage");
    expect(CRC_TRIAGE_START_PROMPT).toContain("CRC");
    expect(buildCrcTriageContext("start")).toEqual({
      patient_subflow: "crc_triage",
      crc_triage: {
        action: "start",
        interaction_source: "patient_crc_triage_tab",
      },
    });
  });

  it("detects completed crc triage findings", () => {
    expect(
      hasCompletedCrcTriage(
        stateWithFindings({
          source_subflow: "crc_triage",
          active_inquiry: false,
          triage_summary: "\u4fbf\u8840\u6301\u7eed\u4e24\u5468\uff0c\u5efa\u8bae\u95e8\u8bca\u8bc4\u4f30\u3002",
          triage_risk_level: "medium",
          triage_disposition: "urgent_gi_clinic",
        }),
      ),
    ).toBe(true);
  });

  it("builds the exact save payload draft for a bleeding example", () => {
    const state = stateWithFindings({
      source_subflow: "crc_triage",
      active_inquiry: false,
      triage_summary: "\u60a3\u8005\u8fd1\u4e24\u5468\u53cd\u590d\u4fbf\u8840\uff0c\u4f34\u6392\u4fbf\u4e60\u60ef\u6539\u53d8\u3002",
      triage_risk_level: "medium",
      triage_disposition: "urgent_gi_clinic",
      known_crc_signals: {
        rectal_bleeding: true,
        bowel_habit_change: true,
        weight_loss: false,
      },
      triage_suggested_tests: ["\u80a0\u955c", "\u8840\u5e38\u89c4", "\u7caa\u4fbf\u9690\u8840"],
      missing_critical_data: ["\u5bb6\u65cf\u53f2", "\u8d2b\u8840\u6307\u6807"],
      symptom_snapshot: {
        chief_symptoms: "\u53cd\u590d\u4fbf\u8840",
        symptom_focus: "\u4fbf\u8840\u4e0e\u6392\u4fbf\u4e60\u60ef\u6539\u53d8",
      },
    });

    expect(buildCrcTriageAssessmentDraft(state)).toEqual({
      record_type: "crc_triage_assessment",
      chief_complaint: "\u53cd\u590d\u4fbf\u8840",
      symptom_group: "\u4fbf\u8840\u4e0e\u6392\u4fbf\u4e60\u60ef\u6539\u53d8",
      risk_level: "medium",
      disposition: "urgent_gi_clinic",
      red_flags: ["bowel_habit_change", "rectal_bleeding"],
      known_crc_signals: {
        rectal_bleeding: true,
        bowel_habit_change: true,
        weight_loss: false,
      },
      suggested_tests: ["\u80a0\u955c", "\u8840\u5e38\u89c4", "\u7caa\u4fbf\u9690\u8840"],
      missing_information: ["\u5bb6\u65cf\u53f2", "\u8d2b\u8840\u6307\u6807"],
      qa_summary: [],
      patient_summary: "\u60a3\u8005\u8fd1\u4e24\u5468\u53cd\u590d\u4fbf\u8840\uff0c\u4f34\u6392\u4fbf\u4e60\u60ef\u6539\u53d8\u3002",
      next_step: "urgent_gi_clinic",
      source_session_id: "session-123",
      source_subflow: "crc_triage",
    });
  });

  it("returns null before crc triage completion", () => {
    expect(
      buildCrcTriageAssessmentDraft(
        stateWithFindings({
          source_subflow: "crc_triage",
          active_inquiry: true,
          triage_summary: "\u4ecd\u9700\u8ffd\u95ee",
          triage_risk_level: "pending",
          triage_disposition: "pending",
        }),
      ),
    ).toBeNull();
  });

  it("does not treat blank completed fields as a completed assessment", () => {
    const state = stateWithFindings({
      source_subflow: "crc_triage",
      active_inquiry: false,
      triage_summary: " ",
      triage_risk_level: "medium",
      triage_disposition: "urgent_gi_clinic",
    });

    expect(hasCompletedCrcTriage(state)).toBe(false);
    expect(buildCrcTriageAssessmentDraft(state)).toBeNull();
  });
});
