import type { SessionState } from "../../app/api/types";
import {
  CRC_TRIAGE_START_PROMPT,
  CRC_TRIAGE_SUBFLOW,
  buildCrcTriageAssessmentDraft,
  buildCrcTriageAnswerContext,
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
    runTrace: null,
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

  it("promotes triage card question_id into crc triage answer context", () => {
    expect(
      buildCrcTriageAnswerContext({
        triage_interaction: {
          question_id: "vitals_shock_or_consciousness",
          field_key: "vitals_shock_or_consciousness",
          selection_mode: "single",
          selected_option_ids: ["option_0"],
          other_text: null,
        },
      }),
    ).toEqual({
      patient_subflow: "crc_triage",
      crc_triage: {
        action: "answer",
        interaction_source: "patient_crc_triage_tab",
        question_id: "vitals_shock_or_consciousness",
      },
    });
  });

  it("uses fallback question id when answer context omits question_id", () => {
    expect(
      buildCrcTriageAnswerContext({}, "vitals_heart_or_breathing"),
    ).toEqual({
      patient_subflow: "crc_triage",
      crc_triage: {
        action: "answer",
        interaction_source: "patient_crc_triage_tab",
        question_id: "vitals_heart_or_breathing",
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
      node_results: [],
      protocol_state: {},
    });
  });

  it("builds assessment draft from protocol assessment with protocol details", () => {
    const state = stateWithFindings({
      source_subflow: "crc_triage",
      active_inquiry: true,
      crc_triage_state: {
        stage: "vitals",
        qa_summary: [
          {
            stage: "vitals",
            question_id: "vitals_shock_or_consciousness",
            question: "\u662f\u5426\u6709\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\uff1f",
            answer: "\u6ca1\u6709",
          },
        ],
        node_results: [
          {
            stage: "vitals",
            title: "\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30",
            risk_level: "low",
            summary: "\u672a\u89c1\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\u7ebf\u7d22\u3002",
            next_step: "\u7ee7\u7eed\u75c7\u72b6\u8be2\u95ee",
          },
        ],
        assessment: {
          record_type: "crc_triage_assessment",
          chief_complaint: "\u4fbf\u8840",
          symptom_group: "\u4fbf\u8840",
          risk_level: "low",
          disposition: "routine_gi_clinic",
          red_flags: ["rectal_bleeding", 7],
          known_crc_signals: null,
          suggested_tests: ["\u80a0\u955c", 12],
          missing_information: "\u5bb6\u65cf\u53f2",
          qa_summary: [{ bad: true }],
          node_results: [{ bad: true }],
          patient_summary: "\u6709\u4fbf\u8840\uff0c\u6682\u65e0\u6025\u5371\u91cd\u7ebf\u7d22\u3002",
          next_step: "routine_gi_clinic",
          source_session_id: "older-session",
          source_subflow: "crc_triage",
        },
      },
    });

    const draft = buildCrcTriageAssessmentDraft(state);

    expect(draft).toEqual({
      record_type: "crc_triage_assessment",
      chief_complaint: "\u4fbf\u8840",
      symptom_group: "\u4fbf\u8840",
      risk_level: "low",
      disposition: "routine_gi_clinic",
      red_flags: ["rectal_bleeding"],
      known_crc_signals: {},
      suggested_tests: ["\u80a0\u955c"],
      missing_information: [],
      qa_summary: [
        {
          stage: "vitals",
          question_id: "vitals_shock_or_consciousness",
          question: "\u662f\u5426\u6709\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\uff1f",
          answer: "\u6ca1\u6709",
        },
      ],
      patient_summary: "\u6709\u4fbf\u8840\uff0c\u6682\u65e0\u6025\u5371\u91cd\u7ebf\u7d22\u3002",
      next_step: "routine_gi_clinic",
      source_session_id: "session-123",
      source_subflow: "crc_triage",
      node_results: [
        {
          stage: "vitals",
          title: "\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30",
          risk_level: "low",
          summary: "\u672a\u89c1\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\u7ebf\u7d22\u3002",
          next_step: "\u7ee7\u7eed\u75c7\u72b6\u8be2\u95ee",
        },
      ],
      protocol_state: {
        stage: "vitals",
        qa_summary: [
          {
            stage: "vitals",
            question_id: "vitals_shock_or_consciousness",
            question: "\u662f\u5426\u6709\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\uff1f",
            answer: "\u6ca1\u6709",
          },
        ],
        node_results: [
          {
            stage: "vitals",
            title: "\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30",
            risk_level: "low",
            summary: "\u672a\u89c1\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\u7ebf\u7d22\u3002",
            next_step: "\u7ee7\u7eed\u75c7\u72b6\u8be2\u95ee",
          },
        ],
      },
    });
    expect(draft?.protocol_state?.assessment).toBeUndefined();
  });

  it("returns null for a partial protocol assessment", () => {
    const state = stateWithFindings({
      source_subflow: "crc_triage",
      active_inquiry: true,
      crc_triage_state: {
        stage: "vitals",
        assessment: {
          record_type: "crc_triage_assessment",
        },
      },
    });

    expect(buildCrcTriageAssessmentDraft(state)).toBeNull();
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
