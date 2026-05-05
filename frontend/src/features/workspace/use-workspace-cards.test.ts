import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { FrontendMessage, JsonObject } from "../../app/api/types";
import {
  deriveWorkspaceCards,
  useWorkspaceCards,
} from "./use-workspace-cards";

type TestSessionState = {
  cards: Record<string, JsonObject>;
  messages: FrontendMessage[];
  findings: Record<string, unknown>;
};

function makeSessionState(overrides: Partial<TestSessionState> = {}): TestSessionState {
  return {
    cards: {},
    messages: [],
    findings: {},
    ...overrides,
  };
}

describe("deriveWorkspaceCards", () => {
  it("merges patient inline cards, strips triage_question_card, and preserves active triage question", () => {
    const patient = makeSessionState({
      cards: {
        triage_card: { type: "triage_card", title: "Triage" },
        triage_question_card: {
          type: "triage_question_card",
          question_id: "question-in-state",
        },
        imaging_card: { type: "imaging_card", title: "Patient imaging" },
      },
      findings: {
        encounter_track: "outpatient_triage",
        active_inquiry: true,
      },
      messages: [
        {
          cursor: "1",
          type: "ai",
          content: "triage follow-up",
          assetRefs: [],
          inlineCards: [
            {
              cardType: "patient_card",
              payload: {
                type: "patient_card",
                title: "Patient from inline",
              },
            },
            {
              cardType: "triage_question_card",
              payload: {
                type: "triage_question_card",
                question_id: "question-inline",
              },
            },
          ],
        },
      ],
    });

    const doctor = makeSessionState({
      cards: {
        patient_card: {
          type: "patient_card",
          title: "Doctor patient",
        },
        triage_card: {
          type: "triage_card",
          title: "Doctor triage",
        },
      },
      findings: {
        encounter_track: null,
        active_inquiry: false,
      },
    });

    const { patientVisibleCards, doctorVisibleCards, activePatientTriageQuestionId } = deriveWorkspaceCards({
      patient,
      doctor,
      isDatabaseDetailActive: false,
    });

    expect(activePatientTriageQuestionId).toBe("question-inline");
    expect(patientVisibleCards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Patient from inline",
      },
      imaging_card: { type: "imaging_card", title: "Patient imaging" },
    });
    expect(doctorVisibleCards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Doctor patient",
      },
      triage_card: {
        type: "triage_card",
        title: "Doctor triage",
      },
    });
  });

  it("hides triage cards for patient when active triage inquiry is running", () => {
    const patient = makeSessionState({
      cards: {
        triage_card: {
          type: "triage_card",
          title: "Triage",
        },
        patient_card: {
          type: "patient_card",
          title: "Patient background",
        },
      },
      findings: {
        encounter_track: "outpatient_triage",
        active_inquiry: true,
      },
    });

    const doctor = makeSessionState({
      cards: {
        triage_card: {
          type: "triage_card",
          title: "Doctor triage",
        },
      },
    });

    const { patientVisibleCards } = deriveWorkspaceCards({
      patient,
      doctor,
    });

    expect(patientVisibleCards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Patient background",
      },
    });
  });

  it("removes overlapping database cards when database detail view is active", () => {
    const patient = makeSessionState({
      cards: {
        patient_card: { type: "patient_card", title: "Patient background" },
        imaging_card: { type: "imaging_card", title: "Imaging" },
        pathology_slide_card: { type: "pathology_slide_card", title: "Pathology" },
      },
    });

    const doctor = makeSessionState({
      cards: {
        patient_card: { type: "patient_card", title: "Doctor patient" },
        pathology_slide_card: { type: "pathology_slide_card", title: "Doctor pathology" },
      },
    });

    const { patientVisibleCards, doctorVisibleCards } = deriveWorkspaceCards({
      patient,
      doctor,
      isDatabaseDetailActive: true,
    });

    expect(patientVisibleCards).toEqual({});
    expect(doctorVisibleCards).toEqual({});
  });
});

describe("useWorkspaceCards", () => {
  it("recomputes cards when triage visibility changes", () => {
    const patientWithoutInquiry = makeSessionState({
      cards: {
        triage_card: {
          type: "triage_card",
          title: "Triage",
        },
        triage_question_card: {
          type: "triage_question_card",
          question_id: "question-1",
        },
        patient_card: {
          type: "patient_card",
          title: "Patient card",
        },
      },
      findings: {
        encounter_track: "outpatient_triage",
        active_inquiry: false,
      },
    });

    const patientWithInquiry = {
      ...patientWithoutInquiry,
      findings: {
        encounter_track: "outpatient_triage",
        active_inquiry: true,
      },
    };

    const doctor = makeSessionState({
      cards: {
        patient_card: {
          type: "patient_card",
          title: "Doctor card",
        },
      },
    });

    const { result, rerender } = renderHook(
      ({ patient }) => useWorkspaceCards({ patient, doctor }),
      {
        initialProps: { patient: patientWithoutInquiry },
      },
    );

    expect(result.current.activePatientTriageQuestionId).toBeNull();
    expect(result.current.patientVisibleCards).toEqual({
      triage_card: {
        type: "triage_card",
        title: "Triage",
      },
      patient_card: {
        type: "patient_card",
        title: "Patient card",
      },
    });

    act(() => {
      rerender({ patient: patientWithInquiry });
    });

    expect(result.current.activePatientTriageQuestionId).toBe("question-1");
    expect(result.current.patientVisibleCards).toEqual({
      patient_card: {
        type: "patient_card",
        title: "Patient card",
      },
    });
  });
});
