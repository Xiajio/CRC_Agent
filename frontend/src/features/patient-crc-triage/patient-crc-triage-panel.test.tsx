import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SessionState } from "../../app/api/types";
import { createInitialSessionState } from "../../app/store/stream-reducer";
import { CRC_TRIAGE_START_PROMPT } from "./crc-triage-context";
import { PatientCrcTriagePanel } from "./patient-crc-triage-panel";

function makeState(findings: Record<string, unknown> = {}): SessionState {
  return {
    ...createInitialSessionState(),
    sessionId: "patient-session",
    findings,
  };
}

function completedState(): SessionState {
  return makeState({
    source_subflow: "crc_triage",
    active_inquiry: false,
    triage_summary: "\u60a3\u8005\u8fd1\u4e24\u5468\u53cd\u590d\u4fbf\u8840\u3002",
    triage_risk_level: "medium",
    triage_disposition: "urgent_gi_clinic",
    triage_suggested_tests: ["\u80a0\u955c", "\u8840\u5e38\u89c4"],
    missing_critical_data: ["\u5bb6\u65cf\u53f2"],
    symptom_snapshot: {
      chief_symptoms: "\u53cd\u590d\u4fbf\u8840",
      symptom_focus: "\u4fbf\u8840",
    },
  });
}

describe("PatientCrcTriagePanel", () => {
  it("starts crc triage with the dedicated context", () => {
    const onStart = vi.fn();

    render(
      <PatientCrcTriagePanel
        sessionState={makeState()}
        disabled={false}
        saveStatus="idle"
        onStart={onStart}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("crc-triage-start"));

    expect(onStart).toHaveBeenCalledWith(CRC_TRIAGE_START_PROMPT, {
      patient_subflow: "crc_triage",
      crc_triage: {
        action: "start",
        interaction_source: "patient_crc_triage_tab",
      },
    });
  });

  it("routes the upload shortcut through the supplied callback", () => {
    const onUploadRequest = vi.fn();

    render(
      <PatientCrcTriagePanel
        sessionState={makeState()}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={onUploadRequest}
        onSaveAssessment={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("crc-triage-upload"));

    expect(onUploadRequest).toHaveBeenCalledTimes(1);
  });

  it("shows completed assessment details and saves the generated payload", () => {
    const onSaveAssessment = vi.fn();

    render(
      <PatientCrcTriagePanel
        sessionState={completedState()}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={onSaveAssessment}
      />,
    );

    expect(screen.getByTestId("crc-triage-summary")).toHaveTextContent("\u53cd\u590d\u4fbf\u8840");
    expect(screen.getByTestId("crc-triage-draft-fields")).toHaveTextContent("urgent_gi_clinic");

    fireEvent.click(screen.getByTestId("crc-triage-save"));

    expect(onSaveAssessment).toHaveBeenCalledWith(
      expect.objectContaining({
        record_type: "crc_triage_assessment",
        chief_complaint: "\u53cd\u590d\u4fbf\u8840",
        source_session_id: "patient-session",
        source_subflow: "crc_triage",
      }),
    );
  });

  it("surfaces save errors for retry", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={completedState()}
        disabled={false}
        saveStatus="error"
        saveErrorMessage="save failed"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("save failed");
    expect(screen.getByTestId("crc-triage-save")).not.toBeDisabled();
  });
});
