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

function completedStateWithProtocolResults(): SessionState {
  return makeState({
    source_subflow: "crc_triage",
    active_inquiry: false,
    triage_summary: "\u60a3\u8005\u8fd1\u4e24\u5468\u53cd\u590d\u4fbf\u8840\uff0c\u4f34\u6392\u4fbf\u4e60\u60ef\u6539\u53d8\u3002",
    triage_risk_level: "medium",
    triage_disposition: "urgent_gi_clinic",
    triage_suggested_tests: ["\u80a0\u955c", "\u8840\u5e38\u89c4"],
    missing_critical_data: ["\u5bb6\u65cf\u53f2"],
    symptom_snapshot: {
      chief_symptoms: "\u53cd\u590d\u4fbf\u8840",
      symptom_focus: "\u4fbf\u8840",
    },
    crc_triage_state: {
      stage: "final",
      qa_summary: [
        {
          stage: "vitals",
          question_id: "vitals_shock_or_consciousness",
          question:
            "\u6700\u8fd1\u6709\u6ca1\u6709\u51fa\u73b0\u5934\u6655\u3001\u773c\u524d\u53d1\u9ed1\u3001\u610f\u8bc6\u6a21\u7cca\u7684\u60c5\u51b5\uff1f",
          answer: "\u6ca1\u6709",
        },
        {
          stage: "red_flags",
          question_id: "red_flags_blood_loss",
          question: "\u4fbf\u8840\u91cf\u662f\u5426\u660e\u663e\u589e\u591a\uff1f",
          answer: "\u4e0d\u660e\u663e",
        },
      ],
      node_results: [
        {
          stage: "vitals",
          title: "\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30",
          risk_level: "\u751f\u547d\u4f53\u5f81\u5e73\u7a33",
          summary: "\u672a\u89c1\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\u7ebf\u7d22\u3002",
          next_step: "\u7ee7\u7eed\u75c7\u72b6\u8be2\u95ee",
        },
        {
          stage: "final",
          title: "\u8282\u70b96\uff1a\u7ec8\u70b9\u8f93\u51fa",
          risk_level: "\u4e2d\u98ce\u9669",
          summary: "\u5efa\u8bae\u5c3d\u5feb\u5b8c\u6210\u6d88\u5316\u95e8\u8bca\u8bc4\u4f30\u3002",
          next_step: "\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca",
        },
      ],
    },
  });
}

describe("PatientCrcTriagePanel", () => {
  const vitalsShockQuestion =
    "\u6700\u8fd1\u6709\u6ca1\u6709\u51fa\u73b0\u5934\u6655\u3001\u773c\u524d\u53d1\u9ed1\u3001\u610f\u8bc6\u6a21\u7cca\uff0c\u6216\u8005\u7a81\u7136\u51fa\u51b7\u6c57\u3001\u9762\u8272\u82cd\u767d\u7684\u60c5\u51b5\uff1f";

  it("renders crc-client aligned progress and current question", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "vitals",
            current_question: {
              id: "vitals_shock_or_consciousness",
              stage: "vitals",
              text: vitalsShockQuestion,
              options: ["\u6ca1\u6709", "\u6709", "\u4e0d\u6e05\u695a"],
            },
          },
        })}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.getByTestId("crc-triage-command-card")).toBeInTheDocument();
    expect(screen.getByTestId("crc-triage-stage-progress")).toHaveTextContent("节点 1/6");
    expect(screen.getByRole("progressbar", { name: "专项问诊进度" })).toHaveAttribute(
      "aria-valuenow",
      "1",
    );
    expect(screen.getByTestId("crc-triage-current-question-card")).toHaveTextContent("当前问题");
    expect(screen.getByText("\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30")).toBeInTheDocument();
    expect(screen.queryByText(/vitals ->/)).not.toBeInTheDocument();
    expect(screen.getByText(vitalsShockQuestion)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "\u6ca1\u6709" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "\u6709" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "\u4e0d\u6e05\u695a" })).toBeInTheDocument();
  });

  it("does not render the start button while a protocol question is active", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "vitals",
            current_question: {
              id: "vitals_shock_or_consciousness",
              stage: "vitals",
              text: vitalsShockQuestion,
              options: ["\u6ca1\u6709", "\u6709", "\u4e0d\u6e05\u695a"],
            },
          },
        })}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("crc-triage-start")).not.toBeInTheDocument();
    expect(screen.getByTestId("crc-triage-upload")).toBeInTheDocument();
  });

  it("ignores invalid option values while rendering valid options", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "vitals",
            current_question: {
              id: "vitals_shock_or_consciousness",
              stage: "vitals",
              text: vitalsShockQuestion,
              options: ["\u6ca1\u6709", 42, null, "\u6709"],
            },
          },
        })}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "\u6ca1\u6709" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "\u6709" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "42" })).not.toBeInTheDocument();
  });

  it("renders without crashing when protocol assessment is partial", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "vitals",
            assessment: {
              record_type: "crc_triage_assessment",
            },
          },
        })}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.getByTestId("crc-triage-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("crc-triage-save")).not.toBeInTheDocument();
  });

  it("answers the current crc triage question with question context", () => {
    const onStart = vi.fn();

    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "vitals",
            current_question: {
              id: "vitals_shock_or_consciousness",
              stage: "vitals",
              text: vitalsShockQuestion,
              options: ["\u6ca1\u6709", "\u6709", "\u4e0d\u6e05\u695a"],
            },
          },
        })}
        disabled={false}
        saveStatus="idle"
        onStart={onStart}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "\u6ca1\u6709" }));

    expect(onStart).toHaveBeenCalledWith("\u6ca1\u6709", {
      patient_subflow: "crc_triage",
      crc_triage: {
        action: "answer",
        interaction_source: "patient_crc_triage_tab",
        question_id: "vitals_shock_or_consciousness",
      },
    });
  });

  it("renders node result summaries from protocol state", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={makeState({
          source_subflow: "crc_triage",
          active_inquiry: true,
          crc_triage_state: {
            stage: "red_flags",
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
        })}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.getByTestId("crc-triage-node-result-card")).toBeInTheDocument();
    expect(screen.getByText("\u8282\u70b91\uff1a\u751f\u547d\u4f53\u5f81\u8bc4\u4f30")).toBeInTheDocument();
    expect(screen.getByText("\u672a\u89c1\u4f11\u514b\u6216\u610f\u8bc6\u6539\u53d8\u7ebf\u7d22\u3002")).toBeInTheDocument();
    expect(screen.getByText("\u4f4e\u98ce\u9669")).toBeInTheDocument();
    expect(screen.getByText(/继续症状询问/)).toBeInTheDocument();
  });

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

    expect(screen.getByTestId("crc-triage-result-summary")).toHaveTextContent("\u53cd\u590d\u4fbf\u8840");
    expect(screen.getByTestId("crc-triage-result-disposition")).toHaveTextContent(
      "\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca",
    );

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

  it("renders the completed assessment as a visual result card", () => {
    render(
      <PatientCrcTriagePanel
        sessionState={completedStateWithProtocolResults()}
        disabled={false}
        saveStatus="idle"
        onStart={vi.fn()}
        onUploadRequest={vi.fn()}
        onSaveAssessment={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("crc-triage-command-card")).not.toBeInTheDocument();
    expect(screen.queryByTestId("crc-triage-draft-fields")).not.toBeInTheDocument();
    expect(screen.getByTestId("crc-triage-result-card")).toBeInTheDocument();
    expect(screen.getByTestId("crc-triage-result-summary")).toHaveTextContent(
      "\u6392\u4fbf\u4e60\u60ef\u6539\u53d8",
    );
    expect(screen.getByTestId("crc-triage-result-risk")).toHaveTextContent("\u4e2d\u98ce\u9669");
    expect(screen.getByTestId("crc-triage-result-disposition")).toHaveTextContent(
      "\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca",
    );
    expect(screen.getByTestId("crc-triage-result-tests")).toHaveTextContent("\u80a0\u955c");
    expect(screen.getByTestId("crc-triage-result-missing")).toHaveTextContent("\u5bb6\u65cf\u53f2");
    expect(screen.getByTestId("crc-triage-result-qa")).toHaveTextContent("\u95ee\u8bca\u8bb0\u5f55");
    expect(screen.getAllByTestId("crc-triage-result-qa-item")).toHaveLength(2);
    expect(screen.getByText(/头晕/)).toBeInTheDocument();
    expect(screen.getByText("\u6ca1\u6709")).toBeInTheDocument();
    expect(screen.getAllByTestId("crc-triage-result-node-card")).toHaveLength(2);
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
