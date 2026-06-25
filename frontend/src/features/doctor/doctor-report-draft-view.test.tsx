import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { FrontendMessage } from "../../app/api/types";
import {
  buildDoctorReportDraftPrompt,
  buildDoctorReportPromptContext,
  DOCTOR_REPORT_DRAFT_ACTIONS,
} from "./doctor-report-draft-utils";
import { DoctorReportDraftView } from "./doctor-report-draft-view";

function message(overrides: Partial<FrontendMessage>): FrontendMessage {
  return {
    cursor: "1",
    type: "ai",
    content: "",
    assetRefs: [],
    ...overrides,
  };
}

describe("DoctorReportDraftView", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("disables report actions when no patient context is available", () => {
    render(
      <DoctorReportDraftView
        registryPatientId={null}
        caseDatabasePatientId={null}
        messages={[]}
        isStreaming={false}
        disabled={false}
        patientContext={null}
        onReportPromptRequest={vi.fn()}
      />,
    );

    expect(screen.getByTestId("doctor-report-draft-view")).toHaveClass("clinical-report-draft-dashboard");
    for (const action of DOCTOR_REPORT_DRAFT_ACTIONS) {
      expect(screen.getByRole("button", { name: action.title })).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: "导出 PDF" })).toBeDisabled();
  });

  it("submits report draft prompts with normalized patient context", () => {
    const onReportPromptRequest = vi.fn();
    render(
      <DoctorReportDraftView
        registryPatientId={42}
        caseDatabasePatientId="7"
        messages={[]}
        isStreaming={false}
        disabled={false}
        patientContext={{ registry_patient_id: "42", case_database_patient_id: 7 }}
        onReportPromptRequest={onReportPromptRequest}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "会诊报告草稿" }));

    expect(onReportPromptRequest).toHaveBeenCalledWith(
      buildDoctorReportDraftPrompt(DOCTOR_REPORT_DRAFT_ACTIONS[1]),
      buildDoctorReportPromptContext({
        registry_patient_id: 42,
        case_database_patient_id: "007",
      }),
    );
  });

  it("previews the latest assistant draft and prints it as a browser PDF path", () => {
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => undefined);
    render(
      <DoctorReportDraftView
        registryPatientId={42}
        caseDatabasePatientId="093"
        messages={[
          message({ cursor: "a1", content: "旧草稿" }),
          message({ cursor: "a2", content: "最新会诊报告草稿" }),
        ]}
        isStreaming={false}
        disabled={false}
        patientContext={{ registry_patient_id: 42, case_database_patient_id: "093" }}
        onReportPromptRequest={vi.fn()}
      />,
    );

    expect(screen.getByText("最新会诊报告草稿")).toBeInTheDocument();

    const exportButton = screen.getByRole("button", { name: "导出 PDF" });
    expect(exportButton).toBeEnabled();
    fireEvent.click(exportButton);

    expect(printSpy).toHaveBeenCalledTimes(1);
  });

  it("disables generation while a report draft is streaming", () => {
    render(
      <DoctorReportDraftView
        registryPatientId={42}
        caseDatabasePatientId="093"
        messages={[message({ content: "已有草稿" })]}
        isStreaming
        disabled={false}
        patientContext={{ registry_patient_id: 42, case_database_patient_id: "093" }}
        onReportPromptRequest={vi.fn()}
      />,
    );

    for (const action of DOCTOR_REPORT_DRAFT_ACTIONS) {
      expect(screen.getByRole("button", { name: action.title })).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: "导出 PDF" })).toBeDisabled();
  });
});
