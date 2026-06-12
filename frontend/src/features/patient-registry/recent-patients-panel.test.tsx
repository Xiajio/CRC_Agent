import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import type { PatientRegistryItem } from "../../app/api/types";
import { RecentPatientsPanel } from "./recent-patients-panel";

type RecentPatientsPanelProps = ComponentProps<typeof RecentPatientsPanel>;

function makePatient(overrides: Partial<PatientRegistryItem>): PatientRegistryItem {
  return {
    patient_id: 101,
    status: "draft",
    updated_at: "2026-04-16T00:00:00Z",
    ...overrides,
  };
}

const patients: PatientRegistryItem[] = [
  makePatient({
    patient_id: 101,
    tumor_location: "直肠",
    clinical_stage: "cT3N1M0",
    mmr_status: "pMMR",
  }),
  makePatient({
    patient_id: 202,
    tumor_location: null,
    clinical_stage: null,
    mmr_status: null,
  }),
];

function renderPanel(overrides: Partial<RecentPatientsPanelProps> = {}) {
  const props: RecentPatientsPanelProps = {
    items: patients,
    previewedPatientId: 202,
    isLoading: false,
    isLoadingPreview: false,
    error: null,
    onPreviewPatient: vi.fn(),
    ...overrides,
  };

  render(<RecentPatientsPanel {...props} />);
  return props;
}

describe("RecentPatientsPanel", () => {
  it("renders preview buttons with the recent-patient class contract and pressed state", () => {
    renderPanel();

    const inactiveButton = screen.getByRole("button", { name: "preview patient 101" });
    const activeButton = screen.getByRole("button", { name: "preview patient 202" });

    expect(inactiveButton).toHaveClass("clinical-list-item", "recent-patient-button");
    expect(inactiveButton).not.toHaveClass("clinical-step-current", "recent-patient-button-active");
    expect(inactiveButton).toHaveAttribute("aria-pressed", "false");
    expect(activeButton).toHaveClass(
      "clinical-list-item",
      "recent-patient-button",
      "clinical-step-current",
      "recent-patient-button-active",
    );
    expect(activeButton).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("直肠 / cT3N1M0 / pMMR")).toBeInTheDocument();
    expect(screen.getByText("暂无摘要")).toBeInTheDocument();
  });

  it("does not render inline style attributes on preview buttons", () => {
    renderPanel();

    for (const button of screen.getAllByRole("button", { name: /preview patient/ })) {
      expect(button).not.toHaveAttribute("style");
    }
  });

  it("calls onPreviewPatient with the selected patient id", () => {
    const onPreviewPatient = vi.fn();
    renderPanel({ onPreviewPatient });

    fireEvent.click(screen.getByRole("button", { name: "preview patient 101" }));

    expect(onPreviewPatient).toHaveBeenCalledWith(101);
  });

  it("renders the loading state without patient buttons", () => {
    renderPanel({ items: [], previewedPatientId: null, isLoading: true });

    expect(screen.getByText("正在加载最近患者...")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /preview patient/ })).not.toBeInTheDocument();
  });

  it("renders the empty state without patient buttons", () => {
    renderPanel({ items: [], previewedPatientId: null });

    expect(screen.getByText("暂无最近患者记录。")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /preview patient/ })).not.toBeInTheDocument();
  });
});
