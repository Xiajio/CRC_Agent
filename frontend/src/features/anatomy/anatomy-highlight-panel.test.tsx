import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AnatomyHighlightPanel } from "./anatomy-highlight-panel";

describe("AnatomyHighlightPanel", () => {
  it("highlights the resolved segment and submits a region prompt with patient context", () => {
    const onPromptRequest = vi.fn();

    render(
      <AnatomyHighlightPanel
        detail={{ patient_id: 7, tumor_location: "乙状结肠" }}
        patientContext={{ registry_patient_id: 7, case_database_patient_id: "093" }}
        onPromptRequest={onPromptRequest}
      />,
    );

    expect(screen.getByText("解剖定位")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "结直肠分段示意图" })).toBeInTheDocument();
    expect(screen.getAllByText("乙状结肠").length).toBeGreaterThan(0);

    const sigmoidButton = screen.getByRole("button", { name: "乙状结肠" });
    expect(sigmoidButton).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(sigmoidButton);

    expect(onPromptRequest).toHaveBeenCalledWith(
      "请针对乙状结肠病灶给出分期与下一步检查建议。",
      {
        registry_patient_id: 7,
        case_database_patient_id: "093",
        anatomy_region_code: "sigmoid_colon",
        anatomy_region_label: "乙状结肠",
        icd_o_topography: "C18.7",
      },
    );
  });

  it("renders a broad colon fallback without enabling a fake precise label", () => {
    render(<AnatomyHighlightPanel detail={{ tumor_location: "colon" }} />);

    expect(screen.getByText("结肠（未细分）")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "升结肠" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "直肠" })).toHaveAttribute("aria-pressed", "false");
  });
});
