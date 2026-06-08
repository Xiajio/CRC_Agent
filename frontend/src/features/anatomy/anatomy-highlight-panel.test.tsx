import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AnatomyHighlightPanel } from "./anatomy-highlight-panel";

describe("AnatomyHighlightPanel", () => {
  it("highlights the whole-body overview and resolved segment, then submits a region prompt with patient context", () => {
    const onPromptRequest = vi.fn();

    render(
      <AnatomyHighlightPanel
        detail={{ patient_id: 7, tumor_location: "乙状结肠" }}
        patientContext={{ registry_patient_id: 7, case_database_patient_id: "093" }}
        onPromptRequest={onPromptRequest}
      />,
    );

    expect(screen.getByText("解剖定位")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "人体定位总览" })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "结直肠分段示意图" })).toBeInTheDocument();

    const wholeBodyRegion = screen.getByRole("button", { name: "腹盆腔结直肠定位区域" });
    expect(wholeBodyRegion).toHaveAttribute("aria-pressed", "true");
    expect(wholeBodyRegion).not.toHaveAttribute("aria-disabled", "true");

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

  it("submits the resolved segment prompt and context from the whole-body overview", () => {
    const onPromptRequest = vi.fn();

    render(
      <AnatomyHighlightPanel
        detail={{ patient_id: 7, tumor_location: "rectum" }}
        patientContext={{ registry_patient_id: 7, case_database_patient_id: "093" }}
        onPromptRequest={onPromptRequest}
      />,
    );

    const wholeBodyRegion = screen.getByRole("button", { name: "腹盆腔结直肠定位区域" });
    const expectedPrompt = "请针对直肠病灶给出分期与下一步检查建议。";
    const expectedContext = {
      registry_patient_id: 7,
      case_database_patient_id: "093",
      anatomy_region_code: "rectum",
      anatomy_region_label: "直肠",
      icd_o_topography: "C20",
    };

    fireEvent.click(wholeBodyRegion);

    expect(onPromptRequest).toHaveBeenCalledWith(expectedPrompt, expectedContext);

    onPromptRequest.mockClear();
    fireEvent.keyDown(wholeBodyRegion, { key: "Enter" });
    expect(onPromptRequest).toHaveBeenCalledWith(expectedPrompt, expectedContext);
  });

  it("preserves ambiguous aggregate context for broad colon whole-body fallback", () => {
    const onPromptRequest = vi.fn();

    render(
      <AnatomyHighlightPanel
        detail={{ patient_id: 7, tumor_location: "colon" }}
        patientContext={{ registry_patient_id: 7 }}
        onPromptRequest={onPromptRequest}
      />,
    );

    const wholeBodyRegion = screen.getByRole("button", { name: "腹盆腔结直肠定位区域" });

    fireEvent.click(wholeBodyRegion);

    expect(onPromptRequest).toHaveBeenCalledWith(
      "请结合结肠（未细分）的结直肠定位总结病灶位置与下一步检查建议。",
      {
        registry_patient_id: 7,
        anatomy_region_codes: [
          "cecum",
          "ascending_colon",
          "hepatic_flexure",
          "transverse_colon",
          "splenic_flexure",
          "descending_colon",
          "sigmoid_colon",
        ],
        anatomy_region_labels: ["盲肠", "升结肠", "肝曲", "横结肠", "脾曲", "降结肠", "乙状结肠"],
        icd_o_topographies: ["C18.0", "C18.2", "C18.3", "C18.4", "C18.5", "C18.6", "C18.7"],
        anatomy_region_scope: "colorectal_multi_segment",
        anatomy_region_summary: "结肠（未细分）",
      },
    );
    expect(onPromptRequest.mock.calls[0]?.[1]).not.toHaveProperty("anatomy_region_code");
    expect(onPromptRequest.mock.calls[0]?.[1]).not.toHaveProperty("anatomy_region_label");
    expect(onPromptRequest.mock.calls[0]?.[1]).not.toHaveProperty("icd_o_topography");
  });

  it("renders an inactive whole-body overview when no location signal is available", () => {
    const onPromptRequest = vi.fn();

    render(<AnatomyHighlightPanel detail={{ tumor_location: null }} onPromptRequest={onPromptRequest} />);

    const wholeBodyRegion = screen.getByRole("button", { name: "腹盆腔结直肠定位区域" });
    expect(wholeBodyRegion).toHaveAttribute("aria-pressed", "false");
    expect(wholeBodyRegion).toHaveAttribute("aria-disabled", "true");
    expect(wholeBodyRegion).toHaveAttribute("tabindex", "-1");
    fireEvent.click(wholeBodyRegion);
    expect(onPromptRequest).not.toHaveBeenCalled();
    expect(screen.getByText("暂未定位肿瘤分段")).toBeInTheDocument();
  });

  it("renders a broad colon fallback without enabling a fake precise label", () => {
    render(<AnatomyHighlightPanel detail={{ tumor_location: "colon" }} />);

    expect(screen.getByText("结肠（未细分）")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "腹盆腔结直肠定位区域" })).toHaveAttribute("aria-pressed", "true");
    for (const label of ["盲肠", "升结肠", "肝曲", "横结肠", "脾曲", "降结肠", "乙状结肠"]) {
      expect(screen.getByRole("button", { name: label })).toHaveAttribute("aria-pressed", "true");
    }
    expect(screen.getByRole("button", { name: "直肠" })).toHaveAttribute("aria-pressed", "false");
  });
});
