import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ClinicalCardsPanel } from "./clinical-cards-panel";

describe("ClinicalCardsPanel", () => {
  it("renders an empty state by default when no cards exist", () => {
    render(<ClinicalCardsPanel cards={{}} selectedCardType={null} emptyMessage="No medical cards." />);

    expect(screen.getByTestId("clinical-empty-state")).toHaveTextContent("No medical cards.");
    expect(screen.getByTestId("clinical-empty-state")).toHaveTextContent("医疗卡片待生成");
    expect(screen.queryByText("cT3N1M0")).not.toBeInTheDocument();
    expect(screen.queryByText("FOLFOX")).not.toBeInTheDocument();
  });

  it("renders real cards from the supplied card map", () => {
    const { container } = render(
      <ClinicalCardsPanel
        cards={{
          evidence_card: { title: "Evidence", summary: "Retrieved guideline evidence." },
          recommendation_card: { title: "Recommendation", summary: "Review treatment options." },
        }}
        selectedCardType="recommendation_card"
      />,
    );

    expect(screen.queryByTestId("clinical-empty-state")).not.toBeInTheDocument();
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText("Recommendation")).toBeInTheDocument();
    expect(container.querySelector(".clinical-medical-card-selected")).toHaveAttribute("aria-current", "true");
  });

  it("renders decision card plan items that use step and rationale fields", () => {
    render(
      <ClinicalCardsPanel
        cards={{
          decision_card: {
            summary: "Stage III low rectal adenocarcinoma, pMMR, cT3N1M0.",
            treatment_plan: [
              {
                step: "Discuss total neoadjuvant therapy in multidisciplinary tumor board.",
                rationale: "cT3N1 low rectal cancer generally requires neoadjuvant treatment before surgery.",
              },
            ],
          },
        }}
        selectedCardType="decision_card"
      />,
    );

    expect(screen.getByText("Discuss total neoadjuvant therapy in multidisciplinary tumor board.")).toBeInTheDocument();
    expect(
      screen.getByText("cT3N1 low rectal cancer generally requires neoadjuvant treatment before surgery."),
    ).toBeInTheDocument();
    expect(screen.queryByText("暂无说明。")).not.toBeInTheDocument();
  });

  it("renders decision card follow_up string items from the backend payload", () => {
    render(
      <ClinicalCardsPanel
        cards={{
          decision_card: {
            summary: "Treatment decision summary.",
            follow_up: ["Repeat CEA every 3 months.", "Schedule surveillance CT."],
          },
        }}
        selectedCardType="decision_card"
      />,
    );

    expect(screen.getByText("Repeat CEA every 3 months.")).toBeInTheDocument();
    expect(screen.getByText("Schedule surveillance CT.")).toBeInTheDocument();
  });

  it("renders imaging card previews with embedded mime types", () => {
    render(
      <ClinicalCardsPanel
        cards={{
          imaging_card: {
            type: "imaging_card",
            data: {
              folder_name: "093",
              total_images: 1,
              images: [
                {
                  image_name: "slice_002.jpg",
                  image_base64: "preview-bytes",
                  image_mime_type: "image/jpeg",
                },
              ],
            },
          },
        }}
        selectedCardType="imaging_card"
      />,
    );

    const previewImages = screen.getAllByAltText("slice_002.jpg");
    expect(previewImages[0]).toHaveAttribute("src", "data:image/jpeg;base64,preview-bytes");
  });
});
