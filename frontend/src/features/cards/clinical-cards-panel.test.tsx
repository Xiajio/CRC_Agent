import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ClinicalCardsPanel } from "./clinical-cards-panel";

describe("ClinicalCardsPanel", () => {
  it("renders an empty state by default when no cards exist", () => {
    render(<ClinicalCardsPanel cards={{}} selectedCardType={null} emptyMessage="No medical cards." />);

    expect(screen.getByTestId("clinical-card-empty-state")).toHaveTextContent("No medical cards.");
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

    expect(screen.queryByTestId("clinical-card-empty-state")).not.toBeInTheDocument();
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
});
