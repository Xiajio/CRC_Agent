import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { JsonObject } from "../../app/api/types";
import { renderCardContent } from "./card-renderers";

const reviewNotice = "\u9700\u4eba\u5de5\u590d\u6838";

function renderMedicalCard(cardType: string, data: Record<string, unknown>) {
  render(
    <div>
      {renderCardContent({
        cardType,
        payload: {
          type: cardType,
          data,
        } as JsonObject,
      })}
    </div>,
  );
}

describe("medical confidence card rendering", () => {
  it("formats tumor detection max confidence as a percentage", () => {
    renderMedicalCard("tumor_detection_card", {
      patient_id: "P-1001",
      max_confidence: 0.876,
    });

    expect(screen.getByText("87.6%")).toBeInTheDocument();
    expect(screen.queryByText("0.876")).not.toBeInTheDocument();
  });

  it("shows a review notice when tumor detection confidence is below threshold", () => {
    renderMedicalCard("tumor_detection_card", {
      patient_id: "P-1002",
      max_confidence: 0.72,
      confidence_threshold: 0.8,
    });

    expect(screen.getByText("72%")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.getByText(reviewNotice)).toBeInTheDocument();
  });

  it("shows a review notice for explicit tumor detection review flags without a threshold", () => {
    renderMedicalCard("tumor_detection_card", {
      patient_id: "P-1003",
      max_confidence: "87.6%",
      needs_review: true,
    });

    expect(screen.getByText("87.6%")).toBeInTheDocument();
    expect(screen.getByText(reviewNotice)).toBeInTheDocument();
  });

  it("formats pathology tumor probability and model confidence as percentages", () => {
    renderMedicalCard("pathology_card", {
      patient_id: "P-2001",
      prediction: "tumor",
      tumor_probability: 0.934,
      confidence: "0.812",
    });

    expect(screen.getByText("93.4%")).toBeInTheDocument();
    expect(screen.getByText("81.2%")).toBeInTheDocument();
  });

  it("uses pathology confidence, not tumor probability, for threshold review", () => {
    renderMedicalCard("pathology_card", {
      patient_id: "P-2002",
      prediction: "tumor",
      tumor_probability: 0.4,
      confidence: 0.91,
      confidence_threshold: 0.8,
    });

    expect(screen.getByText("40%")).toBeInTheDocument();
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.queryByText(reviewNotice)).not.toBeInTheDocument();
  });

  it("does not infer review status or render NaN for invalid confidence values", () => {
    renderMedicalCard("tumor_detection_card", {
      patient_id: "P-1004",
      max_confidence: "not reported",
      confidence_threshold: 0.8,
    });

    expect(screen.getByText("not reported")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.queryByText("NaN%")).not.toBeInTheDocument();
    expect(screen.queryByText(reviewNotice)).not.toBeInTheDocument();
  });
});
