import { describe, expect, it } from "vitest";

import type { JsonObject } from "../../app/api/types";
import { cardTitle } from "./card-renderers-extended";
import { clinicalToneForCard, toClinicalCardViewModels } from "./clinical-card-view-model";

describe("clinical card view model adapter", () => {
  it("returns an empty array for empty input", () => {
    expect(toClinicalCardViewModels({})).toEqual([]);
  });

  it("derives card view model from evidence, recommendation, and assessment cards", () => {
    const cards = {
      recommendation_card: { title: "Recommended next steps", risk_level: "high" } as JsonObject,
      evidence_card: { title: "Supporting evidence summary" } as JsonObject,
      assessment: { stage: "assessment-summary" } as JsonObject,
    };
    const viewModels = toClinicalCardViewModels(cards);

    expect(viewModels).toEqual([
      {
        cardType: "recommendation_card",
        title: cardTitle("recommendation_card", cards.recommendation_card),
        tone: clinicalToneForCard("recommendation_card", 0),
        payload: cards.recommendation_card,
      },
      {
        cardType: "evidence_card",
        title: cardTitle("evidence_card", cards.evidence_card),
        tone: clinicalToneForCard("evidence_card", 1),
        payload: cards.evidence_card,
      },
      {
        cardType: "assessment",
        title: cardTitle("assessment", cards.assessment),
        tone: clinicalToneForCard("assessment", 2),
        payload: cards.assessment,
      },
    ]);
  });

  it("falls back to index-based tones when no keyword match", () => {
    const cards = {
      first_card: {} as JsonObject,
      second_card: {} as JsonObject,
      third_card: {} as JsonObject,
    };
    const viewModels = toClinicalCardViewModels(cards);

    expect(viewModels.map((vm) => vm.tone)).toEqual(["blue", "green", "red"]);
  });

  it("keeps report-class card titles in UTF-8 Chinese", () => {
    const viewModels = toClinicalCardViewModels({
      pathology_card: {} as JsonObject,
      pathology_slide_card: {} as JsonObject,
      radiomics_report_card: {} as JsonObject,
    });

    expect(viewModels.map((vm) => vm.title)).toEqual(["病理报告", "病理切片", "影像组学报告"]);
  });
});
