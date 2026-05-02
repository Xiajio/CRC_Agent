import { describe, expect, it } from "vitest";

import { getVisibleCards } from "../../frontend/src/pages/visible-cards";

describe("getVisibleCards", () => {
  it("hides triage_card during outpatient active inquiry", () => {
    const visibleCards = getVisibleCards(
      {
        triage_card: { type: "triage_card", risk_level: "medium" },
        patient_card: { patient_id: "001" },
      },
      {
        isDatabaseDetailActive: false,
        encounterTrack: "outpatient_triage",
        activeInquiry: true,
      },
    );

    expect(visibleCards).toEqual({
      patient_card: { patient_id: "001" },
    });
  });

  it("keeps triage_card after inquiry completes", () => {
    const visibleCards = getVisibleCards(
      {
        triage_card: { type: "triage_card", risk_level: "medium" },
      },
      {
        isDatabaseDetailActive: false,
        encounterTrack: "outpatient_triage",
        activeInquiry: false,
      },
    );

    expect(visibleCards).toEqual({
      triage_card: { type: "triage_card", risk_level: "medium" },
    });
  });

  it("still removes overlapping database cards while preserving the triage gating rule", () => {
    const visibleCards = getVisibleCards(
      {
        triage_card: { type: "triage_card", risk_level: "medium" },
        patient_card: { patient_id: "001" },
        decision_card: { plan: "A" },
      },
      {
        isDatabaseDetailActive: true,
        encounterTrack: "outpatient_triage",
        activeInquiry: true,
      },
    );

    expect(visibleCards).toEqual({
      decision_card: { plan: "A" },
    });
  });
});
