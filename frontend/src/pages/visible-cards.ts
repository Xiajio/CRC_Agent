import type { JsonObject } from "../app/api/types";

const OVERLAPPING_DATABASE_CARD_TYPES = new Set([
  "patient_card",
  "imaging_card",
  "pathology_slide_card",
]);

type VisibleCardsOptions = {
  isDatabaseDetailActive: boolean;
  encounterTrack: string | null;
  activeInquiry: boolean;
};

export function getVisibleCards(
  cards: Record<string, JsonObject>,
  options: VisibleCardsOptions,
): Record<string, JsonObject> {
  const { isDatabaseDetailActive, encounterTrack, activeInquiry } = options;

  return Object.fromEntries(
    Object.entries(cards).filter(([cardType]) => {
      if (isDatabaseDetailActive && OVERLAPPING_DATABASE_CARD_TYPES.has(cardType)) {
        return false;
      }

      if (cardType === "triage_card" && encounterTrack === "outpatient_triage" && activeInquiry) {
        return false;
      }

      return true;
    }),
  ) as Record<string, JsonObject>;
}
