import type { JsonObject } from "../../app/api/types";
import { cardTitle } from "./card-renderers-extended";

export type ClinicalCardTone = "blue" | "green" | "red";

export type ClinicalCardViewModel = {
  cardType: string;
  title: string;
  tone: ClinicalCardTone;
  payload: JsonObject;
};

export function clinicalToneForCard(cardType: string, index: number): ClinicalCardTone {
  const normalizedType = cardType.toLowerCase();
  if (normalizedType.includes("recommend") || normalizedType.includes("risk")) {
    return "red";
  }
  if (normalizedType.includes("evidence") || normalizedType.includes("reference")) {
    return "green";
  }
  const toneIndex = index % 3;
  return toneIndex === 0 ? "blue" : toneIndex === 1 ? "green" : "red";
}

export function toClinicalCardViewModels(cards: Record<string, JsonObject>): ClinicalCardViewModel[] {
  return Object.entries(cards).map(([cardType, payload], index) => ({
    cardType,
    title: cardTitle(cardType, payload),
    tone: clinicalToneForCard(cardType, index),
    payload,
  }));
}
