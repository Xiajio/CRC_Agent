import { useMemo } from "react";

import { getVisibleCards } from "../../pages/visible-cards";
import type { FrontendMessage, JsonObject, SessionState } from "../../app/api/types";
import { readText } from "./workspace-flow-utils";

export function stripTriageQuestionCard(cards: Record<string, JsonObject>): Record<string, JsonObject> {
  const { triage_question_card: _triageQuestionCard, ...rest } = cards;
  return rest;
}

export function cardsWithInlineCards(
  cards: Record<string, JsonObject>,
  messages: FrontendMessage[],
): Record<string, JsonObject> {
  const merged = { ...cards };
  for (const message of messages) {
    for (const card of message.inlineCards ?? []) {
      merged[card.cardType] = card.payload;
    }
  }
  return merged;
}

export function triageVisibilityContext(findings: SessionState["findings"]): {
  encounterTrack: string | null;
  activeInquiry: boolean;
} {
  const record = findings as Record<string, unknown>;

  return {
    encounterTrack: readText(record.encounter_track),
    activeInquiry: Boolean(record.active_inquiry),
  };
}

export function readActiveTriageQuestionId(cards: SessionState["cards"]): string | null {
  const questionCard = cards.triage_question_card as Record<string, unknown> | undefined;
  return readText(questionCard?.question_id);
}

type WorkspaceCardsInput = {
  patient: Pick<SessionState, "cards" | "messages" | "findings">;
  doctor: Pick<SessionState, "cards" | "messages">;
  isDatabaseDetailActive?: boolean;
};

export type WorkspaceCards = {
  patientVisibleCards: Record<string, JsonObject>;
  doctorVisibleCards: Record<string, JsonObject>;
  activePatientTriageQuestionId: string | null;
};

export function deriveWorkspaceCards({
  patient,
  doctor,
  isDatabaseDetailActive = false,
}: WorkspaceCardsInput): WorkspaceCards {
  const triageContext = triageVisibilityContext(patient.findings);
  const mergedPatientCards = cardsWithInlineCards(patient.cards, patient.messages);
  const mergedDoctorCards = cardsWithInlineCards(doctor.cards, doctor.messages);

  return {
    patientVisibleCards: stripTriageQuestionCard(
      getVisibleCards(mergedPatientCards, {
        encounterTrack: triageContext.encounterTrack,
        activeInquiry: triageContext.activeInquiry,
        isDatabaseDetailActive,
      }),
    ),
    doctorVisibleCards: getVisibleCards(mergedDoctorCards, {
      encounterTrack: null,
      activeInquiry: false,
      isDatabaseDetailActive,
    }),
    activePatientTriageQuestionId: triageContext.activeInquiry ? readActiveTriageQuestionId(mergedPatientCards) : null,
  };
}

export function useWorkspaceCards({
  patient,
  doctor,
  isDatabaseDetailActive = false,
}: WorkspaceCardsInput): WorkspaceCards {
  return useMemo(
    () =>
      deriveWorkspaceCards({
        patient,
        doctor,
        isDatabaseDetailActive,
      }),
    [doctor.cards, doctor.messages, isDatabaseDetailActive, patient.cards, patient.findings, patient.messages],
  );
}
