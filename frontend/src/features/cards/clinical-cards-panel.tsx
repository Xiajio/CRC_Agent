import { useRef } from "react";

import type { JsonObject } from "../../app/api/types";
import { useHighlightFlash } from "../../components/motion/use-highlight-flash";
import { renderCardContent, type CardPromptHandler } from "./card-renderers-extended";
import { toClinicalCardViewModels } from "./clinical-card-view-model";

type ClinicalCardsPanelProps = {
  title?: string;
  emptyMessage?: string;
  cards: Record<string, JsonObject>;
  selectedCardType: string | null;
  onPromptRequest?: CardPromptHandler;
};

export function ClinicalCardsPanel({
  title = "医疗卡片",
  emptyMessage = "暂无医疗卡片。",
  cards,
  selectedCardType,
  onPromptRequest,
}: ClinicalCardsPanelProps) {
  const cardViewModels = toClinicalCardViewModels(cards);
  const panelRef = useRef<HTMLDivElement | null>(null);

  useHighlightFlash(panelRef, cardViewModels.length);

  return (
    <section ref={panelRef} className="clinical-card clinical-medical-cards-panel" data-testid="clinical-cards-panel">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon clinical-grid-icon" aria-hidden="true" />
        <h2>{title}</h2>
      </div>
      {cardViewModels.length > 0 ? (
        <div className="clinical-medical-card-grid">
          {cardViewModels.map((card) => {
            const isSelected = card.cardType === selectedCardType;

            return (
              <article
                key={card.cardType}
                className={`clinical-medical-card clinical-medical-card-${card.tone}${isSelected ? " clinical-medical-card-selected" : ""}`}
                aria-current={isSelected ? "true" : undefined}
              >
                <div className="clinical-medical-card-title-row">
                  <span className="clinical-medical-card-icon" aria-hidden="true" />
                  <h3>{card.title}</h3>
                  <span className="clinical-card-open" aria-hidden="true" />
                </div>
                <div className="clinical-card-rendered-content">
                  {renderCardContent({
                    cardType: card.cardType,
                    payload: card.payload,
                    onPromptRequest,
                  })}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="clinical-card-empty-state" data-testid="clinical-card-empty-state">
          <p>{emptyMessage}</p>
        </div>
      )}
    </section>
  );
}
