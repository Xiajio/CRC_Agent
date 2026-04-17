import { useEffect, useRef, useState } from "react";

import type { JsonObject } from "../../app/api/types";
import { useHighlightFlash } from "../../components/motion/use-highlight-flash";
import { cardTitle, renderCardContent, type CardPromptHandler } from "./card-renderers-extended";

type ClinicalCardsPanelProps = {
  title?: string;
  emptyMessage?: string;
  cards: Record<string, JsonObject>;
  selectedCardType: string | null;
  onPromptRequest?: CardPromptHandler;
};

export function ClinicalCardsPanel({
  title = "临床卡片",
  emptyMessage = "当前暂无临床卡片",
  cards,
  selectedCardType,
  onPromptRequest,
}: ClinicalCardsPanelProps) {
  const cardEntries = Object.entries(cards);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [activeType, setActiveType] = useState<string | null>(
    selectedCardType ?? cardEntries[0]?.[0] ?? null,
  );

  useHighlightFlash(panelRef, cardEntries.length);

  useEffect(() => {
    const nextActiveType = selectedCardType ?? cardEntries[0]?.[0] ?? null;
    setActiveType((current) => {
      if (current && cards[current]) {
        return current;
      }

      return nextActiveType;
    });
  }, [cardEntries, cards, selectedCardType]);

  const activeCard = activeType ? cards[activeType] : null;

  return (
    <div ref={panelRef} className="workspace-card" data-testid="clinical-cards-panel">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}>📇</span> {title}
      </h2>
      {cardEntries.length > 0 ? (
        <div className="workspace-card-grid">
          <ul className="workspace-tab-list">
            {cardEntries.map(([cardType, payload]) => {
              const title = cardTitle(cardType, payload);
              const active = cardType === activeType;

              return (
                <li key={cardType}>
                  <button
                    type="button"
                    className={active ? "workspace-tab workspace-tab-active" : "workspace-tab"}
                    aria-pressed={active}
                    onClick={() => setActiveType(cardType)}
                  >
                    {title}
                  </button>
                </li>
              );
            })}
          </ul>
          {activeCard ? (
            <div className="workspace-card-detail">
              <strong>{cardTitle(activeType ?? "card", activeCard)}</strong>
              {renderCardContent({
                cardType: activeType ?? "card",
                payload: activeCard,
                onPromptRequest,
              })}
            </div>
          ) : null}
        </div>
      ) : (
        <p className="workspace-copy">{emptyMessage}</p>
      )}
    </div>
  );
}
