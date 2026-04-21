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
  const [isExpanded, setIsExpanded] = useState(true);

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
    <div ref={panelRef} className="workspace-card" data-testid="clinical-cards-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          margin: 0,
          fontSize: "1.125rem",
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "none",
          border: "none",
          padding: 0,
          cursor: "pointer",
          width: "100%",
          textAlign: "left",
          color: "inherit",
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "1.25rem" }}>📇</span> {title}
        </span>
        <span
          style={{
            fontSize: "0.875rem",
            color: "#6b7280",
            transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s ease-in-out",
            display: "inline-block",
          }}
        >
          ▼
        </span>
      </button>
      
      {isExpanded ? (
        cardEntries.length > 0 ? (
          <div className="workspace-card-grid">
            <ul className="workspace-tab-list" style={{ display: "flex", gap: "8px", padding: 0, margin: "0 0 16px 0", listStyle: "none", borderBottom: "1px solid #e5e7eb" }}>
              {cardEntries.map(([cardType, payload]) => {
                const title = cardTitle(cardType, payload);
                const active = cardType === activeType;

                return (
                  <li key={cardType}>
                    <button
                      type="button"
                      className={active ? "workspace-tab workspace-tab-active" : "workspace-tab"}
                      aria-pressed={active}
                      style={{
                        padding: "8px 16px",
                        background: "none",
                        border: "none",
                        borderBottom: active ? "2px solid #3b82f6" : "2px solid transparent",
                        color: active ? "#3b82f6" : "#6b7280",
                        fontWeight: active ? 600 : 400,
                        cursor: "pointer",
                        fontSize: "0.875rem",
                        marginBottom: "-1px"
                      }}
                      onClick={() => setActiveType(cardType)}
                    >
                      {title}
                    </button>
                  </li>
                );
              })}
            </ul>
            {activeCard ? (
              <div className="workspace-card-detail" style={{ padding: "8px 0" }}>
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
        )
      ) : null}
    </div>
  );
}
