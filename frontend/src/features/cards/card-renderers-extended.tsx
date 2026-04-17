import type { ReactNode } from "react";

import type { JsonObject } from "../../app/api/types";
import {
  cardTitle as baseCardTitle,
  renderCardContent as baseRenderCardContent,
  type CardPromptHandler,
} from "./card-renderers";

type CardRendererContext = {
  cardType: string;
  payload: JsonObject;
  onPromptRequest?: CardPromptHandler;
};

const TRIAGE_RISK_LABELS: Record<string, string> = {
  low: "\u4f4e\u98ce\u9669",
  medium: "\u4e2d\u98ce\u9669",
  high: "\u9ad8\u98ce\u9669",
};

const TRIAGE_DISPOSITION_LABELS: Record<string, string> = {
  observe: "\u89c2\u5bdf\u968f\u8bbf",
  routine_gi_clinic: "\u5e38\u89c4\u6d88\u5316\u95e8\u8bca",
  urgent_gi_clinic: "\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca",
  emergency: "\u6025\u8bca\u5c31\u533b",
  enter_crc_flow: "\u8fdb\u5165CRC\u4e34\u5e8a\u8bc4\u4f30",
};

function asObject(value: unknown): JsonObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value as JsonObject;
}

function asString(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return null;
}

function triageRiskLabel(value: unknown): string | null {
  const raw = asString(value);
  if (!raw) {
    return null;
  }
  return TRIAGE_RISK_LABELS[raw] ?? raw;
}

function triageDispositionLabel(value: unknown): string | null {
  const raw = asString(value);
  if (!raw) {
    return null;
  }
  return TRIAGE_DISPOSITION_LABELS[raw] ?? raw;
}

function renderDisclosure(title: string, payload: JsonObject) {
  return (
    <details className="workspace-card-disclosure">
      <summary>{title}</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function renderMetaItems(items: Array<{ label: string; value: string | number | null | undefined }>) {
  const visibleItems = items.filter((item) => item.value !== null && item.value !== undefined && String(item.value) !== "");

  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <dl className="workspace-definition-list workspace-definition-list-compact">
      {visibleItems.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{String(item.value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderTriageCard(payload: JsonObject) {
  const data = asObject(payload.data) ?? payload;
  const symptomSnapshot = asObject(data.symptom_snapshot);
  const summary =
    asString(data.summary) ??
    "\u5df2\u751f\u6210\u95e8\u8bca\u5206\u8bca\u7ed3\u679c\uff0c\u53ef\u6839\u636e\u98ce\u9669\u5206\u7ea7\u548c\u5efa\u8bae\u68c0\u67e5\u7ee7\u7eed\u5904\u7406\u3002";
  const riskLevel = triageRiskLabel(data.risk_level);
  const disposition = triageDispositionLabel(data.disposition);
  const chiefSymptoms =
    asString(data.chief_symptoms) ??
    asString(symptomSnapshot?.chief_symptoms);
  const suggestedTests = Array.isArray(data.suggested_tests)
    ? data.suggested_tests.map((item) => asString(item)).filter((item): item is string => Boolean(item))
    : [];

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">{"\u95e8\u8bca\u5206\u8bca"}</p>
        <p className="workspace-copy workspace-copy-tight">{summary}</p>
        {renderMetaItems([
          { label: "\u98ce\u9669\u5206\u7ea7", value: riskLevel },
          { label: "\u5efa\u8bae\u53bb\u5411", value: disposition },
          { label: "\u4e3b\u8981\u75c7\u72b6", value: chiefSymptoms },
        ])}
      </div>
      {suggestedTests.length > 0 ? (
        <div className="workspace-card-section">
          <strong>{"\u5efa\u8bae\u68c0\u67e5"}</strong>
          <ul className="workspace-list">
            {suggestedTests.map((item) => (
              <li key={item} className="workspace-list-item">
                {item}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {renderDisclosure("\u67e5\u770b\u539f\u59cb\u6570\u636e", payload)}
    </>
  );
}

export type { CardPromptHandler };

export function cardTitle(cardType: string, payload: JsonObject): string {
  if (cardType === "triage_card") {
    return asString(payload.title) ?? "\u95e8\u8bca\u5206\u8bca";
  }

  return baseCardTitle(cardType, payload);
}

export function renderCardContent({ cardType, payload, onPromptRequest }: CardRendererContext): ReactNode {
  if (cardType === "triage_card") {
    return renderTriageCard(payload);
  }

  return baseRenderCardContent({ cardType, payload, onPromptRequest });
}
