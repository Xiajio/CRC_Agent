import { useEffect, useState, type ReactNode } from "react";

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
  isInteractive?: boolean;
};

type TriageSelectionMode = "single" | "multiple";

type TriageQuestionOption = {
  id: string;
  label: string;
  submitText: string;
  exclusive: boolean;
  requiresFreeText: boolean;
};

type TriageQuestionCardPayload = JsonObject & {
  question_id?: unknown;
  field_key?: unknown;
  prompt?: unknown;
  help_text?: unknown;
  selection_mode?: unknown;
  options?: unknown;
  allow_other?: unknown;
  other_label?: unknown;
  other_placeholder?: unknown;
  submit_label?: unknown;
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

function asBoolean(value: unknown): boolean {
  return value === true;
}

function asSelectionMode(value: unknown): TriageSelectionMode | null {
  const raw = asString(value);
  if (raw === "single" || raw === "multiple") {
    return raw;
  }

  return null;
}

function normalizeTriageQuestionOption(value: unknown): TriageQuestionOption | null {
  const option = asObject(value);
  if (!option) {
    return null;
  }

  const id = asString(option.id);
  const label = asString(option.label);
  if (!id || !label) {
    return null;
  }

  return {
    id,
    label,
    submitText: asString(option.submit_text) ?? label,
    exclusive: asBoolean(option.exclusive),
    requiresFreeText: asBoolean(option.requires_free_text),
  };
}

function normalizeTriageQuestionOptions(value: unknown): TriageQuestionOption[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item) => normalizeTriageQuestionOption(item)).filter((item): item is TriageQuestionOption => item !== null);
}

function buildTriageInteractionContext(
  questionId: string,
  fieldKey: string | null,
  selectionMode: TriageSelectionMode,
  selectedOptionIds: string[],
  otherText: string | null,
): Record<string, unknown> {
  return {
    triage_interaction: {
      question_id: questionId,
      field_key: fieldKey,
      selection_mode: selectionMode,
      selected_option_ids: selectedOptionIds,
      other_text: otherText,
    },
  };
}

function buildTriageAnswerText(selectedOptions: TriageQuestionOption[], otherText: string): string {
  const fragments = selectedOptions.map((option) => option.submitText);
  const trimmedOther = otherText.trim();
  if (trimmedOther) {
    fragments.push(trimmedOther);
  }

  return fragments.join("; ");
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
    <dl className="workspace-definition-list workspace-definition-list-compact" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px 16px", margin: "8px 0" }}>
      {visibleItems.map((item) => (
        <div key={item.label} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          <dt style={{ fontSize: "0.75rem", color: "#6b7280", fontWeight: 500 }}>{item.label}</dt>
          <dd style={{ margin: 0, fontSize: "0.875rem", color: "#111827", fontWeight: 500, wordBreak: "break-word" }}>{String(item.value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function TriageQuestionCardView({
  payload,
  onPromptRequest,
  isInteractive = true,
}: {
  payload: TriageQuestionCardPayload;
  onPromptRequest?: CardPromptHandler;
  isInteractive?: boolean;
}) {
  const questionId = asString(payload.question_id) ?? "triage-question";
  const fieldKey = asString(payload.field_key);
  const prompt = asString(payload.prompt) ?? "请选择一个答案。";
  const helpText = asString(payload.help_text);
  const selectionMode = asSelectionMode(payload.selection_mode) ?? "single";
  const options = normalizeTriageQuestionOptions(payload.options);
  const allowOther = asBoolean(payload.allow_other);
  const otherLabel = asString(payload.other_label) ?? "其他";
  const otherPlaceholder = asString(payload.other_placeholder) ?? "补充说明";
  const submitLabel = asString(payload.submit_label) ?? "提交答案";
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [otherText, setOtherText] = useState("");
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    setSelectedIds([]);
    setOtherText("");
    setSubmitted(false);
  }, [questionId]);

  const otherSelected = selectedIds.includes("other");
  const selectedOptions = options.filter((option) => selectedIds.includes(option.id));
  const selectedNonTextOptions = selectedOptions.filter((option) => !option.requiresFreeText);
  const requiresText = otherSelected || selectedOptions.some((option) => option.requiresFreeText);
  const canSubmit =
    selectionMode === "multiple"
      ? selectedNonTextOptions.length > 0 || (requiresText && otherText.trim().length > 0)
      : requiresText && otherText.trim().length > 0;
  const statusMessage = submitted && !isInteractive
    ? "已提交，当前已进入下一题。"
    : submitted
      ? "已提交。"
      : !isInteractive
        ? "当前追问已失效。"
        : null;

  const toggleSelection = (option: TriageQuestionOption) => {
    if (submitted || !isInteractive) {
      return;
    }

    if (selectionMode === "single") {
      setSelectedIds([option.id]);
      if (option.requiresFreeText) {
        return;
      }

      if (onPromptRequest) {
        onPromptRequest(
          option.submitText,
          buildTriageInteractionContext(questionId, fieldKey, selectionMode, [option.id], null),
        );
        setSubmitted(true);
      }
      return;
    }

    setSelectedIds((current) => {
      if (option.exclusive) {
        return [option.id];
      }

      const withoutExclusive = current.filter((id) => !options.find((item) => item.id === id)?.exclusive);
      const withoutOption = withoutExclusive.filter((id) => id !== option.id);

      if (current.includes(option.id)) {
        return withoutOption;
      }

      return [...withoutOption, option.id];
    });
  };

  const toggleOther = () => {
    if (submitted || !isInteractive) {
      return;
    }

    if (selectionMode === "single") {
      setSelectedIds(["other"]);
      return;
    }

    setSelectedIds((current) => {
      if (current.includes("other")) {
        return current.filter((id) => id !== "other");
      }

      const withoutExclusive = current.filter((id) => !options.find((item) => item.id === id)?.exclusive);
      return [...withoutExclusive, "other"];
    });
  };

  const submitAnswer = () => {
    if (submitted || !isInteractive || !onPromptRequest) {
      return;
    }

    const chosenOptions = options.filter((option) => selectedIds.includes(option.id));
    const answerText = buildTriageAnswerText(chosenOptions, otherText);
    const nextSelectedIds = [
      ...chosenOptions.map((option) => option.id),
      ...(otherSelected || otherText.trim() ? ["other"] : []),
    ];

    onPromptRequest(
      answerText,
      buildTriageInteractionContext(questionId, fieldKey, selectionMode, nextSelectedIds, otherText.trim() || null),
    );
    setSubmitted(true);
  };

  return (
    <div className="workspace-card-section">
      <p className="workspace-copy workspace-copy-tight">{prompt}</p>
      {helpText ? <p className="workspace-copy workspace-copy-tight">{helpText}</p> : null}
      {options.length > 0 ? (
        <div className="workspace-action-row">
          {options.map((option) => {
            const active = selectedIds.includes(option.id);
            return (
              <button
                key={option.id}
                type="button"
                className={active ? "workspace-secondary-button workspace-action-button workspace-button-active" : "workspace-secondary-button workspace-action-button"}
                disabled={submitted || !isInteractive}
                onClick={() => toggleSelection(option)}
              >
                {option.label}
              </button>
            );
          })}
          {allowOther ? (
            <button
              type="button"
              className={otherSelected ? "workspace-secondary-button workspace-action-button workspace-button-active" : "workspace-secondary-button workspace-action-button"}
              disabled={submitted || !isInteractive}
              onClick={toggleOther}
            >
              {otherLabel}
            </button>
          ) : null}
        </div>
      ) : null}
      {(otherSelected || selectedOptions.some((option) => option.requiresFreeText)) && !submitted ? (
        <div className="workspace-card-section">
          <input
            aria-label={otherLabel}
            type="text"
            placeholder={otherPlaceholder}
            value={otherText}
            onChange={(event) => setOtherText(event.target.value)}
            disabled={submitted || !isInteractive}
          />
          {selectionMode === "single" ? (
            <button
              type="button"
              className="workspace-secondary-button workspace-action-button"
              disabled={submitted || !isInteractive || !canSubmit}
              onClick={submitAnswer}
            >
              {submitLabel}
            </button>
          ) : null}
        </div>
      ) : null}
      {selectionMode === "multiple" && !submitted ? (
        <button
          type="button"
          className="workspace-secondary-button workspace-action-button"
          disabled={submitted || !isInteractive || !canSubmit}
          onClick={submitAnswer}
        >
          {submitLabel}
        </button>
      ) : null}
      {statusMessage ? <p className="workspace-copy workspace-copy-tight">{statusMessage}</p> : null}
    </div>
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
    asString(symptomSnapshot?.chief_symptoms) ??
    (Array.isArray(symptomSnapshot?.chief_symptoms)
      ? symptomSnapshot.chief_symptoms
          .map((item) => asString(item))
          .filter((item): item is string => Boolean(item))
          .join("\u3001")
      : null);
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

  if (cardType === "triage_question_card") {
    return asString(payload.title) ?? "\u95e8\u8bca\u5206\u8bca\u8ffd\u95ee";
  }

  return baseCardTitle(cardType, payload);
}

export function renderCardContent({ cardType, payload, onPromptRequest, isInteractive }: CardRendererContext): ReactNode {
  if (cardType === "triage_question_card") {
    return (
      <TriageQuestionCardView
        payload={payload as TriageQuestionCardPayload}
        onPromptRequest={onPromptRequest}
        isInteractive={isInteractive}
      />
    );
  }

  if (cardType === "triage_card") {
    return renderTriageCard(payload);
  }

  return baseRenderCardContent({ cardType, payload, onPromptRequest });
}
