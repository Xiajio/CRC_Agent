import type { JsonObject } from "../api/types";

const DEFAULT_CRITIC_FEEDBACK = "评审未通过该建议。";
const THINKING_BLOCK_PATTERN = /<think\b[^>]*>[\s\S]*?<\/think>/gi;
const CLOSING_THINKING_TAG = "</think>";

function stripThinkingText(value: string): string {
  let text = value.replace(THINKING_BLOCK_PATTERN, "").trim();
  const closingIndex = text.toLowerCase().lastIndexOf(CLOSING_THINKING_TAG);
  if (closingIndex >= 0) {
    text = text.slice(closingIndex + CLOSING_THINKING_TAG.length).trim();
  }
  return text;
}

function parseJsonObject(candidate: string): JsonObject | null {
  try {
    const parsed = JSON.parse(candidate);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as JsonObject)
      : null;
  } catch {
    return null;
  }
}

function extractFirstJsonObject(value: string): JsonObject | null {
  let start = value.indexOf("{");
  while (start >= 0) {
    let depth = 0;
    let inString = false;
    let escaped = false;

    for (let index = start; index < value.length; index += 1) {
      const char = value[index];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === "\"") {
          inString = false;
        }
        continue;
      }

      if (char === "\"") {
        inString = true;
      } else if (char === "{") {
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          const parsed = parseJsonObject(value.slice(start, index + 1));
          if (parsed) {
            return parsed;
          }
          break;
        }
      }
    }

    start = value.indexOf("{", start + 1);
  }

  return null;
}

function feedbackFromJsonPayload(payload: JsonObject): string | null {
  const feedback = payload.feedback;
  if (typeof feedback === "string" && feedback.trim()) {
    return feedback.trim();
  }
  if (typeof feedback === "number" || typeof feedback === "boolean") {
    return String(feedback);
  }
  return null;
}

function normalizeDisplayText(value: string): string {
  return value
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function formatCriticFeedback(value: unknown, fallback = DEFAULT_CRITIC_FEEDBACK): string {
  if (typeof value !== "string" || !value.trim()) {
    return fallback;
  }

  const withoutThinking = stripThinkingText(value);
  const directPayload = parseJsonObject(withoutThinking);
  const extractedPayload = directPayload ?? extractFirstJsonObject(withoutThinking);
  const extractedFeedback = extractedPayload ? feedbackFromJsonPayload(extractedPayload) : null;
  const readable = extractedFeedback ?? withoutThinking;
  return normalizeDisplayText(stripThinkingText(readable)) || fallback;
}

export function compactClinicalEventDetail(value: unknown, maxChars = 180): string | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  const readable = formatCriticFeedback(value, "").replace(/\s+/g, " ").trim();
  if (!readable) {
    return null;
  }
  return readable.length > maxChars ? `${readable.slice(0, maxChars).trimEnd()}...` : readable;
}
