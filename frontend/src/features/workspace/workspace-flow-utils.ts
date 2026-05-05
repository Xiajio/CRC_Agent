import { ApiClientError } from "../../app/api/client";
import type { FrontendMessage, JsonObject, Scene, SessionState } from "../../app/api/types";

export const DEFAULT_UPLOAD_MAX_BYTES = 25 * 1024 * 1024;

export function readFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

export function readText(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  return null;
}

export function readUploadMaxBytes(): number {
  const importMetaEnv = import.meta as ImportMeta & {
    env?: Record<string, string | boolean | undefined>;
  };
  const configured = readFiniteNumber(importMetaEnv.env?.VITE_API_UPLOAD_MAX_BYTES);
  return configured && configured > 0 ? configured : DEFAULT_UPLOAD_MAX_BYTES;
}

export function formatUploadSize(bytes: number): string {
  const units: Array<[string, number]> = [
    ["GB", 1024 * 1024 * 1024],
    ["MB", 1024 * 1024],
    ["KB", 1024],
  ];

  for (const [unit, size] of units) {
    if (bytes >= size) {
      const value = bytes / size;
      const formatted = Number.isInteger(value) ? String(value) : value.toFixed(1).replace(/\.0$/, "");
      return `${formatted} ${unit}`;
    }
  }

  return `${bytes} ${String.fromCharCode(0x5B57, 0x8282)}`;
}

export function uploadTooLargeMessage(maxBytes = readUploadMaxBytes()): string {
  return `\u6587\u4ef6\u8fc7\u5927\uff0c\u6700\u5927\u4e0a\u4f20\u5927\u5c0f\u4e3a ${formatUploadSize(maxBytes)}\u3002`;
}

export function readUploadMaxBytesFromError(error: ApiClientError): number | null {
  const detail =
    typeof error.detail === "object" && error.detail && "detail" in error.detail
      ? (error.detail as { detail: unknown }).detail
      : error.message;
  const source = typeof detail === "string" ? detail : error.message;
  const match = /maximum size is (\d+) bytes/i.exec(source);
  if (!match) {
    return null;
  }

  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export function readWorkspaceErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    if (error.status === 413) {
      return uploadTooLargeMessage(readUploadMaxBytesFromError(error) ?? readUploadMaxBytes());
    }
    return error.message;
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "\u5de5\u4f5c\u533a\u8bf7\u6c42\u5931\u8d25\u3002";
}

export function isNotFoundApiError(error: unknown): boolean {
  if (error instanceof ApiClientError) {
    return error.status === 404;
  }

  if (typeof error === "object" && error !== null && "status" in error) {
    return (error as { status?: unknown }).status === 404;
  }

  return false;
}

export function isAbortError(error: unknown): boolean {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "AbortError";
  }

  return error instanceof Error && error.name === "AbortError";
}

export function nextLocalCursor(messages: FrontendMessage[]): string {
  const numericCursors = messages
    .map((message) => Number.parseInt(message.cursor, 10))
    .filter((value) => Number.isFinite(value));

  if (numericCursors.length === 0) {
    return "1";
  }

  return String(Math.max(...numericCursors) + 1);
}

export function appendOptimisticUserMessage(state: SessionState, content: string): SessionState {
  return {
    ...state,
    messages: [
      ...state.messages,
      {
        cursor: nextLocalCursor(state.messages),
        type: "human",
        content,
        assetRefs: [],
      },
    ],
    messagesTotal: Math.max(state.messagesTotal, state.messages.length + 1),
    lastError: null,
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
  };
}

export const DOCTOR_WORKFLOW_KEYWORDS = [
  "\u4e34\u5e8a\u8bc4\u4f30",
  "\u6cbb\u7597\u5efa\u8bae",
  "\u6cbb\u7597\u65b9\u6848",
  "\u8bc1\u636e\u4f9d\u636e",
  "\u57fa\u4e8e\u8bc1\u636e",
  "\u7ba1\u7406\u5efa\u8bae",
  "\u4f1a\u8bca\u62a5\u544a",
  "\u751f\u6210\u62a5\u544a",
  "\u6307\u5357",
  "recommendation",
  "treatment",
  "assessment",
  "evidence",
  "guideline",
  "report",
];

export function shouldPrimeDoctorWorkflow(scene: Scene, prompt: string): boolean {
  if (scene !== "doctor") {
    return false;
  }

  const normalized = prompt.toLowerCase();
  return DOCTOR_WORKFLOW_KEYWORDS.some((keyword) => normalized.includes(keyword.toLowerCase()));
}

export function createClinicalRoadmapScaffold(): JsonObject[] {
  return [
    { id: "intent", title: "intent", status: "completed" },
    { id: "planner", title: "planner", status: "in_progress" },
    { id: "assessment", title: "assessment", status: "waiting" },
    { id: "decision", title: "decision", status: "waiting" },
    { id: "citation", title: "citation", status: "waiting" },
    { id: "evaluator", title: "evaluator", status: "waiting" },
    { id: "finalize", title: "finalize", status: "waiting" },
  ];
}

export function createClinicalPlanScaffold(): JsonObject[] {
  return [
    { id: "collect-context", title: "collect context", status: "completed" },
    { id: "retrieve-guidelines", title: "retrieve guidelines", status: "in_progress" },
    { id: "query-case-database", title: "query case database", status: "pending" },
    { id: "generate-assessment", title: "generate clinical assessment", status: "pending" },
    { id: "generate-recommendation", title: "generate treatment recommendation", status: "pending" },
    { id: "finalize-report", title: "finalize report", status: "pending" },
  ];
}

export function primeDoctorClinicalWorkflow(state: SessionState, prompt: string): SessionState {
  if (!shouldPrimeDoctorWorkflow("doctor", prompt)) {
    return state;
  }

  return {
    ...state,
    roadmap: state.roadmap.length > 0 ? state.roadmap : createClinicalRoadmapScaffold(),
    plan: state.plan.length > 0 ? state.plan : createClinicalPlanScaffold(),
  };
}

export function resolveActiveError({
  pageError,
  turnError,
  uploadError,
  sessionState,
  bootstrapError,
}: {
  pageError: string | null;
  turnError: string | null;
  uploadError: string | null;
  sessionState: SessionState;
  bootstrapError: string | null;
}): string | null {
  if (pageError) {
    return pageError;
  }
  if (turnError) {
    return turnError;
  }
  if (uploadError) {
    return uploadError;
  }
  if (sessionState.lastError?.message) {
    return sessionState.lastError.message;
  }
  return bootstrapError;
}
