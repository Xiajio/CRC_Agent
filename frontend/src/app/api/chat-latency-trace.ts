import type { JsonObject, Scene, StreamEvent, TraceStartEvent, TraceStepEvent, TraceSummaryEvent } from "./types";

export type ChatLatencyTraceStatus = "active" | "completed" | "aborted" | "error" | "superseded";

export type ChatLatencyTracePrimary =
  | "frontend_render_tail"
  | "network_or_first_byte"
  | "server_orchestration"
  | "retrieval"
  | "llm_startup"
  | "llm_generation"
  | "mixed"
  | "unknown";

export interface ChatLatencyTraceCandidate {
  key: ChatLatencyTracePrimary;
  ms: number;
  share: number;
}

export interface ChatLatencyTraceSummary {
  traceId: string | null;
  sessionId: string | null;
  runId: string | null;
  graphPath: string[] | null;
  status: ChatLatencyTraceStatus;
  serverTotalMs: number | null;
  routerMs: number | null;
  retrievalMs: number | null;
  llmStartupMs: number | null;
  llmGenerationMs: number | null;
  streamFlushTailMs: number | null;
  serverOrchestrationMs: number | null;
  estimatedClientServerOffsetMs: number | null;
  firstServerEventAtMs: number | null;
  flushControlled: boolean | null;
  model: string | null;
  hasThinking: boolean | null;
  responseChars: number | null;
  toolCalls: number | null;
  retrievalHitCount: number | null;
  promptText: string | null;
  uploadsCount: number | null;
  contextKeys: string[] | null;
}

export interface ChatLatencyTraceRecord {
  traceId: string;
  scene: Scene | null;
  promptText: string | null;
  clientWallClockAtSubmit: string | null;
  submitAt: number | null;
  firstEventReceivedAt: number | null;
  firstDeltaReceivedAt: number | null;
  messageDoneReceivedAt: number | null;
  uiCommittedAt: number | null;
  abortedAt: number | null;
  errorAt: number | null;
  status: ChatLatencyTraceStatus;
  statusSource: "frontend" | "backend" | null;
  submitOrder: number | null;
  backendTraceStart: TraceStartEvent | null;
  backendTraceSteps: TraceStepEvent[];
  backendTraceSummary: TraceSummaryEvent | null;
  backendSummary: ChatLatencyTraceSummary | null;
  uploadsCount: number | null;
  contextKeys: string[] | null;
  sequence: number;
  updatedAt: number;
}

export interface ChatLatencyTraceAnalysis {
  traceId: string;
  status: ChatLatencyTraceStatus;
  derived: {
    ttftMs: number | null;
    renderTailMs: number | null;
    uiCompleteMs: number | null;
    serverTotalMs: number | null;
    routerMs: number | null;
    retrievalMs: number | null;
    llmStartupMs: number | null;
    llmGenerationMs: number | null;
    streamFlushTailMs: number | null;
    serverOrchestrationMs: number | null;
    serverOrchestrationOtherMs: number | null;
    networkOrFirstByteMs: number | null;
    networkOrFirstByteShare: number | null;
    accountedShare: number | null;
    unaccountedShare: number | null;
    estimatedClientServerOffsetMs: number | null;
    serverToClientFirstEventMs: number | null;
  };
  diagnosis: {
    primary: ChatLatencyTracePrimary;
    candidateKeys: ChatLatencyTracePrimary[];
    candidates: ChatLatencyTraceCandidate[];
    secondaryFactors: Array<"simple_prompt" | "thinking_enabled" | "response_too_long">;
    simplePromptHeuristic: boolean;
  };
}

export interface ChatLatencyTraceStore {
  recordClientSubmit(input: {
    traceId: string;
    scene: Scene;
    promptText: string;
    clientWallClockAtSubmit?: string | null;
    submitAt: number;
    uploadsCount?: number | null;
    contextKeys?: string[] | null;
  }): void;
  recordClientMessageDone(traceId: string, messageDoneReceivedAt: number): void;
  recordClientUiComplete(traceId: string, uiCommittedAt: number): void;
  recordClientAbort(traceId: string, abortedAt: number): void;
  recordClientError(traceId: string, errorAt: number): void;
  markSuperseded(traceId: string, supersededAt: number): void;
  recordStreamObservation(traceId: string, event: StreamEvent, receivedAt: number): void;
  recordBackendTraceStart(event: TraceStartEvent): void;
  recordBackendTraceStep(event: TraceStepEvent): void;
  recordBackendTraceSummary(event: TraceSummaryEvent): void;
  recordBackendSummary(traceIdOrSummary: string | ChatLatencyTraceSummary, summaryMaybe?: ChatLatencyTraceSummary): void;
  getTrace(traceId: string): ChatLatencyTraceRecord | null;
  getLatestTrace(): ChatLatencyTraceRecord | null;
  toLatestTraceJson(): string;
  toAllTracesJson(): string;
}

const OFFSET_PLAUSIBILITY_LIMIT_MS = 5 * 60 * 1000;
const SIMPLE_PROMPT_LIMIT = 20;
const RESPONSE_TOO_LONG_LIMIT = 300;

function clampShare(value: number): number {
  return Math.max(0, Math.min(value, 1));
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseIsoMillis(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }

  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function positiveOrNull(value: number | null | undefined): number | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, value);
}

function computeStatusPriority(status: ChatLatencyTraceStatus): number {
  switch (status) {
    case "active":
      return 0;
    case "completed":
      return 1;
    case "aborted":
      return 2;
    case "error":
      return 3;
    case "superseded":
      return 4;
  }
}

function mergeStatus(current: ChatLatencyTraceRecord, nextStatus: ChatLatencyTraceStatus, source: "frontend" | "backend"): void {
  if (current.status === "superseded") {
    return;
  }

  if (nextStatus === "completed" && source === "frontend") {
    current.status = "completed";
    current.statusSource = "frontend";
    return;
  }

  if (current.status === "completed" && current.statusSource === "frontend" && nextStatus !== "superseded") {
    return;
  }

  if (current.status === "aborted" && current.statusSource === "frontend" && nextStatus !== "superseded") {
    return;
  }

  if (nextStatus === "active") {
    return;
  }

  if (computeStatusPriority(nextStatus) >= computeStatusPriority(current.status)) {
    current.status = nextStatus;
    current.statusSource = source;
  }
}

function getTraceTtftMs(record: ChatLatencyTraceRecord): number | null {
  if (record.submitAt === null) {
    return null;
  }
  if (record.firstDeltaReceivedAt !== null) {
    return Math.max(record.firstDeltaReceivedAt - record.submitAt, 0);
  }
  if (record.messageDoneReceivedAt !== null) {
    return Math.max(record.messageDoneReceivedAt - record.submitAt, 0);
  }
  return null;
}

function traceSortKey(record: ChatLatencyTraceRecord): number {
  return record.submitOrder ?? record.sequence;
}

function maybeAddCandidate(
  candidates: ChatLatencyTraceCandidate[],
  {
    key,
    ms,
    uiCompleteMs,
    thresholdMs,
    thresholdShare,
    extraGate = true,
  }: {
    key: Exclude<ChatLatencyTracePrimary, "mixed" | "unknown">;
    ms: number | null;
    uiCompleteMs: number | null;
    thresholdMs: number;
    thresholdShare: number;
    extraGate?: boolean;
  },
): void {
  if (!extraGate || ms === null || uiCompleteMs === null || uiCompleteMs <= 0) {
    return;
  }

  const share = clampShare(ms / uiCompleteMs);
  if (ms > thresholdMs || share > thresholdShare) {
    candidates.push({ key, ms, share });
  }
}

export function isSimplePromptHeuristic(input: {
  promptText: string | null;
  uploadsCount: number | null | undefined;
  contextKeys: string[] | null | undefined;
}): boolean {
  const promptText = input.promptText?.trim() ?? "";
  return promptText.length > 0
    && promptText.length <= SIMPLE_PROMPT_LIMIT
    && (input.uploadsCount ?? 0) === 0
    && (input.contextKeys?.length ?? 0) === 0;
}

function resolveTransportMetrics(summary: ChatLatencyTraceSummary, submitAt: number | null) {
  const estimatedOffset = summary.estimatedClientServerOffsetMs;
  if (estimatedOffset === null || Math.abs(estimatedOffset) > OFFSET_PLAUSIBILITY_LIMIT_MS) {
    return {
      estimatedClientServerOffsetMs: null as number | null,
      serverToClientFirstEventMs: null as number | null,
    };
  }

  if (summary.firstServerEventAtMs === null || submitAt === null) {
    return {
      estimatedClientServerOffsetMs: estimatedOffset,
      serverToClientFirstEventMs: null,
    };
  }

  const normalizedFirstEventAt = summary.firstServerEventAtMs - estimatedOffset;
  return {
    estimatedClientServerOffsetMs: estimatedOffset,
    serverToClientFirstEventMs: Math.max(normalizedFirstEventAt - submitAt, 0),
  };
}

function diffMs(laterMs: number | null, earlierMs: number | null): number | null {
  if (laterMs === null || earlierMs === null) {
    return null;
  }
  return Math.max(laterMs - earlierMs, 0);
}

function latestStepAt(steps: TraceStepEvent[], name: string): number | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step.name === name) {
      return parseIsoMillis(step.at);
    }
  }
  return null;
}

function firstStepAt(steps: TraceStepEvent[], name: string): number | null {
  for (const step of steps) {
    if (step.name === name) {
      return parseIsoMillis(step.at);
    }
  }
  return null;
}

function booleanAttr(attrs: JsonObject | undefined, key: string): boolean | null {
  const value = attrs?.[key];
  return typeof value === "boolean" ? value : null;
}

function buildBackendSummaryFromTrace(record: ChatLatencyTraceRecord): ChatLatencyTraceSummary | null {
  const start = record.backendTraceStart;
  const summary = record.backendTraceSummary;

  if (!start && !summary && record.backendTraceSteps.length === 0) {
    return record.backendSummary;
  }

  const serverReceivedAtMs = parseIsoMillis(start?.server_received_at ?? null);
  const graphStartedAtMs = parseIsoMillis(start?.graph_started_at ?? null);
  const streamDoneAtMs = latestStepAt(record.backendTraceSteps, "stream.done");
  const messageDoneAtMs = latestStepAt(record.backendTraceSteps, "message.done");
  const llmRequestStartedAtMs = latestStepAt(record.backendTraceSteps, "llm.request.started");
  const llmFirstTokenAtMs = latestStepAt(record.backendTraceSteps, "llm.first_token");
  const routerDoneAtMs = firstStepAt(record.backendTraceSteps, "router.done");
  const retrievalDoneAtMs = firstStepAt(record.backendTraceSteps, "retrieval.done");
  const firstServerEventAtMs = firstStepAt(record.backendTraceSteps, "stream.first_byte");
  const summaryAtMs = parseIsoMillis(summary?.at ?? null);

  const clientWallClockAtSubmitMs = parseIsoMillis(record.clientWallClockAtSubmit);

  return {
    traceId: summary?.trace_id ?? start?.trace_id ?? record.traceId,
    sessionId: summary?.session_id ?? start?.session_id ?? null,
    runId: summary?.run_id ?? start?.run_id ?? null,
    graphPath: summary?.graph_path ?? start?.graph_path ?? null,
    status: summary?.status ?? record.status,
    serverTotalMs: diffMs(streamDoneAtMs ?? summaryAtMs, serverReceivedAtMs),
    routerMs: diffMs(routerDoneAtMs, graphStartedAtMs),
    retrievalMs: diffMs(retrievalDoneAtMs, routerDoneAtMs),
    llmStartupMs: diffMs(llmFirstTokenAtMs, llmRequestStartedAtMs),
    llmGenerationMs: diffMs(messageDoneAtMs, llmFirstTokenAtMs),
    streamFlushTailMs: diffMs(streamDoneAtMs ?? summaryAtMs, messageDoneAtMs),
    serverOrchestrationMs: diffMs(llmRequestStartedAtMs, serverReceivedAtMs),
    estimatedClientServerOffsetMs:
      serverReceivedAtMs !== null && clientWallClockAtSubmitMs !== null
        ? serverReceivedAtMs - clientWallClockAtSubmitMs
        : null,
    firstServerEventAtMs,
    flushControlled: booleanAttr(start?.attrs, "flush_controlled"),
    model: summary?.model ?? null,
    hasThinking: summary?.has_thinking ?? null,
    responseChars: summary?.response_chars ?? null,
    toolCalls: summary?.tool_calls ?? null,
    retrievalHitCount: summary?.retrieval_hit_count ?? null,
    promptText: record.promptText,
    uploadsCount: record.uploadsCount,
    contextKeys: record.contextKeys,
  };
}

function syncBackendSummary(record: ChatLatencyTraceRecord): void {
  record.backendSummary = buildBackendSummaryFromTrace(record);
}

function buildAnalysis(record: ChatLatencyTraceRecord): ChatLatencyTraceAnalysis {
  const summary = record.backendSummary;
  const ttftMs = getTraceTtftMs(record);
  const uiCompleteMs =
    record.submitAt !== null && record.uiCommittedAt !== null
      ? Math.max(record.uiCommittedAt - record.submitAt, 0)
      : null;
  const renderTailMs =
    record.messageDoneReceivedAt !== null && record.uiCommittedAt !== null
      ? Math.max(record.uiCommittedAt - record.messageDoneReceivedAt, 0)
      : null;
  const serverTotalMs = summary?.serverTotalMs ?? null;
  const routerMs = summary?.routerMs ?? null;
  const retrievalMs = summary?.retrievalMs ?? null;
  const llmStartupMs = summary?.llmStartupMs ?? null;
  const llmGenerationMs = summary?.llmGenerationMs ?? null;
  const streamFlushTailMs = summary?.streamFlushTailMs ?? null;
  const serverOrchestrationMs = summary?.serverOrchestrationMs ?? null;
  const serverOrchestrationOtherMs =
    serverOrchestrationMs !== null && retrievalMs !== null
      ? Math.max(serverOrchestrationMs - retrievalMs, 0)
      : null;

  const transport = summary ? resolveTransportMetrics(summary, record.submitAt) : {
    estimatedClientServerOffsetMs: null,
    serverToClientFirstEventMs: null,
  };
  const networkOrFirstByteMs = transport.serverToClientFirstEventMs;

  let accountedMs = 0;
  for (const value of [renderTailMs, retrievalMs, serverOrchestrationOtherMs, llmStartupMs, llmGenerationMs]) {
    accountedMs += positiveOrNull(value) ?? 0;
  }

  const accountedShare =
    uiCompleteMs !== null && uiCompleteMs > 0 ? clampShare(accountedMs / uiCompleteMs) : null;
  const unaccountedShare = accountedShare === null ? null : clampShare(1 - accountedShare);

  const simplePromptHeuristic = isSimplePromptHeuristic({
    promptText: record.promptText ?? summary?.promptText ?? null,
    uploadsCount: record.uploadsCount ?? summary?.uploadsCount ?? null,
    contextKeys: record.contextKeys ?? summary?.contextKeys ?? null,
  });

  const candidateShares: ChatLatencyTraceCandidate[] = [];
  maybeAddCandidate(candidateShares, {
    key: "frontend_render_tail",
    ms: renderTailMs,
    uiCompleteMs,
    thresholdMs: 150,
    thresholdShare: 0.10,
  });
  maybeAddCandidate(candidateShares, {
    key: "network_or_first_byte",
    ms: networkOrFirstByteMs,
    uiCompleteMs,
    thresholdMs: 500,
    thresholdShare: 0,
    extraGate:
      summary?.flushControlled === true
      && summary.serverOrchestrationMs !== null
      && summary.serverOrchestrationMs < 500
      && networkOrFirstByteMs !== null
      && networkOrFirstByteMs > 500,
  });
  maybeAddCandidate(candidateShares, {
    key: "server_orchestration",
    ms: serverOrchestrationOtherMs,
    uiCompleteMs,
    thresholdMs: 1000,
    thresholdShare: 0.25,
  });
  maybeAddCandidate(candidateShares, {
    key: "retrieval",
    ms: retrievalMs,
    uiCompleteMs,
    thresholdMs: 800,
    thresholdShare: 0.25,
  });
  maybeAddCandidate(candidateShares, {
    key: "llm_startup",
    ms: llmStartupMs,
    uiCompleteMs,
    thresholdMs: 1500,
    thresholdShare: 0.20,
  });
  maybeAddCandidate(candidateShares, {
    key: "llm_generation",
    ms: llmGenerationMs,
    uiCompleteMs,
    thresholdMs: 4000,
    thresholdShare: 0,
    extraGate: (summary?.responseChars ?? 0) > 0,
  });

  const secondaryFactors: ChatLatencyTraceAnalysis["diagnosis"]["secondaryFactors"] = [];
  if (simplePromptHeuristic) {
    secondaryFactors.push("simple_prompt");
  }
  if (summary?.hasThinking) {
    secondaryFactors.push("thinking_enabled");
  }
  if (simplePromptHeuristic && (summary?.responseChars ?? 0) > RESPONSE_TOO_LONG_LIMIT) {
    secondaryFactors.push("response_too_long");
  }

  const sortedCandidates = [...candidateShares].sort((left, right) => right.share - left.share);
  let primary: ChatLatencyTracePrimary = "unknown";
  if (sortedCandidates.length === 0) {
    primary = uiCompleteMs !== null ? "mixed" : "unknown";
  } else {
    const topCandidate = sortedCandidates[0];
    const secondCandidate = sortedCandidates[1] ?? null;
    primary = topCandidate.key;

    if (accountedShare !== null && accountedShare < 0.85 && topCandidate.key !== "network_or_first_byte") {
      primary = "mixed";
    } else if (topCandidate.share < 0.5) {
      primary = "mixed";
    } else if (secondCandidate !== null && Math.abs(topCandidate.share - secondCandidate.share) <= 0.15) {
      primary = "mixed";
    }
  }

  return {
    traceId: record.traceId,
    status: record.status === "active" && summary ? summary.status : record.status,
    derived: {
      ttftMs,
      renderTailMs,
      uiCompleteMs,
      serverTotalMs,
      routerMs,
      retrievalMs,
      llmStartupMs,
      llmGenerationMs,
      streamFlushTailMs,
      serverOrchestrationMs,
      serverOrchestrationOtherMs,
      networkOrFirstByteMs,
      networkOrFirstByteShare:
        networkOrFirstByteMs !== null && uiCompleteMs !== null && uiCompleteMs > 0
          ? clampShare(networkOrFirstByteMs / uiCompleteMs)
          : null,
      accountedShare,
      unaccountedShare,
      estimatedClientServerOffsetMs: transport.estimatedClientServerOffsetMs,
      serverToClientFirstEventMs: transport.serverToClientFirstEventMs,
    },
    diagnosis: {
      primary,
      candidateKeys: candidateShares.map((entry) => entry.key),
      candidates: sortedCandidates,
      secondaryFactors,
      simplePromptHeuristic,
    },
  };
}

function ensureRecord(registry: ChatLatencyTraceStoreImpl, traceId: string): ChatLatencyTraceRecord {
  const existing = registry.records.get(traceId);
  if (existing) {
    return existing;
  }

  const record: ChatLatencyTraceRecord = {
    traceId,
    scene: null,
    promptText: null,
    clientWallClockAtSubmit: null,
    submitAt: null,
    firstEventReceivedAt: null,
    firstDeltaReceivedAt: null,
    messageDoneReceivedAt: null,
    uiCommittedAt: null,
    abortedAt: null,
    errorAt: null,
    status: "active",
    statusSource: null,
    submitOrder: null,
    backendTraceStart: null,
    backendTraceSteps: [],
    backendTraceSummary: null,
    backendSummary: null,
    uploadsCount: null,
    contextKeys: null,
    sequence: registry.sequence += 1,
    updatedAt: registry.sequence,
  };
  registry.records.set(traceId, record);
  return record;
}

class ChatLatencyTraceStoreImpl implements ChatLatencyTraceStore {
  records = new Map<string, ChatLatencyTraceRecord>();
  sequence = 0;
  submitSequence = 0;

  recordClientSubmit(input: {
    traceId: string;
    scene: Scene;
    promptText: string;
    clientWallClockAtSubmit?: string | null;
    submitAt: number;
    uploadsCount?: number | null;
    contextKeys?: string[] | null;
  }): void {
    const record = ensureRecord(this, input.traceId);
    record.scene = input.scene;
    record.promptText = input.promptText;
    record.clientWallClockAtSubmit = input.clientWallClockAtSubmit ?? record.clientWallClockAtSubmit;
    record.submitAt = input.submitAt;
    record.uploadsCount = input.uploadsCount ?? null;
    record.contextKeys = input.contextKeys ?? null;
    record.status = "active";
    record.statusSource = "frontend";
    if (record.submitOrder === null) {
      record.submitOrder = ++this.submitSequence;
    }
    record.updatedAt = ++this.sequence;
  }

  recordClientMessageDone(traceId: string, messageDoneReceivedAt: number): void {
    const record = ensureRecord(this, traceId);
    record.messageDoneReceivedAt = messageDoneReceivedAt;
    record.updatedAt = ++this.sequence;
  }

  recordClientUiComplete(traceId: string, uiCommittedAt: number): void {
    const record = ensureRecord(this, traceId);
    record.uiCommittedAt = uiCommittedAt;
    mergeStatus(record, "completed", "frontend");
    record.updatedAt = ++this.sequence;
  }

  recordClientAbort(traceId: string, abortedAt: number): void {
    const record = ensureRecord(this, traceId);
    record.abortedAt = abortedAt;
    mergeStatus(record, "aborted", "frontend");
    record.updatedAt = ++this.sequence;
  }

  recordClientError(traceId: string, errorAt: number): void {
    const record = ensureRecord(this, traceId);
    record.errorAt = errorAt;
    mergeStatus(record, "error", "frontend");
    record.updatedAt = ++this.sequence;
  }

  markSuperseded(traceId: string, supersededAt: number): void {
    const record = ensureRecord(this, traceId);
    record.status = "superseded";
    record.statusSource = "frontend";
    record.abortedAt = record.abortedAt ?? supersededAt;
    record.updatedAt = ++this.sequence;
  }

  recordStreamObservation(traceId: string, event: StreamEvent, receivedAt: number): void {
    const backendTraceId =
      "trace_id" in event && typeof event.trace_id === "string" && event.trace_id.trim()
        ? event.trace_id
        : traceId;
    const record = ensureRecord(this, backendTraceId);
    if (record.firstEventReceivedAt === null) {
      record.firstEventReceivedAt = receivedAt;
    }
    if (event.type === "message.delta" && record.firstDeltaReceivedAt === null) {
      record.firstDeltaReceivedAt = receivedAt;
    }
    if (event.type === "trace.start") {
      this.recordBackendTraceStart(event);
      return;
    }
    if (event.type === "trace.step") {
      this.recordBackendTraceStep(event);
      return;
    }
    if (event.type === "trace.summary") {
      this.recordBackendTraceSummary(event);
      return;
    }
    record.updatedAt = ++this.sequence;
  }

  recordBackendTraceStart(event: TraceStartEvent): void {
    const traceId = event.trace_id;
    if (!traceId) {
      return;
    }

    const record = ensureRecord(this, traceId);
    record.backendTraceStart = event;
    if (record.scene === null) {
      record.scene = event.scene;
    }
    syncBackendSummary(record);
    record.updatedAt = ++this.sequence;
  }

  recordBackendTraceStep(event: TraceStepEvent): void {
    const traceId = event.trace_id;
    if (!traceId) {
      return;
    }

    const record = ensureRecord(this, traceId);
    record.backendTraceSteps = [...record.backendTraceSteps, event];
    syncBackendSummary(record);
    record.updatedAt = ++this.sequence;
  }

  recordBackendTraceSummary(event: TraceSummaryEvent): void {
    const traceId = event.trace_id;
    if (!traceId) {
      return;
    }

    const record = ensureRecord(this, traceId);
    record.backendTraceSummary = event;
    if (record.scene === null) {
      record.scene = event.scene;
    }
    mergeStatus(record, event.status, "backend");
    syncBackendSummary(record);
    record.updatedAt = ++this.sequence;
  }

  recordBackendSummary(traceIdOrSummary: string | ChatLatencyTraceSummary, summaryMaybe?: ChatLatencyTraceSummary): void {
    const summary = typeof traceIdOrSummary === "string" ? summaryMaybe : traceIdOrSummary;
    const traceId = typeof traceIdOrSummary === "string" ? traceIdOrSummary : traceIdOrSummary.traceId;
    if (!summary || !traceId) {
      return;
    }

    const record = ensureRecord(this, traceId);
    record.backendSummary = summary;
    mergeStatus(record, summary.status, "backend");
    record.updatedAt = ++this.sequence;
  }

  getTrace(traceId: string): ChatLatencyTraceRecord | null {
    return this.records.get(traceId) ?? null;
  }

  getLatestTrace(): ChatLatencyTraceRecord | null {
    let latest: ChatLatencyTraceRecord | null = null;
    for (const record of this.records.values()) {
      if (latest === null || traceSortKey(record) > traceSortKey(latest)) {
        latest = record;
      }
    }
    return latest;
  }

  toLatestTraceJson(): string {
    const latest = this.getLatestTrace();
    return JSON.stringify(latest ? serializeRecord(latest) : null);
  }

  toAllTracesJson(): string {
    const traces = [...this.records.values()]
      .sort((left, right) => traceSortKey(left) - traceSortKey(right))
      .map((record) => serializeRecord(record));

    return JSON.stringify({
      traces,
    });
  }
}

function serializeRecord(record: ChatLatencyTraceRecord) {
  return {
    traceId: record.traceId,
    scene: record.scene,
    promptText: record.promptText,
    clientWallClockAtSubmit: record.clientWallClockAtSubmit,
    submitAt: record.submitAt,
    firstEventReceivedAt: record.firstEventReceivedAt,
    firstDeltaReceivedAt: record.firstDeltaReceivedAt,
    messageDoneReceivedAt: record.messageDoneReceivedAt,
    uiCommittedAt: record.uiCommittedAt,
    abortedAt: record.abortedAt,
    errorAt: record.errorAt,
    status: record.status,
    statusSource: record.statusSource,
    uploadsCount: record.uploadsCount,
    contextKeys: record.contextKeys,
    backendTraceStart: record.backendTraceStart,
    backendTraceSteps: record.backendTraceSteps,
    backendTraceSummary: record.backendTraceSummary,
    backendSummary: record.backendSummary,
    analysis: buildAnalysis(record),
  };
}

export function buildChatLatencyTraceAnalysis(record: ChatLatencyTraceRecord): ChatLatencyTraceAnalysis {
  return buildAnalysis(record);
}

export function createChatLatencyTraceStore(): ChatLatencyTraceStore {
  return new ChatLatencyTraceStoreImpl();
}

export const chatLatencyTraceStore = createChatLatencyTraceStore();
