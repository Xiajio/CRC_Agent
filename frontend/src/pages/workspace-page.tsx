import { useEffect, useMemo, useRef, useState } from "react";

import { buildChatLatencyTraceAnalysis, createChatLatencyTraceStore } from "../app/api/chat-latency-trace";
import { ApiClientError } from "../app/api/client";
import { generateTraceId } from "../app/api/generate-trace-id";
import type { ChatTurnRequest, FrontendMessage, JsonObject, Scene, SessionResponse, SessionState } from "../app/api/types";
import type { StreamTraceTap } from "../app/api/stream";
import { mergeMessageHistory, reduceStreamEvent } from "../app/store/stream-reducer";
import { useApiClient } from "../app/providers";
import { ConversationPanel, type ConversationLatencyStatus } from "../features/chat/conversation-panel";
import { ClinicalTopNav } from "../components/layout/clinical-top-nav";
import { DoctorSceneShell } from "../features/doctor/doctor-scene-shell";
import { ExecutionPlanPanel } from "../features/execution-plan/execution-plan-panel";
import { PatientBackgroundPanel } from "../features/cards/patient-background-panel";
import { PatientIdentityPanel } from "../features/patient-identity/patient-identity-panel";
import { RoadmapPanel } from "../features/roadmap/roadmap-panel";
import { UploadsPanel } from "../features/uploads/uploads-panel";
import { useDatabaseWorkbench } from "../features/database/use-database-workbench";
import { usePatientRegistry } from "../features/patient-registry/use-patient-registry";
import { useRegistryBrowser } from "../features/patient-registry/use-registry-browser";
import { useSceneSessions } from "../features/workspace/use-scene-sessions";
import { getVisibleCards } from "./visible-cards";

type SceneDrafts = Record<Scene, string>;
type TurnLatencyProbeStatus = "streaming" | "message_done" | "ui_complete" | "aborted" | "error";

type TurnLatencyProbe = {
  sequence: number;
  scene: Scene;
  traceId: string;
  prompt: string;
  status: TurnLatencyProbeStatus;
  startedAt: number;
  messageDoneAt: number | null;
  renderCommittedAt: number | null;
  assistantMessageId: string | null;
  assistantCursor: string | null;
  finalContentText: string | null;
  uiCompleteMs: number | null;
  errorMessage: string | null;
};

type RecentCompletedProbes = Record<Scene, TurnLatencyProbe | null>;

const DEFAULT_UPLOAD_MAX_BYTES = 25 * 1024 * 1024;

type ChatLatencyDebugSurface = {
  readonly latestTrace: unknown;
  readonly traceHistory: unknown[];
  readonly latestDiagnosis: unknown;
  toLatestTraceJson(): string;
  toAllTracesJson(): string;
};

declare global {
  interface Window {
    __chatLatency?: ChatLatencyDebugSurface;
  }
}

function readFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function readText(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  return null;
}

function readUploadMaxBytes(): number {
  const importMetaEnv = import.meta as ImportMeta & {
    env?: Record<string, string | boolean | undefined>;
  };
  const configured = readFiniteNumber(importMetaEnv.env?.VITE_API_UPLOAD_MAX_BYTES);
  return configured && configured > 0 ? configured : DEFAULT_UPLOAD_MAX_BYTES;
}

function formatUploadSize(bytes: number): string {
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

  return `${bytes} 字节`;
}

function uploadTooLargeMessage(maxBytes = readUploadMaxBytes()): string {
  return `文件过大，最大上传大小为 ${formatUploadSize(maxBytes)}。`;
}

function readUploadMaxBytesFromError(error: ApiClientError): number | null {
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

function isChatLatencyDebugEnabled(): boolean {
  if (typeof window !== "undefined") {
    try {
      if (window.localStorage.getItem("chatLatencyDebug") === "1") {
        return true;
      }
    } catch {
      // Ignore storage access failures and fall back to env-only debug mode.
    }
  }

  const importMetaEnv = import.meta as ImportMeta & {
    env?: Record<string, string | boolean | undefined>;
  };
  return importMetaEnv.env?.VITE_CHAT_LATENCY_DEBUG === "true";
}

function readErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    if (error.status === 413) {
      return uploadTooLargeMessage(readUploadMaxBytesFromError(error) ?? readUploadMaxBytes());
    }
    return error.message;
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "工作区请求失败。";
}

function isNotFoundApiError(error: unknown): boolean {
  if (error instanceof ApiClientError) {
    return error.status === 404;
  }

  if (typeof error === "object" && error !== null && "status" in error) {
    return (error as { status?: unknown }).status === 404;
  }

  return false;
}

function isAbortError(error: unknown): boolean {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "AbortError";
  }

  return error instanceof Error && error.name === "AbortError";
}

function nextLocalCursor(messages: FrontendMessage[]): string {
  const numericCursors = messages
    .map((message) => Number.parseInt(message.cursor, 10))
    .filter((value) => Number.isFinite(value));

  if (numericCursors.length === 0) {
    return "1";
  }

  return String(Math.max(...numericCursors) + 1);
}

function appendOptimisticUserMessage(state: SessionState, content: string): SessionState {
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

const DOCTOR_WORKFLOW_KEYWORDS = [
  "临床评估",
  "治疗建议",
  "治疗方案",
  "证据依据",
  "基于证据",
  "管理建议",
  "会诊报告",
  "生成报告",
  "指南",
  "recommendation",
  "treatment",
  "assessment",
  "evidence",
  "guideline",
  "report",
];

function shouldPrimeDoctorWorkflow(scene: Scene, prompt: string): boolean {
  if (scene !== "doctor") {
    return false;
  }

  const normalized = prompt.toLowerCase();
  return DOCTOR_WORKFLOW_KEYWORDS.some((keyword) => normalized.includes(keyword.toLowerCase()));
}

function createClinicalRoadmapScaffold(): JsonObject[] {
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

function createClinicalPlanScaffold(): JsonObject[] {
  return [
    { id: "collect-context", title: "collect context", status: "completed" },
    { id: "retrieve-guidelines", title: "retrieve guidelines", status: "in_progress" },
    { id: "query-case-database", title: "query case database", status: "pending" },
    { id: "generate-assessment", title: "generate clinical assessment", status: "pending" },
    { id: "generate-recommendation", title: "generate treatment recommendation", status: "pending" },
    { id: "finalize-report", title: "finalize report", status: "pending" },
  ];
}

function primeDoctorClinicalWorkflow(state: SessionState, prompt: string): SessionState {
  if (!shouldPrimeDoctorWorkflow("doctor", prompt)) {
    return state;
  }

  return {
    ...state,
    roadmap: state.roadmap.length > 0 ? state.roadmap : createClinicalRoadmapScaffold(),
    plan: state.plan.length > 0 ? state.plan : createClinicalPlanScaffold(),
  };
}

function stripTriageQuestionCard(cards: Record<string, JsonObject>): Record<string, JsonObject> {
  const { triage_question_card: _triageQuestionCard, ...rest } = cards;
  return rest;
}

function cardsWithInlineCards(
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

function triageVisibilityContext(findings: SessionState["findings"]): {
  encounterTrack: string | null;
  activeInquiry: boolean;
} {
  const record = findings as Record<string, unknown>;

  return {
    encounterTrack: readText(record.encounter_track),
    activeInquiry: Boolean(record.active_inquiry),
  };
}

function readActiveTriageQuestionId(cards: SessionState["cards"]): string | null {
  const questionCard = cards.triage_question_card as Record<string, unknown> | undefined;
  return readText(questionCard?.question_id);
}

type PatientWorkspaceTab = "profile" | "upload";

const PATIENT_NAV_ITEMS = [
  { key: "profile", label: "资料填写" },
  { key: "symptoms", label: "症状", disabled: true },
  { key: "upload", label: "上传" },
  { key: "care-plan", label: "照护计划", disabled: true },
];

const PRODUCTION_PATIENT_NAV_ITEMS = PATIENT_NAV_ITEMS.filter((item) => !item.disabled);

function currentSceneError(
  sceneError: string | null,
  sessionState: SessionState,
  bootstrapError: string | null,
): string | null {
  if (sceneError) {
    return sceneError;
  }

  if (sessionState.lastError?.message) {
    return sessionState.lastError.message;
  }

  return bootstrapError;
}

function isProbeIncomplete(probe: TurnLatencyProbe | null): probe is TurnLatencyProbe {
  return probe !== null && probe.status !== "ui_complete" && probe.status !== "aborted" && probe.status !== "error";
}

function conversationLatencyStatusForScene(
  scene: Scene,
  activeProbe: TurnLatencyProbe | null,
  recentCompletedProbes: RecentCompletedProbes,
): ConversationLatencyStatus | null {
  const recentCompletedProbe = recentCompletedProbes[scene];
  if (activeProbe && activeProbe.scene === scene && (activeProbe.status === "streaming" || activeProbe.status === "message_done")) {
    return { kind: "streaming" };
  }

  if (recentCompletedProbe && recentCompletedProbe.scene === scene && recentCompletedProbe.status === "ui_complete" && recentCompletedProbe.uiCompleteMs !== null) {
    return {
      kind: "completed",
      uiCompleteMs: recentCompletedProbe.uiCompleteMs,
    };
  }

  return null;
}

export function WorkspacePage() {
  const apiClient = useApiClient();
  const activeStreamRef = useRef<AbortController | null>(null);
  const streamSequenceRef = useRef(0);
  const activeProbeRef = useRef<TurnLatencyProbe | null>(null);
  const traceStoreRef = useRef(createChatLatencyTraceStore());
  const {
    activeScene,
    setActiveScene,
    bootstrapStatus,
    bootstrapError,
    patient,
    doctor,
    applyResponseToScene,
  } = useSceneSessions();

  const [drafts, setDrafts] = useState<SceneDrafts>({
    patient: "",
    doctor: "",
  });
  const [patientWorkspaceTab, setPatientWorkspaceTab] = useState<PatientWorkspaceTab>("profile");
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [recentCompletedProbes, setRecentCompletedProbes] = useState<RecentCompletedProbes>({
    patient: null,
    doctor: null,
  });

  const activeController = activeScene === "patient" ? patient : doctor;
  const activeSessionState = activeController.state;
  const setActiveSessionState = activeController.setState;
  const doctorPatientId = readFiniteNumber(doctor.state.currentPatientId);
  const { encounterTrack: patientEncounterTrack, activeInquiry: patientActiveInquiry } =
    triageVisibilityContext(patient.state.findings);
  const activePatientTriageQuestionId = patientActiveInquiry
    ? readActiveTriageQuestionId(patient.state.cards)
    : null;

  const patientRegistry = usePatientRegistry({
    enabled: activeScene === "doctor",
    currentPatientId: doctorPatientId,
  });

  const registryBrowser = useRegistryBrowser({
    enabled: activeScene === "doctor",
  });

  const databaseWorkbench = useDatabaseWorkbench({
    autoBootstrap: activeScene === "doctor",
    bootstrapKey: "doctor-historical-workbench",
  });

  const patientVisibleCards = useMemo(
    () =>
      stripTriageQuestionCard(
        getVisibleCards(cardsWithInlineCards(patient.state.cards, patient.state.messages), {
          isDatabaseDetailActive: false,
          encounterTrack: patientEncounterTrack,
          activeInquiry: patientActiveInquiry,
        }),
      ),
    [patient.state.cards, patient.state.messages, patientEncounterTrack, patientActiveInquiry],
  );

  const doctorVisibleCards = useMemo(
    () =>
      getVisibleCards(cardsWithInlineCards(doctor.state.cards, doctor.state.messages), {
        isDatabaseDetailActive: false,
        encounterTrack: null,
        activeInquiry: false,
      }),
    [doctor.state.cards, doctor.state.messages],
  );

  function emitTraceConsoleSummary(traceId: string) {
    if (!isChatLatencyDebugEnabled()) {
      return;
    }

    const trace = traceStoreRef.current.getTrace(traceId);
    if (!trace) {
      return;
    }

    const analysis = buildChatLatencyTraceAnalysis(trace);
    console.debug({
      traceId,
      uiCompleteMs: analysis.derived.uiCompleteMs,
      ttftMs: analysis.derived.ttftMs,
      renderTailMs: analysis.derived.renderTailMs,
      primaryBottleneck: analysis.diagnosis.primary,
      secondaryFactors: analysis.diagnosis.secondaryFactors,
    });
  }

  function markProbeAborted(probe: TurnLatencyProbe, abortedAt: number) {
    activeProbeRef.current = {
      ...probe,
      status: "aborted",
    };
    traceStoreRef.current.recordClientAbort(probe.traceId, abortedAt);
  }

  function markProbeSuperseded(probe: TurnLatencyProbe, supersededAt: number) {
    activeProbeRef.current = {
      ...probe,
      status: "aborted",
    };
    traceStoreRef.current.markSuperseded(probe.traceId, supersededAt);
  }

  useEffect(() => {
    setSceneError(null);
    if (activeScene === "doctor") {
      setUploadStatus(null);
    }
  }, [activeScene]);

  useEffect(() => {
    if (typeof window === "undefined" || !isChatLatencyDebugEnabled()) {
      return;
    }

    const debugSurface: ChatLatencyDebugSurface = {
      get latestTrace() {
        const latest = traceStoreRef.current.getLatestTrace();
        return latest ? JSON.parse(traceStoreRef.current.toLatestTraceJson()) : null;
      },
      get traceHistory() {
        const payload = JSON.parse(traceStoreRef.current.toAllTracesJson()) as { traces?: unknown[] };
        return payload.traces ?? [];
      },
      get latestDiagnosis() {
        const latest = traceStoreRef.current.getLatestTrace();
        return latest ? buildChatLatencyTraceAnalysis(latest) : null;
      },
      toLatestTraceJson() {
        return traceStoreRef.current.toLatestTraceJson();
      },
      toAllTracesJson() {
        return traceStoreRef.current.toAllTracesJson();
      },
    };

    window.__chatLatency = debugSurface;
    return () => {
      if (window.__chatLatency === debugSurface) {
        delete window.__chatLatency;
      }
    };
  }, []);

  useEffect(() => {
    const probe = activeProbeRef.current;
    if (!isProbeIncomplete(probe)) {
      return;
    }
    if (probe.scene === activeScene) {
      return;
    }

    activeProbeRef.current = null;
  }, [activeScene]);

  useEffect(() => {
    const probe = activeProbeRef.current;
    if (!isProbeIncomplete(probe)) {
      return;
    }

    const relevantState = probe.scene === "patient" ? patient.state : doctor.state;
    const errorMessage = relevantState.lastError?.message;
    if (!errorMessage) {
      return;
    }

    const errorAt = performance.now();
    activeProbeRef.current = {
      ...probe,
      status: "error",
      errorMessage,
    };
    traceStoreRef.current.recordClientError(probe.traceId, errorAt);
  }, [patient.state.lastError, doctor.state.lastError, patient.state, doctor.state]);

  useEffect(() => {
    const probe = activeProbeRef.current;
    if (!probe || probe.status !== "message_done") {
      return;
    }
    if (probe.sequence !== streamSequenceRef.current) {
      markProbeAborted(probe, performance.now());
      return;
    }

    const relevantState = probe.scene === "patient" ? patient.state : doctor.state;
    let assistantCursor = probe.assistantCursor;
    let targetMessage: FrontendMessage | undefined;

    if (probe.assistantMessageId) {
      targetMessage = relevantState.messages.find(
        (message) => message.type === "ai" && message.id === probe.assistantMessageId,
      );
      assistantCursor = targetMessage?.cursor ?? assistantCursor;
    } else {
      assistantCursor = assistantCursor ?? relevantState.latestAssistantMessageCursor ?? null;
      targetMessage = assistantCursor
        ? relevantState.messages.find(
            (message) => message.type === "ai" && message.cursor === assistantCursor,
          )
        : undefined;
    }

    if (!targetMessage) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      const currentProbe = activeProbeRef.current;
      if (!currentProbe || currentProbe.sequence !== probe.sequence || currentProbe.status !== "message_done") {
        return;
      }

      const renderCommittedAt = performance.now();
      const completedProbe: TurnLatencyProbe = {
        ...currentProbe,
        assistantCursor,
        renderCommittedAt,
        status: "ui_complete",
      };
      completedProbe.uiCompleteMs = renderCommittedAt - completedProbe.startedAt;
      activeProbeRef.current = completedProbe;
      traceStoreRef.current.recordClientUiComplete(completedProbe.traceId, renderCommittedAt);
      emitTraceConsoleSummary(completedProbe.traceId);
      setRecentCompletedProbes((current) => ({
        ...current,
        [completedProbe.scene]: completedProbe,
      }));
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [
    patient.state.messages,
    patient.state.latestAssistantMessageCursor,
    doctor.state.messages,
    doctor.state.latestAssistantMessageCursor,
  ]);

  useEffect(() => {
    const sessionId = activeSessionState.sessionId;
    if (
      bootstrapStatus !== "ready"
      || !sessionId
      || isStreaming
      || activeSessionState.contextMaintenance?.status !== "running"
    ) {
      return;
    }

    let cancelled = false;
    const timer = window.setInterval(() => {
      void apiClient.getSession(sessionId).then(
        (response) => {
          if (cancelled) {
            return;
          }
          applyResponseToScene(activeScene, response);
        },
        () => undefined,
      );
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    activeScene,
    activeSessionState.contextMaintenance?.status,
    activeSessionState.sessionId,
    apiClient,
    applyResponseToScene,
    bootstrapStatus,
    isStreaming,
  ]);

  function handleSceneSwitch(scene: Scene) {
    if (scene === activeScene) {
      return;
    }

    activeStreamRef.current?.abort();
    activeStreamRef.current = null;
    if (isProbeIncomplete(activeProbeRef.current)) {
      markProbeAborted(activeProbeRef.current, performance.now());
      activeProbeRef.current = null;
    }
    setIsStreaming(false);
    setSceneError(null);
    setActiveScene(scene);
  }

  function handlePatientNavSelect(key: string) {
    if (key === "profile" || key === "upload") {
      setPatientWorkspaceTab(key);
    }
  }

  function updateDraft(scene: Scene, value: string) {
    setDrafts((current) => ({
      ...current,
      [scene]: value,
    }));
  }

  async function loadMessageHistory() {
    const sessionId = activeSessionState.sessionId;
    const before = activeSessionState.messagesNextBeforeCursor;

    if (!sessionId || !before) {
      return;
    }

    setIsLoadingHistory(true);
    setSceneError(null);

    try {
      const history = await apiClient.getMessages(sessionId, before, 20);
      setActiveSessionState((current) => mergeMessageHistory(current, history, "prepend"));
    } catch (error) {
      setSceneError(readErrorMessage(error));
    } finally {
      setIsLoadingHistory(false);
    }
  }

  async function submitMessage(
    scene: Scene,
    prompt: string,
    context?: Record<string, unknown>,
  ) {
    const sceneController = scene === "patient" ? patient : doctor;
    const sessionId = sceneController.state.sessionId;
    const normalizedPrompt = prompt.trim();

    if (!sessionId || !normalizedPrompt) {
      return;
    }

    const traceId = generateTraceId();
    const clientWallClockAtSubmit = new Date().toISOString();
    const startedAt = performance.now();
    const controller = new AbortController();
    const sequence = streamSequenceRef.current + 1;
    const currentProbe = activeProbeRef.current;
    if (isProbeIncomplete(currentProbe)) {
      markProbeSuperseded(currentProbe, startedAt);
    }
    streamSequenceRef.current = sequence;
    activeProbeRef.current = {
      sequence,
      scene,
      traceId,
      prompt: normalizedPrompt,
      status: "streaming",
      startedAt,
      messageDoneAt: null,
      renderCommittedAt: null,
      assistantMessageId: null,
      assistantCursor: null,
      finalContentText: null,
      uiCompleteMs: null,
      errorMessage: null,
    };
    traceStoreRef.current.recordClientSubmit({
      traceId,
      scene,
      promptText: normalizedPrompt,
      clientWallClockAtSubmit,
      submitAt: startedAt,
      uploadsCount: Object.keys(sceneController.state.uploadedAssets ?? {}).length,
      contextKeys: context ? Object.keys(context).sort() : [],
    });
    setRecentCompletedProbes((current) => ({
      ...current,
      [scene]: null,
    }));
    activeStreamRef.current?.abort();
    activeStreamRef.current = controller;
    setIsStreaming(true);
    setSceneError(null);

    sceneController.setState((current) => {
      const withUserMessage = appendOptimisticUserMessage(current, normalizedPrompt);
      return scene === "doctor"
        ? primeDoctorClinicalWorkflow(withUserMessage, normalizedPrompt)
        : withUserMessage;
    });

    const request: ChatTurnRequest = {
      message: {
        role: "user",
        content: normalizedPrompt,
      },
      trace_id: traceId,
      ...(context ? { context } : {}),
    };
    const traceTap: StreamTraceTap = (event, receivedAt) => {
      traceStoreRef.current.recordStreamObservation(traceId, event, receivedAt);
    };

    try {
      await apiClient.streamTurn(sessionId, request, (event) => {
        if (event.type === "message.done" && activeProbeRef.current?.sequence === sequence) {
          const messageDoneAt = performance.now();
          activeProbeRef.current = {
            ...activeProbeRef.current,
            status: "message_done",
            messageDoneAt,
            assistantMessageId: event.message_id ?? null,
            finalContentText: typeof event.content === "string" ? event.content : null,
          };
          traceStoreRef.current.recordClientMessageDone(traceId, messageDoneAt);
        }
        sceneController.setState((current) => reduceStreamEvent(current, event));
      }, controller.signal, traceTap);
    } catch (error) {
      if (!isAbortError(error)) {
        const message = readErrorMessage(error);
        const currentProbeAfterError = activeProbeRef.current;
        if (isProbeIncomplete(currentProbeAfterError) && currentProbeAfterError.sequence === sequence) {
          const errorAt = performance.now();
          activeProbeRef.current = {
            ...currentProbeAfterError,
            status: "error",
            errorMessage: message,
          };
          traceStoreRef.current.recordClientError(currentProbeAfterError.traceId, errorAt);
        }
        sceneController.setState((current) =>
          reduceStreamEvent(current, {
            type: "error",
            code: "STREAM_REQUEST_FAILED",
            message,
            recoverable: true,
          }),
        );
        setSceneError(message);
      }
    } finally {
      if (streamSequenceRef.current === sequence) {
        activeStreamRef.current = null;
        setIsStreaming(false);
      }
    }
  }

  async function submitPrompt() {
    const sessionId = activeSessionState.sessionId;
    const prompt = drafts[activeScene].trim();

    if (!sessionId || !prompt) {
      return;
    }

    updateDraft(activeScene, "");
    void submitMessage(activeScene, prompt);
  }

  async function handleUpload(file: File) {
    const sessionId = patient.state.sessionId;

    if (!sessionId) {
      setSceneError("患者会话尚未准备好上传。");
      return;
    }

    const uploadMaxBytes = readUploadMaxBytes();
    if (file.size > uploadMaxBytes) {
      setSceneError(uploadTooLargeMessage(uploadMaxBytes));
      setUploadStatus(null);
      return;
    }

    setIsUploading(true);
    setSceneError(null);
    setUploadStatus(`正在上传 ${file.name}...`);

    try {
      const response = await apiClient.uploadFile(sessionId, file);
      patient.setState((current) => ({
        ...current,
        uploadedAssets: {
          ...current.uploadedAssets,
          [String(response.asset_id)]: {
            filename: response.filename,
            derived: response.derived,
          },
        },
      }));
      const refreshed = await apiClient.getSession(sessionId);
      applyResponseToScene("patient", refreshed);
      patient.setState((current) => ({
        ...current,
        patientIdentity: refreshed.snapshot.patient_identity ?? null,
      }));
      setUploadStatus(`已上传 ${response.filename}`);
    } catch (error) {
      setSceneError(readErrorMessage(error));
      setUploadStatus(null);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleResetActiveScene() {
    const sessionId = activeSessionState.sessionId;

    if (!sessionId) {
      return;
    }

    activeStreamRef.current?.abort();
    activeStreamRef.current = null;
    if (isProbeIncomplete(activeProbeRef.current)) {
      markProbeAborted(activeProbeRef.current, performance.now());
    }
    activeProbeRef.current = null;
    setRecentCompletedProbes((current) => ({
      ...current,
      [activeScene]: null,
    }));
    setIsStreaming(false);
    setSceneError(null);

    const applyResetResponse = (response: SessionResponse) => {
      applyResponseToScene(activeScene, response);
      if (activeScene === "patient") {
        patient.setState((current) => ({
          ...current,
          patientIdentity: response.snapshot.patient_identity ?? null,
        }));
      }
      updateDraft(activeScene, "");
      if (activeScene === "patient") {
        setUploadStatus(null);
      }
      setSceneError(null);
    };

    try {
      const response = await apiClient.resetSession(sessionId);
      applyResetResponse(response);
    } catch (error) {
      if (isNotFoundApiError(error)) {
        try {
          const response = await apiClient.createSession(activeScene);
          applyResetResponse(response);
        } catch (replacementError) {
          setSceneError(readErrorMessage(replacementError));
        }
        return;
      }
      setSceneError(readErrorMessage(error));
    }
  }

  async function handleBindDoctorPatient(patientId: number): Promise<boolean> {
    const sessionId = doctor.state.sessionId;

    if (!sessionId) {
      setSceneError("医生会话尚未准备好绑定患者。");
      return false;
    }

    setSceneError(null);

    try {
      const response = await patientRegistry.bindPatient(sessionId, patientId);
      applyResponseToScene("doctor", response);
      return true;
    } catch (error) {
      setSceneError(readErrorMessage(error));
      return false;
    }
  }

  if (bootstrapStatus === "loading") {
    return <main className="workspace-shell"><div className="workspace-card">正在加载工作区...</div></main>;
  }

  if (bootstrapStatus === "error") {
    return (
      <main className="workspace-shell">
        <div className="workspace-card">
          <h2>工作区初始化失败</h2>
          <p className="workspace-copy workspace-copy-alert">{bootstrapError ?? "未知初始化错误。"}</p>
        </div>
      </main>
    );
  }

  const activeError = currentSceneError(sceneError, activeSessionState, bootstrapError);
  const activeDraft = drafts[activeScene];
  const patientLatencyStatus = conversationLatencyStatusForScene("patient", activeProbeRef.current, recentCompletedProbes);
  const doctorLatencyStatus = conversationLatencyStatusForScene("doctor", activeProbeRef.current, recentCompletedProbes);

  const topNavActions = (
    <button
      type="button"
      className="clinical-reset-button"
      onClick={() => void handleResetActiveScene()}
    >
      重置当前场景
    </button>
  );

  const patientUploadsPanel = (
    <UploadsPanel
      uploadedAssets={patient.state.uploadedAssets}
      disabled={isUploading || isStreaming}
      statusMessage={uploadStatus}
      onUpload={(file) => void handleUpload(file)}
    />
  );

  if (activeScene === "doctor") {
    return (
      <DoctorSceneShell
        toolbar={topNavActions}
        onSwitchScene={() => handleSceneSwitch("patient")}
        currentPatientId={doctorPatientId}
        patientRegistry={patientRegistry}
        databaseWorkbench={databaseWorkbench}
        registryBrowser={registryBrowser}
        messages={doctor.state.messages}
        draft={drafts.doctor}
        statusNode={doctor.state.statusNode}
        isStreaming={isStreaming}
        isLoadingHistory={isLoadingHistory}
        canLoadHistory={Boolean(doctor.state.messagesNextBeforeCursor)}
        disabled={isStreaming || isUploading}
        errorMessage={activeError}
        latencyStatus={doctorLatencyStatus}
        roadmap={doctor.state.roadmap}
        stage={doctor.state.stage}
        plan={doctor.state.plan}
        cards={doctorVisibleCards}
        references={doctor.state.references}
        critic={doctor.state.critic}
        eventLog={doctor.state.eventLog}
        onLoadHistory={() => void loadMessageHistory()}
        onDraftChange={(value) => updateDraft("doctor", value)}
        onSubmit={() => void submitPrompt()}
        onSetCurrentPatient={handleBindDoctorPatient}
      />
    );
  }

  return (
    <main className="clinical-app-shell clinical-app-shell-patient">
      <ClinicalTopNav
        brandLabel="临床助手"
        navLabel="患者工作台"
        items={PRODUCTION_PATIENT_NAV_ITEMS}
        activeKey={patientWorkspaceTab}
        onSelect={handlePatientNavSelect}
        actions={topNavActions}
        statusLabel="安全会话"
        statusTone="safe"
        profileLabel="患者"
        profileAriaLabel="doctor scene"
        onProfileClick={() => handleSceneSwitch("doctor")}
        className="clinical-top-nav-patient"
      />
      <div className="clinical-patient-dashboard" data-testid="workspace-layout">
        <aside className="clinical-patient-left-column" data-testid="workspace-left-rail">
          <div className="workspace-panel-stack">
            <PatientIdentityPanel
              sessionId={patient.state.sessionId}
              patientIdentity={patient.state.patientIdentity ?? null}
              onSaved={(identity) => {
                patient.setState((current) => ({
                  ...current,
                  patientIdentity: identity,
                }));
              }}
            />
            <PatientBackgroundPanel
              title="患者背景信息"
              emptyMessage="当前暂无患者背景信息"
              cards={patientVisibleCards}
            />
          </div>
        </aside>
        <section className="clinical-patient-center-column" data-testid="workspace-center">
          <div className="workspace-panel-stack">
            {patientWorkspaceTab === "upload" ? (
              patientUploadsPanel
            ) : (
              <ConversationPanel
                messages={patient.state.messages}
                draft={activeDraft}
                activeTriageQuestionId={activePatientTriageQuestionId}
                statusNode={patient.state.statusNode}
                isStreaming={isStreaming}
                isLoadingHistory={isLoadingHistory}
                canLoadHistory={Boolean(patient.state.messagesNextBeforeCursor)}
                disabled={isStreaming || isUploading}
                errorMessage={activeError}
                latencyStatus={patientLatencyStatus}
                onLoadHistory={() => void loadMessageHistory()}
                onDraftChange={(value) => updateDraft("patient", value)}
                onSubmit={() => void submitPrompt()}
                onCardPromptRequest={(prompt: string, context?: Record<string, unknown>) =>
                  void submitMessage("patient", prompt, context)
                }
              />
            )}
          </div>
        </section>
        <aside className="clinical-patient-right-column" data-testid="workspace-right">
          <div className="workspace-panel-stack">
            {patientWorkspaceTab === "profile" ? patientUploadsPanel : null}
          </div>
        </aside>
      </div>
    </main>
  );

}



