import { useEffect, useMemo, useRef, useState } from "react";

import { ApiClientError } from "../app/api/client";
import type { FrontendMessage, JsonObject, Scene, SessionState } from "../app/api/types";
import { mergeMessageHistory, reduceStreamEvent } from "../app/store/stream-reducer";
import { useApiClient } from "../app/providers";
import { WorkspaceLayout } from "../components/layout/workspace-layout";
import { ClinicalCardsPanel } from "../features/cards/clinical-cards-panel";
import { ConversationPanel } from "../features/chat/conversation-panel";
import { DoctorSceneShell } from "../features/doctor/doctor-scene-shell";
import { ExecutionPlanPanel } from "../features/execution-plan/execution-plan-panel";
import { RoadmapPanel } from "../features/roadmap/roadmap-panel";
import { UploadsPanel } from "../features/uploads/uploads-panel";
import { useDatabaseWorkbench } from "../features/database/use-database-workbench";
import { usePatientRegistry } from "../features/patient-registry/use-patient-registry";
import { useRegistryBrowser } from "../features/patient-registry/use-registry-browser";
import { useSceneSessions } from "../features/workspace/use-scene-sessions";
import { getVisibleCards } from "./visible-cards";

type SceneDrafts = Record<Scene, string>;

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

function readErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message;
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Workspace request failed.";
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
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
  };
}

function stripTriageQuestionCard(cards: Record<string, JsonObject>): Record<string, JsonObject> {
  const { triage_question_card: _triageQuestionCard, ...rest } = cards;
  return rest;
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

function sceneLabel(scene: Scene): string {
  return scene === "patient" ? "👤 患者端" : "🩺 医生端";
}

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

export function WorkspacePage() {
  const apiClient = useApiClient();
  const activeStreamRef = useRef<AbortController | null>(null);
  const streamSequenceRef = useRef(0);
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
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

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
        getVisibleCards(patient.state.cards, {
          isDatabaseDetailActive: false,
          encounterTrack: patientEncounterTrack,
          activeInquiry: patientActiveInquiry,
        }),
      ),
    [patient.state.cards, patientEncounterTrack, patientActiveInquiry],
  );

  const doctorVisibleCards = useMemo(
    () =>
      getVisibleCards(doctor.state.cards, {
        isDatabaseDetailActive: false,
        encounterTrack: null,
        activeInquiry: false,
      }),
    [doctor.state.cards],
  );

  useEffect(() => {
    setSceneError(null);
    if (activeScene === "doctor") {
      setUploadStatus(null);
    }
  }, [activeScene]);

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
    setIsStreaming(false);
    setSceneError(null);
    setActiveScene(scene);
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

    const controller = new AbortController();
    const sequence = streamSequenceRef.current + 1;
    streamSequenceRef.current = sequence;
    activeStreamRef.current?.abort();
    activeStreamRef.current = controller;
    setIsStreaming(true);
    setSceneError(null);

    sceneController.setState((current) => appendOptimisticUserMessage(current, normalizedPrompt));

    try {
      await apiClient.streamTurn(
        sessionId,
        context
          ? {
              message: {
                role: "user",
                content: normalizedPrompt,
              },
              context,
            }
          : {
              message: {
                role: "user",
                content: normalizedPrompt,
              },
            },
        (event) => {
          sceneController.setState((current) => reduceStreamEvent(current, event));
        },
        controller.signal,
      );
    } catch (error) {
      if (!isAbortError(error)) {
        setSceneError(readErrorMessage(error));
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
      setSceneError("Patient session is not ready for uploads.");
      return;
    }

    setIsUploading(true);
    setSceneError(null);
    setUploadStatus(`Uploading ${file.name}...`);

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
      setUploadStatus(`Uploaded ${response.filename}`);
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
    setIsStreaming(false);
    setSceneError(null);

    try {
      const response = await apiClient.resetSession(sessionId);
      applyResponseToScene(activeScene, response);
      updateDraft(activeScene, "");
      if (activeScene === "patient") {
        setUploadStatus(null);
      }
    } catch (error) {
      setSceneError(readErrorMessage(error));
    }
  }

  async function handleBindDoctorPatient(patientId: number): Promise<boolean> {
    const sessionId = doctor.state.sessionId;

    if (!sessionId) {
      setSceneError("Doctor session is not ready for patient binding.");
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
    return <main className="workspace-shell"><div className="workspace-card">Loading workspace...</div></main>;
  }

  if (bootstrapStatus === "error") {
    return (
      <main className="workspace-shell">
        <div className="workspace-card">
          <h2>Workspace bootstrap failed</h2>
          <p className="workspace-copy workspace-copy-alert">{bootstrapError ?? "Unknown bootstrap error."}</p>
        </div>
      </main>
    );
  }

  const activeError = currentSceneError(sceneError, activeSessionState, bootstrapError);
  const activeDraft = drafts[activeScene];

  const toolbar = (
    <>
      <button
        type="button"
        className={activeScene === "patient" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => handleSceneSwitch("patient")}
        aria-label="patient scene"
      >
        {sceneLabel("patient")}
      </button>
      <button
        type="button"
        className={activeScene === "doctor" ? "workspace-primary-button" : "workspace-secondary-button"}
        onClick={() => handleSceneSwitch("doctor")}
        aria-label="doctor scene"
      >
        {sceneLabel("doctor")}
      </button>
      <button
        type="button"
        className="workspace-secondary-button"
        onClick={() => void handleResetActiveScene()}
      >
        🔄 重置当前场景
      </button>
    </>
  );

  if (activeScene === "doctor") {
    return (
      <DoctorSceneShell
        toolbar={toolbar}
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
        roadmap={doctor.state.roadmap}
        stage={doctor.state.stage}
        plan={doctor.state.plan}
        cards={doctorVisibleCards}
        references={doctor.state.references}
        onLoadHistory={() => void loadMessageHistory()}
        onDraftChange={(value) => updateDraft("doctor", value)}
        onSubmit={() => void submitPrompt()}
        onSetCurrentPatient={handleBindDoctorPatient}
      />
    );
  }

  return (
      <WorkspaceLayout
      toolbar={(
        <div className="workspace-toolbar-row" style={{ display: "flex", gap: "12px", flexWrap: "wrap", alignItems: "center" }}>
          {toolbar}
        </div>
      )}
      leftRail={(
        <div className="workspace-panel-stack">
          <UploadsPanel
            uploadedAssets={patient.state.uploadedAssets}
            disabled={isUploading || isStreaming}
            statusMessage={uploadStatus}
            onUpload={(file) => void handleUpload(file)}
          />
        </div>
      )}
      centerWorkspace={(
            <div className="workspace-panel-stack">
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
                onLoadHistory={() => void loadMessageHistory()}
                onDraftChange={(value) => updateDraft("patient", value)}
                onSubmit={() => void submitPrompt()}
                onCardPromptRequest={(prompt: string, context?: Record<string, unknown>) =>
                  void submitMessage("patient", prompt, context)
                }
              />
            </div>
          )}
      rightInspector={(
        <div className="workspace-panel-stack">
          <ClinicalCardsPanel 
            title="患者背景信息" 
            emptyMessage="当前暂无患者背景信息" 
            cards={patientVisibleCards} 
            selectedCardType={null} 
          />
        </div>
      )}
    />
  );
}
