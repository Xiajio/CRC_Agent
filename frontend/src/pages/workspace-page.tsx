import { useEffect, useRef, useState } from "react";

import { buildChatLatencyTraceAnalysis, createChatLatencyTraceStore } from "../app/api/chat-latency-trace";
import type { FrontendMessage, Scene } from "../app/api/types";
import { useApiClient } from "../app/providers";
import { ConversationPanel } from "../features/chat/conversation-panel";
import { ClinicalTopNav } from "../components/layout/clinical-top-nav";
import { DoctorSceneShell } from "../features/doctor/doctor-scene-shell";
import { PatientBackgroundPanel } from "../features/cards/patient-background-panel";
import { PatientIdentityPanel } from "../features/patient-identity/patient-identity-panel";
import { UploadsPanel } from "../features/uploads/uploads-panel";
import { useDatabaseWorkbench } from "../features/database/use-database-workbench";
import { usePatientRegistry } from "../features/patient-registry/use-patient-registry";
import { useRegistryBrowser } from "../features/patient-registry/use-registry-browser";
import { usePatientWorkspaceNav } from "../features/workspace/use-patient-workspace-nav";
import { useSceneSessions } from "../features/workspace/use-scene-sessions";
import { useWorkspaceCards } from "../features/workspace/use-workspace-cards";
import { useTurnLatencyProbe } from "../features/workspace/use-turn-latency-probe";
import { useWorkspaceStreamingTurn } from "../features/workspace/use-workspace-streaming-turn";
import { usePatientUploads } from "../features/workspace/use-patient-uploads";
import {
  primeDoctorClinicalWorkflow,
  readFiniteNumber,
  readWorkspaceErrorMessage,
  resolveActiveError,
} from "../features/workspace/workspace-flow-utils";

type SceneDrafts = Record<Scene, string>;

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

export function WorkspacePage() {
  const apiClient = useApiClient();
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
  const [sceneError, setSceneError] = useState<string | null>(null);
  const {
    activeProbeRef,
    activeProbe,
    beginTurn,
    clearActiveProbe,
    clearScene,
    markAborted,
    markError,
    markMessageDone,
    markUiComplete,
    latencyStatusForScene,
  } = useTurnLatencyProbe();
  const patientUploads = usePatientUploads({
    apiClient,
    patientSessionId: patient.state.sessionId,
    setPatientState: patient.setState,
    applyPatientResponse: (response) => applyResponseToScene("patient", response),
  });
  const patientTurn = useWorkspaceStreamingTurn({
    scene: "patient",
    apiClient,
    sessionState: patient.state,
    setSessionState: patient.setState,
    applySessionResponse: (response) => applyResponseToScene("patient", response),
    traceStoreRef,
    latencyProbe: {
      activeProbeRef,
      beginTurn,
      clearScene,
      markAborted,
      markError,
      clearActiveProbe,
      markMessageDone,
      markUiComplete,
    },
  });
  const doctorTurn = useWorkspaceStreamingTurn({
    scene: "doctor",
    apiClient,
    sessionState: doctor.state,
    setSessionState: doctor.setState,
    applySessionResponse: (response) => applyResponseToScene("doctor", response),
    traceStoreRef,
    latencyProbe: {
      activeProbeRef,
      beginTurn,
      clearScene,
      markAborted,
      markError,
      clearActiveProbe,
      markMessageDone,
      markUiComplete,
    },
    primeInitialState: primeDoctorClinicalWorkflow,
  });

  const activeSessionController = activeScene === "patient" ? patient : doctor;
  const activeSessionState = activeSessionController.state;
  const activeTurn = activeScene === "patient" ? patientTurn : doctorTurn;
  const doctorPatientId = readFiniteNumber(doctor.state.currentPatientId);
  const patientNav = usePatientWorkspaceNav();
  const workspaceCards = useWorkspaceCards({
    patient: patient.state,
    doctor: doctor.state,
  });

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

  useEffect(() => {
    setSceneError(null);
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
    const probe = activeProbe;
    if (!probe || probe.status === "ui_complete" || probe.status === "aborted" || probe.status === "error") {
      return;
    }

    const relevantState = probe.scene === "patient" ? patient.state : doctor.state;
    const errorMessage = relevantState.lastError?.message;
    if (!errorMessage) {
      return;
    }

    const errorAt = performance.now();
    markError({
      sequence: probe.sequence,
      scene: probe.scene,
      at: errorAt,
      message: errorMessage,
    });
    traceStoreRef.current.recordClientError(probe.traceId, errorAt);
  }, [activeProbe, patient.state.lastError, doctor.state.lastError, patient.state, doctor.state, markError]);

  useEffect(() => {
    const probe = activeProbe;
    if (!probe || probe.status !== "message_done") {
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
      const currentProbe = activeProbe;
      if (!currentProbe || currentProbe.sequence !== probe.sequence || currentProbe.status !== "message_done") {
        return;
      }

      const renderCommittedAt = performance.now();
      markUiComplete({
        sequence: currentProbe.sequence,
        scene: currentProbe.scene,
        at: renderCommittedAt,
        assistantCursor,
      });
      traceStoreRef.current.recordClientUiComplete(currentProbe.traceId, renderCommittedAt);
      emitTraceConsoleSummary(currentProbe.traceId);
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [
    patient.state.messages,
    patient.state.latestAssistantMessageCursor,
    doctor.state.messages,
    doctor.state.latestAssistantMessageCursor,
    activeProbe,
    markUiComplete,
  ]);

  useEffect(() => {
    const sessionId = activeSessionState.sessionId;
    if (
      bootstrapStatus !== "ready"
      || !sessionId
      || activeTurn.isStreaming
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
    activeTurn.isStreaming,
  ]);

  function handleSceneSwitch(scene: Scene) {
    if (scene === activeScene) {
      return;
    }

    activeTurn.abortActiveTurn("scene_switch");
    setSceneError(null);
    if (scene === "doctor") {
      patientUploads.clearUploadStatus();
    }
    setActiveScene(scene);
  }

  function updateDraft(scene: Scene, value: string) {
    setDrafts((current) => ({
      ...current,
      [scene]: value,
    }));
  }

  async function submitPrompt() {
    const prompt = drafts[activeScene].trim();
    if (!activeSessionState.sessionId || !prompt) {
      return;
    }

    updateDraft(activeScene, "");
    await activeTurn.submitPrompt(prompt);
  }

  async function handleResetActiveScene() {
    const didReset = await activeTurn.resetScene();
    if (!didReset) {
      return;
    }

    if (activeScene === "patient") {
      patientUploads.resetUploadState();
    }
    setSceneError(null);
    updateDraft(activeScene, "");
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
      setSceneError(readWorkspaceErrorMessage(error));
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

  const activeError = resolveActiveError({
    pageError: sceneError,
    turnError: activeTurn.errorMessage,
    uploadError: activeScene === "patient" ? patientUploads.errorMessage : null,
    sessionState: activeSessionState,
    bootstrapError,
  });
  const activeDraft = drafts[activeScene];
  const patientLatencyStatus = latencyStatusForScene("patient");
  const doctorLatencyStatus = latencyStatusForScene("doctor");

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
      disabled={patientUploads.isUploading || patientTurn.isStreaming}
      statusMessage={patientUploads.uploadStatus}
      onUpload={(file) => void patientUploads.uploadFile(file)}
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
        isStreaming={doctorTurn.isStreaming}
        isLoadingHistory={doctorTurn.isLoadingHistory}
        canLoadHistory={Boolean(doctor.state.messagesNextBeforeCursor)}
        disabled={doctorTurn.isStreaming || patientUploads.isUploading}
        errorMessage={activeError}
        latencyStatus={doctorLatencyStatus}
        roadmap={doctor.state.roadmap}
        stage={doctor.state.stage}
        plan={doctor.state.plan}
        cards={workspaceCards.doctorVisibleCards}
        references={doctor.state.references}
        onLoadHistory={() => void doctorTurn.loadMessageHistory()}
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
        items={patientNav.navItems}
        activeKey={patientNav.activeTab}
        onSelect={patientNav.selectTab}
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
              cards={workspaceCards.patientVisibleCards}
            />
          </div>
        </aside>
        <section className="clinical-patient-center-column" data-testid="workspace-center">
          <div className="workspace-panel-stack">
            {patientNav.activeTab === "upload" ? (
              patientUploadsPanel
            ) : (
              <ConversationPanel
                messages={patient.state.messages}
                draft={activeDraft}
                activeTriageQuestionId={workspaceCards.activePatientTriageQuestionId}
                statusNode={patient.state.statusNode}
                isStreaming={patientTurn.isStreaming}
                isLoadingHistory={patientTurn.isLoadingHistory}
                canLoadHistory={Boolean(patient.state.messagesNextBeforeCursor)}
                disabled={patientTurn.isStreaming || patientUploads.isUploading}
                errorMessage={activeError}
                latencyStatus={patientLatencyStatus}
                onLoadHistory={() => void patientTurn.loadMessageHistory()}
                onDraftChange={(value) => updateDraft("patient", value)}
                onSubmit={() => void submitPrompt()}
                onCardPromptRequest={(prompt: string, context?: Record<string, unknown>) =>
                  void patientTurn.submitPrompt(prompt, context)
                }
              />
            )}
          </div>
        </section>
        <aside className="clinical-patient-right-column" data-testid="workspace-right">
          <div className="workspace-panel-stack">
            {patientNav.activeTab === "profile" ? patientUploadsPanel : null}
          </div>
        </aside>
      </div>
    </main>
  );

}





