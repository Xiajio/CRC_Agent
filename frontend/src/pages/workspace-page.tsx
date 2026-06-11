import { useEffect, useMemo, useRef, useState } from "react";

import { buildChatLatencyTraceAnalysis, createChatLatencyTraceStore } from "../app/api/chat-latency-trace";
import type { FrontendMessage, Scene, SessionState } from "../app/api/types";
import { useApiClient } from "../app/providers";
import { ConversationPanel } from "../features/chat/conversation-panel";
import { ClinicalTopNav } from "../components/layout/clinical-top-nav";
import { useDocumentTheme } from "../components/layout/use-document-theme";
import { Card } from "../components/ui";
import { DoctorSceneShell } from "../features/doctor/doctor-scene-shell";
import type { CardPatientContext } from "../features/cards/card-renderers-extended";
import { PatientBackgroundPanel } from "../features/cards/patient-background-panel";
import { PatientIdentityPanel } from "../features/patient-identity/patient-identity-panel";
import { UploadsPanel } from "../features/uploads/uploads-panel";
import { useDatabaseWorkbench } from "../features/database/use-database-workbench";
import { usePatientRegistry } from "../features/patient-registry/use-patient-registry";
import { useRegistryBrowser } from "../features/patient-registry/use-registry-browser";
import {
  PATIENT_ASSISTANT_TAB,
  PATIENT_PROFILE_TAB,
  PATIENT_UPLOAD_TAB,
  usePatientWorkspaceNav,
} from "../features/workspace/use-patient-workspace-nav";
import { useSceneSessions } from "../features/workspace/use-scene-sessions";
import { usePatientUploads } from "../features/workspace/use-patient-uploads";
import { SessionRecoveryBanner } from "../features/workspace/session-recovery-banner";
import { useTurnLatencyProbe } from "../features/workspace/use-turn-latency-probe";
import { useWorkspaceCards } from "../features/workspace/use-workspace-cards";
import { useWorkspaceStreamingTurn } from "../features/workspace/use-workspace-streaming-turn";
import { buildReplayDemoContext } from "../features/workspace/demo-mode";
import { CLINICAL_DOCTOR_SCENE_ARIA_LABEL } from "../app/clinical/clinical-copy";
import {
  primeDoctorClinicalWorkflow,
  readFiniteNumber,
  readText,
  readWorkspaceErrorMessage,
  resolveActiveError,
} from "../features/workspace/workspace-flow-utils";

type SceneDrafts = Record<Scene, string>;

const PATIENT_ASSISTANT_QUICK_ACTIONS = [
  {
    id: "explain-report",
    label: "解释检查报告",
    prompt: "请帮我解释检查报告，说明关键指标、异常项和下一步需要补充的信息。",
  },
  {
    id: "add-symptoms",
    label: "补充病情信息",
    prompt: "我想补充病情信息，请引导我说明症状、持续时间和既往检查。",
  },
  {
    id: "treatment-options",
    label: "了解治疗建议",
    prompt: "我想了解治疗建议，请用患者能理解的语言说明不同方案的区别。",
  },
];

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

function sessionPatientContext(state: SessionState): CardPatientContext | undefined {
  const registryPatientId = readFiniteNumber(state.registryPatientId);
  const caseDatabasePatientId = readText(state.caseDatabasePatientId);
  const context: CardPatientContext = {};

  if (registryPatientId !== null) {
    context.registry_patient_id = registryPatientId;
  }
  if (caseDatabasePatientId !== null) {
    context.case_database_patient_id = caseDatabasePatientId;
  }

  return Object.keys(context).length > 0 ? context : undefined;
}

function formatCaseDatabasePatientId(patientId: number | string): string {
  return String(patientId).trim().padStart(3, "0");
}

function caseDatabasePatientIdFromPrompt(prompt: string): string | null {
  const match = prompt.match(/(?:患者|病人|patient|case)\s*#?\s*0*(\d{1,4})/i);
  if (!match) {
    return null;
  }
  return formatCaseDatabasePatientId(match[1]);
}

function promptPatientContext(scene: Scene, state: SessionState, prompt: string): CardPatientContext | undefined {
  const context = { ...(sessionPatientContext(state) ?? {}) };

  if (scene === "doctor" && context.case_database_patient_id == null) {
    const caseDatabasePatientId = caseDatabasePatientIdFromPrompt(prompt);
    if (caseDatabasePatientId !== null) {
      context.case_database_patient_id = caseDatabasePatientId;
    }
  }

  return Object.keys(context).length > 0 ? context : undefined;
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
    recoverScene,
    recoveryNotice,
    dismissRecoveryNotice,
  } = useSceneSessions();

  useDocumentTheme(activeScene === "doctor" ? "doctor-cockpit" : "patient-care");

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
  const latencyProbe = useMemo(
    () => ({
      activeProbeRef,
      beginTurn,
      clearScene,
      markAborted,
      markError,
      clearActiveProbe,
      markMessageDone,
      markUiComplete,
    }),
    [
      activeProbeRef,
      beginTurn,
      clearActiveProbe,
      clearScene,
      markAborted,
      markError,
      markMessageDone,
      markUiComplete,
    ],
  );
  const patientUploads = usePatientUploads({
    apiClient,
    patientSessionId: patient.state.sessionId,
    setPatientState: patient.setState,
    applyPatientResponse: (response) => applyResponseToScene("patient", response),
    onSessionExpired: recoverScene,
  });
  const patientTurn = useWorkspaceStreamingTurn({
    scene: "patient",
    apiClient,
    sessionState: patient.state,
    setSessionState: patient.setState,
    applySessionResponse: (response) => applyResponseToScene("patient", response),
    traceStoreRef,
    latencyProbe,
    onSessionExpired: recoverScene,
  });
  const doctorTurn = useWorkspaceStreamingTurn({
    scene: "doctor",
    apiClient,
    sessionState: doctor.state,
    setSessionState: doctor.setState,
    applySessionResponse: (response) => applyResponseToScene("doctor", response),
    traceStoreRef,
    latencyProbe,
    primeInitialState: primeDoctorClinicalWorkflow,
    onSessionExpired: recoverScene,
  });

  const activeSessionController = activeScene === "patient" ? patient : doctor;
  const activeSessionState = activeSessionController.state;
  const activeTurn = activeScene === "patient" ? patientTurn : doctorTurn;
  const registryPatientId = readFiniteNumber(doctor.state.registryPatientId);
  const doctorPatientContext = sessionPatientContext(doctor.state);
  const patientPatientContext = sessionPatientContext(patient.state);
  const patientNav = usePatientWorkspaceNav();
  const workspaceCards = useWorkspaceCards({
    patient: patient.state,
    doctor: doctor.state,
  });

  const patientRegistry = usePatientRegistry({
    enabled: activeScene === "doctor",
    registryPatientId,
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
    if (activeScene === "doctor") {
      patientUploads.clearUploadStatus();
    }
    patientUploads.clearError();
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
        (error) => {
          if (cancelled) {
            return;
          }
          // Session vanished server-side (most likely after a backend restart):
          // stop polling and let useSceneSessions recover so the user is not
          // looping on the stale id forever.
          if (error && (error as { status?: number }).status === 404) {
            cancelled = true;
            window.clearInterval(timer);
            void recoverScene(activeScene);
          }
        },
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
    recoverScene,
  ]);

  function handleSceneSwitch(scene: Scene) {
    if (scene === activeScene) {
      return;
    }

    activeTurn.abortActiveTurn("scene_switch");
    setSceneError(null);
    patientUploads.clearError();
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
    const baseContext = promptPatientContext(activeScene, activeSessionState, prompt);
    await activeTurn.submitPrompt(prompt, buildReplayDemoContext(activeScene, prompt, baseContext));
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

  function handleSetCurrentCaseDatabasePatient(patientId: number) {
    const caseDatabasePatientId = formatCaseDatabasePatientId(patientId);
    setSceneError(null);
    doctor.setState((current) => ({
      ...current,
      caseDatabasePatientId,
      currentPatientId: caseDatabasePatientId,
    }));
  }

  if (bootstrapStatus === "loading") {
    return <main className="clinical-page-shell"><Card variant="clinical-panel">正在加载工作区...</Card></main>;
  }

  if (bootstrapStatus === "error") {
    return (
      <main className="clinical-page-shell">
        <Card variant="clinical-panel">
          <h2>工作区初始化失败</h2>
          <p className="clinical-copy clinical-copy-alert">{bootstrapError ?? "未知初始化错误。"}</p>
        </Card>
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

  const recoveryBanner = recoveryNotice ? (
    <SessionRecoveryBanner message={recoveryNotice} onDismiss={dismissRecoveryNotice} />
  ) : null;

  if (activeScene === "doctor") {
    return (
      <>
        {recoveryBanner}
        <DoctorSceneShell
          toolbar={topNavActions}
        onSwitchScene={() => handleSceneSwitch("patient")}
        registryPatientId={registryPatientId}
        caseDatabasePatientId={doctor.state.caseDatabasePatientId}
        patientContext={doctorPatientContext}
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
        critic={doctor.state.critic}
        eventLog={doctor.state.eventLog}
        onLoadHistory={() => void doctorTurn.loadMessageHistory()}
        onDraftChange={(value) => updateDraft("doctor", value)}
        onSubmit={() => void submitPrompt()}
        onSetCurrentPatient={handleBindDoctorPatient}
        onSetCurrentCaseDatabasePatient={handleSetCurrentCaseDatabasePatient}
        onCardPromptRequest={(prompt: string, context?: Record<string, unknown>) =>
          void doctorTurn.submitPrompt(prompt, buildReplayDemoContext("doctor", prompt, context))
        }
      />
      </>
    );
  }

  const patientActiveTab = patientNav.activeTab;
  const patientIsAssistant = patientActiveTab === PATIENT_ASSISTANT_TAB;
  const patientIsProfile = patientActiveTab === PATIENT_PROFILE_TAB;
  const patientIsUpload = patientActiveTab === PATIENT_UPLOAD_TAB;

  const patientProfilePanel = (
    <div className="clinical-panel-stack clinical-patient-profile-stack">
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
  );

  const patientAssistantPanel = (
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
      emptyStateVariant="patient-assistant"
      quickActions={PATIENT_ASSISTANT_QUICK_ACTIONS}
      onQuickActionSelect={(prompt) => updateDraft("patient", prompt)}
      onUploadRequest={() => patientNav.selectTab(PATIENT_UPLOAD_TAB)}
      onLoadHistory={() => void patientTurn.loadMessageHistory()}
      onDraftChange={(value) => updateDraft("patient", value)}
      onSubmit={() => void submitPrompt()}
      patientContext={patientPatientContext}
      onCardPromptRequest={(prompt: string, context?: Record<string, unknown>) =>
        void patientTurn.submitPrompt(prompt, buildReplayDemoContext("patient", prompt, context))
      }
    />
  );

  const patientCenterContent = patientIsUpload
    ? patientUploadsPanel
    : patientIsProfile
      ? patientProfilePanel
      : patientAssistantPanel;
  const patientPanelError = activeError && !patientIsAssistant ? (
    <p className="clinical-copy clinical-copy-alert clinical-error-copy" data-testid="patient-active-error">
      {activeError}
    </p>
  ) : null;

  return (
    <main className="clinical-app-shell clinical-app-shell-patient">
      {recoveryBanner}
      <ClinicalTopNav
        brandLabel="临床助手"
        brandLogoVariant="light"
        navLabel="患者工作台"
        items={patientNav.navItems}
        activeKey={patientNav.activeTab}
        onSelect={patientNav.selectTab}
        actions={topNavActions}
        statusLabel="安全会话"
        statusTone="safe"
        profileLabel="患者"
        profileAriaLabel={CLINICAL_DOCTOR_SCENE_ARIA_LABEL}
        onProfileClick={() => handleSceneSwitch("doctor")}
        className="clinical-top-nav-patient"
      />
      <div
        className={[
          "clinical-patient-dashboard",
          `clinical-patient-dashboard-${patientActiveTab}`,
          patientIsAssistant && "clinical-patient-dashboard-assistant-home",
        ].filter(Boolean).join(" ")}
        data-testid="workspace-layout"
      >
        <aside
          className="clinical-patient-left-column clinical-patient-left-column-collapsed"
          data-testid="workspace-left-rail"
          aria-hidden="true"
        />
        <section className="clinical-patient-center-column" data-testid="workspace-center">
          <div className="clinical-panel-stack">
            {patientPanelError}
            {patientCenterContent}
          </div>
        </section>
        <aside
          className="clinical-patient-right-column clinical-patient-right-column-collapsed"
          data-testid="workspace-right"
          data-panel-state="closed"
          aria-hidden="true"
        />
      </div>
    </main>
  );

}
