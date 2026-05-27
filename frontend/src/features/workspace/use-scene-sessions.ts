import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from "react";

import { ApiClientError } from "../../app/api/client";
import type { Scene, SessionResponse, SessionState } from "../../app/api/types";
import { useApiClient } from "../../app/providers";
import { createInitialSessionState, hydrateSessionState } from "../../app/store/stream-reducer";

export type SceneBootstrapStatus = "loading" | "ready" | "error";

export const PATIENT_SESSION_STORAGE_KEY = "langg.workspace.patient-session-id";
export const DOCTOR_SESSION_STORAGE_KEY = "langg.workspace.doctor-session-id";
const ACTIVE_SCENE_STORAGE_KEY = "langg.workspace.active-scene";

export interface SceneSessionController {
  scene: Scene;
  state: SessionState;
  setState: Dispatch<SetStateAction<SessionState>>;
}

function readPersistedSessionId(storageKey: string): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return window.localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function persistSessionId(storageKey: string, sessionId: string): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(storageKey, sessionId);
  } catch {
    return;
  }
}

function clearPersistedSessionId(storageKey: string): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.removeItem(storageKey);
  } catch {
    return;
  }
}

function isScene(value: string | null): value is Scene {
  return value === "patient" || value === "doctor";
}

function readPersistedActiveScene(): Scene {
  if (typeof window === "undefined") {
    return "patient";
  }

  try {
    const persistedScene = window.localStorage.getItem(ACTIVE_SCENE_STORAGE_KEY);
    return isScene(persistedScene) ? persistedScene : "patient";
  } catch {
    return "patient";
  }
}

function readErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "工作区初始化失败。";
}

function sceneStorageKey(scene: Scene): string {
  return scene === "patient" ? PATIENT_SESSION_STORAGE_KEY : DOCTOR_SESSION_STORAGE_KEY;
}

type LoadedSessionResult = {
  response: SessionResponse | null;
  stale: boolean;
};

export const SESSION_RECOVERY_NOTICE =
  "后端会话已失效，已为您创建新会话；上一轮历史已无法继续，请重新发送或重新上传。";

export function useSceneSessions() {
  const apiClient = useApiClient();
  const [activeScene, setActiveSceneState] = useState<Scene>(() => readPersistedActiveScene());
  const [bootstrapStatus, setBootstrapStatus] = useState<SceneBootstrapStatus>("loading");
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [bootstrapRecoveredScenes, setBootstrapRecoveredScenes] = useState<Scene[]>([]);
  const [patientState, setPatientState] = useState<SessionState>(() => createInitialSessionState());
  const [doctorState, setDoctorState] = useState<SessionState>(() => createInitialSessionState());
  const [recoveryNotice, setRecoveryNotice] = useState<string | null>(null);

  const setActiveScene = useCallback((scene: Scene) => {
    setActiveSceneState(scene);
    persistSessionId(ACTIVE_SCENE_STORAGE_KEY, scene);
  }, []);

  const applyResponseToScene = useCallback((scene: Scene, response: SessionResponse) => {
    persistSessionId(sceneStorageKey(scene), response.session_id);
    const setState = scene === "patient" ? setPatientState : setDoctorState;
    setState((current) => hydrateSessionState(current, response));
  }, []);

  const recoverScene = useCallback(
    async (scene: Scene): Promise<SessionResponse | null> => {
      try {
        const response = await apiClient.createSession(scene);
        applyResponseToScene(scene, response);
        setRecoveryNotice(SESSION_RECOVERY_NOTICE);
        return response;
      } catch {
        // The caller is responsible for surfacing the original error to the
        // user; we deliberately swallow the recovery failure so that we never
        // shadow the actionable upstream message.
        return null;
      }
    },
    [apiClient, applyResponseToScene],
  );

  const dismissRecoveryNotice = useCallback(() => {
    setRecoveryNotice(null);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadStoredSession(scene: Scene): Promise<LoadedSessionResult> {
      const persistedSessionId = readPersistedSessionId(sceneStorageKey(scene));
      if (!persistedSessionId) {
        return { response: null, stale: false };
      }

      try {
        return {
          response: await apiClient.getSession(persistedSessionId),
          stale: false,
        };
      } catch (error) {
        if (error instanceof ApiClientError && error.status === 404) {
          return { response: null, stale: true };
        }
        throw error;
      }
    }

    async function bootstrap() {
      setBootstrapStatus("loading");
      setBootstrapError(null);

      try {
        const [patientLoaded, doctorLoaded] = await Promise.all([
          loadStoredSession("patient"),
          loadStoredSession("doctor"),
        ]);

        let patientResponse = patientLoaded.response;
        let doctorResponse = doctorLoaded.response;

        if (patientLoaded.stale) {
          clearPersistedSessionId(PATIENT_SESSION_STORAGE_KEY);
        }
        if (doctorLoaded.stale) {
          clearPersistedSessionId(DOCTOR_SESSION_STORAGE_KEY);
        }

        const recovered: Scene[] = [];
        if (patientResponse === null) {
          patientResponse = await apiClient.createSession("patient");
          if (patientLoaded.stale) {
            recovered.push("patient");
          }
        }
        if (doctorResponse === null) {
          doctorResponse = await apiClient.createSession("doctor");
          if (doctorLoaded.stale) {
            recovered.push("doctor");
          }
        }

        if (cancelled || patientResponse === null || doctorResponse === null) {
          return;
        }

        applyResponseToScene("patient", patientResponse);
        applyResponseToScene("doctor", doctorResponse);
        setBootstrapRecoveredScenes(recovered);
        if (recovered.length > 0) {
          setRecoveryNotice(SESSION_RECOVERY_NOTICE);
        }
        setBootstrapStatus("ready");
      } catch (error) {
        if (cancelled) {
          return;
        }
        setBootstrapError(readErrorMessage(error));
        setBootstrapStatus("error");
      }
    }

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, [apiClient, applyResponseToScene]);

  const patient: SceneSessionController = {
    scene: "patient",
    state: patientState,
    setState: setPatientState,
  };

  const doctor: SceneSessionController = {
    scene: "doctor",
    state: doctorState,
    setState: setDoctorState,
  };

  return {
    activeScene,
    setActiveScene,
    bootstrapStatus,
    bootstrapError,
    bootstrapRecoveredScenes,
    patient,
    doctor,
    applyResponseToScene,
    recoverScene,
    recoveryNotice,
    dismissRecoveryNotice,
  };
}
