import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from "react";

import { ApiClientError } from "../../app/api/client";
import type { Scene, SessionResponse, SessionState } from "../../app/api/types";
import { useApiClient } from "../../app/providers";
import { createInitialSessionState, hydrateSessionState } from "../../app/store/stream-reducer";

export type SceneBootstrapStatus = "loading" | "ready" | "error";

export const PATIENT_SESSION_STORAGE_KEY = "langg.workspace.patient-session-id";
export const DOCTOR_SESSION_STORAGE_KEY = "langg.workspace.doctor-session-id";

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

export function useSceneSessions() {
  const apiClient = useApiClient();
  const [activeScene, setActiveScene] = useState<Scene>("doctor");
  const [bootstrapStatus, setBootstrapStatus] = useState<SceneBootstrapStatus>("loading");
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [patientState, setPatientState] = useState<SessionState>(() => createInitialSessionState());
  const [doctorState, setDoctorState] = useState<SessionState>(() => createInitialSessionState());

  const applyResponseToScene = useCallback((scene: Scene, response: SessionResponse) => {
    persistSessionId(sceneStorageKey(scene), response.session_id);
    const setState = scene === "patient" ? setPatientState : setDoctorState;
    setState((current) => hydrateSessionState(current, response));
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

        if (patientResponse === null) {
          patientResponse = await apiClient.createSession("patient");
        }
        if (doctorResponse === null) {
          doctorResponse = await apiClient.createSession("doctor");
        }

        if (cancelled || patientResponse === null || doctorResponse === null) {
          return;
        }

        applyResponseToScene("patient", patientResponse);
        applyResponseToScene("doctor", doctorResponse);
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
    patient,
    doctor,
    applyResponseToScene,
  };
}
