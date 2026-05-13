import "@testing-library/jest-dom/vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PATIENT_SESSION_STORAGE_KEY, DOCTOR_SESSION_STORAGE_KEY, useSceneSessions } from "./use-scene-sessions";
import { AppProviders } from "../../app/providers";
import { buildApiClientStub, makeNotFoundError, makeSessionResponse } from "../../test/test-utils";

const ACTIVE_SCENE_STORAGE_KEY = "langg.workspace.active-scene";

function renderSceneSessions(apiClient = buildApiClientStub()) {
  return renderHook(() => useSceneSessions(), {
    wrapper: ({ children }) => <AppProviders apiClient={apiClient}>{children}</AppProviders>,
  });
}

describe("useSceneSessions", () => {
  afterEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it("defaults to the patient scene when no active scene is persisted", async () => {
    const { result } = renderSceneSessions();

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));
    expect(result.current.activeScene).toBe("patient");
  });

  it("restores the persisted doctor scene", async () => {
    window.localStorage.setItem(ACTIVE_SCENE_STORAGE_KEY, "doctor");

    const { result } = renderSceneSessions();

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));
    expect(result.current.activeScene).toBe("doctor");
  });

  it("ignores invalid persisted active scene values", async () => {
    window.localStorage.setItem(ACTIVE_SCENE_STORAGE_KEY, "database");

    const { result } = renderSceneSessions();

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));
    expect(result.current.activeScene).toBe("patient");
  });

  it("persists active scene switches", async () => {
    const { result } = renderSceneSessions();

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));
    act(() => {
      result.current.setActiveScene("doctor");
    });
    expect(result.current.activeScene).toBe("doctor");
    expect(window.localStorage.getItem(ACTIVE_SCENE_STORAGE_KEY)).toBe("doctor");

    act(() => {
      result.current.setActiveScene("patient");
    });
    expect(result.current.activeScene).toBe("patient");
    expect(window.localStorage.getItem(ACTIVE_SCENE_STORAGE_KEY)).toBe("patient");
  });

  it("recreates only the expired scene session while preserving the other restored scene", async () => {
    window.localStorage.setItem(PATIENT_SESSION_STORAGE_KEY, "stale-patient-session");
    window.localStorage.setItem(DOCTOR_SESSION_STORAGE_KEY, "doctor-session");

    const apiClient = buildApiClientStub({
      getSession: vi.fn(async (sessionId: string) => {
        if (sessionId === "stale-patient-session") {
          throw makeNotFoundError();
        }
        return makeSessionResponse({
          session_id: sessionId,
          scene: "doctor",
          patient_id: 202,
          snapshot: { registry_patient_id: 202 },
        });
      }),
      createSession: vi.fn(async (scene) =>
        makeSessionResponse({
          scene,
          session_id: scene === "patient" ? "new-patient-session" : "unexpected-doctor-session",
          patient_id: scene === "patient" ? 101 : null,
          snapshot: { registry_patient_id: scene === "patient" ? 101 : null },
        }),
      ),
    });

    const { result } = renderSceneSessions(apiClient);

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));

    expect(apiClient.getSession).toHaveBeenCalledTimes(2);
    expect(apiClient.createSession).toHaveBeenCalledTimes(1);
    expect(apiClient.createSession).toHaveBeenCalledWith("patient");
    expect(result.current.patient.state.sessionId).toBe("new-patient-session");
    expect(result.current.doctor.state.sessionId).toBe("doctor-session");
    expect(result.current.patient.state.registryPatientId).toBe(101);
    expect(result.current.patient.state.currentPatientId).toBeNull();
    expect(result.current.doctor.state.registryPatientId).toBe(202);
    expect(result.current.doctor.state.caseDatabasePatientId).toBeNull();
    expect(result.current.doctor.state.currentPatientId).toBeNull();
    expect(window.localStorage.getItem(PATIENT_SESSION_STORAGE_KEY)).toBe("new-patient-session");
    expect(window.localStorage.getItem(DOCTOR_SESSION_STORAGE_KEY)).toBe("doctor-session");
  });
});
