import "@testing-library/jest-dom/vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PATIENT_SESSION_STORAGE_KEY, DOCTOR_SESSION_STORAGE_KEY, useSceneSessions } from "./use-scene-sessions";
import { AppProviders } from "../../app/providers";
import { buildApiClientStub, makeNotFoundError, makeSessionResponse } from "../../test/test-utils";

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
          snapshot: { current_patient_id: 202 },
        });
      }),
      createSession: vi.fn(async (scene) =>
        makeSessionResponse({
          scene,
          session_id: scene === "patient" ? "new-patient-session" : "unexpected-doctor-session",
          patient_id: scene === "patient" ? 101 : null,
          snapshot: { current_patient_id: scene === "patient" ? 101 : null },
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
    expect(result.current.doctor.state.currentPatientId).toBe(202);
    expect(window.localStorage.getItem(PATIENT_SESSION_STORAGE_KEY)).toBe("new-patient-session");
    expect(window.localStorage.getItem(DOCTOR_SESSION_STORAGE_KEY)).toBe("doctor-session");
  });
});
