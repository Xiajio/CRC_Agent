import { fireEvent, renderHook, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { useSceneSessions } from "../../frontend/src/features/workspace/use-scene-sessions";
import {
  buildApiClientStub,
  buildAppWrapper,
  makeDatabaseSearchResponse,
  makeDatabaseStatsResponse,
  makeNotFoundError,
  makePatientRegistryAlertsResponse,
  makePatientRegistryDetail,
  makePatientRegistryListResponse,
  makePatientRegistryRecordsResponse,
  makeSessionResponse,
  renderWorkspaceWithSceneSessions,
} from "./test-utils";

const PATIENT_SESSION_KEY = "langg.workspace.patient-session-id";
const DOCTOR_SESSION_KEY = "langg.workspace.doctor-session-id";

function installStorageMock() {
  const store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => {
        store.set(key, String(value));
      },
      removeItem: (key: string) => {
        store.delete(key);
      },
      clear: () => {
        store.clear();
      },
    },
  });
}

describe("scene-driven workspace", () => {
  beforeEach(() => {
    installStorageMock();
  });

  test("bootstraps patient and doctor sessions and recreates both when one persisted id returns 404", async () => {
    window.localStorage.setItem(PATIENT_SESSION_KEY, "stale-patient");
    window.localStorage.setItem(DOCTOR_SESSION_KEY, "stale-doctor");

    const getSession = vi.fn()
      .mockRejectedValueOnce(makeNotFoundError())
      .mockResolvedValueOnce(makeSessionResponse({ scene: "doctor", session_id: "stale-doctor" }));
    const createSession = vi.fn(async (scene: "patient" | "doctor") =>
      makeSessionResponse({ scene, session_id: `new-${scene}` }));
    const apiClient = buildApiClientStub({
      getSession,
      createSession,
    });

    const { result } = renderHook(() => useSceneSessions(), {
      wrapper: buildAppWrapper(apiClient),
    });

    await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));

    expect(createSession).toHaveBeenCalledTimes(2);
    expect(createSession.mock.calls.map((call) => call[0]).sort()).toEqual(["doctor", "patient"]);
    expect(window.localStorage.getItem(PATIENT_SESSION_KEY)).toBe("new-patient");
    expect(window.localStorage.getItem(DOCTOR_SESSION_KEY)).toBe("new-doctor");
  });

  test("doctor consultation general mode renders conversation and database registry CTA", async () => {
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    expect(await screen.findByTestId("general-clinical-mode")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-panel")).toBeInTheDocument();
    expect(screen.getByTestId("clinical-cards-panel")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open database > patient registry/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /open database > patient registry/i }));

    expect(await screen.findByRole("button", { name: /historical case base/i })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /patient registry/i })).toBeInTheDocument();
    expect(await screen.findByTestId("recent-patients-panel")).toBeInTheDocument();
  });

  test("registry preview does not bind until set as current patient", async () => {
    const bindPatient = vi.fn(async (sessionId: string, patientId: number) =>
      makeSessionResponse({
        scene: "doctor",
        session_id: sessionId,
        patient_id: patientId,
        snapshot: { current_patient_id: patientId },
      }));
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
      getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
      getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
      getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
      bindPatient,
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    fireEvent.click(await screen.findByRole("button", { name: /open database > patient registry/i }));
    fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));

    expect(await screen.findByTestId("registry-preview-panel")).toHaveTextContent(/#33/);
    expect(bindPatient).not.toHaveBeenCalled();
  });

  test("set as current patient binds and preserves registry preview state across tabs", async () => {
    const bindPatient = vi.fn(async (sessionId: string, patientId: number) =>
      makeSessionResponse({
        scene: "doctor",
        session_id: sessionId,
        patient_id: patientId,
        snapshot: { current_patient_id: patientId },
      }));
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
      getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
      getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
      getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
      bindPatient,
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    fireEvent.click(await screen.findByRole("button", { name: /open database > patient registry/i }));
    fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));
    fireEvent.click(await screen.findByRole("button", { name: /set current patient 33/i }));

    await waitFor(() => expect(bindPatient).toHaveBeenCalledWith("sess-doctor", 33));
    expect(await screen.findByRole("button", { name: /consultation workspace/i })).toBeInTheDocument();
    expect(await screen.findByRole("region", { name: /current patient summary/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /database/i }));
    expect(await screen.findByTestId("registry-preview-panel")).toHaveTextContent(/#33/);
  });

  test("registry browser deletes the previewed patient and refreshes the recent list", async () => {
    const deletePatientRegistryPatient = vi.fn(async () => ({
      patient_id: 33,
      deleted_records: 1,
      deleted_assets: 1,
      deleted_asset_paths: [],
      record_ids: [77],
    }));
    const getRecentPatients = vi.fn()
      .mockResolvedValueOnce(makePatientRegistryListResponse())
      .mockResolvedValueOnce(makePatientRegistryListResponse({ items: [], total: 0 }));
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients,
      getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
      getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
      getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
      deletePatientRegistryPatient,
    });
    vi.stubGlobal("confirm", vi.fn(() => true));

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    fireEvent.click(await screen.findByRole("button", { name: /open database > patient registry/i }));
    fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));
    fireEvent.click(await screen.findByRole("button", { name: /delete patient 33/i }));

    await waitFor(() => expect(deletePatientRegistryPatient).toHaveBeenCalledWith(33));
    await waitFor(() => expect(screen.queryByRole("button", { name: /preview patient 33/i })).not.toBeInTheDocument());
  });

  test("registry browser clears the registry and empties the recent list", async () => {
    const clearPatientRegistry = vi.fn(async () => ({
      deleted_patients: 1,
      deleted_records: 1,
      deleted_assets: 1,
      patient_ids: [33],
      deleted_asset_paths: [],
    }));
    const getRecentPatients = vi.fn()
      .mockResolvedValueOnce(makePatientRegistryListResponse())
      .mockResolvedValueOnce(makePatientRegistryListResponse({ items: [], total: 0 }));
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients,
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
      clearPatientRegistry,
    });
    vi.stubGlobal("confirm", vi.fn(() => true));

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    fireEvent.click(await screen.findByRole("button", { name: /open database > patient registry/i }));
    fireEvent.click(await screen.findByRole("button", { name: /clear registry/i }));

    await waitFor(() => expect(clearPatientRegistry).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByRole("button", { name: /preview patient 33/i })).not.toBeInTheDocument());
  });

  test("doctor database tab keeps historical case base available", async () => {
    const getDatabaseStats = vi.fn(async () => makeDatabaseStatsResponse());
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
      getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
      getDatabaseStats,
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
    fireEvent.click(screen.getByRole("button", { name: /^database$/i }));

    await waitFor(() => expect(getDatabaseStats).toHaveBeenCalled());
    expect(await screen.findByTestId("database-workbench-panel")).toBeInTheDocument();
  });

  test("bound registry patient does not trigger historical detail loading on consultation view", async () => {
    const getDatabaseCaseDetail = vi.fn(async () => ({
      patient_id: "33",
      case_record: { patient_id: 33, clinical_stage: "cT3N1M0" },
      available_data: { case_info: true, imaging: false, pathology_slides: false },
      cards: {},
    }));
    const apiClient = buildApiClientStub({
      createSession: vi.fn(async (scene: "patient" | "doctor") =>
        makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : 33 })),
      getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
      getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
      getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
      getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
      getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
      searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
      getDatabaseCaseDetail,
    });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));

    expect(await screen.findByRole("region", { name: /current patient summary/i })).toBeInTheDocument();
    expect(getDatabaseCaseDetail).not.toHaveBeenCalled();
  });
});
