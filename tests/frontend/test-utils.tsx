import type { PropsWithChildren, ReactElement } from "react";
import { render } from "@testing-library/react";
import { vi } from "vitest";

import { AppProviders } from "../../frontend/src/app/providers";
import type { ApiClient } from "../../frontend/src/app/api/client";
import { ApiClientError } from "../../frontend/src/app/api/client";
import type {
  DatabaseCaseDetailResponse,
  DatabaseSearchResponse,
  DatabaseStatsResponse,
  PatientRegistryAlertsResponse,
  PatientRegistryDetail,
  PatientRegistryListResponse,
  PatientRegistryRecordsResponse,
  RecoverySnapshot,
  Scene,
  SessionResponse,
} from "../../frontend/src/app/api/types";
import { WorkspacePage } from "../../frontend/src/pages/workspace-page";

export function makeSessionResponse(
  overrides: Partial<SessionResponse> & { snapshot?: Partial<RecoverySnapshot> } = {},
): SessionResponse {
  const scene = overrides.scene ?? "patient";
  const patientId = overrides.patient_id ?? (scene === "patient" ? 101 : null);
  return {
    session_id: overrides.session_id ?? `sess-${scene}`,
    thread_id: overrides.thread_id ?? `thread-${scene}`,
    scene,
    patient_id: patientId,
    snapshot_version: overrides.snapshot_version ?? 0,
    snapshot: {
      snapshot_version: overrides.snapshot?.snapshot_version ?? overrides.snapshot_version ?? 0,
      messages: overrides.snapshot?.messages ?? [],
      messages_total: overrides.snapshot?.messages_total ?? 0,
      messages_next_before_cursor: overrides.snapshot?.messages_next_before_cursor ?? null,
      cards: overrides.snapshot?.cards ?? [],
      roadmap: overrides.snapshot?.roadmap ?? [],
      findings: overrides.snapshot?.findings ?? {},
      patient_profile: overrides.snapshot?.patient_profile ?? null,
      stage: overrides.snapshot?.stage ?? null,
      assessment_draft: overrides.snapshot?.assessment_draft ?? null,
      current_patient_id: overrides.snapshot?.current_patient_id ?? patientId,
      references: overrides.snapshot?.references ?? [],
      plan: overrides.snapshot?.plan ?? [],
      critic: overrides.snapshot?.critic ?? null,
      safety_alert: overrides.snapshot?.safety_alert ?? null,
      uploaded_assets: overrides.snapshot?.uploaded_assets ?? {},
      context_maintenance: overrides.snapshot?.context_maintenance ?? null,
      context_state: overrides.snapshot?.context_state ?? null,
    },
    runtime: overrides.runtime ?? {
      runner_mode: "real",
      fixture_case: null,
    },
  };
}

export function makeDatabaseStatsResponse(
  overrides: Partial<DatabaseStatsResponse> = {},
): DatabaseStatsResponse {
  return {
    total_cases: overrides.total_cases ?? 2,
    gender_distribution: overrides.gender_distribution ?? { male: 1, female: 1 },
    age_statistics: overrides.age_statistics ?? { min: 45, max: 60, mean: 52.5 },
    tumor_location_distribution: overrides.tumor_location_distribution ?? { rectum: 2 },
    ct_stage_distribution: overrides.ct_stage_distribution ?? { cT3: 1, cT2: 1 },
    mmr_status_distribution: overrides.mmr_status_distribution ?? { pMMR: 1, dMMR: 1 },
    cea_statistics: overrides.cea_statistics ?? { min: 2, max: 8, mean: 5 },
  };
}

export function makeDatabaseSearchResponse(
  overrides: Partial<DatabaseSearchResponse> = {},
): DatabaseSearchResponse {
  return {
    items: overrides.items ?? [
      {
        patient_id: 33,
        age: 52,
        gender: "female",
        ecog_score: 1,
        tumor_location: "rectum",
        clinical_stage: "cT3N1M0",
        mmr_status: "dMMR",
      },
    ],
    total: overrides.total ?? 1,
    page: overrides.page ?? 1,
    page_size: overrides.page_size ?? 20,
    applied_filters: overrides.applied_filters ?? {},
    warnings: overrides.warnings ?? [],
  };
}

export function makePatientRegistryListResponse(
  overrides: Partial<PatientRegistryListResponse> = {},
): PatientRegistryListResponse {
  return {
    items: overrides.items ?? [
      {
        patient_id: 33,
        status: "draft",
        created_by_session_id: "sess-patient",
        updated_at: "2026-04-16T00:00:00Z",
        tumor_location: "rectum",
        mmr_status: "dMMR",
        clinical_stage: "cT3N1M0",
      },
    ],
    total: overrides.total ?? ((overrides.items ?? []).length || 1),
  };
}

export function makePatientRegistryDetail(
  overrides: Partial<PatientRegistryDetail> = {},
): PatientRegistryDetail {
  return {
    patient_id: overrides.patient_id ?? 33,
    status: overrides.status ?? "draft",
    created_by_session_id: overrides.created_by_session_id ?? "sess-patient",
    created_at: overrides.created_at ?? "2026-04-16T00:00:00Z",
    updated_at: overrides.updated_at ?? "2026-04-16T00:00:00Z",
    chief_complaint: overrides.chief_complaint ?? "rectal bleeding",
    age: overrides.age ?? 52,
    gender: overrides.gender ?? "female",
    tumor_location: overrides.tumor_location ?? "rectum",
    mmr_status: overrides.mmr_status ?? "dMMR",
    clinical_stage: overrides.clinical_stage ?? "cT3N1M0",
    t_stage: overrides.t_stage ?? "T3",
    n_stage: overrides.n_stage ?? "N1",
    m_stage: overrides.m_stage ?? "M0",
  };
}

export function makePatientRegistryRecordsResponse(
  overrides: Partial<PatientRegistryRecordsResponse> = {},
): PatientRegistryRecordsResponse {
  return {
    items: overrides.items ?? [],
  };
}

export function makePatientRegistryAlertsResponse(
  overrides: Partial<PatientRegistryAlertsResponse> = {},
): PatientRegistryAlertsResponse {
  return {
    items: overrides.items ?? [],
  };
}

export function buildApiClientStub(overrides: Partial<ApiClient> = {}): ApiClient {
  const createSession = vi.fn(async (scene: Scene) => makeSessionResponse({ scene }));
  const getSession = vi.fn(async (sessionId: string) => makeSessionResponse({ session_id: sessionId }));
  const getMessages = vi.fn(async () => ({
    session_id: "sess-patient",
    thread_id: "thread-patient",
    snapshot_version: 0,
    messages_total: 0,
    next_before_cursor: null,
    messages: [],
  }));
  const streamTurn = vi.fn(async () => undefined);
  const uploadFile = vi.fn(async () => ({
    asset_id: "1",
    filename: "report.pdf",
    content_type: "application/pdf",
    size: 4,
    sha256: "sha",
    reused: false,
    derived: { record_id: 1 },
  }));
  const resetSession = vi.fn(async (sessionId: string) => makeSessionResponse({ session_id: sessionId, scene: "patient" }));
  const bindPatient = vi.fn(async (sessionId: string, patientId: number) =>
    makeSessionResponse({
      session_id: sessionId,
      scene: "doctor",
      patient_id: patientId,
      snapshot: { current_patient_id: patientId },
    }),
  );
  const getDatabaseStats = vi.fn(async () => makeDatabaseStatsResponse());
  const searchDatabaseCases = vi.fn(async () => makeDatabaseSearchResponse());
  const getDatabaseCaseDetail = vi.fn(async (patientId: number): Promise<DatabaseCaseDetailResponse> => ({
    patient_id: String(patientId),
    case_record: { patient_id: patientId, clinical_stage: "cT3N1M0" },
    available_data: { case_info: true, imaging: false, pathology_slides: false },
    cards: {},
  }));
  const upsertDatabaseCase = vi.fn(async () => ({
    patient_id: "33",
    case_record: { patient_id: 33 },
    available_data: { case_info: true, imaging: false, pathology_slides: false },
    cards: {},
  }));
  const parseDatabaseQueryIntent = vi.fn(async (query: string) => ({
    query,
    normalized_query: query,
    filters: {},
    unsupported_terms: [],
    warnings: [],
  }));
  const getRecentPatients = vi.fn(async () => makePatientRegistryListResponse());
  const searchPatientRegistry = vi.fn(async () => makePatientRegistryListResponse());
  const getPatientRegistryDetail = vi.fn(async () => makePatientRegistryDetail());
  const getPatientRecords = vi.fn(async () => makePatientRegistryRecordsResponse());
  const getPatientRegistryAlerts = vi.fn(async () => makePatientRegistryAlertsResponse());
  const deletePatientRegistryPatient = vi.fn(async () => ({
    patient_id: 33,
    deleted_records: 1,
    deleted_assets: 1,
    deleted_asset_paths: [],
    record_ids: [],
  }));
  const clearPatientRegistry = vi.fn(async () => ({
    deleted_patients: 1,
    deleted_records: 1,
    deleted_assets: 1,
    patient_ids: [33],
    deleted_asset_paths: [],
  }));

  return {
    createSession,
    getSession,
    getMessages,
    streamTurn,
    uploadFile,
    resetSession,
    bindPatient,
    getDatabaseStats,
    searchDatabaseCases,
    getDatabaseCaseDetail,
    upsertDatabaseCase,
    parseDatabaseQueryIntent,
    getRecentPatients,
    searchPatientRegistry,
    getPatientRegistryDetail,
    getPatientRecords,
    getPatientRegistryAlerts,
    deletePatientRegistryPatient,
    clearPatientRegistry,
    ...overrides,
  };
}

export function buildStreamingApiClientStub(overrides: Partial<ApiClient> = {}): ApiClient {
  return buildApiClientStub(overrides);
}

export function buildAppWrapper(apiClient: ApiClient) {
  return function AppWrapper({ children }: PropsWithChildren): ReactElement {
    return <AppProviders apiClient={apiClient}>{children}</AppProviders>;
  };
}

export function renderWorkspaceWithSceneSessions(apiClient: ApiClient) {
  return render(
    <AppProviders apiClient={apiClient}>
      <WorkspacePage />
    </AppProviders>,
  );
}

export function makeNotFoundError(detail = "Session not found") {
  return new ApiClientError(404, detail, { detail });
}
