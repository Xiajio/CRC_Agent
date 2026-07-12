import type { PropsWithChildren, ReactElement } from "react";
import { render } from "@testing-library/react";
import { vi } from "vitest";

import { AppProviders } from "../app/providers";
import type { ApiClient } from "../app/api/client";
import { ApiClientError } from "../app/api/client";
import type {
  AdminReleaseClosureResponse,
  AdminReleaseDashboardResponse,
  AdminReleaseExecutionResponse,
  AdminReleaseGovernanceResponse,
  AdminReleaseMonitoringResponse,
  AdminRulesResponse,
  AdminToolManifestResponse,
  DatabaseCaseDetailResponse,
  DatabaseSearchResponse,
  DatabaseStatsResponse,
  DoctorActionTraceResponse,
  DoctorReviewResponse,
  PatientIdentitySnapshot,
  PatientCareCardsResponse,
  PatientRegistryAlertsResponse,
  PatientRegistryDetail,
  PatientRegistryListResponse,
  PatientRegistryRecordsResponse,
  RecoverySnapshot,
  SaveCrcTriageAssessmentResponse,
  Scene,
  SessionResponse,
} from "../app/api/types";
import { WorkspacePage } from "../pages/workspace-page";

export function makeSessionResponse(
  overrides: Omit<Partial<SessionResponse>, "snapshot"> & { snapshot?: Partial<RecoverySnapshot> } = {},
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
      patient_identity: overrides.snapshot?.patient_identity ?? null,
      stage: overrides.snapshot?.stage ?? null,
      assessment_draft: overrides.snapshot?.assessment_draft ?? null,
      case_database_patient_id: overrides.snapshot?.case_database_patient_id ?? null,
      registry_patient_id: overrides.snapshot?.registry_patient_id ?? patientId,
      current_patient_id: overrides.snapshot?.current_patient_id ?? null,
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
    total: overrides.total ?? ((overrides.items ?? []).length || 1),
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

export function makePatientCareCardsResponse(
  overrides: Partial<PatientCareCardsResponse> = {},
): PatientCareCardsResponse {
  return {
    focusMetrics: overrides.focusMetrics ?? [],
    periodicChecks: overrides.periodicChecks ?? [],
    dailyActions: overrides.dailyActions ?? [],
  };
}

export function makeDoctorReviewResponse(
  overrides: Partial<DoctorReviewResponse> = {},
): DoctorReviewResponse {
  return {
    patient_id: overrides.patient_id ?? 101,
    session_id: overrides.session_id ?? "sess-doctor",
    feature_flag: overrides.feature_flag ?? "doctor_review_cockpit_v0",
    timeline: overrides.timeline ?? [],
    assertions: overrides.assertions ?? [],
    draft: overrides.draft ?? {
      draft_id: "draft-101",
      sections: [
        {
          section_id: "risk_summary",
          text: "Traceable risk summary.",
          provenance: [],
          verification_status: "traceable",
        },
      ],
    },
    available_actions: overrides.available_actions ?? [
      "accept",
      "edit",
      "reject",
      "escalate",
      "request_evidence",
      "mark_unsafe",
    ],
  };
}

export function makeDoctorActionTraceResponse(
  overrides: Partial<DoctorActionTraceResponse> = {},
): DoctorActionTraceResponse {
  return {
    patient_id: overrides.patient_id ?? 101,
    patient_version: overrides.patient_version ?? 1,
    projection_version: overrides.projection_version ?? 1,
    event_ids: overrides.event_ids ?? ["event-1"],
    snapshot_changed: overrides.snapshot_changed ?? false,
    trace: overrides.trace ?? {
      trace_id: "doctor_trace_1",
      patient_id: overrides.patient_id ?? 101,
      session_id: "sess-doctor",
      action_type: "accept",
      target_object: null,
      target_refs: {
        assertion_id: "assertion-1",
      },
      before_after: null,
      reason_code: "unsupported_claim",
      reviewer_role: "physician_reviewer",
      deidentified: true,
      timestamp: "2026-06-29T04:00:00Z",
    },
  };
}

export function makePatientRegistryAlertsResponse(
  overrides: Partial<PatientRegistryAlertsResponse> = {},
): PatientRegistryAlertsResponse {
  return {
    items: overrides.items ?? [],
  };
}

export function makeAdminReleaseExecutionResponse(
  overrides: Partial<AdminReleaseExecutionResponse> = {},
): AdminReleaseExecutionResponse {
  return {
    governance: overrides.governance ?? {
      active_intent_id: null,
      derived_status: null,
      required_approvals_complete: false,
      rollback_plan_id: null,
    },
    preflight: overrides.preflight ?? {
      release: { allowed: false, reasons: ["no active governance intent"] },
      rollback: { allowed: false, reasons: ["no successful release execution exists"] },
    },
    feature_flag_state: overrides.feature_flag_state ?? null,
    requests: overrides.requests ?? [],
    results: overrides.results ?? [],
    audit_events: overrides.audit_events ?? [],
    integrity: overrides.integrity ?? {
      status: "verified",
      warnings: [],
    },
    runtime: overrides.runtime ?? {
      auth: "admin",
      source: "reports/release_execution",
      mode: "controlled_local_execution",
    },
  };
}

export function makeAdminReleaseMonitoringResponse(
  overrides: Partial<AdminReleaseMonitoringResponse> = {},
): AdminReleaseMonitoringResponse {
  return {
    status: overrides.status ?? "idle",
    latest_release: overrides.latest_release ?? null,
    required_checks: overrides.required_checks ?? [],
    checks: overrides.checks ?? [],
    alerts: overrides.alerts ?? [],
    rollback_trigger_candidate: overrides.rollback_trigger_candidate ?? null,
    acknowledgements: overrides.acknowledgements ?? [],
    integrity: overrides.integrity ?? {
      status: "verified",
      warnings: [],
    },
    runtime: overrides.runtime ?? {
      auth: "admin",
      source: "reports/release_monitoring",
      mode: "post_release_monitoring",
    },
  };
}

export function makeAdminReleaseClosureResponse(
  overrides: Partial<AdminReleaseClosureResponse> = {},
): AdminReleaseClosureResponse {
  return {
    status: overrides.status ?? "idle",
    latest_release: overrides.latest_release ?? null,
    closure_gate: overrides.closure_gate ?? {
      allowed: false,
      status: "idle",
      reasons: [],
      checks: [],
    },
    latest_closure: overrides.latest_closure ?? null,
    latest_evidence_package: overrides.latest_evidence_package ?? null,
    closures: overrides.closures ?? [],
    evidence_packages: overrides.evidence_packages ?? [],
    integrity: overrides.integrity ?? {
      status: "verified",
      warnings: [],
    },
    runtime: overrides.runtime ?? {
      auth: "admin",
      source: "reports/release_closure",
      mode: "post_release_closure",
    },
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
    asset_url: "/api/sessions/sess-patient/assets/1",
    filename: "report.pdf",
    content_type: "application/pdf",
    size: 4,
    sha256: "sha",
    reused: false,
    derived: { record_id: 1 },
  }));
  const downloadSessionAsset = vi.fn(async () => new Blob(["asset"], { type: "application/octet-stream" }));
  const resetSession = vi.fn(async (sessionId: string) => makeSessionResponse({ session_id: sessionId, scene: "patient" }));
  const bindPatient = vi.fn(async (sessionId: string, patientId: number) =>
    makeSessionResponse({
      session_id: sessionId,
      scene: "doctor",
      patient_id: patientId,
      snapshot: {
        registry_patient_id: patientId,
      },
    }),
  );
  const saveSessionPatientIdentity = vi.fn(async (sessionId: string, patient_name: string, patient_number: string) =>
    makeSessionResponse({
      session_id: sessionId,
      scene: "patient",
      snapshot: {
        patient_identity: {
          patient_name,
          patient_number,
          identity_locked: true,
        } satisfies PatientIdentitySnapshot,
      },
    }),
  );
  const saveCrcTriageAssessment = vi.fn(async (): Promise<SaveCrcTriageAssessmentResponse> => ({
    patient_id: 101,
    patient_version: 1,
    projection_version: 1,
    event_ids: ["event-1"],
    record_id: 1,
    reused: false,
  }));
  const getSessionPatientRecords = vi.fn(async () => makePatientRegistryRecordsResponse());
  const getSessionCareCards = vi.fn(async () => makePatientCareCardsResponse());
  const getDoctorReview = vi.fn(async () => makeDoctorReviewResponse());
  const recordDoctorActionTrace = vi.fn(async () => makeDoctorActionTraceResponse());
  const getDatabaseStats = vi.fn(async () => makeDatabaseStatsResponse());
  const searchDatabaseCases = vi.fn(async () => makeDatabaseSearchResponse());
  const getDatabaseCaseDetail = vi.fn(async (patientId: number): Promise<DatabaseCaseDetailResponse> => ({
    patient_id: patientId,
    case_record: { patient_id: patientId, clinical_stage: "cT3N1M0" },
    available_data: { case_info: true, imaging: false, pathology_slides: false },
    cards: {},
  }));
  const upsertDatabaseCase = vi.fn(async () => ({
    patient_id: 33,
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
  const getAdminTools = vi.fn(async (): Promise<AdminToolManifestResponse> => ({
    tools: [],
    groups: [],
    runtime: {
      web_search_enabled: false,
      auth: "admin",
      source: "src.tools.manifest",
    },
  }));
  const getAdminRules = vi.fn(async (): Promise<AdminRulesResponse> => ({
    policy_id: "crc_safety_policy_v0",
    version: "2026-06-29.0",
    status: "draft",
    applies_to: "patient_crc_triage",
    severity_order: ["emergency", "urgent", "backfill", "routine"],
    rules: [],
    source_path: "config/safety_policy.yaml",
    note: "read-only projection; not editable from admin UI",
  }));
  const getAdminReleaseDashboard = vi.fn(async (): Promise<AdminReleaseDashboardResponse> => ({
    version_chain: {
      agent_policy_version: "agent_policy_20260629_0",
      clinical_safety_policy_version: "crc_safety_policy_v0",
      evidence_index_version: "rag_crc_guideline_20260620",
      judge_rubric_version: "crc_rubric_v0",
    },
    release_decision: "feature_flag_or_pass",
    rollback_target: "agent_policy_20260624_0",
    human_signoff: {
      required: true,
      status: "missing",
      reason: "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    },
    summary: {
      hard_fail_count: 0,
      p0_cases_total: 5,
      p0_cases_passed: 5,
      literature_claims: 3,
      literature_isolation_violations: 0,
      clinical_rag_ingest_enabled: false,
    },
    runs: [],
    blocking_gates: [],
    disabled_actions: [],
    runtime: {
      auth: "admin",
      source: "reports/static_release_artifacts",
      mode: "read_only",
    },
  }));
  const getAdminReleaseGovernance = vi.fn(async (): Promise<AdminReleaseGovernanceResponse> => ({
    dashboard_snapshot: {
      release_decision: "feature_flag_or_pass",
      rollback_target: "agent_policy_20260624_0",
      hard_fail_count: 0,
      literature_status: "shadow_only",
    },
    intents: [],
    active_intent: null,
    approvals: [],
    required_approvals: [
      {
        role: "release_manager",
        status: "missing",
        latest_decision: null,
      },
      {
        role: "clinical_safety_reviewer",
        status: "missing",
        latest_decision: null,
      },
    ],
    rollback_plan: null,
    audit_events: [],
    integrity: {
      status: "verified",
      warnings: [],
    },
    disabled_execution_actions: [
      {
        id: "execute_release",
        label: "Execute release",
        disabled: true,
        reason: "Step 12 records governance only.",
      },
      {
        id: "execute_rollback",
        label: "Execute rollback",
        disabled: true,
        reason: "Rollback execution requires a later execution-path design.",
      },
    ],
    runtime: {
      auth: "admin",
      source: "reports/release_governance",
      mode: "audit_only",
    },
  }));
  const getAdminReleaseExecution = vi.fn(async (): Promise<AdminReleaseExecutionResponse> =>
    makeAdminReleaseExecutionResponse(),
  );
  const getAdminReleaseMonitoring = vi.fn(async (): Promise<AdminReleaseMonitoringResponse> =>
    makeAdminReleaseMonitoringResponse(),
  );
  const getAdminReleaseClosure = vi.fn(async (): Promise<AdminReleaseClosureResponse> =>
    makeAdminReleaseClosureResponse(),
  );
  const createAdminReleaseIntent = vi.fn(async () => getAdminReleaseGovernance());
  const recordAdminReleaseApproval = vi.fn(async () => getAdminReleaseGovernance());
  const recordAdminReleaseRollbackPlan = vi.fn(async () => getAdminReleaseGovernance());
  const cancelAdminReleaseIntent = vi.fn(async () => getAdminReleaseGovernance());
  const executeAdminRelease = vi.fn(async () => makeAdminReleaseExecutionResponse());
  const executeAdminReleaseRollback = vi.fn(async () => makeAdminReleaseExecutionResponse());
  const recordAdminReleaseMonitoringCheck = vi.fn(async () => makeAdminReleaseMonitoringResponse());
  const acknowledgeAdminReleaseMonitoringAlert = vi.fn(async () => makeAdminReleaseMonitoringResponse());
  const recordAdminReleaseClosure = vi.fn(async () => makeAdminReleaseClosureResponse());
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
    getAdminTools,
    getAdminRules,
    getAdminReleaseDashboard,
    getAdminReleaseGovernance,
    getAdminReleaseExecution,
    getAdminReleaseMonitoring,
    getAdminReleaseClosure,
    createAdminReleaseIntent,
    recordAdminReleaseApproval,
    recordAdminReleaseRollbackPlan,
    cancelAdminReleaseIntent,
    executeAdminRelease,
    executeAdminReleaseRollback,
    recordAdminReleaseMonitoringCheck,
    acknowledgeAdminReleaseMonitoringAlert,
    recordAdminReleaseClosure,
    createSession,
    getSession,
    getMessages,
    streamTurn,
    uploadFile,
    downloadSessionAsset,
    resetSession,
    bindPatient,
    saveSessionPatientIdentity,
    saveCrcTriageAssessment,
    getSessionPatientRecords,
    getSessionCareCards,
    getDoctorReview,
    recordDoctorActionTrace,
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

export function renderWithProviders(ui: ReactElement, apiClient: ApiClient) {
  return render(ui, { wrapper: buildAppWrapper(apiClient) });
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
