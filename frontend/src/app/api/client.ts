import { streamJsonEvents, type FetchLike, type StreamTraceTap } from "./stream";
import type {
  AdminAcknowledgeReleaseMonitoringAlertRequest,
  AdminCancelReleaseIntentRequest,
  AdminCreateReleaseIntentRequest,
  AdminExecuteReleaseRequest,
  AdminRecordReleaseMonitoringCheckRequest,
  AdminReleaseDashboardResponse,
  AdminReleaseExecutionResponse,
  AdminReleaseGovernanceResponse,
  AdminReleaseMonitoringResponse,
  AdminRecordReleaseApprovalRequest,
  AdminRecordReleaseRollbackPlanRequest,
  AdminToolManifestResponse,
  ChatTurnRequest,
  DatabaseCaseDetailResponse,
  DatabaseQueryIntentResponse,
  DatabaseSearchRequest,
  DatabaseSearchResponse,
  DatabaseStatsResponse,
  DatabaseUpsertRequest,
  DoctorActionTraceRequest,
  DoctorActionTraceResponse,
  DoctorReviewResponse,
  MessageHistoryResponse,
  PatientRegistryAlertsResponse,
  PatientRegistryClearResponse,
  PatientRegistryDeleteResponse,
  PatientCareCardsResponse,
  PatientRegistryDetail,
  PatientRegistryListResponse,
  PatientRegistryRecordsResponse,
  PatientRegistrySearchRequest,
  Scene,
  SaveCrcTriageAssessmentRequest,
  SaveCrcTriageAssessmentResponse,
  SessionResponse,
  StreamEvent,
  UploadResponse,
} from "./types";

export class ApiClientError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiClientOptions {
  baseUrl?: string;
  fetchImpl?: FetchLike;
  headers?: HeadersInit;
}

function buildUrl(path: string, baseUrl?: string, query?: URLSearchParams): string {
  const suffix = query && query.toString() ? `?${query.toString()}` : "";
  if (!baseUrl) {
    return `${path}${suffix}`;
  }
  return `${baseUrl.replace(/\/$/, "")}${path}${suffix}`;
}

async function parseJsonOrNull(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await parseJsonOrNull(response);
    const message =
      typeof detail === "object" && detail && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiClientError(response.status, message, detail);
  }

  return (await response.json()) as T;
}

function buildJsonHeaders(defaultHeaders?: HeadersInit): Headers {
  const headers = new Headers(defaultHeaders);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

export interface ApiClient {
  getAdminTools(): Promise<AdminToolManifestResponse>;
  getAdminReleaseDashboard(): Promise<AdminReleaseDashboardResponse>;
  getAdminReleaseGovernance(): Promise<AdminReleaseGovernanceResponse>;
  getAdminReleaseExecution(): Promise<AdminReleaseExecutionResponse>;
  getAdminReleaseMonitoring(): Promise<AdminReleaseMonitoringResponse>;
  createAdminReleaseIntent(request: AdminCreateReleaseIntentRequest): Promise<AdminReleaseGovernanceResponse>;
  recordAdminReleaseApproval(
    intentId: string,
    request: AdminRecordReleaseApprovalRequest,
  ): Promise<AdminReleaseGovernanceResponse>;
  recordAdminReleaseRollbackPlan(
    intentId: string,
    request: AdminRecordReleaseRollbackPlanRequest,
  ): Promise<AdminReleaseGovernanceResponse>;
  cancelAdminReleaseIntent(
    intentId: string,
    request: AdminCancelReleaseIntentRequest,
  ): Promise<AdminReleaseGovernanceResponse>;
  executeAdminRelease(request: AdminExecuteReleaseRequest): Promise<AdminReleaseExecutionResponse>;
  executeAdminReleaseRollback(request: AdminExecuteReleaseRequest): Promise<AdminReleaseExecutionResponse>;
  recordAdminReleaseMonitoringCheck(
    request: AdminRecordReleaseMonitoringCheckRequest,
  ): Promise<AdminReleaseMonitoringResponse>;
  acknowledgeAdminReleaseMonitoringAlert(
    alertId: string,
    request: AdminAcknowledgeReleaseMonitoringAlertRequest,
  ): Promise<AdminReleaseMonitoringResponse>;
  createSession(scene: Scene): Promise<SessionResponse>;
  getSession(sessionId: string, messageLimit?: number): Promise<SessionResponse>;
  getMessages(sessionId: string, before?: string | number | null, limit?: number): Promise<MessageHistoryResponse>;
  streamTurn(
    sessionId: string,
    request: ChatTurnRequest,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
    traceTap?: StreamTraceTap,
  ): Promise<void>;
  uploadFile(sessionId: string, file: File): Promise<UploadResponse>;
  downloadSessionAsset(sessionId: string, assetId: string): Promise<Blob>;
  resetSession(sessionId: string): Promise<SessionResponse>;
  bindPatient(sessionId: string, patientId: number): Promise<SessionResponse>;
  saveSessionPatientIdentity(
    sessionId: string,
    patient_name: string,
    patient_number: string,
  ): Promise<SessionResponse>;
  saveCrcTriageAssessment(
    sessionId: string,
    request: SaveCrcTriageAssessmentRequest,
  ): Promise<SaveCrcTriageAssessmentResponse>;
  getDoctorReview(sessionId: string): Promise<DoctorReviewResponse>;
  recordDoctorActionTrace(
    sessionId: string,
    request: DoctorActionTraceRequest,
  ): Promise<DoctorActionTraceResponse>;
  getSessionPatientRecords(sessionId: string): Promise<PatientRegistryRecordsResponse>;
  getSessionCareCards(sessionId: string): Promise<PatientCareCardsResponse>;
  getDatabaseStats(): Promise<DatabaseStatsResponse>;
  searchDatabaseCases(request: DatabaseSearchRequest): Promise<DatabaseSearchResponse>;
  getDatabaseCaseDetail(patientId: number): Promise<DatabaseCaseDetailResponse>;
  upsertDatabaseCase(request: DatabaseUpsertRequest): Promise<DatabaseCaseDetailResponse>;
  parseDatabaseQueryIntent(query: string): Promise<DatabaseQueryIntentResponse>;
  getRecentPatients(limit?: number): Promise<PatientRegistryListResponse>;
  searchPatientRegistry(request: PatientRegistrySearchRequest): Promise<PatientRegistryListResponse>;
  getPatientRegistryDetail(patientId: number): Promise<PatientRegistryDetail>;
  getPatientRecords(patientId: number): Promise<PatientRegistryRecordsResponse>;
  getPatientRegistryAlerts(patientId: number): Promise<PatientRegistryAlertsResponse>;
  deletePatientRegistryPatient(patientId: number): Promise<PatientRegistryDeleteResponse>;
  clearPatientRegistry(): Promise<PatientRegistryClearResponse>;
}

export function createApiClient(options: ApiClientOptions = {}): ApiClient {
  const fetchImpl = options.fetchImpl ?? fetch;
  const defaultHeaders = options.headers;
  const baseUrl = options.baseUrl;

  return {
    async getAdminTools() {
      const response = await fetchImpl(buildUrl("/api/admin/tools", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminToolManifestResponse>(response);
    },

    async getAdminReleaseDashboard() {
      const response = await fetchImpl(buildUrl("/api/admin/release-dashboard", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminReleaseDashboardResponse>(response);
    },

    async getAdminReleaseGovernance() {
      const response = await fetchImpl(buildUrl("/api/admin/release-governance", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminReleaseGovernanceResponse>(response);
    },

    async getAdminReleaseExecution() {
      const response = await fetchImpl(buildUrl("/api/admin/release-execution", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminReleaseExecutionResponse>(response);
    },

    async getAdminReleaseMonitoring() {
      const response = await fetchImpl(buildUrl("/api/admin/release-monitoring", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
    },

    async createAdminReleaseIntent(request) {
      const response = await fetchImpl(buildUrl("/api/admin/release-governance/intents", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<AdminReleaseGovernanceResponse>(response);
    },

    async recordAdminReleaseApproval(intentId, request) {
      const encodedIntentId = encodeURIComponent(intentId);
      const response = await fetchImpl(
        buildUrl(`/api/admin/release-governance/intents/${encodedIntentId}/approvals`, baseUrl),
        {
          method: "POST",
          headers: buildJsonHeaders(defaultHeaders),
          body: JSON.stringify(request),
        },
      );
      return parseJsonResponse<AdminReleaseGovernanceResponse>(response);
    },

    async recordAdminReleaseRollbackPlan(intentId, request) {
      const encodedIntentId = encodeURIComponent(intentId);
      const response = await fetchImpl(
        buildUrl(`/api/admin/release-governance/intents/${encodedIntentId}/rollback-plan`, baseUrl),
        {
          method: "POST",
          headers: buildJsonHeaders(defaultHeaders),
          body: JSON.stringify(request),
        },
      );
      return parseJsonResponse<AdminReleaseGovernanceResponse>(response);
    },

    async cancelAdminReleaseIntent(intentId, request) {
      const encodedIntentId = encodeURIComponent(intentId);
      const response = await fetchImpl(
        buildUrl(`/api/admin/release-governance/intents/${encodedIntentId}/cancel`, baseUrl),
        {
          method: "POST",
          headers: buildJsonHeaders(defaultHeaders),
          body: JSON.stringify(request),
        },
      );
      return parseJsonResponse<AdminReleaseGovernanceResponse>(response);
    },

    async executeAdminRelease(request) {
      const response = await fetchImpl(buildUrl("/api/admin/release-execution/release", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<AdminReleaseExecutionResponse>(response);
    },

    async executeAdminReleaseRollback(request) {
      const response = await fetchImpl(buildUrl("/api/admin/release-execution/rollback", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<AdminReleaseExecutionResponse>(response);
    },

    async recordAdminReleaseMonitoringCheck(request) {
      const response = await fetchImpl(buildUrl("/api/admin/release-monitoring/checks", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
    },

    async acknowledgeAdminReleaseMonitoringAlert(alertId, request) {
      const encodedAlertId = encodeURIComponent(alertId);
      const response = await fetchImpl(
        buildUrl(`/api/admin/release-monitoring/alerts/${encodedAlertId}/acknowledge`, baseUrl),
        {
          method: "POST",
          headers: buildJsonHeaders(defaultHeaders),
          body: JSON.stringify(request),
        },
      );
      return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
    },

    async createSession(scene) {
      const response = await fetchImpl(buildUrl("/api/sessions", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify({ scene }),
      });
      return parseJsonResponse<SessionResponse>(response);
    },

    async getSession(sessionId, messageLimit) {
      const params = new URLSearchParams();
      if (messageLimit !== undefined) {
        params.set("message_limit", String(messageLimit));
      }
      const response = await fetchImpl(
        buildUrl(`/api/sessions/${sessionId}`, baseUrl, params),
        { headers: defaultHeaders },
      );
      return parseJsonResponse<SessionResponse>(response);
    },

    async getMessages(sessionId, before, limit) {
      const params = new URLSearchParams();
      if (before !== undefined && before !== null) {
        params.set("before", String(before));
      }
      if (limit !== undefined) {
        params.set("limit", String(limit));
      }
      const response = await fetchImpl(
        buildUrl(`/api/sessions/${sessionId}/messages`, baseUrl, params),
        { headers: defaultHeaders },
      );
      return parseJsonResponse<MessageHistoryResponse>(response);
    },

    async streamTurn(sessionId, request, onEvent, signal, traceTap) {
      await streamJsonEvents({
        fetchImpl,
        url: buildUrl(`/api/sessions/${sessionId}/messages/stream`, baseUrl),
        body: request,
        headers: defaultHeaders,
        signal,
        onEvent,
        traceTap,
      });
    },

    async uploadFile(sessionId, file) {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/uploads`, baseUrl), {
        method: "POST",
        body: formData,
        headers: defaultHeaders,
      });
      return parseJsonResponse<UploadResponse>(response);
    },

    async downloadSessionAsset(sessionId, assetId) {
      const encodedSessionId = encodeURIComponent(sessionId);
      const encodedAssetId = encodeURIComponent(assetId);
      const response = await fetchImpl(
        buildUrl(`/api/sessions/${encodedSessionId}/assets/${encodedAssetId}`, baseUrl),
        { headers: defaultHeaders },
      );
      if (!response.ok) {
        const detail = await parseJsonOrNull(response);
        const message =
          typeof detail === "object" && detail && "detail" in detail
            ? String((detail as { detail: unknown }).detail)
            : `Request failed with status ${response.status}`;
        throw new ApiClientError(response.status, message, detail);
      }
      return response.blob();
    },

    async resetSession(sessionId) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/reset`, baseUrl), {
        method: "POST",
        headers: defaultHeaders,
      });
      return parseJsonResponse<SessionResponse>(response);
    },

    async bindPatient(sessionId, patientId) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}`, baseUrl), {
        method: "PATCH",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify({ patient_id: patientId }),
      });
      return parseJsonResponse<SessionResponse>(response);
    },

    async saveSessionPatientIdentity(sessionId, patient_name, patient_number) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/identity`, baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify({
          patient_name,
          patient_number,
        }),
      });
      return parseJsonResponse<SessionResponse>(response);
    },

    async saveCrcTriageAssessment(sessionId, request) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/crc-triage/assessments`, baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<SaveCrcTriageAssessmentResponse>(response);
    },

    async getDoctorReview(sessionId) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/doctor-review`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<DoctorReviewResponse>(response);
    },

    async recordDoctorActionTrace(sessionId, request) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/doctor-review/action-traces`, baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<DoctorActionTraceResponse>(response);
    },

    async getSessionPatientRecords(sessionId) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/patient-records`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryRecordsResponse>(response);
    },

    async getSessionCareCards(sessionId) {
      const response = await fetchImpl(buildUrl(`/api/sessions/${sessionId}/care-cards`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientCareCardsResponse>(response);
    },

    async getDatabaseStats() {
      const response = await fetchImpl(buildUrl("/api/database/stats", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<DatabaseStatsResponse>(response);
    },

    async searchDatabaseCases(request) {
      const response = await fetchImpl(buildUrl("/api/database/cases/search", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<DatabaseSearchResponse>(response);
    },

    async getDatabaseCaseDetail(patientId) {
      const response = await fetchImpl(buildUrl(`/api/database/cases/${patientId}`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<DatabaseCaseDetailResponse>(response);
    },

    async upsertDatabaseCase(request) {
      const response = await fetchImpl(buildUrl("/api/database/cases/upsert", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<DatabaseCaseDetailResponse>(response);
    },

    async parseDatabaseQueryIntent(query) {
      const response = await fetchImpl(buildUrl("/api/database/query-intent", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify({ query }),
      });
      return parseJsonResponse<DatabaseQueryIntentResponse>(response);
    },

    async getRecentPatients(limit = 5) {
      const params = new URLSearchParams({ limit: String(limit) });
      const response = await fetchImpl(
        buildUrl("/api/patient-registry/patients/recent", baseUrl, params),
        { headers: defaultHeaders },
      );
      return parseJsonResponse<PatientRegistryListResponse>(response);
    },

    async searchPatientRegistry(request) {
      const response = await fetchImpl(buildUrl("/api/patient-registry/patients/search", baseUrl), {
        method: "POST",
        headers: buildJsonHeaders(defaultHeaders),
        body: JSON.stringify(request),
      });
      return parseJsonResponse<PatientRegistryListResponse>(response);
    },

    async getPatientRegistryDetail(patientId) {
      const response = await fetchImpl(buildUrl(`/api/patient-registry/patients/${patientId}`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryDetail>(response);
    },

    async getPatientRecords(patientId) {
      const response = await fetchImpl(buildUrl(`/api/patient-registry/patients/${patientId}/records`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryRecordsResponse>(response);
    },

    async getPatientRegistryAlerts(patientId) {
      const response = await fetchImpl(buildUrl(`/api/patient-registry/patients/${patientId}/alerts`, baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryAlertsResponse>(response);
    },

    async deletePatientRegistryPatient(patientId) {
      const response = await fetchImpl(buildUrl(`/api/patient-registry/patients/${patientId}`, baseUrl), {
        method: "DELETE",
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryDeleteResponse>(response);
    },

    async clearPatientRegistry() {
      const response = await fetchImpl(buildUrl("/api/patient-registry/patients", baseUrl), {
        method: "DELETE",
        headers: defaultHeaders,
      });
      return parseJsonResponse<PatientRegistryClearResponse>(response);
    },
  };
}
