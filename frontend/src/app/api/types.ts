export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonObject
  | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue | unknown;
}

export type Scene = "patient" | "doctor";

export interface AssetRef {
  asset_id: string;
  name?: string;
}

export interface InlineCard {
  card_type: string;
  payload: JsonObject;
}

export interface SessionMessage {
  cursor: string;
  type: string;
  content: unknown;
  id?: string | null;
  name?: string | null;
  tool_call_id?: string | null;
  status?: string | null;
  asset_refs: AssetRef[];
  inline_cards?: InlineCard[];
}

export interface CardUpsertEvent {
  type: "card.upsert";
  card_type: string;
  payload: JsonObject;
  source_channel: "state" | "findings" | "message_kwargs";
}

export interface StatusNodeEvent {
  type: "status.node";
  node: string;
}

export interface MessageDoneEvent {
  type: "message.done";
  role: "assistant";
  content: unknown;
  thinking?: string | null;
  message_id?: string | null;
  node?: string | null;
  inline_cards?: InlineCard[] | null;
}

export interface MessageDeltaEvent {
  type: "message.delta";
  message_id: string;
  node?: string | null;
  delta: string;
}

export interface StageUpdateEvent {
  type: "stage.update";
  stage: string;
}

export interface PatientProfileUpdateEvent {
  type: "patient_profile.update";
  profile: JsonObject;
}

export interface CriticVerdictEvent {
  type: "critic.verdict";
  verdict: string;
  feedback?: string | null;
  iteration_count?: number | null;
}

export interface RoadmapUpdateEvent {
  type: "roadmap.update";
  roadmap: JsonObject[];
}

export interface PlanUpdateEvent {
  type: "plan.update";
  plan: JsonObject[];
}

export interface SafetyAlertEvent {
  type: "safety.alert";
  message: string;
  blocking: true;
}

export interface FindingsPatchEvent {
  type: "findings.patch";
  patch: JsonObject;
}

export interface ReferencesAppendEvent {
  type: "references.append";
  items: JsonObject[];
}

export interface ErrorEvent {
  type: "error";
  code: string;
  message: string;
  recoverable: boolean;
}

export interface ContextMaintenanceEvent {
  type: "context.maintenance";
  status: "running" | "completed" | "failed";
  message: string;
}

export interface DoneEvent {
  type: "done";
  thread_id: string;
  run_id: string;
  snapshot_version: number;
}

export type StreamEvent =
  | CardUpsertEvent
  | StatusNodeEvent
  | MessageDeltaEvent
  | MessageDoneEvent
  | StageUpdateEvent
  | PatientProfileUpdateEvent
  | CriticVerdictEvent
  | RoadmapUpdateEvent
  | PlanUpdateEvent
  | SafetyAlertEvent
  | FindingsPatchEvent
  | ReferencesAppendEvent
  | ErrorEvent
  | ContextMaintenanceEvent
  | DoneEvent;

export interface ContextMaintenanceState {
  status: "running" | "completed" | "failed";
  message: string;
  error?: string | null;
}

export interface ContextStateSnapshot {
  summary_memory?: string | null;
  structured_summary?: JsonObject | null;
  summary_memory_cursor?: number | null;
  [key: string]: unknown;
}

export interface RecoverySnapshot {
  snapshot_version: number;
  messages: SessionMessage[];
  messages_total: number;
  messages_next_before_cursor: string | null;
  cards: CardUpsertEvent[];
  roadmap: JsonObject[];
  findings: JsonObject;
  patient_profile: JsonObject | null;
  stage: string | null;
  assessment_draft: unknown;
  current_patient_id: string | number | null;
  references: JsonObject[];
  plan: JsonObject[];
  critic: JsonObject | null;
  safety_alert: JsonObject | null;
  uploaded_assets: Record<string, unknown>;
  context_maintenance: ContextMaintenanceState | null;
  context_state: ContextStateSnapshot | null;
}

export interface RuntimeInfo {
  runner_mode: string;
  fixture_case: string | null;
}

export interface SessionResponse {
  session_id: string;
  thread_id: string;
  scene: Scene;
  patient_id: number | null;
  snapshot_version: number;
  snapshot: RecoverySnapshot;
  runtime: RuntimeInfo;
}

export interface MessageHistoryResponse {
  session_id: string;
  thread_id: string;
  snapshot_version: number;
  messages_total: number;
  next_before_cursor: string | null;
  messages: SessionMessage[];
}

export interface ChatTurnRequest {
  message: {
    role: "user";
    content: string;
  };
  context?: Record<string, unknown>;
}

export interface UploadResponse {
  asset_id: string;
  filename: string;
  content_type: string;
  size: number;
  sha256: string;
  reused: boolean;
  derived: {
    record_id?: number | null;
    patient_id?: number | null;
    document_type?: string | null;
    ingest_decision?: string | null;
    medical_card_created?: boolean;
    sqlite_record_id?: number | null;
    [key: string]: unknown;
  };
}

export interface PatientRegistryItem {
  patient_id: number;
  status: string;
  created_by_session_id?: string | null;
  updated_at: string;
  tumor_location?: string | null;
  mmr_status?: string | null;
  clinical_stage?: string | null;
}

export interface PatientRegistryListResponse {
  items: PatientRegistryItem[];
  total: number;
}

export interface PatientRegistrySearchRequest {
  patient_id?: number | null;
  tumor_location?: string | null;
  mmr_status?: string | null;
  clinical_stage?: string | null;
  limit?: number;
}

export interface PatientRegistryDetail extends PatientRegistryItem {
  created_at: string;
  chief_complaint?: string | null;
  age?: number | null;
  gender?: string | null;
  t_stage?: string | null;
  n_stage?: string | null;
  m_stage?: string | null;
}

export interface PatientRegistryRecord {
  record_id: number;
  patient_id: number;
  asset_id: number;
  record_type: string;
  document_type?: string;
  ingest_decision?: string;
  snapshot_contributed?: boolean;
  conflict_detected?: boolean;
  normalized_payload_json?: JsonValue | unknown;
  summary_text: string;
  source: string;
  snapshot_meta_json?: JsonValue | unknown;
  created_at: string;
}

export interface PatientRegistryRecordsResponse {
  items: PatientRegistryRecord[];
}

export interface PatientRegistryAlert {
  kind: string;
  message: string;
  patient_id?: number;
  record_id?: number | null;
  field_name?: string | null;
  field_names?: string[];
  document_type?: string | null;
  created_at?: string | null;
}

export interface PatientRegistryAlertsResponse {
  items: PatientRegistryAlert[];
}

export interface PatientRegistryDeleteResponse {
  patient_id: number;
  deleted_records: number;
  deleted_assets: number;
  deleted_asset_paths: string[];
  record_ids: number[];
}

export interface PatientRegistryClearResponse {
  deleted_patients: number;
  deleted_records: number;
  deleted_assets: number;
  patient_ids: number[];
  deleted_asset_paths: string[];
}

export type DatabaseSortField =
  | "patient_id"
  | "age"
  | "gender"
  | "ecog_score"
  | "tumor_location"
  | "histology_type"
  | "clinical_stage"
  | "cea_level"
  | "mmr_status";

export type DatabaseSortDirection = "asc" | "desc";

export interface DatabaseFilters {
  patient_id?: number | null;
  tumor_location: string[];
  ct_stage: string[];
  cn_stage: string[];
  histology_type: string[];
  mmr_status: string[];
  age_min?: number | null;
  age_max?: number | null;
  cea_max?: number | null;
  family_history?: boolean | null;
  biopsy_confirmed?: boolean | null;
  ecog_min?: number | null;
  ecog_max?: number | null;
}

export interface DatabasePagination {
  page: number;
  page_size: number;
}

export interface DatabaseSort {
  field: DatabaseSortField;
  direction: DatabaseSortDirection;
}

export interface DatabaseSearchRequest {
  filters: DatabaseFilters;
  pagination: DatabasePagination;
  sort: DatabaseSort;
}

export interface DatabaseCaseRow {
  patient_id: number;
  gender?: string | null;
  age?: number | null;
  ecog_score?: number | null;
  tumor_location?: string | null;
  histology_type?: string | null;
  clinical_stage?: string | null;
  cea_level?: number | null;
  mmr_status?: string | null;
  chief_complaint?: string | null;
  symptom_duration?: string | null;
  family_history?: boolean | null;
  family_history_details?: string | null;
  biopsy_confirmed?: boolean | null;
  biopsy_details?: string | null;
  risk_factors?: string[];
  [key: string]: unknown;
}

export interface DatabaseSearchResponse {
  items: DatabaseCaseRow[];
  total: number;
  page: number;
  page_size: number;
  applied_filters: Record<string, unknown>;
  warnings: string[];
}

export interface DatabaseNumericStatistics {
  min?: number | null;
  max?: number | null;
  mean?: number | null;
}

export interface DatabaseStatsResponse {
  total_cases: number;
  gender_distribution: Record<string, number>;
  age_statistics?: DatabaseNumericStatistics | null;
  tumor_location_distribution: Record<string, number>;
  ct_stage_distribution: Record<string, number>;
  mmr_status_distribution: Record<string, number>;
  cea_statistics?: DatabaseNumericStatistics | null;
  [key: string]: unknown;
}

export interface DatabaseAvailableData {
  case_info: boolean;
  imaging: boolean;
  pathology_slides: boolean;
}

export interface DatabaseCaseDetailResponse {
  patient_id: string;
  case_record: JsonObject | null;
  available_data: DatabaseAvailableData;
  cards: Record<string, JsonObject>;
}

export interface DatabaseUpsertRequest {
  record: JsonObject;
}

export interface DatabaseQueryIntentResponse {
  query: string;
  normalized_query: string;
  filters: Partial<DatabaseFilters>;
  unsupported_terms: string[];
  warnings: string[];
}

export type DatabaseWorkbenchMode = "stats" | "search" | "detail" | "edit";

export interface DatabaseWorkbenchContext {
  visible: boolean;
  mode: DatabaseWorkbenchMode;
  query_text?: string | null;
  filters?: Partial<DatabaseFilters> | null;
  selected_patient_id?: number | null;
}

export interface FrontendMessage {
  cursor: string;
  type: string;
  content: unknown;
  thinking?: string | null;
  id?: string;
  name?: string;
  toolCallId?: string;
  status?: string;
  assetRefs: AssetRef[];
  node?: string | null;
  inlineCards?: Array<{
    cardType: string;
    payload: JsonObject;
  }>;
}

export interface SafetyAlertState {
  message: string;
  blocking: true;
}

export interface SessionState {
  sessionId: string | null;
  threadId: string | null;
  snapshotVersion: number;
  runtime: RuntimeInfo | null;
  messages: FrontendMessage[];
  messagesTotal: number;
  messagesNextBeforeCursor: string | null;
  cards: Record<string, JsonObject>;
  roadmap: JsonObject[];
  findings: JsonObject;
  patientProfile: JsonObject | null;
  stage: string | null;
  references: JsonObject[];
  plan: JsonObject[];
  critic: JsonObject | null;
  safetyAlert: SafetyAlertState | null;
  assessmentDraft: unknown;
  currentPatientId: string | number | null;
  uploadedAssets: Record<string, unknown>;
  contextMaintenance: ContextMaintenanceState | null;
  contextState: ContextStateSnapshot | null;
  statusNode: string | null;
  lastError: { code: string; message: string; recoverable: boolean } | null;
  activeRunId: string | null;
  pendingInlineCards: Array<{
    cardType: string;
    payload: JsonObject;
  }>;
  latestAssistantMessageCursor: string | null;
  streamingMessageCursors: Record<string, string>;
}

