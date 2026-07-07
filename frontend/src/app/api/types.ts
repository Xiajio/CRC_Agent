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

export type AdminToolCategory =
  | "clinical"
  | "rag"
  | "web"
  | "database"
  | "imaging"
  | "pathology"
  | "tumor"
  | "utility"
  | "formatting";

export type AdminToolRegistry =
  | "graph"
  | "graph_web"
  | "executor"
  | "database_node"
  | "optional";

export type AdminToolRouteTarget =
  | "knowledge"
  | "case_database"
  | "rad_agent"
  | "path_agent"
  | "web_search"
  | "tool_executor"
  | "decision";

export type AdminToolGraphScope =
  | "doctor"
  | "patient"
  | "both"
  | "node_local"
  | "executor_only";

export type AdminToolState = "available" | "candidate" | "internal" | "disabled";

export interface AdminToolItem {
  name: string;
  category: AdminToolCategory;
  registries: AdminToolRegistry[];
  route_targets: AdminToolRouteTarget[];
  graph_scope: AdminToolGraphScope;
  planner_aliases: string[];
  requires_web: boolean;
  available: boolean;
  state: AdminToolState;
}

export interface AdminToolGroup {
  category: AdminToolCategory;
  count: number;
  available_count: number;
}

export interface AdminToolManifestResponse {
  tools: AdminToolItem[];
  groups: AdminToolGroup[];
  runtime: {
    web_search_enabled: boolean;
    auth: "admin";
    source: "src.tools.manifest";
  };
}

export type AdminReleaseRunStatus = "pass" | "fail" | "shadow_only" | "missing" | "invalid";
export type AdminReleaseGateState = "pass" | "locked" | "warning" | "blocked" | "missing";
export type AdminReleaseHumanSignoffStatus = "missing" | "recorded_elsewhere" | "not_required";

export interface AdminReleaseDashboardVersionChain {
  agent_policy_version: string | null;
  clinical_safety_policy_version: string | null;
  evidence_index_version: string | null;
  judge_rubric_version: string | null;
}

export interface AdminReleaseDashboardHumanSignoff {
  required: boolean;
  status: AdminReleaseHumanSignoffStatus;
  reason: string;
}

export interface AdminReleaseDashboardSummary {
  hard_fail_count: number;
  p0_cases_total: number;
  p0_cases_passed: number;
  literature_claims: number;
  literature_isolation_violations: number;
  clinical_rag_ingest_enabled: boolean;
}

export interface AdminReleaseDashboardRun {
  run_id: string;
  kind: "p0_crc_harness" | "release_safety" | "literature_shadow_harness";
  status: AdminReleaseRunStatus;
  source_path: string;
  hard_fail_count: number;
}

export interface AdminReleaseDashboardGate {
  id: string;
  label: string;
  state: AdminReleaseGateState;
  reason: string;
}

export interface AdminReleaseDashboardDisabledAction {
  id: "record_human_signoff" | "publish_feature_flag" | "rollback_release";
  label: string;
  reason: string;
}

export interface AdminReleaseDashboardResponse {
  version_chain: AdminReleaseDashboardVersionChain;
  release_decision: string;
  rollback_target: string | null;
  human_signoff: AdminReleaseDashboardHumanSignoff;
  summary: AdminReleaseDashboardSummary;
  runs: AdminReleaseDashboardRun[];
  blocking_gates: AdminReleaseDashboardGate[];
  disabled_actions: AdminReleaseDashboardDisabledAction[];
  runtime: {
    auth: "admin";
    source: "reports/static_release_artifacts";
    mode: "read_only";
  };
}

export type AdminReleaseTargetScope = "shadow" | "feature_flag_candidate";
export type AdminReleaseIntentStatus = "draft" | "pending_approval" | "approved" | "rejected" | "cancelled";
export type AdminReleaseCreateIntentStatus = "draft" | "pending_approval";
export type AdminReleaseApproverRole = "release_manager" | "clinical_safety_reviewer" | "evidence_reviewer";
export type AdminReleaseApprovalDecision = "approve" | "reject" | "request_changes";
export type AdminReleaseApprovalStatus = "missing" | "approved" | "rejected" | "changes_requested";
export type AdminReleaseRollbackPlanStatus = "proposed" | "accepted";
export type AdminReleaseIntegrityStatus = "verified" | "failed";

export interface AdminReleaseGovernanceDashboardSnapshot {
  version_chain?: Partial<AdminReleaseDashboardVersionChain>;
  release_decision?: string | null;
  rollback_target?: string | null;
  hard_fail_count?: number | null;
  literature_isolation_violations?: number | null;
  clinical_rag_ingest_enabled?: boolean | null;
  literature_status?: AdminReleaseRunStatus | string | null;
}

export interface AdminReleaseIntent {
  intent_id: string;
  source_release_report_id: string;
  source_report_path: string;
  harness_run_ids: string[];
  literature_run_id: string | null;
  version_chain: Record<string, JsonValue | unknown>;
  release_decision_snapshot: string;
  rollback_target: string;
  requested_by: string;
  requested_at: string;
  target_scope: AdminReleaseTargetScope;
  status: AdminReleaseIntentStatus;
  derived_status: AdminReleaseIntentStatus;
  blocking_summary: Record<string, JsonValue | unknown>;
}

export interface AdminReleaseApproval {
  approval_id: string;
  intent_id: string;
  approver_role: AdminReleaseApproverRole;
  decision: AdminReleaseApprovalDecision;
  reason: string;
  signed_by: string;
  signed_at: string;
  required: boolean;
}

export interface AdminReleaseRequiredApproval {
  role: AdminReleaseApproverRole;
  status: AdminReleaseApprovalStatus;
  latest_decision: AdminReleaseApprovalDecision | null;
  approval_id?: string;
  signed_by?: string;
  signed_at?: string;
}

export interface AdminReleaseRollbackPlan {
  rollback_plan_id: string;
  intent_id: string;
  rollback_target: string;
  owner: string;
  status: AdminReleaseRollbackPlanStatus;
  verification_steps: string[];
  created_at: string;
}

export interface AdminReleaseAuditEvent {
  event_id: string;
  intent_id: string;
  event_type: "intent_created" | "approval_recorded" | "rollback_plan_recorded" | "intent_cancelled" | "governance_read";
  actor: string;
  timestamp: string;
  payload_hash: string;
  previous_event_hash: string;
  event_hash: string;
}

export interface AdminReleaseGovernanceIntegrity {
  status: AdminReleaseIntegrityStatus;
  warnings: string[];
  affected_intent_ids?: string[];
  global_failure?: boolean;
}

export interface AdminReleaseGovernanceDisabledAction {
  id: "execute_release" | "execute_rollback";
  label: string;
  disabled: true;
  reason: string;
}

export interface AdminReleaseGovernanceResponse {
  dashboard_snapshot: AdminReleaseGovernanceDashboardSnapshot;
  intents: AdminReleaseIntent[];
  active_intent: AdminReleaseIntent | null;
  approvals: AdminReleaseApproval[];
  required_approvals: AdminReleaseRequiredApproval[];
  rollback_plan: AdminReleaseRollbackPlan | null;
  audit_events: AdminReleaseAuditEvent[];
  integrity: AdminReleaseGovernanceIntegrity;
  disabled_execution_actions: AdminReleaseGovernanceDisabledAction[];
  runtime: {
    auth: "admin";
    source: "reports/release_governance";
    mode: "audit_only";
  };
}

export interface AdminReleaseExecutionPreflightAction {
  allowed: boolean;
  reasons: string[];
}

export interface AdminReleaseExecutionGovernanceSummary {
  active_intent_id: string | null;
  derived_status: AdminReleaseIntentStatus | null;
  required_approvals_complete: boolean;
  rollback_plan_id: string | null;
}

export interface AdminFeatureFlagState {
  flag_name: "doctor_review_cockpit_v0";
  enabled: boolean;
  scope: "feature_flag_candidate";
  source_intent_id: string;
  source_execution_id: string;
  rollback_target: string;
  updated_by: string;
  updated_at: string;
}

export interface AdminReleaseExecutionRequestRecord {
  execution_id: string;
  intent_id: string;
  action: "release" | "rollback";
  requested_by: string;
  requested_at: string;
  idempotency_key: string;
  reason: string;
  expected_governance_hash: string;
  expected_rollback_plan_id: string;
  target_flag_state: Record<string, JsonValue | unknown>;
  rollback_target: string | null;
}

export interface AdminReleaseExecutionResultRecord {
  result_id: string;
  execution_id: string;
  intent_id: string;
  action: "release" | "rollback";
  status: "succeeded" | "failed";
  started_at: string;
  finished_at: string;
  actor: string;
  previous_flag_state: AdminFeatureFlagState | null;
  new_flag_state: AdminFeatureFlagState | null;
  failure_reason: string | null;
}

export interface AdminReleaseExecutionAuditEvent {
  event_id: string;
  execution_id: string;
  intent_id: string;
  event_type:
    | "release_requested"
    | "release_succeeded"
    | "release_failed"
    | "rollback_requested"
    | "rollback_succeeded"
    | "rollback_failed"
    | "execution_read";
  actor: string;
  timestamp: string;
  payload_hash: string;
  previous_event_hash: string;
  event_hash: string;
}

export interface AdminReleaseExecutionResponse {
  governance: AdminReleaseExecutionGovernanceSummary;
  preflight: {
    release: AdminReleaseExecutionPreflightAction;
    rollback: AdminReleaseExecutionPreflightAction;
  };
  feature_flag_state: AdminFeatureFlagState | null;
  requests: AdminReleaseExecutionRequestRecord[];
  results: AdminReleaseExecutionResultRecord[];
  audit_events: AdminReleaseExecutionAuditEvent[];
  integrity: AdminReleaseGovernanceIntegrity;
  runtime: {
    auth: "admin";
    source: "reports/release_execution";
    mode: "controlled_local_execution";
  };
}

export type AdminReleaseMonitoringStatus = "idle" | "monitoring" | "rolled_back";
export type AdminReleaseMonitoringCheckType =
  | "execution_integrity"
  | "governance_drift"
  | "p0_harness_replay"
  | "agent_admin_smoke"
  | "doctor_review_smoke"
  | "literature_isolation"
  | "manual_operator_note";
export type AdminReleaseMonitoringCheckStatus = "pass" | "warning" | "fail";
export type AdminReleaseMonitoringRequiredCheckStatus = AdminReleaseMonitoringCheckStatus | "missing";
export type AdminReleaseMonitoringAlertSeverity = "info" | "warning" | "critical";
export type AdminReleaseMonitoringAlertCategory =
  | "missing_required_check"
  | "post_release_check_failed"
  | "execution_integrity_failed"
  | "governance_drift"
  | "feature_flag_state_mismatch"
  | "rollback_ready";
export type AdminReleaseMonitoringRecommendedAction =
  | "observe"
  | "investigate"
  | "prepare_rollback"
  | "execute_step13_rollback";
export type AdminReleaseMonitoringAcknowledgementDisposition =
  | "investigating"
  | "accepted_risk"
  | "rollback_started_elsewhere"
  | "false_positive";

export interface AdminReleaseMonitoringLatestRelease {
  intent_id: string;
  execution_id: string;
  released_at: string;
  flag_enabled: boolean;
  rollback_plan_id: string | null;
}

export interface AdminReleaseMonitoringRequiredCheck {
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringRequiredCheckStatus;
  latest_check_id: string | null;
  reason: string;
}

export interface AdminReleaseMonitoringCheckRecord {
  check_id: string;
  intent_id: string;
  execution_id: string;
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringCheckStatus;
  observed_by: string;
  observed_at: string;
  summary: string;
  evidence_refs: string[];
  metrics: Record<string, JsonValue>;
  idempotency_key: string;
}

export interface AdminReleaseMonitoringAlert {
  alert_id: string;
  intent_id: string;
  execution_id: string;
  severity: AdminReleaseMonitoringAlertSeverity;
  category: AdminReleaseMonitoringAlertCategory;
  status: "active" | "acknowledged";
  message: string;
  source_check_ids: string[];
  recommended_action: AdminReleaseMonitoringRecommendedAction;
  created_at: string;
}

export interface AdminReleaseRollbackTriggerCandidate {
  candidate_id: string;
  intent_id: string;
  execution_id: string;
  source_alert_ids: string[];
  recommended_action: "execute_step13_rollback";
  rollback_plan_id: string;
  rollback_target: string;
  reason: string;
  created_at: string;
}

export interface AdminReleaseMonitoringAcknowledgement {
  acknowledgement_id: string;
  alert_id: string;
  intent_id: string;
  execution_id: string;
  acknowledged_by: string;
  acknowledged_at: string;
  disposition: AdminReleaseMonitoringAcknowledgementDisposition;
  reason: string;
}

export interface AdminReleaseMonitoringResponse {
  status: AdminReleaseMonitoringStatus;
  latest_release: AdminReleaseMonitoringLatestRelease | null;
  required_checks: AdminReleaseMonitoringRequiredCheck[];
  checks: AdminReleaseMonitoringCheckRecord[];
  alerts: AdminReleaseMonitoringAlert[];
  rollback_trigger_candidate: AdminReleaseRollbackTriggerCandidate | null;
  acknowledgements: AdminReleaseMonitoringAcknowledgement[];
  integrity: AdminReleaseGovernanceIntegrity;
  runtime: {
    auth: "admin";
    source: "reports/release_monitoring";
    mode: "post_release_monitoring";
  };
}

export interface AdminRecordReleaseMonitoringCheckRequest {
  intent_id: string;
  execution_id: string;
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringCheckStatus;
  observed_by: string;
  summary: string;
  evidence_refs: string[];
  metrics: Record<string, JsonValue>;
  idempotency_key: string;
}

export interface AdminAcknowledgeReleaseMonitoringAlertRequest {
  acknowledged_by: string;
  disposition: AdminReleaseMonitoringAcknowledgementDisposition;
  reason: string;
}

export type AdminReleaseClosureStatus = "idle" | "ready_to_close" | "blocked" | "closed" | "rolled_back_closed";
export type AdminReleaseClosureRecordStatus = "accepted" | "accepted_with_observations" | "rolled_back";
export type AdminReleaseClosureGateCheckStatus = "pass" | "warning" | "fail";

export interface AdminReleaseClosureGateCheck {
  name: string;
  status: AdminReleaseClosureGateCheckStatus;
  reason: string;
}

export interface AdminReleaseClosureGate {
  allowed: boolean;
  status: AdminReleaseClosureStatus;
  reasons: string[];
  checks: AdminReleaseClosureGateCheck[];
  allowed_statuses?: AdminReleaseClosureRecordStatus[];
  blocked_status_reasons?: Partial<Record<AdminReleaseClosureRecordStatus, string[]>>;
}

export interface AdminReleaseClosureLatestRelease {
  intent_id: string;
  release_execution_id: string;
  released_at: string | null;
  rollback_execution_id: string | null;
  rolled_back_at: string | null;
}

export interface AdminReleaseClosureRecord {
  closure_id: string;
  intent_id: string;
  release_execution_id: string;
  rollback_execution_id: string | null;
  closure_status: AdminReleaseClosureRecordStatus;
  closed_by: string;
  closed_at: string;
  rationale: string;
  evidence_package_id: string;
  idempotency_key: string;
}

export interface AdminReleaseEvidencePackage {
  package_id: string;
  closure_id: string;
  intent_id: string;
  release_execution_id: string;
  rollback_execution_id: string | null;
  generated_by: string;
  generated_at: string;
  closure_status: AdminReleaseClosureRecordStatus;
  summary: string;
  source_refs: string[];
  artifact_refs: string[];
  snapshot_hashes: Record<string, string>;
}

export interface AdminReleaseClosureResponse {
  status: AdminReleaseClosureStatus;
  latest_release: AdminReleaseClosureLatestRelease | null;
  closure_gate: AdminReleaseClosureGate;
  latest_closure: AdminReleaseClosureRecord | null;
  latest_evidence_package: AdminReleaseEvidencePackage | null;
  closures: AdminReleaseClosureRecord[];
  evidence_packages: AdminReleaseEvidencePackage[];
  integrity: { status: "verified" | "failed"; warnings: string[] };
  runtime: { auth: "admin"; source: "reports/release_closure"; mode: "post_release_closure" };
}

export interface AdminRecordReleaseClosureRequest {
  intent_id: string;
  release_execution_id: string;
  closure_status: AdminReleaseClosureRecordStatus;
  closed_by: string;
  rationale: string;
  idempotency_key: string;
}

export interface AdminExecuteReleaseRequest {
  intent_id: string;
  requested_by: string;
  idempotency_key: string;
  reason: string;
  expected_rollback_plan_id: string;
}

export interface AdminCreateReleaseIntentRequest {
  requested_by: string;
  target_scope: AdminReleaseTargetScope;
  status: AdminReleaseCreateIntentStatus;
  reason: string;
}

export interface AdminRecordReleaseApprovalRequest {
  approver_role: AdminReleaseApproverRole;
  decision: AdminReleaseApprovalDecision;
  reason: string;
  signed_by: string;
}

export interface AdminRecordReleaseRollbackPlanRequest {
  owner: string;
  status: AdminReleaseRollbackPlanStatus;
  verification_steps: string[];
}

export interface AdminCancelReleaseIntentRequest {
  actor: string;
  reason: string;
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
  requires_human_review?: boolean;
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

export interface TraceStartEvent {
  type: "trace.start";
  trace_id?: string | null;
  scene: Scene;
  session_id: string;
  run_id: string;
  server_received_at: string;
  graph_started_at: string;
  graph_path: string[];
  attrs: JsonObject;
}

export interface TraceStepEvent {
  type: "trace.step";
  trace_id?: string | null;
  name: string;
  at: string;
  session_id: string;
  run_id: string;
  attrs: JsonObject;
}

export interface TraceSummaryEvent {
  type: "trace.summary";
  trace_id?: string | null;
  run_id: string;
  session_id: string;
  scene: Scene;
  at: string;
  status: "completed" | "error" | "aborted";
  graph_path: string[];
  model?: string | null;
  has_thinking: boolean;
  response_chars: number;
  tool_calls: number;
  retrieval_hit_count: number;
  response_tokens: number | null;
  attrs: JsonObject;
}

export type TraceEvent = TraceStartEvent | TraceStepEvent | TraceSummaryEvent;

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
  | TraceEvent
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
  patient_identity?: PatientIdentitySnapshot | null;
  stage: string | null;
  assessment_draft: unknown;
  case_database_patient_id?: string | null;
  registry_patient_id?: number | null;
  current_patient_id?: string | number | null;
  references: JsonObject[];
  plan: JsonObject[];
  critic: JsonObject | null;
  safety_alert: JsonObject | null;
  uploaded_assets: Record<string, unknown>;
  context_maintenance: ContextMaintenanceState | null;
  context_state: ContextStateSnapshot | null;
}

export interface PatientIdentitySnapshot {
  patient_name: string | null;
  patient_number: string | null;
  identity_locked: boolean;
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
  trace_id?: string;
  context?: Record<string, unknown>;
}

export interface CrcTriageQuestion {
  id: string;
  stage: string;
  text: string;
  options: string[];
  [key: string]: unknown;
}

export interface CrcTriageQaItem {
  stage: string;
  question_id: string;
  question: string | null;
  answer: string;
  [key: string]: unknown;
}

export interface CrcTriageNodeResult {
  stage: string;
  title: string;
  risk_level: string;
  summary: string;
  next_step: string;
  [key: string]: unknown;
}

export interface CrcTriageProtocolState {
  stage?: string;
  current_question?: CrcTriageQuestion | null;
  qa_summary?: CrcTriageQaItem[];
  node_results?: CrcTriageNodeResult[];
  active_inquiry?: boolean;
  assessment?: Partial<CrcTriageAssessmentPayload>;
  [key: string]: unknown;
}

export interface CrcTriageAssessmentPayload {
  record_type: "crc_triage_assessment";
  chief_complaint: string;
  symptom_group: string;
  risk_level: string;
  disposition: string;
  red_flags: string[];
  known_crc_signals: Record<string, unknown>;
  suggested_tests: string[];
  missing_information: string[];
  qa_summary: Array<Record<string, unknown>>;
  node_results?: CrcTriageNodeResult[];
  protocol_state?: Record<string, unknown>;
  patient_summary: string;
  next_step: string;
  source_session_id: string;
  source_subflow: "crc_triage";
}

export interface SaveCrcTriageAssessmentRequest {
  assessment: CrcTriageAssessmentPayload;
}

export interface SaveCrcTriageAssessmentResponse {
  patient_id: number;
  patient_version: number;
  projection_version: number;
  event_ids: string[];
  record_id: number;
  reused: boolean;
}

export interface ClinicalAssertionRef {
  kind: string;
  id: string;
  field?: string | null;
}

export type ClinicalAssertionSource =
  | "triage"
  | "patient_upload"
  | "doctor_note"
  | "database_snapshot"
  | "care_card"
  | "model_draft";

export type ClinicalAssertionFactType =
  | "condition_signal"
  | "symptom"
  | "risk_disposition"
  | "missing_information"
  | "test_status"
  | "safety_rule_match"
  | "document_fact";

export type ClinicalAssertionReviewedStatus =
  | "unreviewed"
  | "accepted"
  | "edited"
  | "rejected"
  | "needs_evidence"
  | "unsafe";

export interface ClinicalAssertionPayload {
  assertion_id: string;
  patient_id: string;
  session_id?: string | null;
  source: ClinicalAssertionSource;
  source_record_id?: string | null;
  source_assessment_id?: string | null;
  normalized_fact: JsonObject;
  evidence_refs: ClinicalAssertionRef[];
  confidence: string;
  reviewed_status: ClinicalAssertionReviewedStatus;
  safety_policy_version?: string | null;
  created_from_projection_version?: string | null;
}

export interface DoctorReviewTimelineItem {
  item_id: string;
  kind: string;
  title: string;
  created_at: string;
  assertion_refs: string[];
}

export interface DoctorReviewProvenanceRef {
  kind: string;
  assertion_id?: string | null;
  record_id?: string | null;
  safety_policy_version?: string | null;
  id?: string | null;
  field?: string | null;
}

export interface DoctorReviewDraftSection {
  section_id: string;
  text: string;
  provenance: DoctorReviewProvenanceRef[];
  verification_status: "traceable" | "model_generated_unverified";
}

export interface DoctorReviewDraft {
  draft_id: string;
  sections: DoctorReviewDraftSection[];
}

export interface DoctorReviewResponse {
  patient_id: number;
  session_id: string;
  feature_flag: "doctor_review_cockpit_v0";
  timeline: DoctorReviewTimelineItem[];
  assertions: ClinicalAssertionPayload[];
  draft: DoctorReviewDraft;
  available_actions: DoctorActionType[];
}

export type DoctorActionType =
  | "accept"
  | "edit"
  | "reject"
  | "escalate"
  | "request_evidence"
  | "mark_unsafe";

export type DoctorReasonCode =
  | "fact_wrong"
  | "missing_red_flag"
  | "unsupported_claim"
  | "bad_tone"
  | "workflow_mismatch"
  | "citation_not_traceable"
  | "missing_information"
  | "unsafe_disposition"
  | "evidence_conflict"
  | "template_mismatch";

export interface DoctorActionTargetRefs {
  draft_id?: string | null;
  assertion_id?: string | null;
  assessment_id?: string | null;
  record_id?: string | null;
  care_card_id?: string | null;
  citation_id?: string | null;
}

export interface DoctorActionBeforeAfter {
  before: string;
  after: string;
}

export interface DoctorActionTraceRequest {
  action_type: DoctorActionType;
  target_object?: string | null;
  target_refs?: DoctorActionTargetRefs;
  before_after?: DoctorActionBeforeAfter | null;
  reason_code: DoctorReasonCode;
  reviewer_role?: string;
}

export interface DoctorActionTrace {
  trace_id: string;
  patient_id: number;
  session_id: string;
  action_type: DoctorActionType;
  target_object?: string | null;
  target_refs: DoctorActionTargetRefs;
  before_after?: DoctorActionBeforeAfter | null;
  reason_code: DoctorReasonCode;
  reviewer_role: string;
  deidentified: boolean;
  timestamp: string;
}

export interface DoctorActionTraceResponse {
  patient_id: number;
  trace: DoctorActionTrace;
  event_ids: string[];
  patient_version: number;
  projection_version: number;
  snapshot_changed?: boolean;
}

export interface UploadResponse {
  asset_id: string;
  asset_url: string;
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
  tumor_region_code?: string | null;
  tumor_region_codes?: string[];
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
  tumor_region_code?: string | null;
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
  tumor_region_code?: string | null;
  tumor_region_codes?: string[];
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
  clinical_assertions?: ClinicalAssertionPayload[];
  clinical_assertion_refs?: string[];
}

export interface PatientRegistryRecordsResponse {
  items: PatientRegistryRecord[];
}

export interface PatientCareCardsResponse {
  focusMetrics: string[];
  periodicChecks: string[];
  dailyActions: string[];
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
  patient_id: number;
  case_record: JsonObject | null;
  available_data: DatabaseAvailableData;
  cards: Record<string, JsonObject>;
}

export type DatabaseUpsertMode = "full" | "partial";

export interface DatabaseUpsertRequest {
  record: JsonObject;
  mode?: DatabaseUpsertMode;
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

export type ClinicalEventLogKind =
  | "node"
  | "stage"
  | "roadmap"
  | "critic"
  | "plan"
  | "references"
  | "error"
  | "done";

export type ClinicalEventLogTone = "neutral" | "success" | "warning" | "error";

export interface ClinicalEventLogEntry {
  id: string;
  kind: ClinicalEventLogKind;
  title: string;
  detail?: string | null;
  tone: ClinicalEventLogTone;
  requiresHumanReview?: boolean;
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
  patientIdentity?: PatientIdentitySnapshot | null;
  stage: string | null;
  references: JsonObject[];
  plan: JsonObject[];
  critic: JsonObject | null;
  safetyAlert: SafetyAlertState | null;
  assessmentDraft: unknown;
  caseDatabasePatientId: string | null;
  registryPatientId: number | null;
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
  eventLog: ClinicalEventLogEntry[];
}
