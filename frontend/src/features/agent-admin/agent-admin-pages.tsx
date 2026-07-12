import { useEffect, useState, type CSSProperties, type FormEvent } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpenCheck,
  Clock3,
  DatabaseZap,
  FileText,
  GitBranch,
  KeyRound,
  Layers3,
  ListChecks,
  Route,
  ServerCog,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";

import type {
  AdminReleaseApprovalDecision,
  AdminReleaseApprovalStatus,
  AdminReleaseClosureRecordStatus,
  AdminReleaseClosureResponse,
  AdminReleaseApproverRole,
  AdminReleaseCreateIntentStatus,
  AdminRecordReleaseClosureRequest,
  AdminReleaseDashboardResponse,
  AdminExecuteReleaseRequest,
  AdminReleaseExecutionResponse,
  AdminReleaseGateState,
  AdminReleaseGovernanceResponse,
  AdminReleaseHumanSignoffStatus,
  AdminReleaseIntegrityStatus,
  AdminReleaseMonitoringAcknowledgementDisposition,
  AdminReleaseMonitoringCheckStatus,
  AdminReleaseMonitoringCheckType,
  AdminReleaseMonitoringRequiredCheckStatus,
  AdminReleaseMonitoringResponse,
  AdminReleaseRollbackPlanStatus,
  AdminReleaseRunStatus,
  AdminReleaseTargetScope,
  ClinicalEventLogEntry,
  JsonValue,
  Scene,
  SessionState,
} from "../../app/api/types";
import {
  AgentAdminMetricStrip,
  AgentAdminPanel,
  AgentAdminDisabledAction,
  AgentAdminSplitWorkbench,
  AgentAdminSourceBadge,
  AgentAdminStateIcon,
} from "./agent-admin-components";
import {
  AGENT_ADMIN_TASKS,
  asRecord,
  buildEvidenceRows,
  buildLearningReadiness,
  buildMemoryAutomationSummary,
  buildMemoryLayerRows,
  buildMemoryLifecycleRows,
  buildMemoryVisualizationRows,
  buildPermissionRows,
  buildRuleCatalogGroups,
  buildRuleCatalogRows,
  buildRuleGroupRows,
  buildSessionSummary,
  buildToolInventoryRows,
  buildToolReachabilityRows,
  buildLiveTraceRows,
  formatSnapshot,
  readText,
  sessionStatus,
  type AgentAdminTraceRow,
  type AgentAdminTaskId,
} from "./agent-admin-model";
import type { AgentAdminReleaseDashboardResource, AgentAdminRulesResource, AgentAdminToolsResource } from "./agent-admin-view";
import type {
  AgentAdminReleaseClosureActions,
  AgentAdminReleaseClosureActionState,
  AgentAdminReleaseClosureResource,
  AgentAdminReleaseExecutionActions,
  AgentAdminReleaseExecutionActionState,
  AgentAdminReleaseExecutionResource,
  AgentAdminReleaseGovernanceActions,
  AgentAdminReleaseGovernanceActionState,
  AgentAdminReleaseGovernanceResource,
  AgentAdminReleaseMonitoringActions,
  AgentAdminReleaseMonitoringActionState,
  AgentAdminReleaseMonitoringResource,
} from "./agent-admin-view";

type AgentAdminPagesProps = {
  activeTaskId: AgentAdminTaskId;
  activeScene: Scene;
  patient: SessionState;
  doctor: SessionState;
  onNavigateTask: (taskId: AgentAdminTaskId) => void;
  toolsResource: AgentAdminToolsResource;
  rulesResource: AgentAdminRulesResource;
  releaseDashboardResource: AgentAdminReleaseDashboardResource;
  releaseGovernanceResource: AgentAdminReleaseGovernanceResource;
  releaseExecutionResource: AgentAdminReleaseExecutionResource;
  releaseMonitoringResource: AgentAdminReleaseMonitoringResource;
  releaseClosureResource: AgentAdminReleaseClosureResource;
  releaseGovernanceActionState: AgentAdminReleaseGovernanceActionState;
  releaseExecutionActionState: AgentAdminReleaseExecutionActionState;
  releaseMonitoringActionState: AgentAdminReleaseMonitoringActionState;
  releaseClosureActionState: AgentAdminReleaseClosureActionState;
  releaseGovernanceActions: AgentAdminReleaseGovernanceActions;
  releaseExecutionActions: AgentAdminReleaseExecutionActions;
  releaseMonitoringActions: AgentAdminReleaseMonitoringActions;
  releaseClosureActions: AgentAdminReleaseClosureActions;
};

type SessionRecentEvent = {
  key: string;
  text: string;
};

function latencyBarStyle(row: AgentAdminTraceRow): CSSProperties | undefined {
  return row.latencyPercent !== null ? { width: `${row.latencyPercent}%` } : undefined;
}

function latencyAvailabilityLabel(rows: AgentAdminTraceRow[]): string {
  const traceRows = rows.filter((row) => row.source === "runTrace");
  if (traceRows.length === 0) {
    return "eventLog fallback: latency unavailable";
  }

  const timedRows = traceRows.filter((row) => row.latency !== null).length;
  if (timedRows === traceRows.length) {
    return "所有 runTrace steps 都包含真实耗时";
  }
  if (timedRows > 0) {
    return "部分节点缺少 duration_ms / elapsed_ms";
  }
  return "runTrace 已到达；节点耗时尚未写入 duration_ms / elapsed_ms";
}

function recentSessionEventsForScene(
  sceneKey: Scene,
  label: "患者" | "医生",
  eventLog: ClinicalEventLogEntry[],
): SessionRecentEvent[] {
  return eventLog.slice(-5).map((event) => ({
    key: `${sceneKey}-${event.id}`,
    text: `${label}: ${event.title}`,
  }));
}

function watchedSession(activeScene: Scene, patient: SessionState, doctor: SessionState) {
  return activeScene === "doctor" ? doctor : patient;
}

export function AgentAdminTaskPages({
  activeTaskId,
  activeScene,
  patient,
  doctor,
  onNavigateTask,
  toolsResource,
  rulesResource,
  releaseDashboardResource,
  releaseGovernanceResource,
  releaseExecutionResource,
  releaseMonitoringResource,
  releaseClosureResource,
  releaseGovernanceActionState,
  releaseExecutionActionState,
  releaseMonitoringActionState,
  releaseClosureActionState,
  releaseGovernanceActions,
  releaseExecutionActions,
  releaseMonitoringActions,
  releaseClosureActions,
}: AgentAdminPagesProps) {
  const activeTask = AGENT_ADMIN_TASKS.find((task) => task.id === activeTaskId) ?? AGENT_ADMIN_TASKS[0];
  const ActiveTaskIcon = activeTask.icon;

  return (
    <section
      className={`agent-admin-task-page agent-admin-task-page-${activeTaskId}`}
      data-testid="agent-admin-task-page"
      data-task-id={activeTaskId}
      aria-label={`${activeTask.label}页面`}
    >
      <div className="agent-admin-task-page-header">
        <div>
          <span>后台子任务 / {activeTask.label}</span>
          <h1>{activeTask.detailTitle}</h1>
          <p>{activeTask.description}</p>
        </div>
        <ActiveTaskIcon size={24} aria-hidden="true" />
      </div>
      {activeTaskId === "overview" ? (
        <AgentAdminOverviewPage
          activeScene={activeScene}
          patient={patient}
          doctor={doctor}
          onNavigateTask={onNavigateTask}
          toolsResource={toolsResource}
        />
      ) : activeTaskId === "sessions" ? (
        <SessionsPage patient={patient} doctor={doctor} />
      ) : activeTaskId === "memory" ? (
        <MemoryPage patient={patient} doctor={doctor} />
      ) : activeTaskId === "rules" ? (
        <RulesPage rulesResource={rulesResource} />
      ) : activeTaskId === "tools" ? (
        <ToolsPage toolsResource={toolsResource} />
      ) : activeTaskId === "learning" ? (
        <LearningPage />
      ) : activeTaskId === "trace" ? (
        <TracePage activeScene={activeScene} patient={patient} doctor={doctor} />
      ) : activeTaskId === "evidence" ? (
        <EvidencePage activeScene={activeScene} patient={patient} doctor={doctor} />
      ) : activeTaskId === "release" ? (
        <ReleasePage
          releaseDashboardResource={releaseDashboardResource}
          releaseGovernanceResource={releaseGovernanceResource}
          releaseExecutionResource={releaseExecutionResource}
          releaseMonitoringResource={releaseMonitoringResource}
          releaseClosureResource={releaseClosureResource}
          releaseGovernanceActionState={releaseGovernanceActionState}
          releaseExecutionActionState={releaseExecutionActionState}
          releaseMonitoringActionState={releaseMonitoringActionState}
          releaseClosureActionState={releaseClosureActionState}
          releaseGovernanceActions={releaseGovernanceActions}
          releaseExecutionActions={releaseExecutionActions}
          releaseMonitoringActions={releaseMonitoringActions}
          releaseClosureActions={releaseClosureActions}
        />
      ) : activeTaskId === "read-only" ? (
        <ReadOnlyPage activeScene={activeScene} patient={patient} doctor={doctor} />
      ) : (
        <AgentAdminFallbackPage activeScene={activeScene} state={watchedSession(activeScene, patient, doctor)} />
      )}
    </section>
  );
}

function AgentAdminOverviewPage({
  activeScene,
  patient,
  doctor,
  onNavigateTask,
  toolsResource,
}: Omit<
  AgentAdminPagesProps,
  | "activeTaskId"
  | "rulesResource"
  | "releaseDashboardResource"
  | "releaseGovernanceResource"
  | "releaseExecutionResource"
  | "releaseMonitoringResource"
  | "releaseClosureResource"
  | "releaseGovernanceActionState"
  | "releaseExecutionActionState"
  | "releaseMonitoringActionState"
  | "releaseClosureActionState"
  | "releaseGovernanceActions"
  | "releaseExecutionActions"
  | "releaseMonitoringActions"
  | "releaseClosureActions"
>) {
  const watchedState = watchedSession(activeScene, patient, doctor);
  const patientSession = buildSessionSummary("患者", patient);
  const doctorSession = buildSessionSummary("医生", doctor);
  const ruleGroups = buildRuleGroupRows();
  const runtimeToolCount = toolsResource.status === "success" ? toolsResource.data.tools.length : null;
  const references = buildEvidenceRows(watchedState);
  const activePlan = watchedState.plan.slice(0, 4);
  const status = sessionStatus(watchedState);
  const timelineSteps = buildLiveTraceRows(watchedState).slice(-8);
  const recentRun = watchedState.runTrace;
  const metrics = [
    { label: "活跃会话", value: `${patient.sessionId ? 1 : 0}/${doctor.sessionId ? 1 : 0}`, tone: "red" as const },
    { label: "患者快照", value: patientSession.snapshot, detail: patientSession.sessionId, tone: "neutral" as const },
    { label: "医生快照", value: doctorSession.snapshot, detail: doctorSession.sessionId, tone: "neutral" as const },
    { label: "当前状态", value: status, detail: watchedState.activeRunId ?? "idle", tone: status === "error" ? "warning" as const : "success" as const },
    {
      label: "最近 run",
      value: recentRun?.runId ?? watchedState.activeRunId ?? "idle",
      detail: recentRun ? `${recentRun.status ?? "unknown"} / graph ${recentRun.graphPath.length}` : "runTrace pending",
      tone: recentRun?.status === "error" || recentRun?.status === "aborted" ? "warning" as const : "neutral" as const,
    },
    {
      label: "可用工具",
      value: runtimeToolCount === null ? "n/a (catalog)" : String(runtimeToolCount),
      detail: runtimeToolCount === null ? undefined : "runtime-api",
      tone: "red" as const,
    },
    { label: "规则组", value: String(ruleGroups.length), tone: "neutral" as const },
  ];
  const recentChanges = [
    `${activeScene === "doctor" ? "医生" : "患者"}会话 ${watchedState.sessionId ?? "未创建"} / ${formatSnapshot(watchedState)}`,
    `Active Run ${watchedState.activeRunId ?? "idle"}`,
    `证据池 ${references[0]?.title ?? "等待 references.append"}`,
  ];

  return (
    <>
      <AgentAdminMetricStrip metrics={metrics} />
      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="run health" title="运行健康时间线" icon={Clock3} action={<AgentAdminSourceBadge source="live" />}>
            <div className="agent-admin-timeline">
              {timelineSteps.map((step) => (
                <article key={step.id} className={`agent-admin-timeline-row agent-admin-timeline-row-${step.state}`}>
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state={step.state} />
                    {step.name}
                  </span>
                  <span
                    className={`agent-admin-latency-bar${step.state === "active" ? " agent-admin-latency-bar-active" : ""}`}
                    aria-label={step.latency ? `节点耗时 ${step.latency}` : step.state === "active" ? "节点运行中，耗时未记录" : "节点耗时未记录"}
                  >
                    {step.latencyPercent !== null ? <i style={latencyBarStyle(step)} /> : null}
                  </span>
                  <span>{step.detail}</span>
                  <strong>{step.latency ?? "—"}</strong>
                </article>
              ))}
            </div>
            <div className="agent-admin-plan-list">
              <span>当前计划</span>
              {activePlan.length > 0 ? (
                activePlan.map((step, index) => {
                  const record = asRecord(step) ?? {};
                  return (
                    <p key={readText(record.id, `plan-${index}`)}>
                      <GitBranch size={14} aria-hidden="true" />
                      {readText(record.title ?? record.id, `计划步骤 ${index + 1}`)}
                      <em>{readText(record.status, "pending")}</em>
                    </p>
                  );
                })
              ) : (
                <p>
                  <GitBranch size={14} aria-hidden="true" />
                  等待 plan.update 事件
                  <em>idle</em>
                </p>
              )}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <>
            <AgentAdminPanel eyebrow="graph state" title="当前图状态" icon={ServerCog}>
              <div className="agent-admin-session-grid">
                <div>
                  <span>患者 Session</span>
                  <strong>{patientSession.sessionId}</strong>
                  <small>snapshot {patientSession.snapshot} / {patientSession.status}</small>
                </div>
                <div>
                  <span>医生 Session</span>
                  <strong>{doctorSession.sessionId}</strong>
                  <small>snapshot {doctorSession.snapshot} / {doctorSession.status}</small>
                </div>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="attention queue" title="风险队列" icon={AlertTriangle}>
              <div className="agent-admin-detail-list">
                <button type="button" onClick={() => onNavigateTask("trace")}>
                  <Clock3 size={15} aria-hidden="true" /> 链路延迟与节点状态
                </button>
                <button type="button" onClick={() => onNavigateTask("evidence")}>
                  <FileText size={15} aria-hidden="true" /> 引用置信度与来源
                </button>
                <button type="button" onClick={() => onNavigateTask("read-only")}>
                  <ShieldCheck size={15} aria-hidden="true" /> 只读边界与权限
                </button>
              </div>
            </AgentAdminPanel>
          </>
        }
      />
      <AgentAdminPanel eyebrow="change feed" title="最近变化" icon={FileText}>
        <div className="agent-admin-detail-list">
          {recentChanges.map((change) => (
            <span key={change}>{change}</span>
          ))}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function SessionsPage({ patient, doctor }: Pick<AgentAdminPagesProps, "patient" | "doctor">) {
  const patientSession = buildSessionSummary("患者", patient);
  const doctorSession = buildSessionSummary("医生", doctor);
  const comparisonRows = [
    { label: "session id", patient: patientSession.sessionId, doctor: doctorSession.sessionId },
    { label: "snapshot", patient: patientSession.snapshot, doctor: doctorSession.snapshot },
    { label: "status", patient: patientSession.status, doctor: doctorSession.status },
    { label: "active run", patient: patientSession.activeRunId, doctor: doctorSession.activeRunId },
    { label: "current node", patient: patientSession.currentNode, doctor: doctorSession.currentNode },
  ];
  const recentEvents = [
    ...recentSessionEventsForScene("patient", "患者", patient.eventLog),
    ...recentSessionEventsForScene("doctor", "医生", doctor.eventLog),
  ];

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <div className="agent-admin-session-grid">
            <AgentAdminPanel eyebrow="patient context" title="患者 Session" icon={ServerCog}>
              <div className="agent-admin-detail-list">
                <span>{patientSession.sessionId}</span>
                <span>snapshot {patientSession.snapshot}</span>
                <span>{patientSession.status}</span>
                <span>{patientSession.activeRunId}</span>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="comparison" title="差异对比" icon={GitBranch}>
              <div className="agent-admin-timeline">
                {comparisonRows.map((row) => (
                  <article key={row.label} className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state={row.patient === row.doctor ? "success" : "active"} />
                      {row.label}
                    </span>
                    <span>{row.patient}</span>
                    <strong>{row.doctor}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>
          </div>
        }
        secondary={
          <AgentAdminPanel eyebrow="doctor context" title="医生 Session" icon={ServerCog}>
            <div className="agent-admin-detail-list">
              <span>{doctorSession.sessionId}</span>
              <span>snapshot {doctorSession.snapshot}</span>
              <span>{doctorSession.status}</span>
              <span>{doctorSession.activeRunId}</span>
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="event stream" title="recent session events" icon={Clock3}>
        <div className="agent-admin-detail-list">
          {recentEvents.length > 0
            ? recentEvents.map((event) => (
                <span key={event.key}>{event.text}</span>
              ))
            : <span>尚无 SSE 事件</span>}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function memoryStateIconState(state: string): "success" | "warning" | "ready" | "active" | "disabled" | "idle" {
  if (state === "active" || state === "completed") {
    return "success";
  }
  if (state === "running") {
    return "active";
  }
  if (state === "failed" || state === "stale" || state === "unstructured") {
    return "warning";
  }
  if (state === "empty" || state === "idle") {
    return "ready";
  }
  return "idle";
}

function MemoryPage({ patient, doctor }: Pick<AgentAdminPagesProps, "patient" | "doctor">) {
  const metrics = buildMemoryAutomationSummary(patient, doctor);
  const layerRows = buildMemoryLayerRows(patient, doctor);
  const lifecycleRows = buildMemoryLifecycleRows(patient, doctor);
  const visualizationRows = buildMemoryVisualizationRows(patient, doctor);
  const failedRows = visualizationRows.filter((row) => row.state === "failed");
  const inspectorRows = failedRows.length > 0 ? failedRows : visualizationRows.slice(0, 4);

  return (
    <>
      <AgentAdminPanel eyebrow="session boundary" title="这是会话上下文记忆，不是模型权重训练" icon={ShieldCheck}>
        <div className="agent-admin-detail-list">
          <span>patient {patient.sessionId ?? "未创建"}</span>
          <span>doctor {doctor.sessionId ?? "未创建"}</span>
          <span>memory is scoped to the current session snapshot and context maintenance pipeline</span>
        </div>
      </AgentAdminPanel>

      <AgentAdminPanel eyebrow="memory health" title="记忆健康" icon={Activity}>
        <AgentAdminMetricStrip metrics={metrics} />
      </AgentAdminPanel>

      <div className="agent-admin-memory-workbench">
        <AgentAdminPanel eyebrow="memory layers" title="记忆分层导航" icon={Layers3}>
          <div className="agent-admin-memory-layer-list">
            {layerRows.map((row) => (
              <article key={row.sourceKey} className={`agent-admin-memory-layer-row agent-admin-memory-layer-row-${row.state}`}>
                <span className="agent-admin-timeline-node">
                  <AgentAdminStateIcon state={memoryStateIconState(row.state)} />
                  {row.label}
                </span>
                <small>{row.sourceKey}</small>
                <strong>患者 {row.patientCount} / 医生 {row.doctorCount}</strong>
                <em>{row.retentionReason}</em>
              </article>
            ))}
          </div>
        </AgentAdminPanel>

        <AgentAdminPanel eyebrow="automation lifecycle" title="自动化维护流水线" icon={Sparkles}>
          <div className="agent-admin-memory-pipeline">
            {lifecycleRows.map((row) => (
              <article key={row.stage} className={`agent-admin-timeline-row agent-admin-timeline-row-${memoryStateIconState(row.state)}`}>
                <span className="agent-admin-timeline-node">
                  <AgentAdminStateIcon state={memoryStateIconState(row.state)} />
                  {row.stage}
                </span>
                <span>{row.explanation}</span>
                <strong>{row.state}</strong>
                <small>患者 {row.patient}</small>
                <small>医生 {row.doctor}</small>
              </article>
            ))}
          </div>
        </AgentAdminPanel>

        <AgentAdminPanel eyebrow="source / audit" title="来源与维护审计" icon={ServerCog}>
          <div className="agent-admin-detail-list">
            <span>patient maintenance {patient.contextMaintenance?.status ?? "idle"}</span>
            <span>{patient.contextMaintenance?.message ?? "patient context maintenance message empty"}</span>
            <span>{patient.contextMaintenance?.error ?? "patient context maintenance error empty"}</span>
            <span>doctor maintenance {doctor.contextMaintenance?.status ?? "idle"}</span>
            <span>{doctor.contextMaintenance?.message ?? "doctor context maintenance message empty"}</span>
            <span>{doctor.contextMaintenance?.error ?? "doctor context maintenance error empty"}</span>
            {inspectorRows.map((row) => (
              <span key={`${row.source}-${row.type}-${row.content}-${row.state}`}>
                <DatabaseZap size={15} aria-hidden="true" />
                {row.source} / {row.type} / {row.state} / {row.retentionReason}
              </span>
            ))}
          </div>
        </AgentAdminPanel>
      </div>

      <AgentAdminPanel eyebrow="current memory" title="当前记忆可视化" icon={FileText}>
        <div className="agent-admin-memory-table" role="table" aria-label="当前记忆可视化">
          <div className="agent-admin-memory-table-row agent-admin-memory-table-head" role="row">
            <span role="columnheader">记忆内容</span>
            <span role="columnheader">类型</span>
            <span role="columnheader">来源</span>
            <span role="columnheader">保留原因</span>
            <span role="columnheader">状态</span>
          </div>
          {visualizationRows.map((row) => (
            <article
              key={`${row.source}-${row.type}-${row.content}-${row.state}`}
              className={`agent-admin-memory-table-row agent-admin-memory-table-row-${row.state}`}
              role="row"
            >
              <span role="cell">{row.content}</span>
              <span role="cell">{row.type}</span>
              <span role="cell">{row.source}</span>
              <span role="cell">{row.retentionReason}</span>
              <strong role="cell">{row.state}</strong>
            </article>
          ))}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function RulesPage({ rulesResource }: { rulesResource: AgentAdminRulesResource }) {
  const rulesResponse = rulesResource.status === "success" ? rulesResource.data : null;
  const shouldRenderCatalog = rulesResource.status === "idle" || rulesResource.status === "error";
  const rulesByGroup = rulesResponse ? buildRuleCatalogGroups(rulesResponse) : shouldRenderCatalog ? buildRuleCatalogGroups() : [];
  const ruleRows = rulesResponse ? buildRuleCatalogRows(rulesResponse) : shouldRenderCatalog ? buildRuleCatalogRows() : [];
  const inspectedRule = ruleRows[0];
  const sourceBadge = rulesResource.status === "success" ? "runtime-api" : "catalog";
  const sourceStatus =
    rulesResource.status === "success"
      ? `${rulesResource.data.policy_id} / ${rulesResource.data.version} / ${rulesResource.data.status}`
      : rulesResource.status === "loading"
        ? "reading rules API"
        : rulesResource.status === "error"
          ? `rules API unavailable${rulesResource.error.status ? ` (${rulesResource.error.status})` : ""}: ${rulesResource.error.message}`
          : "catalog fallback";
  const note = rulesResource.status === "success" ? rulesResource.data.note : "静态目录占位；规则仅展示，不可在后台编辑";

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel
              eyebrow="group summary"
              title="规则分组"
              icon={ListChecks}
              action={<AgentAdminSourceBadge source={sourceBadge} />}
            >
              <div className="agent-admin-detail-list">
                <span>{sourceStatus}</span>
                <span>{note}</span>
              </div>
              <div className="agent-admin-timeline">
                {rulesByGroup.length > 0 ? (
                  rulesByGroup.map((group) => (
                    <article key={`${group.name}:${group.disposition}`} className="agent-admin-timeline-row agent-admin-timeline-row-success">
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state="success" />
                        {group.name}
                      </span>
                      <span>
                        {group.name} / {group.disposition} / {group.count}
                      </span>
                      <strong>{group.count}</strong>
                    </article>
                  ))
                ) : (
                  <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state="ready" />
                      reading rules API
                    </span>
                    <span>runtime rules</span>
                    <strong>loading</strong>
                  </article>
                )}
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="catalog tree" title="规则目录" icon={GitBranch}>
              <div className="agent-admin-detail-list">
                {rulesByGroup.map((group) => (
                  <span key={`${group.name}:${group.disposition}`}>
                    <strong>{group.name}</strong> / {(group.rules ?? []).map((rule) => rule.id).join(" / ")}
                  </span>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="rule inspector" title="规则检查器" icon={ServerCog}>
            <div className="agent-admin-detail-list">
              {inspectedRule ? (
                <>
                  <span>{inspectedRule.id}</span>
                  <span>{inspectedRule.label}</span>
                  <span>group: {inspectedRule.group}</span>
                  <span>disposition: {inspectedRule.disposition}</span>
                  <span>state: {inspectedRule.state}</span>
                  <span>policy_id: {inspectedRule.policyId}</span>
                  <span>version: {inspectedRule.version}</span>
                  <span>hard_fail_if_missed: {String(inspectedRule.hardFailIfMissed)}</span>
                  <span>{inspectedRule.conditionSummary}</span>
                </>
              ) : (
                <>
                  <span>reading rules API</span>
                  <span>runtime rules</span>
                  <span>loading</span>
                </>
              )}
            </div>
          </AgentAdminPanel>
        }
      />
    </>
  );
}

function ToolsPage({ toolsResource }: { toolsResource: AgentAdminToolsResource }) {
  const manifest = toolsResource.status === "success" ? toolsResource.data : null;
  const shouldRenderFallback = toolsResource.status === "idle" || toolsResource.status === "error";
  const toolRows = manifest ? buildToolInventoryRows(manifest) : shouldRenderFallback ? buildToolInventoryRows() : [];
  const reachabilityRows = manifest
    ? buildToolReachabilityRows(manifest)
    : shouldRenderFallback
      ? buildToolReachabilityRows()
      : [];
  const inspectedTool = toolRows[0];
  const sourceBadge =
    toolsResource.status === "success" ? "runtime-api" : toolsResource.status === "loading" ? "unavailable" : "catalog";
  const sourceStatus =
    toolsResource.status === "success"
      ? "runtime manifest"
      : toolsResource.status === "loading"
        ? "reading runtime manifest"
        : toolsResource.status === "error"
          ? `runtime manifest unavailable${toolsResource.error.status ? ` (${toolsResource.error.status})` : ""}: ${toolsResource.error.message}`
          : "非运行时清单";
  const fallbackStatus = toolsResource.status === "error" ? "非运行时清单" : null;
  const runtimeFlag = manifest ? `WEB_SEARCH_ENABLED ${String(manifest.runtime.web_search_enabled)}` : null;
  const emptyToolLabel =
    toolsResource.status === "success" ? "runtime manifest returned no tools" : "reading runtime manifest";
  const emptyGroupLabel =
    toolsResource.status === "success" ? "runtime manifest returned no groups" : "reading runtime manifest";
  const emptyState = toolsResource.status === "success" ? "empty" : "loading";

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel
              eyebrow="filters"
              title="工具筛选"
              icon={GitBranch}
              action={<AgentAdminSourceBadge source={sourceBadge} />}
            >
              <div className="agent-admin-detail-list">
                <span>{sourceStatus}</span>
                {fallbackStatus ? <span>{fallbackStatus}</span> : null}
                {runtimeFlag ? <span>{runtimeFlag}</span> : null}
                {reachabilityRows.map((group) => (
                  <span key={group.name}>
                    {group.name} / {group.count} / {group.status}
                  </span>
                ))}
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="inventory table" title="工具清单" icon={Wrench}>
              <div className="agent-admin-timeline">
                {toolRows.length > 0 ? (
                  toolRows.map((tool) => (
                    <article key={tool.name} className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state={tool.available ? "success" : "ready"} />
                        {tool.name}
                      </span>
                      <span>{tool.group}</span>
                      <strong>{tool.state}</strong>
                    </article>
                  ))
                ) : (
                  <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state="ready" />
                      {emptyToolLabel}
                    </span>
                    <span>runtime manifest</span>
                    <strong>{emptyState}</strong>
                  </article>
                )}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="dependency inspector" title="依赖检查" icon={ServerCog}>
            <div className="agent-admin-detail-list">
              {inspectedTool ? (
                <>
                  <span>{inspectedTool.name}</span>
                  <span>group: {inspectedTool.group}</span>
                  <span>state: {inspectedTool.state}</span>
                  <span>{inspectedTool.dependency}</span>
                  <span>registries: {inspectedTool.registries}</span>
                  <span>route targets: {inspectedTool.routeTargets}</span>
                  <span>graph scope: {inspectedTool.graphScope}</span>
                  <span>tool_executor required for executor dispatch</span>
                </>
              ) : (
                <>
                  <span>{emptyToolLabel}</span>
                  <span>runtime manifest</span>
                  <span>{emptyState}</span>
                </>
              )}
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="reachability map" title="可达性矩阵" icon={Route}>
        <div className="agent-admin-timeline">
          {reachabilityRows.length > 0 ? (
            reachabilityRows.map((group) => (
              <article key={group.name} className="agent-admin-timeline-row agent-admin-timeline-row-success">
                <span className="agent-admin-timeline-node">
                  <AgentAdminStateIcon state="success" />
                  {group.name}
                </span>
                <span>{group.status}</span>
                <strong>{group.count}</strong>
              </article>
            ))
          ) : (
            <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
              <span className="agent-admin-timeline-node">
                <AgentAdminStateIcon state="ready" />
                {emptyGroupLabel}
              </span>
              <span>runtime manifest</span>
              <strong>{emptyState}</strong>
            </article>
          )}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function LearningPage() {
  const readinessRows = buildLearningReadiness();
  const pipelineStages = ["发现论文", "去重", "打分", "摘要", "人工审核", "写入知识库", "生成学习报告"];
  const artifacts = [
    "learned artifact preview / paper digest card",
    "candidate guideline delta / oncology evidence watch",
    "knowledge base write preview / no mutation in Phase 1",
  ];

  return (
    <>
      <AgentAdminMetricStrip
        metrics={readinessRows.map((row) => ({
          label: row.label,
          value: row.value,
          detail: row.state,
          tone: row.state === "disabled" ? "warning" : "neutral",
        }))}
      />
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="source readiness" title="source readiness" icon={BookOpenCheck}>
              <div className="agent-admin-detail-list">
                {readinessRows.map((row) => (
                  <span key={row.label}>
                    <strong>{row.label}</strong> / {row.value} / {row.state}
                  </span>
                ))}
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="learning pipeline" title="学习流水线" icon={GitBranch}>
              <div className="agent-admin-timeline">
                {pipelineStages.map((stage, index) => (
                  <article key={stage} className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state={index < 2 ? "success" : "ready"} />
                      {stage}
                    </span>
                    <span>{index < 2 ? "readiness checked" : "waiting for scheduler"}</span>
                    <strong>{index + 1}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="actions" title="disabled actions" icon={ShieldCheck}>
            <div className="agent-admin-detail-list">
              <AgentAdminDisabledAction label="Run now" reason="一期不执行每日任务" />
              <AgentAdminDisabledAction label="Write knowledge base" reason="Phase 1 read-only" />
              <AgentAdminDisabledAction label="Enable scheduler" reason="scheduler disabled / config needed" />
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="preview" title="learned artifacts preview" icon={FileText}>
        <div className="agent-admin-detail-list">
          {artifacts.map((artifact) => (
            <span key={artifact}>{artifact}</span>
          ))}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function TracePage({ activeScene, patient, doctor }: Pick<AgentAdminPagesProps, "activeScene" | "patient" | "doctor">) {
  const state = watchedSession(activeScene, patient, doctor);
  const traceEvents = buildLiveTraceRows(state);
  const runTrace = state.runTrace;
  const timedTraceSteps = traceEvents.filter((event) => event.source === "runTrace" && event.latency !== null).length;

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="timeline events" title="执行事件时间线" icon={Clock3}>
            <div className="agent-admin-timeline">
              {traceEvents.map((event) => (
                <article key={event.id} className={`agent-admin-timeline-row agent-admin-timeline-row-${event.state}`}>
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state={event.state} />
                    {event.name}
                  </span>
                  <span>{event.detail}</span>
                  <strong>{event.latency ?? "—"}</strong>
                </article>
              ))}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <AgentAdminPanel eyebrow="latency panel" title="latency panel" icon={Activity}>
            <div className="agent-admin-detail-list">
              {runTrace ? (
                <>
                  <span>runTrace run {runTrace.runId ?? state.activeRunId ?? "unknown"}</span>
                  <span>runTrace status {runTrace.status ?? "unknown"}</span>
                  <span>graphPath {runTrace.graphPath.length > 0 ? runTrace.graphPath.join(" / ") : "empty"}</span>
                  <span>step count {runTrace.steps.length}</span>
                  <span>real latencies {timedTraceSteps}/{runTrace.steps.length}</span>
                  <span>{latencyAvailabilityLabel(traceEvents)}</span>
                </>
              ) : (
                <>
                  <span>active run {state.activeRunId ?? "idle"}</span>
                  <span>status node {state.statusNode ?? "idle"}</span>
                  <span>snapshot version {state.snapshotVersion}</span>
                  <span>eventLog length {state.eventLog.length}</span>
                  <span>eventLog fallback: latency unavailable</span>
                </>
              )}
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="event table" title="event table" icon={ListChecks}>
        <div className="agent-admin-timeline">
          {traceEvents.map((event) => (
            <article key={event.id} className="agent-admin-timeline-row agent-admin-timeline-row-ready">
              <span>{event.name}</span>
              <span>{event.detail}</span>
              <strong>{event.latency ?? "—"}</strong>
            </article>
          ))}
        </div>
      </AgentAdminPanel>
    </>
  );
}

function EvidencePage({ activeScene, patient, doctor }: Pick<AgentAdminPagesProps, "activeScene" | "patient" | "doctor">) {
  const state = watchedSession(activeScene, patient, doctor);
  const evidenceRows = buildEvidenceRows(state);

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="evidence filters" title="citation coverage" icon={GitBranch}>
              <div className="agent-admin-detail-list">
                <span>citation coverage / {evidenceRows.length} visible references</span>
                <span>retrieval profile / RAG / latest session snapshot</span>
                <span>confidence threshold / 70% review band</span>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="evidence table" title="证据池" icon={FileText}>
              <div className="agent-admin-timeline">
                {evidenceRows.map((row) => (
                  <article key={`${row.title}-${row.source}`} className="agent-admin-timeline-row agent-admin-timeline-row-success">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state="success" />
                      {row.title}
                    </span>
                    <span>{row.source}</span>
                    <strong>{row.confidence}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="retrieval profile" title="retrieval profile" icon={ServerCog}>
            <div className="agent-admin-detail-list">
              <span>scene {activeScene}</span>
              <span>session {state.sessionId ?? "未创建"}</span>
              <span>active run {state.activeRunId ?? "idle"}</span>
              <span>source preview {evidenceRows[0]?.title ?? "empty"}</span>
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="RAG pipeline" title="RAG pipeline" icon={Route}>
        <div className="agent-admin-detail-list">
          <span>query rewrite / retrieve / rank / cite / answer</span>
          <span>source preview / {evidenceRows.map((row) => row.source).join(" / ")}</span>
          <span>references are read-only frontend snapshots</span>
        </div>
      </AgentAdminPanel>
    </>
  );
}

function formatReleaseLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function formatReleaseValue(value: string | number | boolean | null): string {
  return value === null ? "not set" : String(value);
}

function releaseRowState(
  state: AdminReleaseRunStatus | AdminReleaseGateState | AdminReleaseHumanSignoffStatus,
): "success" | "warning" | "ready" | "active" | "disabled" | "idle" {
  if (state === "pass" || state === "recorded_elsewhere" || state === "not_required") {
    return "success";
  }
  if (state === "fail" || state === "blocked" || state === "locked" || state === "missing" || state === "invalid") {
    return "warning";
  }
  if (state === "shadow_only" || state === "warning") {
    return "active";
  }
  return "ready";
}

function ReleasePage({
  releaseDashboardResource,
  releaseGovernanceResource,
  releaseExecutionResource,
  releaseMonitoringResource,
  releaseClosureResource,
  releaseGovernanceActionState,
  releaseExecutionActionState,
  releaseMonitoringActionState,
  releaseClosureActionState,
  releaseGovernanceActions,
  releaseExecutionActions,
  releaseMonitoringActions,
  releaseClosureActions,
}: {
  releaseDashboardResource: AgentAdminReleaseDashboardResource;
  releaseGovernanceResource: AgentAdminReleaseGovernanceResource;
  releaseExecutionResource: AgentAdminReleaseExecutionResource;
  releaseMonitoringResource: AgentAdminReleaseMonitoringResource;
  releaseClosureResource: AgentAdminReleaseClosureResource;
  releaseGovernanceActionState: AgentAdminReleaseGovernanceActionState;
  releaseExecutionActionState: AgentAdminReleaseExecutionActionState;
  releaseMonitoringActionState: AgentAdminReleaseMonitoringActionState;
  releaseClosureActionState: AgentAdminReleaseClosureActionState;
  releaseGovernanceActions: AgentAdminReleaseGovernanceActions;
  releaseExecutionActions: AgentAdminReleaseExecutionActions;
  releaseMonitoringActions: AgentAdminReleaseMonitoringActions;
  releaseClosureActions: AgentAdminReleaseClosureActions;
}) {
  if (releaseDashboardResource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="release artifacts" title="Release Dashboard" icon={GitBranch}>
        <div className="agent-admin-detail-list">
          <span>reading release dashboard</span>
          <span>runtime release artifacts are loading from the admin API</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (releaseDashboardResource.status === "error") {
    const status = releaseDashboardResource.error.status ? ` (${releaseDashboardResource.error.status})` : "";
    return (
      <AgentAdminPanel eyebrow="release artifacts" title="Release Dashboard" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release dashboard unavailable{status}: {releaseDashboardResource.error.message}</span>
          <span>admin shell remains read-only and available</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (releaseDashboardResource.status === "success") {
    return (
      <ReleaseSuccessPage
        dashboard={releaseDashboardResource.data}
        releaseGovernanceResource={releaseGovernanceResource}
        releaseExecutionResource={releaseExecutionResource}
        releaseMonitoringResource={releaseMonitoringResource}
        releaseClosureResource={releaseClosureResource}
        releaseGovernanceActionState={releaseGovernanceActionState}
        releaseExecutionActionState={releaseExecutionActionState}
        releaseMonitoringActionState={releaseMonitoringActionState}
        releaseClosureActionState={releaseClosureActionState}
        releaseGovernanceActions={releaseGovernanceActions}
        releaseExecutionActions={releaseExecutionActions}
        releaseMonitoringActions={releaseMonitoringActions}
        releaseClosureActions={releaseClosureActions}
      />
    );
  }

  return (
    <AgentAdminPanel eyebrow="release artifacts" title="Release Dashboard" icon={GitBranch}>
      <div className="agent-admin-detail-list">
        <span>select Release to read committed release artifacts</span>
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseSuccessPage({
  dashboard,
  releaseGovernanceResource,
  releaseExecutionResource,
  releaseMonitoringResource,
  releaseClosureResource,
  releaseGovernanceActionState,
  releaseExecutionActionState,
  releaseMonitoringActionState,
  releaseClosureActionState,
  releaseGovernanceActions,
  releaseExecutionActions,
  releaseMonitoringActions,
  releaseClosureActions,
}: {
  dashboard: AdminReleaseDashboardResponse;
  releaseGovernanceResource: AgentAdminReleaseGovernanceResource;
  releaseExecutionResource: AgentAdminReleaseExecutionResource;
  releaseMonitoringResource: AgentAdminReleaseMonitoringResource;
  releaseClosureResource: AgentAdminReleaseClosureResource;
  releaseGovernanceActionState: AgentAdminReleaseGovernanceActionState;
  releaseExecutionActionState: AgentAdminReleaseExecutionActionState;
  releaseMonitoringActionState: AgentAdminReleaseMonitoringActionState;
  releaseClosureActionState: AgentAdminReleaseClosureActionState;
  releaseGovernanceActions: AgentAdminReleaseGovernanceActions;
  releaseExecutionActions: AgentAdminReleaseExecutionActions;
  releaseMonitoringActions: AgentAdminReleaseMonitoringActions;
  releaseClosureActions: AgentAdminReleaseClosureActions;
}) {
  const versionRows = Object.entries(dashboard.version_chain);
  const summary = dashboard.summary;
  const summaryMetrics = [
    {
      label: "Hard fails",
      value: String(summary.hard_fail_count),
      detail: "release gate",
      tone: summary.hard_fail_count === 0 ? ("success" as const) : ("warning" as const),
    },
    {
      label: "P0 cases",
      value: `${summary.p0_cases_passed}/${summary.p0_cases_total}`,
      detail: "passed",
      tone: summary.p0_cases_passed === summary.p0_cases_total ? ("success" as const) : ("warning" as const),
    },
    {
      label: "Literature claims",
      value: String(summary.literature_claims),
      detail: `${summary.literature_isolation_violations} isolation violations`,
      tone: summary.literature_isolation_violations === 0 ? ("success" as const) : ("warning" as const),
    },
    {
      label: "Clinical RAG ingest",
      value: String(summary.clinical_rag_ingest_enabled),
      detail: "read-only",
      tone: summary.clinical_rag_ingest_enabled ? ("warning" as const) : ("neutral" as const),
    },
  ];

  return (
    <>
      <AgentAdminMetricStrip metrics={summaryMetrics} />
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="version chain" title="Release Dashboard" icon={GitBranch}>
              <div className="agent-admin-timeline">
                {versionRows.map(([key, value]) => (
                  <article key={key} className="agent-admin-timeline-row agent-admin-timeline-row-success">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state="success" />
                      {formatReleaseLabel(key)}
                    </span>
                    <span>committed artifact</span>
                    <strong>{formatReleaseValue(value)}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>

            <AgentAdminPanel eyebrow="harness runs" title="Harness run ledger" icon={ListChecks}>
              <div className="agent-admin-timeline">
                {dashboard.runs.map((run) => {
                  const rowState = releaseRowState(run.status);
                  return (
                    <article
                      key={`${run.kind}-${run.source_path}-${run.run_id}`}
                      className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}
                    >
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state={rowState} />
                        {run.run_id}
                      </span>
                      <span>{run.kind} / {run.source_path}</span>
                      <strong>{run.status} / hard fails {run.hard_fail_count}</strong>
                    </article>
                  );
                })}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <>
            <AgentAdminPanel eyebrow="decision" title="Release decision" icon={ShieldCheck}>
              <div className="agent-admin-detail-list">
                <span>release decision / {dashboard.release_decision}</span>
                <span>rollback target / {formatReleaseValue(dashboard.rollback_target)}</span>
                <span>runtime / {dashboard.runtime.auth} / {dashboard.runtime.source} / {dashboard.runtime.mode}</span>
              </div>
            </AgentAdminPanel>

            <AgentAdminPanel eyebrow="human sign-off" title="Sign-off readiness" icon={KeyRound}>
              <div className="agent-admin-detail-list">
                <span>required / {String(dashboard.human_signoff.required)}</span>
                <span>status / {dashboard.human_signoff.status}</span>
                <span>{dashboard.human_signoff.reason}</span>
              </div>
            </AgentAdminPanel>
          </>
        }
      />

      <AgentAdminPanel eyebrow="blocking gates" title="Blocking gates" icon={AlertTriangle}>
        <div className="agent-admin-timeline">
          {dashboard.blocking_gates.map((gate) => {
            const rowState = releaseRowState(gate.state);
            return (
              <article key={gate.id} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
                <span className="agent-admin-timeline-node">
                  <AgentAdminStateIcon state={rowState} />
                  {gate.label}
                </span>
                <span>{gate.reason}</span>
                <strong>{gate.state}</strong>
              </article>
            );
          })}
        </div>
      </AgentAdminPanel>

      <AgentAdminPanel eyebrow="read-only actions" title="Disabled release controls" icon={KeyRound}>
        <div className="agent-admin-detail-list">
          {dashboard.disabled_actions.map((action) => (
            <button key={action.id} type="button" className="agent-admin-disabled-action" disabled>
              <KeyRound size={15} aria-hidden="true" />
              <span>{action.label}</span>
              <small>{action.reason}</small>
            </button>
          ))}
        </div>
      </AgentAdminPanel>

      <ReleaseGovernanceSection
        resource={releaseGovernanceResource}
        actionState={releaseGovernanceActionState}
        actions={releaseGovernanceActions}
        showDisabledExecutionControls={releaseExecutionResource.status === "idle"}
      />

      <ReleaseExecutionSection
        resource={releaseExecutionResource}
        actionState={releaseExecutionActionState}
        actions={releaseExecutionActions}
      />

      <ReleaseMonitoringSection
        resource={releaseMonitoringResource}
        actionState={releaseMonitoringActionState}
        actions={releaseMonitoringActions}
      />

      <ReleaseClosureSection
        resource={releaseClosureResource}
        actionState={releaseClosureActionState}
        actions={releaseClosureActions}
      />
    </>
  );
}

function releaseApprovalState(status: AdminReleaseApprovalStatus): "success" | "warning" | "ready" {
  if (status === "approved") {
    return "success";
  }
  if (status === "rejected" || status === "changes_requested") {
    return "warning";
  }
  return "ready";
}

function releaseIntegrityState(status: AdminReleaseIntegrityStatus): "success" | "warning" {
  return status === "verified" ? "success" : "warning";
}

function splitReleaseVerificationSteps(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((step) => step.trim())
    .filter(Boolean);
}

const releaseMonitoringCheckTypes: AdminReleaseMonitoringCheckType[] = [
  "execution_integrity",
  "governance_drift",
  "p0_harness_replay",
  "agent_admin_smoke",
  "doctor_review_smoke",
  "literature_isolation",
  "manual_operator_note",
];

const releaseMonitoringCheckStatuses: AdminReleaseMonitoringCheckStatus[] = ["pass", "warning", "fail"];

const releaseMonitoringAcknowledgementDispositions: AdminReleaseMonitoringAcknowledgementDisposition[] = [
  "investigating",
  "accepted_risk",
  "rollback_started_elsewhere",
  "false_positive",
];

const releaseClosureStatuses: AdminReleaseClosureRecordStatus[] = [
  "accepted",
  "accepted_with_observations",
  "rolled_back",
];

function releaseClosureAllowedStatuses(closure: AdminReleaseClosureResponse): AdminReleaseClosureRecordStatus[] {
  if (closure.closure_gate.allowed_statuses) {
    return closure.closure_gate.allowed_statuses;
  }
  if (!closure.closure_gate.allowed) {
    return [];
  }
  return closure.latest_release?.rollback_execution_id
    ? ["rolled_back"]
    : ["accepted", "accepted_with_observations"];
}

function releaseMonitoringCheckState(
  status: AdminReleaseMonitoringCheckStatus | AdminReleaseMonitoringRequiredCheckStatus,
): "success" | "warning" | "ready" | "active" {
  if (status === "pass") {
    return "success";
  }
  if (status === "fail" || status === "missing") {
    return "warning";
  }
  if (status === "warning") {
    return "active";
  }
  return "ready";
}

function splitMonitoringEvidenceRefs(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((ref) => ref.trim())
    .filter(Boolean);
}

function parseMonitoringMetrics(value: string): { ok: true; metrics: Record<string, JsonValue> } | { ok: false; message: string } {
  if (!value.trim()) {
    return { ok: true, metrics: {} };
  }

  try {
    const parsed = JSON.parse(value) as JsonValue;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, message: "Metrics JSON must be a valid object" };
    }
    return { ok: true, metrics: parsed as Record<string, JsonValue> };
  } catch {
    return { ok: false, message: "Metrics JSON must be a valid object" };
  }
}

function ReleaseMonitoringSection({
  resource,
  actionState,
  actions,
}: {
  resource: AgentAdminReleaseMonitoringResource;
  actionState: AgentAdminReleaseMonitoringActionState;
  actions: AgentAdminReleaseMonitoringActions;
}) {
  if (resource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="post-release monitoring" title="Post-release monitoring" icon={Activity}>
        <div className="agent-admin-detail-list">
          <span>reading release monitoring</span>
          <span>post-release checks and alerts are loading from reports/release_monitoring</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "error") {
    const status = resource.error.status ? ` (${resource.error.status})` : "";
    return (
      <AgentAdminPanel eyebrow="post-release monitoring" title="Post-release monitoring" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release monitoring unavailable{status}: {resource.error.message}</span>
          <span>dashboard governance and execution panels remain available</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "success") {
    return <ReleaseMonitoringPanel monitoring={resource.data} actionState={actionState} actions={actions} />;
  }

  return (
    <AgentAdminPanel eyebrow="post-release monitoring" title="Post-release monitoring" icon={Activity}>
      <div className="agent-admin-detail-list">
        <span>post-release monitoring idle</span>
        <span>monitoring API is unavailable for this admin surface</span>
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseMonitoringPanel({
  monitoring,
  actionState,
  actions,
}: {
  monitoring: AdminReleaseMonitoringResponse;
  actionState: AgentAdminReleaseMonitoringActionState;
  actions: AgentAdminReleaseMonitoringActions;
}) {
  const latestRelease = monitoring.latest_release;
  const actionRunning = actionState.status === "running";
  const canRecordCheck = monitoring.status === "monitoring" && monitoring.latest_release !== null && monitoring.integrity.status === "verified" && !actionRunning;
  const canAcknowledge = monitoring.alerts.length > 0 && monitoring.integrity.status === "verified" && !actionRunning;
  const integrityWarnings = monitoring.integrity.warnings.length > 0 ? monitoring.integrity.warnings : ["no integrity warnings"];

  return (
    <>
      <AgentAdminPanel eyebrow="post-release monitoring" title="Post-release monitoring" icon={Activity}>
        {actionState.status === "running" ? (
          <span className="agent-admin-action-status">{actionState.label} in progress</span>
        ) : null}
        {actionState.status === "error" ? (
          <span className="agent-admin-action-status agent-admin-action-status-error">{actionState.message}</span>
        ) : null}

        <AgentAdminSplitWorkbench
          primary={
            <div className="agent-admin-detail-list">
              <span>monitoring status / {monitoring.status}</span>
              <span>latest release / {latestRelease ? `${latestRelease.intent_id} / ${latestRelease.execution_id}` : "none"}</span>
              <span>released at / {latestRelease?.released_at ?? "not released"}</span>
              <span>flag enabled / {latestRelease ? String(latestRelease.flag_enabled) : "not set"}</span>
              <span>rollback plan / {latestRelease?.rollback_plan_id ?? "none"}</span>
              <span>runtime / {monitoring.runtime.auth} / {monitoring.runtime.source} / {monitoring.runtime.mode}</span>
            </div>
          }
          secondary={
            <div className="agent-admin-detail-list">
              <span>integrity status / {monitoring.integrity.status}</span>
              {integrityWarnings.map((warning) => (
                <span key={warning}>integrity warning / {warning}</span>
              ))}
            </div>
          }
        />
      </AgentAdminPanel>

      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="required checks" title="Required monitoring checks" icon={ListChecks}>
            <div className="agent-admin-timeline">
              {monitoring.required_checks.length > 0 ? (
                monitoring.required_checks.map((check) => {
                  const rowState = releaseMonitoringCheckState(check.status);
                  return (
                    <article key={check.check_type} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state={rowState} />
                        {check.check_type}
                      </span>
                      <span>{check.latest_check_id ?? "no latest check"} / {check.reason}</span>
                      <strong>{check.status}</strong>
                    </article>
                  );
                })
              ) : (
                <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state="ready" />
                    no required checks
                  </span>
                  <span>monitoring has no required checks yet</span>
                  <strong>idle</strong>
                </article>
              )}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <AgentAdminPanel eyebrow="rollback trigger" title="Rollback trigger candidate" icon={AlertTriangle}>
            <div className="agent-admin-detail-list">
              {monitoring.rollback_trigger_candidate ? (
                <div className="agent-admin-detail-list agent-admin-monitoring-trigger">
                  <span>candidate / {monitoring.rollback_trigger_candidate.candidate_id}</span>
                  <span>recommendation / {monitoring.rollback_trigger_candidate.recommended_action}</span>
                  <span>execute_step13_rollback is the explicit backend recommendation for this candidate</span>
                  <span>rollback plan / {monitoring.rollback_trigger_candidate.rollback_plan_id}</span>
                  <span>rollback target / {monitoring.rollback_trigger_candidate.rollback_target}</span>
                  <span>{monitoring.rollback_trigger_candidate.reason}</span>
                </div>
              ) : (
                <span>no rollback trigger candidate</span>
              )}
            </div>
          </AgentAdminPanel>
        }
      />

      <AgentAdminSplitWorkbench
        primary={<ReleaseMonitoringChecksHistory monitoring={monitoring} />}
        secondary={<ReleaseMonitoringAlerts monitoring={monitoring} />}
      />

      <AgentAdminSplitWorkbench
        primary={<ReleaseMonitoringAcknowledgements monitoring={monitoring} />}
        secondary={
          <ReleaseMonitoringForms
            monitoring={monitoring}
            canRecordCheck={canRecordCheck}
            canAcknowledge={canAcknowledge}
            actions={actions}
          />
        }
      />
    </>
  );
}

function ReleaseMonitoringChecksHistory({ monitoring }: { monitoring: AdminReleaseMonitoringResponse }) {
  return (
    <AgentAdminPanel eyebrow="checks history" title="Monitoring checks history" icon={FileText}>
      <div className="agent-admin-timeline">
        {monitoring.checks.length > 0 ? (
          monitoring.checks.map((check) => {
            const rowState = releaseMonitoringCheckState(check.status);
            return (
              <article key={check.check_id} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
                <span className="agent-admin-timeline-node">
                  <AgentAdminStateIcon state={rowState} />
                  {check.check_id}
                </span>
                <span>{check.check_type} / {check.observed_by} / {check.summary}</span>
                <strong>{check.status}</strong>
              </article>
            );
          })
        ) : (
          <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
            <span className="agent-admin-timeline-node">
              <AgentAdminStateIcon state="ready" />
              no monitoring checks
            </span>
            <span>record a required check after release</span>
            <strong>idle</strong>
          </article>
        )}
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseMonitoringAlerts({ monitoring }: { monitoring: AdminReleaseMonitoringResponse }) {
  return (
    <AgentAdminPanel eyebrow="alerts" title="Monitoring alerts" icon={AlertTriangle}>
      <div className="agent-admin-detail-list">
        {monitoring.alerts.length > 0 ? (
          monitoring.alerts.map((alert) => (
            <span
              key={alert.alert_id}
              className={alert.severity === "critical" ? "agent-admin-monitoring-alert-critical" : undefined}
            >
              {alert.alert_id} / {alert.status} / {alert.severity} / {alert.category} / {alert.recommended_action} / {alert.message}
            </span>
          ))
        ) : (
          <span>no monitoring alerts</span>
        )}
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseMonitoringAcknowledgements({ monitoring }: { monitoring: AdminReleaseMonitoringResponse }) {
  return (
    <AgentAdminPanel eyebrow="acknowledgements" title="Monitoring acknowledgements" icon={BookOpenCheck}>
      <div className="agent-admin-detail-list">
        {monitoring.acknowledgements.length > 0 ? (
          monitoring.acknowledgements.map((acknowledgement) => (
            <span key={acknowledgement.acknowledgement_id}>
              {acknowledgement.acknowledgement_id} / {acknowledgement.alert_id} / {acknowledgement.acknowledged_by} / {acknowledgement.disposition} / {acknowledgement.reason}
            </span>
          ))
        ) : (
          <span>no monitoring acknowledgements</span>
        )}
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseMonitoringForms({
  monitoring,
  canRecordCheck,
  canAcknowledge,
  actions,
}: {
  monitoring: AdminReleaseMonitoringResponse;
  canRecordCheck: boolean;
  canAcknowledge: boolean;
  actions: AgentAdminReleaseMonitoringActions;
}) {
  const [checkActor, setCheckActor] = useState("release_monitor");
  const [checkType, setCheckType] = useState<AdminReleaseMonitoringCheckType>("execution_integrity");
  const [checkStatus, setCheckStatus] = useState<AdminReleaseMonitoringCheckStatus>("pass");
  const [checkSummary, setCheckSummary] = useState("");
  const [checkIdempotencyKey, setCheckIdempotencyKey] = useState("");
  const [evidenceRefsText, setEvidenceRefsText] = useState("");
  const [metricsText, setMetricsText] = useState("{}");
  const [formError, setFormError] = useState("");
  const [alertId, setAlertId] = useState(monitoring.alerts[0]?.alert_id ?? "");
  const [ackActor, setAckActor] = useState("release_monitor");
  const [ackDisposition, setAckDisposition] = useState<AdminReleaseMonitoringAcknowledgementDisposition>("investigating");
  const [ackReason, setAckReason] = useState("");
  const latestRelease = monitoring.latest_release;

  async function handleRecordCheck(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError("");
    if (!canRecordCheck || !latestRelease) {
      return;
    }

    const metricsResult = parseMonitoringMetrics(metricsText);
    if (!metricsResult.ok) {
      setFormError(metricsResult.message);
      return;
    }

    await actions.recordCheck({
      intent_id: latestRelease.intent_id,
      execution_id: latestRelease.execution_id,
      check_type: checkType,
      status: checkStatus,
      observed_by: checkActor.trim(),
      summary: checkSummary.trim(),
      evidence_refs: splitMonitoringEvidenceRefs(evidenceRefsText),
      metrics: metricsResult.metrics,
      idempotency_key: checkIdempotencyKey.trim(),
    });
  }

  async function handleAcknowledgeAlert(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError("");
    if (!canAcknowledge) {
      return;
    }

    await actions.acknowledgeAlert(alertId.trim(), {
      acknowledged_by: ackActor.trim(),
      disposition: ackDisposition,
      reason: ackReason.trim(),
    });
  }

  return (
    <AgentAdminPanel eyebrow="monitoring writes" title="Monitoring actions" icon={KeyRound}>
      {formError ? (
        <span className="agent-admin-action-status agent-admin-action-status-error">{formError}</span>
      ) : null}

      <div className="agent-admin-governance-form-grid">
        <form className="agent-admin-governance-form" onSubmit={handleRecordCheck}>
          <h3>Record monitoring check</h3>
          <label htmlFor="release-monitoring-actor">
            <span>Monitoring actor</span>
            <input
              id="release-monitoring-actor"
              value={checkActor}
              onChange={(event) => setCheckActor(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-check-type">
            <span>Monitoring check type</span>
            <select
              id="release-monitoring-check-type"
              value={checkType}
              onChange={(event) => setCheckType(event.target.value as AdminReleaseMonitoringCheckType)}
            >
              {releaseMonitoringCheckTypes.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </label>
          <label htmlFor="release-monitoring-check-status">
            <span>Monitoring check status</span>
            <select
              id="release-monitoring-check-status"
              value={checkStatus}
              onChange={(event) => setCheckStatus(event.target.value as AdminReleaseMonitoringCheckStatus)}
            >
              {releaseMonitoringCheckStatuses.map((status) => (
                <option key={status} value={status}>{status}</option>
              ))}
            </select>
          </label>
          <label htmlFor="release-monitoring-summary">
            <span>Monitoring summary</span>
            <textarea
              id="release-monitoring-summary"
              value={checkSummary}
              onChange={(event) => setCheckSummary(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-idempotency-key">
            <span>Monitoring idempotency key</span>
            <input
              id="release-monitoring-idempotency-key"
              value={checkIdempotencyKey}
              onChange={(event) => setCheckIdempotencyKey(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-evidence-refs">
            <span>Evidence refs</span>
            <textarea
              id="release-monitoring-evidence-refs"
              value={evidenceRefsText}
              onChange={(event) => setEvidenceRefsText(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-metrics-json">
            <span>Metrics JSON</span>
            <textarea
              id="release-monitoring-metrics-json"
              value={metricsText}
              onChange={(event) => setMetricsText(event.target.value)}
            />
          </label>
          <button type="submit" disabled={!canRecordCheck}>Record monitoring check</button>
        </form>

        <form className="agent-admin-governance-form" onSubmit={handleAcknowledgeAlert}>
          <h3>Acknowledge monitoring alert</h3>
          <label htmlFor="release-monitoring-alert-id">
            <span>Alert id</span>
            <input
              id="release-monitoring-alert-id"
              value={alertId}
              onChange={(event) => setAlertId(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-ack-actor">
            <span>Acknowledgement actor</span>
            <input
              id="release-monitoring-ack-actor"
              value={ackActor}
              onChange={(event) => setAckActor(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-monitoring-ack-disposition">
            <span>Acknowledgement disposition</span>
            <select
              id="release-monitoring-ack-disposition"
              value={ackDisposition}
              onChange={(event) => setAckDisposition(event.target.value as AdminReleaseMonitoringAcknowledgementDisposition)}
            >
              {releaseMonitoringAcknowledgementDispositions.map((disposition) => (
                <option key={disposition} value={disposition}>{disposition}</option>
              ))}
            </select>
          </label>
          <label htmlFor="release-monitoring-ack-reason">
            <span>Acknowledgement reason</span>
            <textarea
              id="release-monitoring-ack-reason"
              value={ackReason}
              onChange={(event) => setAckReason(event.target.value)}
              required
            />
          </label>
          <button type="submit" disabled={!canAcknowledge}>Acknowledge alert</button>
        </form>
      </div>
    </AgentAdminPanel>
  );
}

function releaseClosureState(status: AdminReleaseClosureResponse["status"]): "success" | "warning" | "ready" | "active" {
  if (status === "closed" || status === "rolled_back_closed") {
    return "success";
  }
  if (status === "blocked") {
    return "warning";
  }
  if (status === "ready_to_close") {
    return "active";
  }
  return "ready";
}

function releaseClosureStatusSummary(status: AdminReleaseClosureResponse["status"]): string {
  switch (status) {
    case "ready_to_close":
      return "Closure gate is ready for an audited close-out record.";
    case "blocked":
      return "Closure remains blocked until the gate reasons are resolved.";
    case "closed":
      return "Release is closed and the latest evidence package is available.";
    case "rolled_back_closed":
      return "Rollback closure is recorded and the evidence package is sealed.";
    default:
      return "Closure is idle until a releasable execution exists.";
  }
}

function ReleaseClosureSection({
  resource,
  actionState,
  actions,
}: {
  resource: AgentAdminReleaseClosureResource;
  actionState: AgentAdminReleaseClosureActionState;
  actions: AgentAdminReleaseClosureActions;
}) {
  if (resource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="post-release closure" title="Release closure" icon={BookOpenCheck}>
        <div className="agent-admin-detail-list">
          <span>reading release closure</span>
          <span>closure gate and evidence package summaries are loading from reports/release_closure</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "error") {
    const status = resource.error.status ? ` (${resource.error.status})` : "";
    return (
      <AgentAdminPanel eyebrow="post-release closure" title="Release closure" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release closure unavailable{status}: {resource.error.message}</span>
          <span>monitoring remains available while closure reports are unavailable</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "success") {
    return <ReleaseClosurePanel closure={resource.data} actionState={actionState} actions={actions} />;
  }

  return (
    <AgentAdminPanel eyebrow="post-release closure" title="Release closure" icon={BookOpenCheck}>
      <div className="agent-admin-detail-list">
        <span>release closure idle</span>
        <span>closure API is unavailable for this admin surface</span>
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseClosurePanel({
  closure,
  actionState,
  actions,
}: {
  closure: AdminReleaseClosureResponse;
  actionState: AgentAdminReleaseClosureActionState;
  actions: AgentAdminReleaseClosureActions;
}) {
  const latestRelease = closure.latest_release;
  const latestClosure = closure.latest_closure;
  const latestEvidencePackage = closure.latest_evidence_package;
  const closureState = releaseClosureState(closure.status);
  const actionRunning = actionState.status === "running";
  const canSubmit = closure.closure_gate.allowed && latestRelease !== null && closure.integrity.status === "verified" && !actionRunning;
  const integrityWarnings = closure.integrity.warnings.length > 0 ? closure.integrity.warnings : ["no integrity warnings"];

  return (
    <>
      <AgentAdminPanel eyebrow="post-release closure" title="Release closure" icon={BookOpenCheck}>
        {actionState.status === "running" ? (
          <span className="agent-admin-action-status">{actionState.label} in progress</span>
        ) : null}
        {actionState.status === "error" ? (
          <span className="agent-admin-action-status agent-admin-action-status-error">{actionState.message}</span>
        ) : null}

        <AgentAdminSplitWorkbench
          primary={
            <div className="agent-admin-detail-list">
              <span>closure status / {closure.status}</span>
              <strong>{closure.status}</strong>
              <span>closure gate / {closure.closure_gate.status}</span>
              <span className={`agent-admin-release-closure-status agent-admin-release-closure-status-${closureState}`}>
                {releaseClosureStatusSummary(closure.status)}
              </span>
              <span>latest release / {latestRelease ? `${latestRelease.intent_id} / ${latestRelease.release_execution_id}` : "none"}</span>
              <span>released at / {latestRelease?.released_at ?? "not released"}</span>
              <span>rollback execution / {latestRelease?.rollback_execution_id ?? "none"}</span>
              <span>rolled back at / {latestRelease?.rolled_back_at ?? "not rolled back"}</span>
              <span>runtime / {closure.runtime.auth} / {closure.runtime.source} / {closure.runtime.mode}</span>
            </div>
          }
          secondary={
            <div className="agent-admin-detail-list">
              <span>gate allowed / {String(closure.closure_gate.allowed)}</span>
              <span>integrity status / {closure.integrity.status}</span>
              {integrityWarnings.map((warning) => (
                <span key={warning}>integrity warning / {warning}</span>
              ))}
            </div>
          }
        />
      </AgentAdminPanel>

      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="closure gate" title="Closure gate checks" icon={ListChecks}>
            <div className="agent-admin-release-closure-gate">
              {closure.closure_gate.checks.length > 0 ? (
                closure.closure_gate.checks.map((check) => {
                  const rowState =
                    check.status === "pass" ? "success" : check.status === "warning" ? "active" : "warning";
                  return (
                    <article key={check.name} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state={rowState} />
                        {check.name}
                      </span>
                      <span>{check.reason}</span>
                      <strong>{check.status}</strong>
                    </article>
                  );
                })
              ) : (
                <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state="ready" />
                    no closure gate checks
                  </span>
                  <span>closure gate has no recorded checks yet</span>
                  <strong>idle</strong>
                </article>
              )}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <AgentAdminPanel eyebrow="gate reasons" title="Closure gate reasons" icon={AlertTriangle}>
            <div className="agent-admin-detail-list">
              {(closure.closure_gate.reasons.length > 0 ? closure.closure_gate.reasons : ["closure gate clear"]).map((reason) => (
                <span key={reason}>{reason}</span>
              ))}
            </div>
          </AgentAdminPanel>
        }
      />

      <AgentAdminSplitWorkbench
        primary={<ReleaseClosureForms closure={closure} canSubmit={canSubmit} actions={actions} />}
        secondary={
          <AgentAdminPanel eyebrow="latest closure" title="Latest closure and evidence" icon={FileText}>
            <div className="agent-admin-release-closure-package">
              {latestClosure ? (
                <>
                  <span>latest closure / {latestClosure.closure_status} / {latestClosure.closed_by}</span>
                  <span>closed at / {latestClosure.closed_at}</span>
                  <span>rationale / {latestClosure.rationale}</span>
                  <span>evidence package / {latestClosure.evidence_package_id}</span>
                  <span>idempotency key / {latestClosure.idempotency_key}</span>
                </>
              ) : (
                <span>no closure recorded</span>
              )}
              {latestEvidencePackage ? (
                <>
                  <span>package id / {latestEvidencePackage.package_id}</span>
                  <span>generated by / {latestEvidencePackage.generated_by}</span>
                  <span>generated at / {latestEvidencePackage.generated_at}</span>
                  <span>{latestEvidencePackage.summary}</span>
                  {latestEvidencePackage.source_refs.map((ref) => (
                    <span key={ref}>source ref / {ref}</span>
                  ))}
                  {latestEvidencePackage.artifact_refs.map((ref) => (
                    <span key={ref}>artifact ref / {ref}</span>
                  ))}
                  {Object.entries(latestEvidencePackage.snapshot_hashes).map(([name, hash]) => (
                    <span key={name}>{name} / {hash}</span>
                  ))}
                </>
              ) : (
                <span>no evidence package generated</span>
              )}
            </div>
          </AgentAdminPanel>
        }
      />
    </>
  );
}

function ReleaseClosureForms({
  closure,
  canSubmit,
  actions,
}: {
  closure: AdminReleaseClosureResponse;
  canSubmit: boolean;
  actions: AgentAdminReleaseClosureActions;
}) {
  const [closureActor, setClosureActor] = useState("release_manager");
  const [closureRationale, setClosureRationale] = useState("");
  const [closureIdempotencyKey, setClosureIdempotencyKey] = useState("");
  const latestRelease = closure.latest_release;
  const rolledBackRelease = latestRelease?.rollback_execution_id !== null && latestRelease?.rollback_execution_id !== undefined;
  const allowedStatuses = releaseClosureAllowedStatuses(closure);
  const defaultClosureStatus: AdminReleaseClosureRecordStatus = rolledBackRelease
    ? "rolled_back"
    : allowedStatuses.includes("accepted")
      ? "accepted"
      : allowedStatuses[0] ?? "accepted";
  const [closureStatus, setClosureStatus] = useState<AdminReleaseClosureRecordStatus>(defaultClosureStatus);
  const effectiveClosureStatus = allowedStatuses.includes(closureStatus) ? closureStatus : defaultClosureStatus;

  useEffect(() => {
    setClosureStatus(defaultClosureStatus);
  }, [defaultClosureStatus, latestRelease?.release_execution_id]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit || !latestRelease || !allowedStatuses.includes(effectiveClosureStatus)) {
      return;
    }

    const request: AdminRecordReleaseClosureRequest = {
      intent_id: latestRelease.intent_id,
      release_execution_id: latestRelease.release_execution_id,
      closure_status: rolledBackRelease ? "rolled_back" : effectiveClosureStatus,
      closed_by: closureActor.trim(),
      rationale: closureRationale.trim(),
      idempotency_key: closureIdempotencyKey.trim(),
    };
    await actions.recordReleaseClosure(request);
  }

  return (
    <AgentAdminPanel eyebrow="closure writes" title="Closure actions" icon={KeyRound}>
      <form className="agent-admin-governance-form" onSubmit={handleSubmit}>
        <h3>Record closure</h3>
        <label htmlFor="release-closure-actor">
          <span>Closure actor</span>
          <input
            id="release-closure-actor"
            value={closureActor}
            onChange={(event) => setClosureActor(event.target.value)}
            required
          />
        </label>
        <label htmlFor="release-closure-status">
          <span>Closure status</span>
          <select
            id="release-closure-status"
            value={effectiveClosureStatus}
            onChange={(event) => setClosureStatus(event.target.value as AdminReleaseClosureRecordStatus)}
          >
            {releaseClosureStatuses.map((status) => (
              <option
                key={status}
                value={status}
                disabled={!allowedStatuses.includes(status)}
              >
                {status}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor="release-closure-rationale">
          <span>Closure rationale</span>
          <textarea
            id="release-closure-rationale"
            value={closureRationale}
            onChange={(event) => setClosureRationale(event.target.value)}
            required
          />
        </label>
        <label htmlFor="release-closure-idempotency-key">
          <span>Closure idempotency key</span>
          <input
            id="release-closure-idempotency-key"
            value={closureIdempotencyKey}
            onChange={(event) => setClosureIdempotencyKey(event.target.value)}
            required
          />
        </label>
        <button type="submit" disabled={!canSubmit}>Record closure</button>
      </form>
    </AgentAdminPanel>
  );
}

function ReleaseGovernanceSection({
  resource,
  actionState,
  actions,
  showDisabledExecutionControls = true,
}: {
  resource: AgentAdminReleaseGovernanceResource;
  actionState: AgentAdminReleaseGovernanceActionState;
  actions: AgentAdminReleaseGovernanceActions;
  showDisabledExecutionControls?: boolean;
}) {
  if (resource.status === "idle") {
    return null;
  }

  if (resource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="audit governance" title="Release governance" icon={ShieldCheck}>
        <div className="agent-admin-detail-list">
          <span>reading release governance</span>
          <span>audit-only state is loading from the admin API</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "error") {
    const status = resource.error.status ? ` (${resource.error.status})` : "";
    return (
      <AgentAdminPanel eyebrow="audit governance" title="Release governance" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release governance unavailable{status}: {resource.error.message}</span>
          <span>release execution remains disabled</span>
        </div>
      </AgentAdminPanel>
    );
  }

  const governance = resource.data;
  const activeIntent = governance.active_intent;
  const approvalsComplete = governance.required_approvals.filter((approval) => approval.status === "approved").length;
  const rollbackStatus = governance.rollback_plan?.status ?? "missing";
  const governanceMetrics = [
    {
      label: "Active intent",
      value: activeIntent?.derived_status ?? "none",
      detail: activeIntent?.intent_id ?? "no active intent",
      tone: activeIntent ? ("success" as const) : ("neutral" as const),
    },
    {
      label: "Approvals",
      value: `${approvalsComplete}/${governance.required_approvals.length}`,
      detail: "required roles",
      tone: approvalsComplete === governance.required_approvals.length ? ("success" as const) : ("warning" as const),
    },
    {
      label: "Rollback plan",
      value: rollbackStatus,
      detail: governance.rollback_plan?.rollback_target ?? "not recorded",
      tone: rollbackStatus === "accepted" ? ("success" as const) : ("warning" as const),
    },
    {
      label: "Audit integrity",
      value: governance.integrity.status,
      detail: `${governance.audit_events.length} events`,
      tone: releaseIntegrityState(governance.integrity.status),
    },
  ];

  return (
    <>
      <AgentAdminMetricStrip metrics={governanceMetrics} />
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="audit governance" title="Release governance" icon={ShieldCheck}>
              <div className="agent-admin-detail-list">
                <span>active intent / {activeIntent?.intent_id ?? "none"}</span>
                <span>requested by / {activeIntent?.requested_by ?? "not requested"}</span>
                <span>target scope / {activeIntent?.target_scope ?? "not selected"}</span>
                <span>source report / {activeIntent?.source_release_report_id ?? "not linked"}</span>
                <span>derived status / {activeIntent?.derived_status ?? "none"}</span>
              </div>
            </AgentAdminPanel>

            <AgentAdminPanel eyebrow="required approvals" title="Approval ledger" icon={ListChecks}>
              <div className="agent-admin-timeline">
                {governance.required_approvals.map((approval) => {
                  const rowState = releaseApprovalState(approval.status);
                  return (
                    <article key={approval.role} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
                      <span className="agent-admin-timeline-node">
                        <AgentAdminStateIcon state={rowState} />
                        {approval.role}
                      </span>
                      <span>{approval.signed_by ?? approval.latest_decision ?? "waiting for signed approval"}</span>
                      <strong>{approval.status}</strong>
                    </article>
                  );
                })}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <>
            <AgentAdminPanel eyebrow="rollback plan" title="Rollback guardrail" icon={Route}>
              <div className="agent-admin-detail-list">
                <span>target / {governance.rollback_plan?.rollback_target ?? activeIntent?.rollback_target ?? "not recorded"}</span>
                <span>owner / {governance.rollback_plan?.owner ?? "not assigned"}</span>
                <span>status / {governance.rollback_plan?.status ?? "missing"}</span>
                {(governance.rollback_plan?.verification_steps ?? ["verification steps not recorded"]).map((step) => (
                  <span key={step}>{step}</span>
                ))}
              </div>
            </AgentAdminPanel>

            <AgentAdminPanel eyebrow="integrity" title="Audit hash chain" icon={KeyRound}>
              <div className="agent-admin-detail-list">
                <span>status / {governance.integrity.status}</span>
                <span>warnings / {governance.integrity.warnings.length}</span>
                <span>runtime / {governance.runtime.auth} / {governance.runtime.source} / {governance.runtime.mode}</span>
              </div>
            </AgentAdminPanel>
          </>
        }
      />

      <ReleaseGovernanceForms governance={governance} actionState={actionState} actions={actions} />

      {showDisabledExecutionControls ? (
        <AgentAdminSplitWorkbench
          primary={<ReleaseGovernanceAuditTrail governance={governance} />}
          secondary={
            <AgentAdminPanel eyebrow="execution locked" title="Disabled execution controls" icon={KeyRound}>
              <div className="agent-admin-detail-list">
                {governance.disabled_execution_actions.map((action) => (
                  <button key={action.id} type="button" className="agent-admin-disabled-action" disabled>
                    <KeyRound size={15} aria-hidden="true" />
                    <span>{action.label}</span>
                    <small>{action.reason}</small>
                  </button>
                ))}
              </div>
            </AgentAdminPanel>
          }
        />
      ) : (
        <ReleaseGovernanceAuditTrail governance={governance} />
      )}
    </>
  );
}

function ReleaseGovernanceAuditTrail({ governance }: { governance: AdminReleaseGovernanceResponse }) {
  return (
    <AgentAdminPanel eyebrow="audit trail" title="Release governance audit trail" icon={FileText}>
      <div className="agent-admin-timeline">
        {governance.audit_events.length > 0 ? (
          governance.audit_events.map((event) => (
            <article key={event.event_id} className="agent-admin-timeline-row agent-admin-timeline-row-success">
              <span className="agent-admin-timeline-node">
                <AgentAdminStateIcon state="success" />
                {event.event_id}
              </span>
              <span>{event.event_type} / {event.actor}</span>
              <strong>{event.event_hash}</strong>
            </article>
          ))
        ) : (
          <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
            <span className="agent-admin-timeline-node">
              <AgentAdminStateIcon state="ready" />
              no audit events
            </span>
            <span>create intent to start the release governance chain</span>
            <strong>idle</strong>
          </article>
        )}
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseExecutionSection({
  resource,
  actionState,
  actions,
}: {
  resource: AgentAdminReleaseExecutionResource;
  actionState: AgentAdminReleaseExecutionActionState;
  actions: AgentAdminReleaseExecutionActions;
}) {
  if (resource.status === "idle") {
    return null;
  }

  if (resource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="controlled execution" title="Release execution" icon={KeyRound}>
        <div className="agent-admin-detail-list">
          <span>reading release execution</span>
          <span>controlled local execution state is loading from reports/release_execution</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (resource.status === "error") {
    const status = resource.error.status ? ` (${resource.error.status})` : "";
    return (
      <AgentAdminPanel eyebrow="controlled execution" title="Release execution" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release execution unavailable{status}: {resource.error.message}</span>
          <span>feature flag state remains unchanged</span>
        </div>
      </AgentAdminPanel>
    );
  }

  return (
    <ReleaseExecutionPanel
      execution={resource.data}
      actionState={actionState}
      actions={actions}
    />
  );
}

function ReleaseExecutionPanel({
  execution,
  actionState,
  actions,
}: {
  execution: AdminReleaseExecutionResponse;
  actionState: AgentAdminReleaseExecutionActionState;
  actions: AgentAdminReleaseExecutionActions;
}) {
  const [actor, setActor] = useState("release_manager");
  const [reason, setReason] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState("");
  const [expectedRollbackPlan, setExpectedRollbackPlan] = useState(execution.governance.rollback_plan_id ?? "");
  const intentId = execution.governance.active_intent_id ?? "";
  const flagState = execution.feature_flag_state;
  const actionRunning = actionState.status === "running";
  const releaseBlocked = !execution.preflight.release.allowed;
  const rollbackBlocked = !execution.preflight.rollback.allowed;
  const formIncomplete = !intentId || !actor.trim() || !reason.trim() || !idempotencyKey.trim() || !expectedRollbackPlan.trim();

  async function submitExecution(action: "release" | "rollback") {
    if (formIncomplete) {
      return;
    }

    const request: AdminExecuteReleaseRequest = {
      intent_id: intentId,
      requested_by: actor.trim(),
      idempotency_key: idempotencyKey.trim(),
      reason: reason.trim(),
      expected_rollback_plan_id: expectedRollbackPlan.trim(),
    };

    if (action === "release") {
      await actions.executeRelease(request);
      return;
    }

    await actions.executeRollback(request);
  }

  function handleNoopSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
  }

  return (
    <AgentAdminPanel eyebrow="controlled execution" title="Release execution" icon={KeyRound}>
      {actionState.status === "running" ? (
        <span className="agent-admin-action-status">{actionState.label} in progress</span>
      ) : null}
      {actionState.status === "error" ? (
        <span className="agent-admin-action-status agent-admin-action-status-error">{actionState.message}</span>
      ) : null}

      <AgentAdminSplitWorkbench
        primary={
          <div className="agent-admin-detail-list">
            <span>active intent / {execution.governance.active_intent_id ?? "none"}</span>
            <span>derived status / {execution.governance.derived_status ?? "none"}</span>
            <span>approvals complete / {String(execution.governance.required_approvals_complete)}</span>
            <span>rollback plan / {execution.governance.rollback_plan_id ?? "none"}</span>
            <span>release preflight / {execution.preflight.release.allowed ? "allowed" : "blocked"}</span>
            {(execution.preflight.release.reasons.length > 0 ? execution.preflight.release.reasons : ["release preflight clear"]).map((reasonItem) => (
              <span key={`release-${reasonItem}`}>{reasonItem}</span>
            ))}
            <span>rollback preflight / {execution.preflight.rollback.allowed ? "allowed" : "blocked"}</span>
            {(execution.preflight.rollback.reasons.length > 0 ? execution.preflight.rollback.reasons : ["rollback preflight clear"]).map((reasonItem) => (
              <span key={`rollback-${reasonItem}`}>{reasonItem}</span>
            ))}
          </div>
        }
        secondary={
          <div className="agent-admin-detail-list">
            <span>flag / {flagState?.flag_name ?? "not set"}</span>
            <span>enabled / {flagState ? String(flagState.enabled) : "not set"}</span>
            <span>intent / {flagState?.source_intent_id ?? "not set"}</span>
            <span>execution / {flagState?.source_execution_id ?? "not set"}</span>
            <span>updated / {flagState?.updated_at ?? "not set"}</span>
            <span>runtime / {execution.runtime.auth} / {execution.runtime.source} / {execution.runtime.mode}</span>
          </div>
        }
      />

      <form className="agent-admin-governance-form" onSubmit={handleNoopSubmit}>
        <h3>Execute controlled release</h3>
        <label htmlFor="release-execution-actor">
          <span>Execution actor</span>
          <input
            id="release-execution-actor"
            value={actor}
            onChange={(event) => setActor(event.target.value)}
            required
          />
        </label>
        <label htmlFor="release-execution-reason">
          <span>Execution reason</span>
          <textarea
            id="release-execution-reason"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            required
          />
        </label>
        <label htmlFor="release-execution-idempotency-key">
          <span>Idempotency key</span>
          <input
            id="release-execution-idempotency-key"
            value={idempotencyKey}
            onChange={(event) => setIdempotencyKey(event.target.value)}
            required
          />
        </label>
        <label htmlFor="release-execution-rollback-plan">
          <span>Expected rollback plan</span>
          <input
            id="release-execution-rollback-plan"
            value={expectedRollbackPlan}
            onChange={(event) => setExpectedRollbackPlan(event.target.value)}
            required
          />
        </label>
        <div className="agent-admin-execution-action-row">
          <button
            type="button"
            disabled={releaseBlocked || formIncomplete || actionRunning}
            onClick={() => void submitExecution("release")}
          >
            Execute release
          </button>
          <button
            type="button"
            disabled={rollbackBlocked || formIncomplete || actionRunning}
            onClick={() => void submitExecution("rollback")}
          >
            Execute rollback
          </button>
        </div>
      </form>

      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="execution results" title="Release execution results" icon={ListChecks}>
            <div className="agent-admin-timeline">
              {execution.results.length > 0 ? (
                execution.results.map((result) => (
                  <article key={result.result_id} className={`agent-admin-timeline-row agent-admin-timeline-row-${result.status === "succeeded" ? "success" : "warning"}`}>
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state={result.status === "succeeded" ? "success" : "warning"} />
                      {result.result_id}
                    </span>
                    <span>{result.action} / {result.status}</span>
                    <strong>{result.finished_at}</strong>
                  </article>
                ))
              ) : (
                <article className="agent-admin-timeline-row agent-admin-timeline-row-ready">
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state="ready" />
                    no execution results
                  </span>
                  <span>release or rollback has not run</span>
                  <strong>idle</strong>
                </article>
              )}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <AgentAdminPanel eyebrow="execution audit" title="Release execution audit trail" icon={FileText}>
            <div className="agent-admin-detail-list">
              {execution.audit_events.length > 0 ? (
                execution.audit_events.map((event) => (
                  <span key={event.event_id}>{event.event_type} / {event.actor} / {event.event_hash}</span>
                ))
              ) : (
                <span>no execution audit events</span>
              )}
            </div>
          </AgentAdminPanel>
        }
      />
    </AgentAdminPanel>
  );
}

function ReleaseGovernanceForms({
  governance,
  actionState,
  actions,
}: {
  governance: AdminReleaseGovernanceResponse;
  actionState: AgentAdminReleaseGovernanceActionState;
  actions: AgentAdminReleaseGovernanceActions;
}) {
  const activeIntentId = governance.active_intent?.intent_id;
  const [createRequestedBy, setCreateRequestedBy] = useState("release_admin");
  const [createReason, setCreateReason] = useState("");
  const [createTargetScope, setCreateTargetScope] = useState<AdminReleaseTargetScope>("shadow");
  const [createStatus, setCreateStatus] = useState<AdminReleaseCreateIntentStatus>("pending_approval");
  const [approvalRole, setApprovalRole] = useState<AdminReleaseApproverRole>("release_manager");
  const [approvalDecision, setApprovalDecision] = useState<AdminReleaseApprovalDecision>("approve");
  const [approvalReason, setApprovalReason] = useState("");
  const [approvalSignedBy, setApprovalSignedBy] = useState("release_manager");
  const [rollbackOwner, setRollbackOwner] = useState("release_manager");
  const [rollbackStatus, setRollbackStatus] = useState<AdminReleaseRollbackPlanStatus>("proposed");
  const [rollbackSteps, setRollbackSteps] = useState("");
  const [cancelActor, setCancelActor] = useState("release_manager");
  const [cancelReason, setCancelReason] = useState("");
  const actionRunning = actionState.status === "running";
  const activeIntentMissing = !activeIntentId;

  async function handleCreateIntent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await actions.createIntent({
      requested_by: createRequestedBy.trim(),
      target_scope: createTargetScope,
      status: createStatus,
      reason: createReason.trim(),
    });
  }

  async function handleRecordApproval(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeIntentId) {
      return;
    }
    await actions.recordApproval(activeIntentId, {
      approver_role: approvalRole,
      decision: approvalDecision,
      reason: approvalReason.trim(),
      signed_by: approvalSignedBy.trim(),
    });
  }

  async function handleRecordRollbackPlan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeIntentId) {
      return;
    }
    await actions.recordRollbackPlan(activeIntentId, {
      owner: rollbackOwner.trim(),
      status: rollbackStatus,
      verification_steps: splitReleaseVerificationSteps(rollbackSteps),
    });
  }

  async function handleCancelIntent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeIntentId) {
      return;
    }
    await actions.cancelIntent(activeIntentId, {
      actor: cancelActor.trim(),
      reason: cancelReason.trim(),
    });
  }

  return (
    <AgentAdminPanel eyebrow="audit writes" title="Release governance actions" icon={KeyRound}>
      {actionState.status === "running" ? (
        <span className="agent-admin-action-status">{actionState.label} in progress</span>
      ) : null}
      {actionState.status === "error" ? (
        <span className="agent-admin-action-status agent-admin-action-status-error">{actionState.message}</span>
      ) : null}

      <div className="agent-admin-governance-form-grid">
        <form className="agent-admin-governance-form" onSubmit={handleCreateIntent}>
          <h3>Create intent</h3>
          <label htmlFor="release-create-requested-by">
            <span>Requested by</span>
            <input
              id="release-create-requested-by"
              value={createRequestedBy}
              onChange={(event) => setCreateRequestedBy(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-create-reason">
            <span>Intent reason</span>
            <textarea
              id="release-create-reason"
              value={createReason}
              onChange={(event) => setCreateReason(event.target.value)}
              required
            />
          </label>
          <label htmlFor="release-create-target-scope">
            <span>Target scope</span>
            <select
              id="release-create-target-scope"
              value={createTargetScope}
              onChange={(event) => setCreateTargetScope(event.target.value as AdminReleaseTargetScope)}
            >
              <option value="shadow">shadow</option>
              <option value="feature_flag_candidate">feature_flag_candidate</option>
            </select>
          </label>
          <label htmlFor="release-create-status">
            <span>Intent status</span>
            <select
              id="release-create-status"
              value={createStatus}
              onChange={(event) => setCreateStatus(event.target.value as AdminReleaseCreateIntentStatus)}
            >
              <option value="pending_approval">pending_approval</option>
              <option value="draft">draft</option>
            </select>
          </label>
          <button type="submit" disabled={actionRunning}>Create intent</button>
        </form>

        <form className="agent-admin-governance-form" onSubmit={handleRecordApproval}>
          <h3>Record approval</h3>
          <label htmlFor="release-approval-role">
            <span>Approval role</span>
            <select
              id="release-approval-role"
              value={approvalRole}
              onChange={(event) => setApprovalRole(event.target.value as AdminReleaseApproverRole)}
              disabled={activeIntentMissing}
            >
              <option value="release_manager">release_manager</option>
              <option value="clinical_safety_reviewer">clinical_safety_reviewer</option>
              <option value="evidence_reviewer">evidence_reviewer</option>
            </select>
          </label>
          <label htmlFor="release-approval-decision">
            <span>Approval decision</span>
            <select
              id="release-approval-decision"
              value={approvalDecision}
              onChange={(event) => setApprovalDecision(event.target.value as AdminReleaseApprovalDecision)}
              disabled={activeIntentMissing}
            >
              <option value="approve">approve</option>
              <option value="reject">reject</option>
              <option value="request_changes">request_changes</option>
            </select>
          </label>
          <label htmlFor="release-approval-signed-by">
            <span>Signed by</span>
            <input
              id="release-approval-signed-by"
              value={approvalSignedBy}
              onChange={(event) => setApprovalSignedBy(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <label htmlFor="release-approval-reason">
            <span>Approval reason</span>
            <textarea
              id="release-approval-reason"
              value={approvalReason}
              onChange={(event) => setApprovalReason(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <button type="submit" disabled={activeIntentMissing || actionRunning}>Record approval</button>
        </form>

        <form className="agent-admin-governance-form" onSubmit={handleRecordRollbackPlan}>
          <h3>Record rollback plan</h3>
          <label htmlFor="release-rollback-owner">
            <span>Rollback owner</span>
            <input
              id="release-rollback-owner"
              value={rollbackOwner}
              onChange={(event) => setRollbackOwner(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <label htmlFor="release-rollback-status">
            <span>Rollback status</span>
            <select
              id="release-rollback-status"
              value={rollbackStatus}
              onChange={(event) => setRollbackStatus(event.target.value as AdminReleaseRollbackPlanStatus)}
              disabled={activeIntentMissing}
            >
              <option value="proposed">proposed</option>
              <option value="accepted">accepted</option>
            </select>
          </label>
          <label htmlFor="release-rollback-steps">
            <span>Verification steps</span>
            <textarea
              id="release-rollback-steps"
              value={rollbackSteps}
              onChange={(event) => setRollbackSteps(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <button type="submit" disabled={activeIntentMissing || actionRunning}>Record rollback plan</button>
        </form>

        <form className="agent-admin-governance-form" onSubmit={handleCancelIntent}>
          <h3>Cancel intent</h3>
          <label htmlFor="release-cancel-actor">
            <span>Cancel actor</span>
            <input
              id="release-cancel-actor"
              value={cancelActor}
              onChange={(event) => setCancelActor(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <label htmlFor="release-cancel-reason">
            <span>Cancel reason</span>
            <textarea
              id="release-cancel-reason"
              value={cancelReason}
              onChange={(event) => setCancelReason(event.target.value)}
              disabled={activeIntentMissing}
              required
            />
          </label>
          <button type="submit" disabled={activeIntentMissing || actionRunning}>Cancel intent</button>
        </form>
      </div>
    </AgentAdminPanel>
  );
}

function ReadOnlyPage({ activeScene, patient, doctor }: Pick<AgentAdminPagesProps, "activeScene" | "patient" | "doctor">) {
  const permissionRows = buildPermissionRows();
  const activeState = watchedSession(activeScene, patient, doctor);

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="status / auth" title="status/auth panel" icon={ShieldCheck}>
              <div className="agent-admin-detail-list">
                <span>active scene {activeScene}</span>
                <span>patient {patient.sessionId ?? "未创建"}</span>
                <span>doctor {doctor.sessionId ?? "未创建"}</span>
                <span>snapshot {formatSnapshot(activeState)}</span>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="permissions" title="权限矩阵" icon={ListChecks}>
              <div className="agent-admin-timeline">
                {permissionRows.map((row) => (
                  <article
                    key={row.label}
                    className={`agent-admin-timeline-row agent-admin-timeline-row-${row.state === "enabled" ? "success" : "ready"}`}
                  >
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state={row.state === "enabled" ? "success" : "disabled"} />
                      {row.label}
                    </span>
                    <span>{row.reason}</span>
                    <strong>{row.state}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="disabled controls" title="编辑规则" icon={KeyRound}>
            <div className="agent-admin-detail-list">
              {permissionRows
                .filter((row) => row.state === "disabled")
                .map((row) => (
                  <AgentAdminDisabledAction key={row.label} label={row.label} reason={`disabled / ${row.reason}`} />
                ))}
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="boundary" title="只读边界" icon={Route}>
        <div className="agent-admin-detail-list">
          <span>Graph scene 保持为 {activeScene}</span>
          <span>不创建第三种 graph scene</span>
          <span>admin console observes patient / doctor state only</span>
        </div>
      </AgentAdminPanel>
    </>
  );
}

function AgentAdminFallbackPage({
  activeScene,
  state,
}: {
  activeScene: Scene;
  state: SessionState;
}) {
  return (
    <AgentAdminPanel eyebrow="selected page" title="临时页面占位" icon={Route}>
      <div className="agent-admin-detail-list">
        <span>
          <Route size={15} aria-hidden="true" />
          Graph scene 保持为 {activeScene}
        </span>
        <span>
          <Clock3 size={15} aria-hidden="true" />
          Snapshot {formatSnapshot(state)}
        </span>
        <span>
          <ShieldCheck size={15} aria-hidden="true" />
          一期只读，不写入规则或工具状态
        </span>
      </div>
    </AgentAdminPanel>
  );
}
