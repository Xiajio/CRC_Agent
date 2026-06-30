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
  AdminReleaseDashboardResponse,
  AdminReleaseGateState,
  AdminReleaseHumanSignoffStatus,
  AdminReleaseRunStatus,
  Scene,
  SessionState,
} from "../../app/api/types";
import {
  AgentAdminMetricStrip,
  AgentAdminPanel,
  AgentAdminDisabledAction,
  AgentAdminSplitWorkbench,
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
  buildTraceRows,
  formatSnapshot,
  readText,
  sessionStatus,
  type AgentAdminTaskId,
} from "./agent-admin-model";
import type { AgentAdminReleaseDashboardResource, AgentAdminToolsResource } from "./agent-admin-view";

type AgentAdminPagesProps = {
  activeTaskId: AgentAdminTaskId;
  activeScene: Scene;
  patient: SessionState;
  doctor: SessionState;
  onNavigateTask: (taskId: AgentAdminTaskId) => void;
  toolsResource: AgentAdminToolsResource;
  releaseDashboardResource: AgentAdminReleaseDashboardResource;
};

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
  releaseDashboardResource,
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
        />
      ) : activeTaskId === "sessions" ? (
        <SessionsPage patient={patient} doctor={doctor} />
      ) : activeTaskId === "memory" ? (
        <MemoryPage patient={patient} doctor={doctor} />
      ) : activeTaskId === "rules" ? (
        <RulesPage />
      ) : activeTaskId === "tools" ? (
        <ToolsPage toolsResource={toolsResource} />
      ) : activeTaskId === "learning" ? (
        <LearningPage />
      ) : activeTaskId === "trace" ? (
        <TracePage activeScene={activeScene} patient={patient} doctor={doctor} />
      ) : activeTaskId === "evidence" ? (
        <EvidencePage activeScene={activeScene} patient={patient} doctor={doctor} />
      ) : activeTaskId === "release" ? (
        <ReleasePage releaseDashboardResource={releaseDashboardResource} />
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
}: Omit<AgentAdminPagesProps, "activeTaskId" | "toolsResource" | "releaseDashboardResource">) {
  const watchedState = watchedSession(activeScene, patient, doctor);
  const patientSession = buildSessionSummary("患者", patient);
  const doctorSession = buildSessionSummary("医生", doctor);
  const ruleGroups = buildRuleGroupRows();
  const toolRows = buildToolInventoryRows();
  const references = buildEvidenceRows(watchedState);
  const activePlan = watchedState.plan.slice(0, 4);
  const status = sessionStatus(watchedState);
  const metrics = [
    { label: "活跃会话", value: `${patient.sessionId ? 1 : 0}/${doctor.sessionId ? 1 : 0}`, tone: "red" as const },
    { label: "患者快照", value: patientSession.snapshot, detail: patientSession.sessionId, tone: "neutral" as const },
    { label: "医生快照", value: doctorSession.snapshot, detail: doctorSession.sessionId, tone: "neutral" as const },
    { label: "当前状态", value: status, detail: watchedState.activeRunId ?? "idle", tone: status === "error" ? "warning" as const : "success" as const },
    { label: "可用工具", value: String(toolRows.length), tone: "red" as const },
    { label: "规则组", value: String(ruleGroups.length), tone: "neutral" as const },
  ];
  const timelineSteps = [
    { node: "用户请求", state: "success" as const, detail: "80ms" },
    { node: "Planner", state: watchedState.statusNode === "planner" ? "active" as const : "success" as const, detail: "420ms" },
    { node: "Knowledge Retrieval", state: "success" as const, detail: "680ms" },
    { node: "Tool Executor", state: watchedState.statusNode === "tool_executor" ? "active" as const : "ready" as const, detail: "540ms" },
    { node: "Critic", state: watchedState.critic ? "success" as const : "ready" as const, detail: "220ms" },
    { node: "Response", state: watchedState.messages.length > 0 ? "success" as const : "ready" as const, detail: "160ms" },
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
          <AgentAdminPanel eyebrow="run health" title="运行健康时间线" icon={Clock3}>
            <div className="agent-admin-timeline">
              {timelineSteps.map((step) => (
                <article key={step.node} className={`agent-admin-timeline-row agent-admin-timeline-row-${step.state}`}>
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state={step.state} />
                    {step.node}
                  </span>
                  <span className="agent-admin-latency-bar">
                    <i style={{ width: step.state === "ready" ? "32%" : "76%" }} />
                  </span>
                  <strong>{step.detail}</strong>
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
    `patient ${patientSession.sessionId} / ${patientSession.snapshot} / ${patientSession.activeRunId}`,
    `doctor ${doctorSession.sessionId} / ${doctorSession.snapshot} / ${doctorSession.activeRunId}`,
    `maintenance ${patientSession.contextMaintenance} -> ${doctorSession.contextMaintenance}`,
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
          {recentEvents.map((event) => (
            <span key={event}>{event}</span>
          ))}
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

function RulesPage() {
  const rulesByGroup = buildRuleCatalogGroups();
  const inspectedRule = buildRuleCatalogRows()[0];

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="group summary" title="规则分组" icon={ListChecks}>
              <div className="agent-admin-timeline">
                {rulesByGroup.map((group) => (
                  <article key={group.name} className="agent-admin-timeline-row agent-admin-timeline-row-success">
                    <span className="agent-admin-timeline-node">
                      <AgentAdminStateIcon state="success" />
                      {group.name}
                    </span>
                    <span>catalog entries</span>
                    <strong>{group.count}</strong>
                  </article>
                ))}
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="catalog tree" title="规则目录" icon={GitBranch}>
              <div className="agent-admin-detail-list">
                {rulesByGroup.map((group) => (
                  <span key={group.name}>
                    <strong>{group.name}</strong> / {group.rules.map((rule) => rule.id).join(" / ")}
                  </span>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <AgentAdminPanel eyebrow="rule inspector" title="owner module" icon={ServerCog}>
            <div className="agent-admin-detail-list">
              <span>{inspectedRule.id}</span>
              <span>{inspectedRule.label}</span>
              <span>group: {inspectedRule.group}</span>
              <span>state: {inspectedRule.state}</span>
              <span>editable: {String(inspectedRule.editable)}</span>
              <span>owner module: {inspectedRule.ownerModule}</span>
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
  const sourceStatus =
    toolsResource.status === "success"
      ? "runtime manifest"
      : toolsResource.status === "loading"
        ? "reading runtime manifest"
        : toolsResource.status === "error"
          ? `runtime manifest unavailable${toolsResource.error.status ? ` (${toolsResource.error.status})` : ""}: ${toolsResource.error.message}`
          : "fallback inventory";
  const fallbackStatus = toolsResource.status === "error" ? "fallback inventory" : null;
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
            <AgentAdminPanel eyebrow="filters" title="工具筛选" icon={GitBranch}>
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
  const traceEvents = buildTraceRows(state);

  return (
    <>
      <AgentAdminSplitWorkbench
        primary={
          <AgentAdminPanel eyebrow="timeline events" title="执行事件时间线" icon={Clock3}>
            <div className="agent-admin-timeline">
              {traceEvents.map((event) => (
                <article key={event.name} className={`agent-admin-timeline-row agent-admin-timeline-row-${event.state}`}>
                  <span className="agent-admin-timeline-node">
                    <AgentAdminStateIcon state={event.state} />
                    {event.name}
                  </span>
                  <span>{event.detail}</span>
                  <strong>{event.latency}</strong>
                </article>
              ))}
            </div>
          </AgentAdminPanel>
        }
        secondary={
          <AgentAdminPanel eyebrow="latency panel" title="latency panel" icon={Activity}>
            <div className="agent-admin-detail-list">
              <span>active run {state.activeRunId ?? "idle"}</span>
              <span>status node {state.statusNode ?? "idle"}</span>
              <span>scene {activeScene}</span>
              <span>snapshot {formatSnapshot(state)}</span>
            </div>
          </AgentAdminPanel>
        }
      />
      <AgentAdminPanel eyebrow="event table" title="event table" icon={ListChecks}>
        <div className="agent-admin-timeline">
          {traceEvents.map((event) => (
            <article key={`${event.name}-table`} className="agent-admin-timeline-row agent-admin-timeline-row-ready">
              <span>{event.name}</span>
              <span>{event.detail}</span>
              <strong>{event.latency}</strong>
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
}: {
  releaseDashboardResource: AgentAdminReleaseDashboardResource;
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
    return <ReleaseSuccessPage dashboard={releaseDashboardResource.data} />;
  }

  return (
    <AgentAdminPanel eyebrow="release artifacts" title="Release Dashboard" icon={GitBranch}>
      <div className="agent-admin-detail-list">
        <span>select Release to read committed release artifacts</span>
      </div>
    </AgentAdminPanel>
  );
}

function ReleaseSuccessPage({ dashboard }: { dashboard: AdminReleaseDashboardResponse }) {
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
                    <article key={run.run_id} className={`agent-admin-timeline-row agent-admin-timeline-row-${rowState}`}>
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
    </>
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
