import {
  Activity,
  BookOpenCheck,
  Brain,
  FileText,
  Gauge,
  GitBranch,
  KeyRound,
  ListChecks,
  ServerCog,
  Wrench,
  type LucideIcon,
} from "lucide-react";

import type { AdminToolItem, AdminToolManifestResponse, SessionState } from "../../app/api/types";

export type AgentAdminTaskId =
  | "overview"
  | "sessions"
  | "memory"
  | "rules"
  | "tools"
  | "learning"
  | "trace"
  | "evidence"
  | "release"
  | "read-only";

export type AgentAdminTask = {
  id: AgentAdminTaskId;
  label: string;
  detailTitle: string;
  description: string;
  status: string;
  responsibility: string;
  icon: LucideIcon;
};

export type AgentAdminMemoryState = "active" | "empty" | "unstructured" | "stale" | "failed" | "running" | "idle";

type AgentAdminMemoryTone = "red" | "neutral" | "warning" | "success";
type AgentAdminMemorySource = "患者" | "医生" | "患者 / 医生";
type AgentAdminMemorySourceKey =
  | "summary_memory"
  | "immutable_info"
  | "dynamic_info"
  | "anchor_events"
  | "contextMaintenance";

type AgentAdminMemoryLayer = {
  label: "摘要记忆" | "永久事实" | "动态事实" | "锚点事件" | "维护日志";
  sourceKey: AgentAdminMemorySourceKey;
  retentionReason: string;
  emptyContent: string;
};

export type AgentAdminMemoryLayerRow = AgentAdminMemoryLayer & {
  patientCount: number;
  doctorCount: number;
  state: AgentAdminMemoryState;
};

export type AgentAdminMemoryLifecycleRow = {
  stage: "收集" | "摘要" | "结构化" | "同步" | "过期检查";
  state: AgentAdminMemoryState;
  explanation: string;
  patient: string;
  doctor: string;
};

export type AgentAdminMemoryVisualizationRow = {
  content: string;
  type: AgentAdminMemoryLayer["label"];
  source: AgentAdminMemorySource;
  retentionReason: string;
  state: AgentAdminMemoryState;
};

export type AgentAdminMemoryAutomationMetric = {
  id: "summary" | "immutable" | "dynamic" | "anchor" | "maintenance";
  label: string;
  value: string;
  detail: string;
  tone: AgentAdminMemoryTone;
};

const MEMORY_LAYERS: AgentAdminMemoryLayer[] = [
  { label: "摘要记忆", sourceKey: "summary_memory", retentionReason: "压缩会话上下文", emptyContent: "暂无摘要记忆" },
  { label: "永久事实", sourceKey: "immutable_info", retentionReason: "稳定事实", emptyContent: "永久事实未结构化" },
  { label: "动态事实", sourceKey: "dynamic_info", retentionReason: "近期变化", emptyContent: "动态事实未结构化" },
  { label: "锚点事件", sourceKey: "anchor_events", retentionReason: "关键事件", emptyContent: "暂无锚点事件" },
  { label: "维护日志", sourceKey: "contextMaintenance", retentionReason: "自动维护状态", emptyContent: "维护日志为空" },
];

export const AGENT_ADMIN_TASKS: AgentAdminTask[] = [
  {
    id: "overview",
    label: "总览",
    detailTitle: "全局健康",
    description: "全局健康 / active sessions / last snapshot / runtime state",
    status: "live",
    responsibility: "健康、快照、运行态",
    icon: Gauge,
  },
  {
    id: "sessions",
    label: "会话",
    detailTitle: "会话观测",
    description: "patient / doctor watched sessions / stream status / abort state",
    status: "2",
    responsibility: "patient / doctor sessions",
    icon: ServerCog,
  },
  {
    id: "memory",
    label: "记忆",
    detailTitle: "上下文记忆",
    description: "summary memory / 永久事实 / dynamic facts / anchor events / context maintenance",
    status: "ctx",
    responsibility: "summary memory / 永久事实",
    icon: Brain,
  },
  {
    id: "rules",
    label: "规则",
    detailTitle: "永久上下文规则",
    description: "routing / safety / evaluation / memory rules",
    status: "5组",
    responsibility: "routing / safety / memory",
    icon: ListChecks,
  },
  {
    id: "tools",
    label: "工具",
    detailTitle: "工具可用性",
    description: "graph-level / executor / database / web search tools",
    status: "17",
    responsibility: "tool surfaces / availability",
    icon: Wrench,
  },
  {
    id: "learning",
    label: "学习",
    detailTitle: "自主学习准备",
    description: "daily paper readiness / latest research / scheduler config",
    status: "disabled",
    responsibility: "daily paper readiness",
    icon: BookOpenCheck,
  },
  {
    id: "trace",
    label: "Trace",
    detailTitle: "执行时间线",
    description: "node timings / trace events / latency",
    status: "trace",
    responsibility: "node timings / trace events / latency",
    icon: Activity,
  },
  {
    id: "evidence",
    label: "证据",
    detailTitle: "证据池",
    description: "references / RAG traces / retrieved evidence / source confidence",
    status: "rag",
    responsibility: "references / RAG trace",
    icon: FileText,
  },
  {
    id: "release",
    label: "Release",
    detailTitle: "Release Dashboard",
    description: "version chain / harness runs / rollback target / sign-off readiness",
    status: "read-only",
    responsibility: "version chain / harness runs",
    icon: GitBranch,
  },
  {
    id: "read-only",
    label: "设置只读",
    detailTitle: "只读边界",
    description: "admin token / feature flags / read-only boundary",
    status: "lock",
    responsibility: "auth / flags / no mutation",
    icon: KeyRound,
  },
];

export const ADMIN_NAV_ITEMS: Array<{ key: AgentAdminTaskId; label: string }> = [
  { key: "overview", label: "巡检总览" },
  { key: "trace", label: "运行链路" },
  { key: "learning", label: "每日论文准备" },
  { key: "release", label: "Release" },
  { key: "read-only", label: "只读边界" },
];

export const RULE_CATALOG = [
  { id: "routing.intent.knowledge_query", group: "路由规则", label: "知识查询意图路由", state: "enabled" },
  { id: "routing.intent.case_database", group: "路由规则", label: "病例数据库意图路由", state: "enabled" },
  { id: "safety.medical_disclaimer", group: "安全规则", label: "医疗建议安全边界", state: "enabled" },
  { id: "memory.summary_persistence", group: "记忆规则", label: "summary memory 持久化", state: "enabled" },
  { id: "evaluation.critic_required", group: "评估规则", label: "回答前 Critic 检查", state: "enabled" },
];

type AgentAdminToolRowSource = "runtime" | "fallback";

export type AgentAdminToolRow = {
  name: string;
  group: string;
  state: string;
  dependency: string;
  available: boolean;
  registries: string;
  routeTargets: string;
  graphScope: string;
  source: AgentAdminToolRowSource;
};

export type AgentAdminToolGroupRow = {
  name: string;
  count: number;
  status: string;
  availableCount?: number;
  source: AgentAdminToolRowSource;
};

export const FALLBACK_TOOL_INVENTORY = [
  { name: "search_latest_research", group: "Web search", state: "candidate" },
  { name: "query_case_database", group: "Database", state: "available" },
  { name: "get_patient_registry", group: "Database", state: "available" },
  { name: "tool_executor", group: "Executor", state: "available" },
  { name: "clinical_graph", group: "Graph-level", state: "available" },
];

export const TOOL_INVENTORY = FALLBACK_TOOL_INVENTORY;

const TOOL_GROUP_STATUS: Record<string, string> = {
  "Graph-level": "随 graph 构建注入",
  Executor: "tool_executor 可见",
  Database: "case database 节点",
  "Web search": "WEB_SEARCH_ENABLED 控制",
};

export const CATEGORY_LABELS: Record<AdminToolItem["category"], string> = {
  clinical: "Clinical",
  rag: "RAG",
  web: "Web search",
  database: "Database",
  imaging: "Imaging",
  pathology: "Pathology",
  tumor: "Tumor",
  utility: "Utility",
  formatting: "Formatting",
};

export function buildRuleCatalogRows() {
  return RULE_CATALOG.map((rule) => ({
    ...rule,
    editable: false,
    ownerModule: rule.id.split(".")[0],
  }));
}

export function buildRuleCatalogGroups() {
  const ruleRows = buildRuleCatalogRows();

  return buildRuleGroupRows().map((group) => ({
    ...group,
    rules: ruleRows.filter((rule) => rule.group === group.name),
  }));
}

export function dependencyForRuntimeTool(tool: AdminToolItem, manifest: AdminToolManifestResponse): string {
  if (tool.requires_web) {
    return manifest.runtime.web_search_enabled
      ? "WEB_SEARCH_ENABLED controls Web search reachability"
      : "WEB_SEARCH_ENABLED disabled for this runtime";
  }
  if (tool.registries.includes("executor")) {
    return "tool_executor required for executor dispatch";
  }
  if (tool.registries.includes("graph") || tool.registries.includes("graph_web")) {
    return "graph manifest registration required";
  }
  return "available";
}

export function buildToolInventoryRows(manifest?: AdminToolManifestResponse | null): AgentAdminToolRow[] {
  if (manifest) {
    return manifest.tools.map((tool) => ({
      name: tool.name,
      group: CATEGORY_LABELS[tool.category] ?? tool.category,
      state: tool.available ? tool.state : `${tool.state} / unavailable`,
      dependency: dependencyForRuntimeTool(tool, manifest),
      available: tool.available,
      registries: tool.registries.join(" / "),
      routeTargets: tool.route_targets.join(" / "),
      graphScope: tool.graph_scope,
      source: "runtime",
    }));
  }

  return TOOL_INVENTORY.map((tool) => ({
    ...tool,
    dependency:
      tool.group === "Web search"
        ? "WEB_SEARCH_ENABLED controls Web search reachability"
        : tool.group === "Executor"
          ? "tool_executor required for executor dispatch"
          : TOOL_GROUP_STATUS[tool.group] ?? "available",
    available: tool.state === "available",
    registries: "fallback inventory",
    routeTargets: "fallback inventory",
    graphScope: "fallback inventory",
    source: "fallback",
  }));
}

export function buildToolReachabilityRows(manifest?: AdminToolManifestResponse | null) {
  return buildToolGroupRows(manifest);
}

export function buildToolGroupRows(manifest?: AdminToolManifestResponse | null): AgentAdminToolGroupRow[] {
  if (manifest) {
    return manifest.groups.map((group) => {
      const name = CATEGORY_LABELS[group.category] ?? group.category;
      const runtimeStatus =
        group.category === "web"
          ? `WEB_SEARCH_ENABLED ${manifest.runtime.web_search_enabled ? "enabled" : "disabled"}`
          : `${group.available_count}/${group.count} available`;

      return {
        name,
        count: group.count,
        availableCount: group.available_count,
        status: runtimeStatus,
        source: "runtime",
      };
    });
  }

  return ["Graph-level", "Executor", "Database", "Web search"].map((name) => ({
    name,
    count: TOOL_INVENTORY.filter((tool) => tool.group === name).length,
    status: TOOL_GROUP_STATUS[name] ?? "available",
    source: "fallback",
  }));
}

export function buildRuleGroupRows() {
  const counts = RULE_CATALOG.reduce<Record<string, number>>((groups, rule) => {
    const namespace = rule.id.split(".")[0];
    const group =
      namespace === "routing"
        ? "路由规则"
        : namespace === "safety"
          ? "安全规则"
          : namespace === "evaluation"
            ? "评估规则"
            : namespace === "memory"
              ? "记忆规则"
              : rule.group;
    groups[group] = (groups[group] ?? 0) + 1;
    return groups;
  }, {});

  return Object.entries(counts).map(([name, count]) => ({ name, count }));
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function readText(value: unknown, fallback = "未提供"): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return fallback;
}

export function formatSnapshot(state: SessionState): string {
  return state.snapshotVersion > 0 ? `v${state.snapshotVersion}` : "v0";
}

export function sessionStatus(state: SessionState): "error" | "running" | "idle" | "missing" {
  if (state.lastError) {
    return "error";
  }
  if (state.activeRunId || state.statusNode) {
    return "running";
  }
  return state.sessionId ? "idle" : "missing";
}

export function buildSessionSummary(label: string, state: SessionState) {
  return {
    label,
    sessionId: state.sessionId ?? "未创建",
    snapshot: formatSnapshot(state),
    status: sessionStatus(state),
    activeRunId: state.activeRunId ?? "idle",
    currentNode: state.statusNode ?? "idle",
    contextMaintenance: state.contextMaintenance?.status ?? "idle",
  };
}

export function buildMemoryRows(label: string, state: SessionState) {
  const structured = asRecord(state.contextState?.structured_summary);
  const immutable = structured?.immutable_info;
  const dynamic = structured?.dynamic_info;
  const anchors = structured?.anchor_events;

  return [
    {
      label,
      type: "摘要记忆",
      value: readText(state.contextState?.summary_memory, "暂无 summary memory"),
    },
    {
      label,
      type: "永久事实",
      value: formatMemoryCount(immutable, "条永久事实", "永久事实未结构化"),
    },
    {
      label,
      type: "动态事实",
      value: formatMemoryCount(dynamic, "条动态事实", "动态事实未结构化"),
    },
    {
      label,
      type: "锚点事件",
      value: formatMemoryCount(anchors, "个锚点事件", "锚点事件未结构化"),
    },
    {
      label,
      type: "上下文维护",
      value: state.contextMaintenance?.message ?? state.contextMaintenance?.status ?? "idle",
    },
  ];
}

export function buildMemoryFactRows(label: string, state: SessionState) {
  const structured = asRecord(state.contextState?.structured_summary);
  const groups = [
    { type: "永久事实", values: structured?.immutable_info },
    { type: "动态事实", values: structured?.dynamic_info },
    { type: "锚点事件", values: structured?.anchor_events },
  ];

  return groups.flatMap((group) => {
    const values = normalizeMemoryFactValues(group.values);
    return values.length > 0
      ? values.map((value) => ({ label, type: group.type, value }))
      : [{ label, type: group.type, value: "暂无结构化事实" }];
  });
}

function formatMemoryCount(value: unknown, unitLabel: string, unavailableLabel: string): string {
  const count = normalizeMemoryFactValues(value).length;
  return count > 0 ? `${count} ${unitLabel}` : unavailableLabel;
}

function normalizeMemoryFactValues(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => formatMemoryFactValue(item)).filter((item): item is string => Boolean(item));
  }

  const record = asRecord(value);
  if (record) {
    return Object.entries(record)
      .map(([key, entryValue]) => {
        const formatted = formatMemoryFactValue(entryValue);
        return formatted ? `${key}: ${formatted}` : null;
      })
      .filter((item): item is string => Boolean(item));
  }

  const formatted = formatMemoryFactValue(value);
  return formatted ? [formatted] : [];
}

function formatMemoryFactValue(value: unknown): string | null {
  const text = readText(value, "");
  if (text) {
    return text;
  }

  if (typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    const values = value.map((item) => formatMemoryFactValue(item)).filter((item): item is string => Boolean(item));
    return values.length > 0 ? values.join(", ") : null;
  }

  const record = asRecord(value);
  if (record) {
    const summary = ["title", "summary", "event", "description", "text"]
      .map((field) => readText(record[field], ""))
      .find(Boolean);
    return summary ?? stringifyMemoryFact(record);
  }

  return null;
}

function stringifyMemoryFact(value: unknown): string | null {
  try {
    return JSON.stringify(value);
  } catch {
    return null;
  }
}

function structuredSummary(state: SessionState): Record<string, unknown> | null {
  return asRecord(state.contextState?.structured_summary);
}

function hasOwnRecordKey(record: Record<string, unknown> | null, key: string): boolean {
  return Boolean(record && Object.prototype.hasOwnProperty.call(record, key));
}

function hasContextStateKey(state: SessionState, key: string): boolean {
  return Boolean(state.contextState && Object.prototype.hasOwnProperty.call(state.contextState, key));
}

function hasStructuredSummaryValue(state: SessionState): boolean {
  return hasContextStateKey(state, "structured_summary") && state.contextState?.structured_summary != null;
}

function hasMalformedStructuredSummary(state: SessionState): boolean {
  return hasStructuredSummaryValue(state) && !structuredSummary(state);
}

function layerByKey(sourceKey: AgentAdminMemorySourceKey): AgentAdminMemoryLayer {
  return MEMORY_LAYERS.find((layer) => layer.sourceKey === sourceKey) ?? MEMORY_LAYERS[0];
}

function memoryRawValue(state: SessionState, layer: AgentAdminMemoryLayer): unknown {
  if (layer.sourceKey === "summary_memory") {
    return state.contextState?.summary_memory;
  }

  if (layer.sourceKey === "contextMaintenance") {
    return state.contextMaintenance?.error ?? state.contextMaintenance?.message ?? state.contextMaintenance?.status;
  }

  if (hasMalformedStructuredSummary(state)) {
    return state.contextState?.structured_summary;
  }

  return structuredSummary(state)?.[layer.sourceKey];
}

function memorySourcePresent(state: SessionState, layer: AgentAdminMemoryLayer): boolean {
  if (layer.sourceKey === "summary_memory") {
    return hasContextStateKey(state, "summary_memory");
  }

  if (layer.sourceKey === "contextMaintenance") {
    return Boolean(state.contextMaintenance);
  }

  if (hasMalformedStructuredSummary(state)) {
    return true;
  }

  return hasOwnRecordKey(structuredSummary(state), layer.sourceKey);
}

function hasRawMemoryContent(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.length > 0;
  }

  const record = asRecord(value);
  if (record) {
    return Object.keys(record).length > 0;
  }

  if (typeof value === "string") {
    return Boolean(value.trim());
  }

  return typeof value === "number" || typeof value === "boolean";
}

function memoryValuesForLayer(state: SessionState, layer: AgentAdminMemoryLayer): string[] {
  if (layer.sourceKey === "summary_memory") {
    const summary = readText(state.contextState?.summary_memory, "");
    return summary ? [summary] : [];
  }

  if (layer.sourceKey === "contextMaintenance") {
    const message = readText(
      state.contextMaintenance?.error ?? state.contextMaintenance?.message ?? state.contextMaintenance?.status,
      "",
    );
    return message ? [message] : [];
  }

  if (hasMalformedStructuredSummary(state)) {
    return [];
  }

  return normalizeMemoryFactValues(structuredSummary(state)?.[layer.sourceKey]);
}

function countLayerValues(state: SessionState, layer: AgentAdminMemoryLayer): number {
  return memoryValuesForLayer(state, layer).length;
}

function maintenanceState(state: SessionState): AgentAdminMemoryState {
  if (state.contextMaintenance?.status === "failed") {
    return "failed";
  }
  if (state.contextMaintenance?.status === "running") {
    return "running";
  }
  if (state.contextMaintenance?.status === "completed") {
    return "active";
  }
  return "idle";
}

function isSummaryStale(state: SessionState): boolean {
  const cursor = state.contextState?.summary_memory_cursor;
  return Boolean(readText(state.contextState?.summary_memory, "")) && typeof cursor === "number" && cursor < state.snapshotVersion;
}

function memoryStateForLayer(state: SessionState, layer: AgentAdminMemoryLayer): AgentAdminMemoryState {
  if (layer.sourceKey === "contextMaintenance") {
    return maintenanceState(state);
  }

  if (layer.sourceKey === "summary_memory" && isSummaryStale(state)) {
    return "stale";
  }

  const values = memoryValuesForLayer(state, layer);
  if (values.length > 0) {
    return "active";
  }

  if (!memorySourcePresent(state, layer)) {
    return "empty";
  }

  return hasRawMemoryContent(memoryRawValue(state, layer)) ? "unstructured" : "empty";
}

function combineMemoryState(
  patientState: AgentAdminMemoryState,
  doctorState: AgentAdminMemoryState,
): AgentAdminMemoryState {
  const rank: Record<AgentAdminMemoryState, number> = {
    failed: 7,
    stale: 6,
    running: 5,
    unstructured: 4,
    active: 3,
    empty: 2,
    idle: 1,
  };

  return rank[patientState] >= rank[doctorState] ? patientState : doctorState;
}

function stateTone(state: AgentAdminMemoryState): AgentAdminMemoryTone {
  if (state === "failed" || state === "stale" || state === "running" || state === "unstructured") {
    return "warning";
  }
  if (state === "active") {
    return "red";
  }
  return "neutral";
}

function summaryStatus(patient: SessionState, doctor: SessionState): { value: string; state: AgentAdminMemoryState } {
  const states = [patient, doctor];
  const hasSummary = states.some((state) => readText(state.contextState?.summary_memory, ""));
  const stale = states.some((state) => isSummaryStale(state));

  if (stale) {
    return { value: "待刷新", state: "stale" };
  }
  if (hasSummary) {
    return { value: "已生成", state: "active" };
  }
  return { value: "待生成", state: "empty" };
}

function structuredMemoryState(state: SessionState): AgentAdminMemoryState {
  const structured = structuredSummary(state);
  if (!structured && !hasMalformedStructuredSummary(state)) {
    return "empty";
  }

  return combineMemoryState(
    memoryStateForLayer(state, layerByKey("immutable_info")),
    combineMemoryState(
      memoryStateForLayer(state, layerByKey("dynamic_info")),
      memoryStateForLayer(state, layerByKey("anchor_events")),
    ),
  );
}

function sessionCollectionState(state: SessionState): AgentAdminMemoryState {
  return state.sessionId || state.messages.length > 0 || state.snapshotVersion > 0 ? "active" : "empty";
}

function syncState(state: SessionState): AgentAdminMemoryState {
  if (isSummaryStale(state)) {
    return "stale";
  }
  return state.sessionId || state.snapshotVersion > 0 ? "active" : "idle";
}

function collectionLabel(state: SessionState): string {
  return sessionCollectionState(state) === "active" ? `${state.messages.length} messages` : "会话未创建";
}

function syncLabel(state: SessionState): string {
  if (isSummaryStale(state)) {
    return "stale";
  }
  return state.sessionId || state.snapshotVersion > 0 ? `snapshot ${formatSnapshot(state)}` : "会话未创建";
}

function summaryLabel(state: SessionState): string {
  if (readText(state.contextState?.summary_memory, "")) {
    return isSummaryStale(state) ? "summary_memory stale" : "summary_memory ready";
  }
  return "暂无摘要记忆";
}

function structuredLabel(state: SessionState): string {
  const structuredState = structuredMemoryState(state);
  if (structuredState === "active") {
    return "structured_summary ready";
  }
  if (structuredState === "unstructured") {
    return "结构化字段不可读";
  }
  return "结构化摘要未生成";
}

function rowContentForEmptyLayer(layer: AgentAdminMemoryLayer, state: AgentAdminMemoryState): string {
  return state === "unstructured" ? "结构化字段不可读" : layer.emptyContent;
}

function combineSources(current: AgentAdminMemorySource, next: AgentAdminMemorySource): AgentAdminMemorySource {
  return current === next ? current : "患者 / 医生";
}

function mergeVisualizationRows(rows: AgentAdminMemoryVisualizationRow[]): AgentAdminMemoryVisualizationRow[] {
  const merged: AgentAdminMemoryVisualizationRow[] = [];

  for (const row of rows) {
    const existing = merged.find(
      (item) =>
        item.content === row.content &&
        item.type === row.type &&
        item.retentionReason === row.retentionReason &&
        item.state === row.state,
    );
    if (existing) {
      existing.source = combineSources(existing.source, row.source);
    } else {
      merged.push({ ...row });
    }
  }

  return merged;
}

export function buildMemoryAutomationSummary(
  patient: SessionState,
  doctor: SessionState,
): AgentAdminMemoryAutomationMetric[] {
  const summary = summaryStatus(patient, doctor);
  const immutable = layerByKey("immutable_info");
  const dynamic = layerByKey("dynamic_info");
  const anchors = layerByKey("anchor_events");
  const maintenance = combineMemoryState(maintenanceState(patient), maintenanceState(doctor));
  const immutablePatient = countLayerValues(patient, immutable);
  const immutableDoctor = countLayerValues(doctor, immutable);
  const dynamicPatient = countLayerValues(patient, dynamic);
  const dynamicDoctor = countLayerValues(doctor, dynamic);
  const anchorPatient = countLayerValues(patient, anchors);
  const anchorDoctor = countLayerValues(doctor, anchors);

  return [
    { id: "summary", label: "摘要状态", value: summary.value, detail: "summary_memory", tone: stateTone(summary.state) },
    {
      id: "immutable",
      label: "永久事实",
      value: String(immutablePatient + immutableDoctor),
      detail: `患者 ${immutablePatient} / 医生 ${immutableDoctor}`,
      tone: stateTone(combineMemoryState(memoryStateForLayer(patient, immutable), memoryStateForLayer(doctor, immutable))),
    },
    {
      id: "dynamic",
      label: "动态事实",
      value: String(dynamicPatient + dynamicDoctor),
      detail: `患者 ${dynamicPatient} / 医生 ${dynamicDoctor}`,
      tone: stateTone(combineMemoryState(memoryStateForLayer(patient, dynamic), memoryStateForLayer(doctor, dynamic))),
    },
    {
      id: "anchor",
      label: "锚点事件",
      value: String(anchorPatient + anchorDoctor),
      detail: `患者 ${anchorPatient} / 医生 ${anchorDoctor}`,
      tone: stateTone(combineMemoryState(memoryStateForLayer(patient, anchors), memoryStateForLayer(doctor, anchors))),
    },
    {
      id: "maintenance",
      label: "维护状态",
      value: maintenance,
      detail: `${patient.contextMaintenance?.status ?? "idle"} / ${doctor.contextMaintenance?.status ?? "idle"}`,
      tone: stateTone(maintenance),
    },
  ];
}

export function buildMemoryLayerRows(patient: SessionState, doctor: SessionState): AgentAdminMemoryLayerRow[] {
  return MEMORY_LAYERS.map((layer) => ({
    ...layer,
    patientCount: countLayerValues(patient, layer),
    doctorCount: countLayerValues(doctor, layer),
    state: combineMemoryState(memoryStateForLayer(patient, layer), memoryStateForLayer(doctor, layer)),
  }));
}

export function buildMemoryLifecycleRows(patient: SessionState, doctor: SessionState): AgentAdminMemoryLifecycleRow[] {
  const summaryLayer = layerByKey("summary_memory");
  const maintenance = combineMemoryState(maintenanceState(patient), maintenanceState(doctor));

  return [
    {
      stage: "收集",
      state: combineMemoryState(sessionCollectionState(patient), sessionCollectionState(doctor)),
      explanation: "读取 messages、sessionId、snapshotVersion",
      patient: collectionLabel(patient),
      doctor: collectionLabel(doctor),
    },
    {
      stage: "摘要",
      state: combineMemoryState(memoryStateForLayer(patient, summaryLayer), memoryStateForLayer(doctor, summaryLayer)),
      explanation: "检查 contextState.summary_memory 与 summary_memory_cursor",
      patient: summaryLabel(patient),
      doctor: summaryLabel(doctor),
    },
    {
      stage: "结构化",
      state: combineMemoryState(structuredMemoryState(patient), structuredMemoryState(doctor)),
      explanation: "读取 immutable_info、dynamic_info、anchor_events",
      patient: structuredLabel(patient),
      doctor: structuredLabel(doctor),
    },
    {
      stage: "同步",
      state: combineMemoryState(syncState(patient), syncState(doctor)),
      explanation: "比较 snapshotVersion 与 summary_memory_cursor",
      patient: syncLabel(patient),
      doctor: syncLabel(doctor),
    },
    {
      stage: "过期检查",
      state: maintenance,
      explanation: "读取 contextMaintenance.status、message、error",
      patient: patient.contextMaintenance?.status ?? "idle",
      doctor: doctor.contextMaintenance?.status ?? "idle",
    },
  ];
}

export function buildMemoryVisualizationRows(
  patient: SessionState,
  doctor: SessionState,
): AgentAdminMemoryVisualizationRow[] {
  const rows: AgentAdminMemoryVisualizationRow[] = [];
  const sources: Array<{ source: AgentAdminMemorySource; state: SessionState }> = [
    { source: "患者", state: patient },
    { source: "医生", state: doctor },
  ];

  for (const session of sources) {
    for (const layer of MEMORY_LAYERS) {
      const values = memoryValuesForLayer(session.state, layer);
      const state = memoryStateForLayer(session.state, layer);

      if (values.length > 0) {
        rows.push(
          ...values.map((content) => ({
            content,
            type: layer.label,
            source: session.source,
            retentionReason: layer.retentionReason,
            state,
          })),
        );
      } else {
        rows.push({
          content: rowContentForEmptyLayer(layer, state),
          type: layer.label,
          source: session.source,
          retentionReason: layer.retentionReason,
          state,
        });
      }
    }
  }

  return mergeVisualizationRows(rows);
}

export function buildEvidenceRows(state: SessionState): Array<{ title: string; source: string; confidence: string }> {
  const rows = state.references.slice(0, 4).map((reference) => {
    const record = asRecord(reference) ?? {};
    const confidence =
      typeof record.confidence === "number"
        ? `${Math.round(record.confidence * 100)}%`
        : readText(record.score, "待评估");

    return {
      title: readText(record.title ?? record.name ?? record.url, "未命名证据"),
      source: readText(record.source ?? record.type, "RAG"),
      confidence,
    };
  });

  return rows.length > 0
    ? rows
    : [
        { title: "暂无引用，等待 references.append", source: "SSE", confidence: "待生成" },
        { title: "RAG trace 尚未暴露到前端", source: "Phase 2", confidence: "未启用" },
      ];
}

export type AgentAdminTraceRow = {
  name: string;
  detail: string;
  latency: string | null;
  state: "success" | "active" | "ready" | "warning" | "error";
  source: "eventLog" | "runTrace" | "empty";
};

export function buildLiveTraceRows(state: SessionState): AgentAdminTraceRow[] {
  const log = state.eventLog ?? [];
  if (log.length === 0) {
    return [
      {
        name: "暂无执行事件",
        detail: "先在患者/医生端发起一轮对话；后台读取同一会话的 eventLog",
        latency: null,
        state: "ready",
        source: "empty",
      },
    ];
  }

  return log.map((entry) => ({
    name: entry.title,
    detail: entry.detail ?? entry.kind,
    latency: null,
    state:
      entry.tone === "error"
        ? "error"
        : entry.tone === "warning"
          ? "warning"
          : entry.tone === "success"
            ? "success"
            : entry.kind === "node" && state.statusNode === entry.title
              ? "active"
              : "ready",
    source: "eventLog",
  }));
}

export function buildTraceRows(state: SessionState): AgentAdminTraceRow[] {
  return buildLiveTraceRows(state);
}

export function buildLearningReadiness() {
  return [
    { label: "候选工具", value: "search_latest_research", state: "candidate" },
    { label: "调度器配置", value: "scheduler disabled / config needed", state: "disabled" },
    { label: "摄取队列", value: "ingestion queue empty", state: "ready" },
    { label: "论文来源", value: "PubMed / Crossref / arXiv", state: "planned" },
  ];
}

export function buildPermissionRows() {
  return [
    { label: "查看会话", state: "enabled", reason: "read-only observation" },
    { label: "查看证据", state: "enabled", reason: "references are frontend snapshots" },
    { label: "编辑规则", state: "disabled", reason: "Phase 1 不写入规则" },
    { label: "启停工具", state: "disabled", reason: "Phase 1 不写入工具状态" },
    { label: "运行学习任务", state: "disabled", reason: "scheduler disabled / config needed" },
  ];
}
