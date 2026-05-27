import type { ClinicalEventLogEntry, ClinicalEventLogKind } from "../api/types";

const EVENT_KIND_LABELS: Record<ClinicalEventLogKind, string> = {
  node: "运行节点",
  stage: "阶段进展",
  roadmap: "路线图更新",
  critic: "安全复核",
  plan: "执行计划",
  references: "证据引用",
  error: "异常",
  done: "本轮完成",
};

const NODE_LABELS: Record<string, string> = {
  assessment: "临床评估",
  citation: "证据引用",
  critic: "安全复核",
  database_query: "数据库查询",
  decision: "治疗决策",
  evaluator: "评估器",
  finalize: "完成输出",
  intent: "意图识别",
  intent_router: "意图识别",
  memory_manager: "记忆管理",
  outpatient_triage: "门诊分诊",
  planner: "规划器",
  tool_router: "工具路由",
  triage: "门诊分诊",
};

const STATIC_TITLE_LABELS: Record<string, string> = {
  "plan updated": "执行计划已更新",
  "references appended": "证据引用已追加",
  "roadmap updated": "路线图已更新",
  "stream completed": "本轮生成完成",
};

const CRITIC_VERDICT_LABELS: Record<string, string> = {
  APPROVED: "通过",
  APPROVED_WITH_WARNING: "通过，有警示",
  APPROVED_WITH_WARNINGS: "通过，有警示",
  NEEDS_REVIEW: "需复核",
  PENDING_REVIEW: "待复核",
  REJECTED: "未通过",
};

function normalizeKey(value: string): string {
  return value.trim().toLowerCase().replace(/[\s_-]+/g, " ");
}

function lookupLabel(labels: Record<string, string>, value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  return labels[trimmed] ?? labels[trimmed.toLowerCase()] ?? labels[normalizeKey(trimmed)] ?? null;
}

export function formatClinicalEventKind(kind: ClinicalEventLogKind): string {
  return EVENT_KIND_LABELS[kind] ?? kind;
}

export function formatClinicalNodeLabel(value: string): string {
  return lookupLabel(NODE_LABELS, value) ?? value;
}

export function formatClinicalCriticVerdict(verdict: unknown): string | null {
  if (typeof verdict !== "string") {
    return null;
  }

  const trimmed = verdict.trim();
  if (!trimmed) {
    return null;
  }

  return CRITIC_VERDICT_LABELS[trimmed.toUpperCase()] ?? trimmed;
}

function formatCriticEventTitle(title: string): string {
  const match = title.match(/^Critic\s+(.+)$/i);
  const verdictLabel = match ? formatClinicalCriticVerdict(match[1]) : null;
  return verdictLabel && verdictLabel !== match?.[1] ? `安全复核${verdictLabel}` : "安全复核";
}

export function formatClinicalEventTitle(event: Pick<ClinicalEventLogEntry, "kind" | "title">): string {
  const title = event.title.trim();
  if (!title) {
    return formatClinicalEventKind(event.kind);
  }

  if (event.kind === "critic") {
    return formatCriticEventTitle(title);
  }

  if (event.kind === "node" || event.kind === "stage") {
    return formatClinicalNodeLabel(title);
  }

  return lookupLabel(STATIC_TITLE_LABELS, title) ?? formatClinicalNodeLabel(title);
}

export function formatClinicalEventDetail(detail: string | null | undefined): string | null {
  const trimmed = detail?.trim();
  if (!trimmed) {
    return null;
  }

  const stepCount = trimmed.match(/^(\d+)\s+step\(s\)$/i);
  if (stepCount) {
    return `${stepCount[1]} 个步骤`;
  }

  const referenceCount = trimmed.match(/^(\d+)\s+reference\(s\)$/i);
  if (referenceCount) {
    return `${referenceCount[1]} 条引用`;
  }

  return trimmed;
}
