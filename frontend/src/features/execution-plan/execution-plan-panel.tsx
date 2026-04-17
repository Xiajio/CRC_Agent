import type { JsonObject } from "../../app/api/types";

type ExecutionPlanPanelProps = {
  plan: JsonObject[];
  references: JsonObject[];
};

function planLabel(step: JsonObject, index: number): string {
  return typeof step.title === "string" ? step.title : `第 ${index + 1} 项计划`;
}

function planStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "待处理",
    in_progress: "进行中",
    completed: "已完成",
    blocked: "受阻",
    done: "已完成",
  };

  return labels[status] ?? status;
}

export function ExecutionPlanPanel({ plan, references }: ExecutionPlanPanelProps) {
  return (
    <div className="workspace-card">
      <h2>执行计划</h2>
      <p className="workspace-copy">
        当前跟踪 {plan.length} 项计划，关联 {references.length} 条参考依据。
      </p>
      {plan.length > 0 ? (
        <ul className="workspace-list">
          {plan.map((step, index) => (
            <li key={String(step.id ?? index)} className="workspace-list-item">
              <strong>{planLabel(step, index)}</strong>
              <div className="workspace-meta">
                {typeof step.owner === "string" ? <span>负责人：{step.owner}</span> : null}
                {typeof step.status === "string" ? <span>状态：{planStatusLabel(step.status)}</span> : null}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="workspace-copy">当前暂无执行计划</p>
      )}
    </div>
  );
}
