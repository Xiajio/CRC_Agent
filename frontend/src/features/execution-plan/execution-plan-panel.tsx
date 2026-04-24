import type { JsonObject } from "../../app/api/types";

type ExecutionPlanPanelProps = {
  plan: JsonObject[];
  references: JsonObject[];
};

const PLAN_TITLE_LABELS: Record<string, string> = {
  "collect context": "汇总患者上下文",
  "retrieve guidelines": "检索指南",
  "query case database": "查询病例库",
  "generate clinical assessment": "生成临床评估",
  "generate treatment recommendation": "生成治疗建议",
  "run ct summary": "生成 CT 摘要",
  "check citations": "核对引用",
  "finalize report": "完成报告",
};

function planLabel(step: JsonObject, index: number): string {
  const raw = typeof step.title === "string"
    ? step.title
    : typeof step.step === "string"
      ? step.step
      : typeof step.description === "string"
        ? step.description
        : null;
  if (!raw) {
    return `计划步骤 ${index + 1}`;
  }
  return PLAN_TITLE_LABELS[raw.toLowerCase()] ?? raw;
}

function normalizedStatus(status: unknown): string {
  if (typeof status !== "string") {
    return "pending";
  }
  if (status === "done") {
    return "completed";
  }
  return status;
}

function planStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "待处理",
    waiting: "待处理",
    in_progress: "进行中",
    completed: "已完成",
    blocked: "阻塞",
  };

  return labels[status] ?? status;
}

function referenceTitle(reference: JsonObject, index: number): string {
  return typeof reference.title === "string"
    ? reference.title
    : typeof reference.name === "string"
      ? reference.name
      : typeof reference.source === "string"
        ? reference.source
        : `参考资料 ${index + 1}`;
}

function referenceSubtitle(reference: JsonObject): string {
  return typeof reference.snippet === "string"
    ? reference.snippet
    : typeof reference.summary === "string"
      ? reference.summary
      : typeof reference.subtitle === "string"
        ? reference.subtitle
        : "本次运行关联的证据来源";
}

function referenceTag(reference: JsonObject): string {
  return typeof reference.tag === "string"
    ? reference.tag
    : typeof reference.source_type === "string"
      ? reference.source_type
      : typeof reference.source === "string"
        ? reference.source
        : "参考";
}

export function ExecutionPlanPanel({ plan, references }: ExecutionPlanPanelProps) {
  const visiblePlan = plan;
  const visibleReferences = references.slice(0, 2);

  return (
    <>
      <section className="clinical-card clinical-execution-card">
        <div className="clinical-panel-header">
          <span className="clinical-panel-icon clinical-list-icon" aria-hidden="true" />
          <h2>执行计划</h2>
        </div>
        {visiblePlan.length > 0 ? (
          <div className="clinical-plan-table" role="table" aria-label="执行计划">
            <div className="clinical-plan-head" role="row">
              <span>计划步骤</span>
              <span>状态</span>
            </div>
            {visiblePlan.map((step, index) => {
              const status = normalizedStatus(step.status);
              return (
                <div key={String(step.id ?? step.step_id ?? index)} className="clinical-plan-row" role="row">
                  <span className="clinical-plan-index">{index + 1}</span>
                  <span className="clinical-plan-title">{planLabel(step, index)}</span>
                  <span className={`clinical-plan-status clinical-plan-status-${status}`}>
                    {planStatusLabel(status)}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="clinical-empty-note">暂无执行计划。</p>
        )}
      </section>
      <section className="clinical-card clinical-reference-card">
        <div className="clinical-panel-header">
          <span className="clinical-panel-icon clinical-book-icon" aria-hidden="true" />
          <h2>参考列表（前 2 条）</h2>
        </div>
        {visibleReferences.length > 0 ? (
          <>
            <div className="clinical-reference-list">
              {visibleReferences.map((reference, index) => (
                <article key={String(reference.id ?? reference.url ?? referenceTitle(reference, index))} className="clinical-reference-item">
                  <span className="clinical-reference-number">{index + 1}</span>
                  <div>
                    <h3>{referenceTitle(reference, index)}</h3>
                    <p>{referenceSubtitle(reference)}</p>
                  </div>
                  <span className="clinical-reference-tag">{referenceTag(reference)}</span>
                  <span className="clinical-reference-open" aria-hidden="true" />
                </article>
              ))}
            </div>
            <button type="button" className="clinical-view-all-button">查看全部参考资料（{references.length}）</button>
          </>
        ) : (
          <p className="clinical-empty-note">暂无参考资料。</p>
        )}
      </section>
    </>
  );
}
