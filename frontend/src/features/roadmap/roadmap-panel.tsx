import { useRef } from "react";

import type { JsonObject } from "../../app/api/types";
import { useHighlightFlash } from "../../components/motion/use-highlight-flash";

type RoadmapPanelProps = {
  roadmap: JsonObject[];
  stage: string | null;
};

function roadmapLabel(step: JsonObject, index: number): string {
  const title = typeof step.title === "string" ? step.title : typeof step.step_name === "string" ? step.step_name : null;
  const id = typeof step.id === "string" ? step.id : typeof step.step_id === "string" ? step.step_id : null;
  return title ?? id ?? `第 ${index + 1} 步`;
}

function isCurrentStep(step: JsonObject, stage: string | null, index: number): boolean {
  if (!stage) {
    return index === 0;
  }

  const title = typeof step.title === "string" ? step.title : typeof step.step_name === "string" ? step.step_name : null;
  const id = typeof step.id === "string" ? step.id : typeof step.step_id === "string" ? step.step_id : null;
  const status = typeof step.status === "string" ? step.status : null;
  return stage === title || stage === id || stage === status;
}

function roadmapStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "待处理",
    in_progress: "进行中",
    completed: "已完成",
    blocked: "受阻",
    skipped: "已跳过",
  };

  return labels[status] ?? status;
}

function roadmapStatusStyle(status: string): React.CSSProperties {
  switch (status) {
    case "completed":
      return {
        background: "rgba(76, 175, 80, 0.1)",
        color: "#2e7d32",
        padding: "2px 8px",
        borderRadius: "12px",
        fontSize: "0.75rem",
        fontWeight: "bold",
      };
    case "pending":
    case "blocked":
      return {
        background: "rgba(244, 67, 54, 0.1)",
        color: "#c62828",
        padding: "2px 8px",
        borderRadius: "12px",
        fontSize: "0.75rem",
        fontWeight: "bold",
      };
    case "in_progress":
      return {
        background: "rgba(33, 150, 243, 0.1)",
        color: "#1565c0",
        padding: "2px 8px",
        borderRadius: "12px",
        fontSize: "0.75rem",
        fontWeight: "bold",
      };
    case "skipped":
      return {
        background: "rgba(158, 158, 158, 0.1)",
        color: "#616161",
        padding: "2px 8px",
        borderRadius: "12px",
        fontSize: "0.75rem",
        fontWeight: "bold",
      };
    default:
      return {
        fontSize: "0.75rem",
        color: "#91515a",
      };
  }
}

export function RoadmapPanel({ roadmap, stage }: RoadmapPanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);

  useHighlightFlash(panelRef, stage ?? roadmap.length);

  return (
    <div ref={panelRef} className="workspace-card">
      <h2>诊疗路径</h2>
      {roadmap.length > 0 ? (
        <ol className="workspace-step-list">
          {roadmap.map((step, index) => {
            const label = roadmapLabel(step, index);
            const current = isCurrentStep(step, stage, index);

            return (
              <li
                key={String(step.id ?? step.step_id ?? label)}
                aria-label={label}
                aria-current={current ? "step" : undefined}
                className={current ? "workspace-step workspace-step-current" : "workspace-step"}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {typeof step.icon === "string" ? <span>{step.icon}</span> : null}
                    {label}
                  </strong>
                  {typeof step.status === "string" ? (
                    <span style={{ ...roadmapStatusStyle(step.status), flexShrink: 0 }}>
                      {roadmapStatusLabel(step.status)}
                    </span>
                  ) : null}
                </div>
                {typeof step.details === "string" && step.details ? (
                  <p className="workspace-copy workspace-copy-tight" style={{ fontSize: '0.85rem', marginTop: '6px' }}>
                    {step.details}
                  </p>
                ) : null}
                {typeof step.reason === "string" && step.reason ? (
                  <p className="workspace-copy workspace-copy-tight" style={{ fontSize: '0.75rem', marginTop: '4px', opacity: 0.7 }}>
                    {step.reason}
                  </p>
                ) : null}
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="workspace-copy">当前暂无诊疗路径信息</p>
      )}
    </div>
  );
}
