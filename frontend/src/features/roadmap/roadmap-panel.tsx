import { useRef } from "react";

import type { JsonObject } from "../../app/api/types";
import { useHighlightFlash } from "../../components/motion/use-highlight-flash";

type RoadmapPanelProps = {
  roadmap: JsonObject[];
  stage: string | null;
};

const ROADMAP_TITLE_LABELS: Record<string, string> = {
  intent: "意图识别",
  planner: "规划器",
  "tool router": "工具路由",
  "tool-router": "工具路由",
  assessment: "临床评估",
  decision: "决策",
  citation: "引用",
  evaluator: "评估器",
  finalize: "完成",
};

function roadmapLabel(step: JsonObject, index: number): string {
  const title = typeof step.title === "string" ? step.title : typeof step.step_name === "string" ? step.step_name : null;
  const id = typeof step.id === "string" ? step.id : typeof step.step_id === "string" ? step.step_id : null;
  const raw = title ?? id;
  if (!raw) {
    return `步骤 ${index + 1}`;
  }
  return ROADMAP_TITLE_LABELS[raw.toLowerCase()] ?? raw;
}

function roadmapStatus(step: JsonObject): string {
  return typeof step.status === "string" ? step.status : "waiting";
}

function normalizedStatus(status: string): string {
  if (status === "done") {
    return "completed";
  }
  if (status === "pending") {
    return "waiting";
  }
  return status;
}

function isCurrentStep(step: JsonObject, stage: string | null, index: number): boolean {
  if (!stage) {
    return index === 0;
  }

  const normalizedStage = stage.toLowerCase();
  const title = typeof step.title === "string" ? step.title.toLowerCase() : null;
  const id = typeof step.id === "string" ? step.id.toLowerCase() : typeof step.step_id === "string" ? step.step_id.toLowerCase() : null;
  const status = typeof step.status === "string" ? step.status.toLowerCase() : null;
  return normalizedStage === title || normalizedStage === id || normalizedStage === status;
}

function roadmapStatusLabel(status: string): string {
  const normalized = normalizedStatus(status);
  const labels: Record<string, string> = {
    completed: "已完成",
    in_progress: "进行中",
    waiting: "等待中",
    blocked: "阻塞",
    skipped: "已跳过",
  };

  return labels[normalized] ?? status;
}

export function RoadmapPanel({ roadmap, stage }: RoadmapPanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const visibleRoadmap = roadmap;

  useHighlightFlash(panelRef, stage ?? visibleRoadmap.length);

  return (
    <section ref={panelRef} className="clinical-card clinical-roadmap-card">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon clinical-node-icon" aria-hidden="true" />
        <h2>工作流路线图</h2>
      </div>
      {visibleRoadmap.length > 0 ? (
        <ol className="clinical-roadmap-list">
          {visibleRoadmap.map((step, index) => {
            const label = roadmapLabel(step, index);
            const current = isCurrentStep(step, stage, index);
            const status = current && normalizedStatus(roadmapStatus(step)) === "waiting"
              ? "in_progress"
              : normalizedStatus(roadmapStatus(step));

            return (
              <li
                key={String(step.id ?? step.step_id ?? label)}
                className={`clinical-roadmap-step clinical-roadmap-step-${status}`}
                aria-current={current ? "step" : undefined}
              >
                <span className="clinical-roadmap-dot" aria-hidden="true" />
                <span className="clinical-roadmap-label">{label}</span>
                <span className="clinical-roadmap-status">{roadmapStatusLabel(status)}</span>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="clinical-empty-note">暂无工作流路线。</p>
      )}
    </section>
  );
}
