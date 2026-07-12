import type { MouseEvent, ReactNode } from "react";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  CircleDashed,
  type LucideIcon,
} from "lucide-react";

import { classNames } from "../../components/ui";

type AgentAdminTone = "red" | "neutral" | "warning" | "success";

export type AgentAdminDataSource = "live" | "runtime-api" | "catalog" | "unavailable" | "roadmap";

export function AgentAdminSourceBadge({ source }: { source: AgentAdminDataSource }) {
  const label =
    source === "live"
      ? "live session"
      : source === "runtime-api"
        ? "runtime API"
        : source === "catalog"
          ? "static catalog"
          : source === "roadmap"
            ? "roadmap / not wired"
            : "unavailable";

  return (
    <span className={`agent-admin-source-badge agent-admin-source-badge-${source}`} data-source={source}>
      {label}
    </span>
  );
}

type AgentAdminMetric = {
  id?: string;
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: AgentAdminTone;
};

export function AgentAdminMetricStrip({
  metrics,
  className,
}: {
  metrics: AgentAdminMetric[];
  className?: string;
}) {
  return (
    <div className={classNames(["agent-admin-metrics-grid", className])}>
      {metrics.map((metric) => (
        <article
          key={metric.id ?? metric.label}
          className={classNames([
            "agent-admin-metric",
            metric.tone ? `agent-admin-metric-${metric.tone}` : "agent-admin-metric-neutral",
          ])}
        >
          <span>{metric.label}</span>
          <strong>{metric.value}</strong>
          {metric.detail ? <small>{metric.detail}</small> : null}
        </article>
      ))}
    </div>
  );
}

export function AgentAdminStatusChip({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: AgentAdminTone;
  className?: string;
}) {
  return (
    <span className={classNames(["agent-admin-status-chip", `agent-admin-status-chip-${tone}`, className])}>
      <AgentAdminStateIcon state={tone} />
      {children}
    </span>
  );
}

export function AgentAdminPanel({
  eyebrow,
  title,
  icon: Icon,
  action,
  children,
  className,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  icon?: LucideIcon;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={classNames(["agent-admin-panel", className])}>
      <div className="agent-admin-panel-heading">
        <div>
          {eyebrow ? <span>{eyebrow}</span> : null}
          <h2>{title}</h2>
        </div>
        {action ?? (Icon ? <Icon size={20} aria-hidden="true" /> : null)}
      </div>
      {children}
    </section>
  );
}

export function AgentAdminSplitWorkbench({
  primary,
  secondary,
  className,
}: {
  primary: ReactNode;
  secondary: ReactNode;
  className?: string;
}) {
  return (
    <div className={classNames(["agent-admin-split-workbench", className])}>
      <div className="agent-admin-split-primary">{primary}</div>
      <div className="agent-admin-split-secondary">{secondary}</div>
    </div>
  );
}

export function AgentAdminDisabledAction({
  label,
  reason,
  onClick,
  className,
}: {
  label: ReactNode;
  reason: ReactNode;
  onClick?: (event: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}) {
  function handleClick(event: MouseEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.stopPropagation();
    void onClick;
  }

  return (
    <button
      type="button"
      className={classNames(["agent-admin-disabled-action", className])}
      aria-disabled="true"
      onClick={handleClick}
    >
      <Ban size={15} aria-hidden="true" />
      <span>{label}</span>
      <small>{reason}</small>
    </button>
  );
}

export function AgentAdminStateIcon({
  state,
  className,
}: {
  state: AgentAdminTone | "disabled" | "idle" | "active" | "ready" | "error";
  className?: string;
}) {
  const Icon =
    state === "success" || state === "active"
      ? CheckCircle2
      : state === "warning" || state === "red" || state === "error"
        ? AlertTriangle
        : state === "disabled"
          ? Ban
          : CircleDashed;

  return (
    <Icon
      size={15}
      aria-hidden="true"
      className={classNames(["agent-admin-state-icon", `agent-admin-state-icon-${state}`, className])}
    />
  );
}
