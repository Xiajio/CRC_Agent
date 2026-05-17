import type { ReactNode } from "react";

import { classNames } from "../ui";

export type ClinicalEmptyStateIcon =
  | "cards"
  | "chat"
  | "events"
  | "plan"
  | "references"
  | "roadmap"
  | "summary"
  | "uploads";

type ClinicalEmptyStateProps = {
  actionLabel?: string;
  className?: string;
  compact?: boolean;
  icon?: ClinicalEmptyStateIcon;
  message: ReactNode;
  onAction?: () => void;
  title: ReactNode;
};

export function ClinicalEmptyState({
  actionLabel,
  className,
  compact = false,
  icon = "cards",
  message,
  onAction,
  title,
}: ClinicalEmptyStateProps) {
  return (
    <div
      className={classNames([
        "clinical-empty-state",
        compact && "clinical-empty-state-compact",
        className,
      ])}
      data-testid="clinical-empty-state"
    >
      <span
        className={`clinical-empty-state-icon clinical-empty-state-icon-${icon}`}
        data-testid="clinical-empty-state-icon"
        aria-hidden="true"
      />
      <div className="clinical-empty-state-copy">
        <strong className="clinical-empty-state-title">{title}</strong>
        <p className="clinical-empty-state-message">{message}</p>
      </div>
      {actionLabel && onAction ? (
        <button type="button" className="clinical-empty-state-action" onClick={onAction}>
          {actionLabel}
        </button>
      ) : null}
    </div>
  );
}
