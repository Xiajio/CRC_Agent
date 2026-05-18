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

const emptyStateIconPaths: Record<ClinicalEmptyStateIcon, ReactNode> = {
  cards: (
    <>
      <rect x="6" y="7" width="12" height="10" rx="2.5" />
      <path d="M9 11h6" />
    </>
  ),
  chat: (
    <>
      <path d="M5 8.5A4.5 4.5 0 0 1 9.5 4h5A4.5 4.5 0 0 1 19 8.5v2A4.5 4.5 0 0 1 14.5 15H11l-4 3v-3.7A4.5 4.5 0 0 1 5 10.5z" />
    </>
  ),
  events: (
    <>
      <circle cx="7" cy="7" r="2" />
      <circle cx="17" cy="8" r="2" />
      <circle cx="10" cy="17" r="2" />
      <path d="M8.8 7.5 15.2 8M8 8.8l1.3 6.2" />
    </>
  ),
  plan: <path d="M7 7h10M7 12h10M7 17h6" />,
  references: (
    <>
      <path d="M8 5h8a2 2 0 0 1 2 2v12H8a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" />
      <path d="M9 9h6M9 13h5" />
    </>
  ),
  roadmap: (
    <>
      <circle cx="6.5" cy="7" r="2" />
      <circle cx="17.5" cy="17" r="2" />
      <path d="M8.5 7h4a3 3 0 0 1 0 6h-1a3 3 0 0 0 0 6h4" />
    </>
  ),
  summary: (
    <>
      <circle cx="12" cy="7.5" r="3" />
      <path d="M6.5 19c1.2-3.4 3-5 5.5-5s4.3 1.6 5.5 5" />
    </>
  ),
  uploads: (
    <>
      <path d="M12 17V6" />
      <path d="m8.5 9.5 3.5-3.5 3.5 3.5" />
      <path d="M6 18.5h12" />
    </>
  ),
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
      >
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          {emptyStateIconPaths[icon]}
        </svg>
      </span>
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
