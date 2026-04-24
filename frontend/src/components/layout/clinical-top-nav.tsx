import type { ReactNode } from "react";

export type ClinicalNavItem = {
  key: string;
  label: string;
  disabled?: boolean;
};

export function ClinicalNodeLogo() {
  return (
    <svg className="clinical-logo-mark" viewBox="0 0 40 40" aria-hidden="true">
      <line x1="12" y1="10" x2="28" y2="8" />
      <line x1="12" y1="10" x2="9" y2="26" />
      <line x1="28" y1="8" x2="31" y2="25" />
      <line x1="9" y1="26" x2="22" y2="32" />
      <line x1="31" y1="25" x2="22" y2="32" />
      <line x1="12" y1="10" x2="31" y2="25" />
      <circle cx="12" cy="10" r="4" />
      <circle cx="28" cy="8" r="4" />
      <circle cx="9" cy="26" r="4" />
      <circle cx="31" cy="25" r="4" />
      <circle cx="22" cy="32" r="4" />
    </svg>
  );
}

export function ClinicalUserIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 22c1.8-4.8 4.5-7.2 8-7.2s6.2 2.4 8 7.2" />
    </svg>
  );
}

type ClinicalTopNavProps = {
  brandLabel: string;
  navLabel: string;
  items: ClinicalNavItem[];
  activeKey: string;
  onSelect: (key: string) => void;
  actions?: ReactNode;
  actionsLabel?: string;
  statusLabel: string;
  statusTone: "connected" | "safe";
  profileLabel: string;
  profileAriaLabel: string;
  onProfileClick?: () => void;
  className?: string;
};

export function ClinicalTopNav({
  brandLabel,
  navLabel,
  items,
  activeKey,
  onSelect,
  actions,
  actionsLabel = "场景操作",
  statusLabel,
  statusTone,
  profileLabel,
  profileAriaLabel,
  onProfileClick,
  className,
}: ClinicalTopNavProps) {
  const toneClass = statusTone === "safe" ? " clinical-safe-pill" : "";

  return (
    <header className={`clinical-top-nav${className ? ` ${className}` : ""}`} data-testid="workspace-toolbar">
      <div className="clinical-brand-block">
        <ClinicalNodeLogo />
        <span>{brandLabel}</span>
      </div>
      <nav className="clinical-nav-tabs" aria-label={navLabel}>
        {items.map((item) => {
          const isActive = item.key === activeKey;
          const isDisabled = Boolean(item.disabled);

          return (
            <button
              key={item.key}
              type="button"
              className={isActive ? "clinical-nav-tab clinical-nav-tab-active" : "clinical-nav-tab"}
              aria-current={isActive ? "page" : undefined}
              aria-disabled={isDisabled ? "true" : undefined}
              aria-pressed={isActive}
              disabled={isDisabled}
              onClick={() => {
                if (!isDisabled) {
                  onSelect(item.key);
                }
              }}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      {actions ? (
        <div className="clinical-scene-switcher" aria-label={actionsLabel}>
          {actions}
        </div>
      ) : null}
      <div className="clinical-user-area">
        <span className={`clinical-sse-pill${toneClass}`}>
          <span />
          {statusLabel}
        </span>
        <span className="clinical-top-divider" />
        <button
          type="button"
          className="clinical-profile-switch"
          aria-label={profileAriaLabel}
          onClick={onProfileClick}
        >
          <span className="clinical-avatar">
            <ClinicalUserIcon />
          </span>
          <span className="clinical-doctor-name">{profileLabel}</span>
          <span className="clinical-chevron" aria-hidden="true">
            v
          </span>
        </button>
      </div>
    </header>
  );
}
