import type { HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export type TopNavItem = {
  key: string;
  label: string;
  disabled?: boolean;
};

export interface TopNavProps extends Omit<HTMLAttributes<HTMLElement>, "onSelect"> {
  actions?: ReactNode;
  actionsLabel?: string;
  activeKey: string;
  brandLabel: string;
  brandIcon?: ReactNode;
  className?: string;
  items: TopNavItem[];
  navLabel: string;
  onProfileClick?: () => void;
  onSelect: (key: string) => void;
  profileAriaLabel: string;
  profileIcon?: ReactNode;
  profileLabel: string;
  statusLabel?: string;
  statusTone?: "connected" | "safe";
}

export function TopNav({
  actions,
  actionsLabel = "Actions",
  activeKey,
  brandLabel,
  brandIcon,
  className,
  items,
  navLabel,
  onProfileClick,
  onSelect,
  profileAriaLabel,
  profileIcon,
  profileLabel,
  statusLabel,
  statusTone = "connected",
  ...props
}: TopNavProps) {
  return (
    <header
      {...props}
      className={["ui-top-nav", className].filter(Boolean).join(" ")}
      data-testid="workspace-toolbar"
    >
      <div className="ui-top-nav-brand clinical-brand-block">
        {brandIcon}
        <span>{brandLabel}</span>
      </div>
      <nav className="ui-top-nav-tabs clinical-nav-tabs" aria-label={navLabel}>
        {items.map((item) => {
          const isActive = item.key === activeKey;
          const isDisabled = Boolean(item.disabled);

          return (
            <button
              key={item.key}
              type="button"
              className={classNames([
                "ui-top-nav-tab",
                "clinical-nav-tab",
                isActive && "ui-top-nav-tab-active",
                isActive && "clinical-nav-tab-active",
              ])}
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
        <div className="ui-top-nav-actions clinical-scene-switcher" aria-label={actionsLabel}>
          {actions}
        </div>
      ) : null}
      <div className="ui-top-nav-user clinical-user-area">
        {statusLabel ? (
          <span
            className={classNames([
              "ui-status-pill",
              "clinical-sse-pill",
              statusTone === "safe" && "ui-status-pill-safe",
              statusTone === "safe" && "clinical-safe-pill",
            ])}
          >
            <span />
            {statusLabel}
          </span>
        ) : null}
        {statusLabel ? <span className="ui-top-nav-divider clinical-top-divider" /> : null}
        <button
          type="button"
          className="ui-profile-switch clinical-profile-switch"
          aria-label={profileAriaLabel}
          onClick={onProfileClick}
        >
          {profileIcon ? <span className="ui-profile-avatar clinical-avatar">{profileIcon}</span> : null}
          <span className="ui-profile-label clinical-doctor-name">{profileLabel}</span>
          <span className="ui-profile-chevron clinical-chevron" aria-hidden="true">
            v
          </span>
        </button>
      </div>
    </header>
  );
}
