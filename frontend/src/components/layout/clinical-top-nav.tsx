import type { ReactNode } from "react";

import yizhuCompanyLogo from "../../assets/brand/yizhu-company-logo-dark.webp";
import yizhuCompanyLogoLight from "../../assets/brand/yizhu-company-logo-light.webp";
import { TopNav, type TopNavItem } from "../ui";

export type ClinicalNavItem = TopNavItem;
type BrandLogoVariant = "dark" | "light";

export function CompanyBrandLogo({ variant = "dark" }: { variant?: BrandLogoVariant }) {
  const logoSrc = variant === "light" ? yizhuCompanyLogoLight : yizhuCompanyLogo;

  return (
    <img
      className="clinical-company-logo"
      src={logoSrc}
      alt="亿铸科技公司标识"
    />
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
  brandLogoVariant?: BrandLogoVariant;
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
  brandLogoVariant = "dark",
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
  return (
    <TopNav
      brandLabel={brandLabel}
      brandIcon={<CompanyBrandLogo variant={brandLogoVariant} />}
      navLabel={navLabel}
      items={items}
      activeKey={activeKey}
      onSelect={onSelect}
      actions={actions}
      actionsLabel={actionsLabel}
      statusLabel={statusLabel}
      statusTone={statusTone}
      profileLabel={profileLabel}
      profileAriaLabel={profileAriaLabel}
      profileIcon={<ClinicalUserIcon />}
      onProfileClick={onProfileClick}
      className={["clinical-top-nav", className].filter(Boolean).join(" ")}
    />
  );
}
