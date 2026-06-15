import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ClinicalTopNav, type ClinicalNavItem } from "./clinical-top-nav";

const navItems: ClinicalNavItem[] = [
  { key: "consultation", label: "Consultation" },
  { key: "database", label: "Patient Database" },
  { key: "multimodal", label: "Multimodal", disabled: true },
  { key: "reports", label: "Reports", disabled: true },
];

describe("ClinicalTopNav", () => {
  it("renders actions and profile controls", () => {
    const onSelect = vi.fn();
    const onProfileClick = vi.fn();

    render(
      <ClinicalTopNav
        brandLabel="LangGraph Clinical Assistant"
        navLabel="Clinical navigation"
        items={navItems}
        activeKey="consultation"
        onSelect={onSelect}
        actions={<button type="button">Reset Scene</button>}
        statusLabel="SSE Connected"
        statusTone="connected"
        profileLabel="Doctor"
        profileAriaLabel="患者场景"
        onProfileClick={onProfileClick}
      />,
    );

    expect(screen.getByText("LangGraph Clinical Assistant")).toBeInTheDocument();
    const companyLogo = screen.getByRole("img", { name: "亿铸科技公司标识" });
    expect(companyLogo).toBeInTheDocument();
    expect(companyLogo).toHaveAttribute("src", expect.stringContaining("yizhu-company-logo-dark"));
    expect(screen.getByRole("navigation", { name: "Clinical navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Consultation" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "患者场景" })).toHaveClass("clinical-profile-switch");
    expect(screen.getByRole("button", { name: "Reset Scene" })).toBeInTheDocument();
    expect(screen.getByLabelText("场景操作")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "患者场景" }));
    expect(onProfileClick).toHaveBeenCalledTimes(1);
  });

  it("uses native disabled state for unavailable nav items", () => {
    const onSelect = vi.fn();

    render(
      <ClinicalTopNav
        brandLabel="LangGraph Clinical Assistant"
        navLabel="Clinical navigation"
        items={navItems}
        activeKey="consultation"
        onSelect={onSelect}
        statusLabel="SSE Connected"
        statusTone="connected"
        profileLabel="Doctor"
        profileAriaLabel="患者场景"
      />,
    );

    const disabledNav = screen.getByRole("button", { name: "Multimodal" });
    expect(disabledNav).toBeDisabled();
    expect(disabledNav).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(disabledNav);
    expect(onSelect).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Patient Database" }));
    expect(onSelect).toHaveBeenCalledWith("database");
  });

  it("can render the patient-care light company logo variant", () => {
    render(
      <ClinicalTopNav
        brandLabel="LangGraph Clinical Assistant"
        brandLogoVariant="light"
        navLabel="Clinical navigation"
        items={navItems}
        activeKey="consultation"
        onSelect={vi.fn()}
        statusLabel="Safe Mode"
        statusTone="safe"
        profileLabel="Patient"
        profileAriaLabel="医生场景"
        className="clinical-top-nav-patient"
      />,
    );

    expect(screen.getByRole("img", { name: "亿铸科技公司标识" })).toHaveAttribute(
      "src",
      expect.stringContaining("yizhu-company-logo-light"),
    );
  });

  it("renders a custom profile control when the workspace provides one", () => {
    render(
      <ClinicalTopNav
        brandLabel="Agent Admin"
        navLabel="Agent admin navigation"
        items={navItems}
        activeKey="consultation"
        onSelect={vi.fn()}
        statusLabel="Read only"
        statusTone="safe"
        profileLabel="Admin"
        profileAriaLabel="切换工作台"
        profileControl={<button type="button">后台切换菜单</button>}
      />,
    );

    expect(screen.getByRole("button", { name: "后台切换菜单" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "切换工作台" })).not.toBeInTheDocument();
  });
});
