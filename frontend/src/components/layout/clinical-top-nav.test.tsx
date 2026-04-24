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
        profileAriaLabel="patient scene"
        onProfileClick={onProfileClick}
      />,
    );

    expect(screen.getByText("LangGraph Clinical Assistant")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Clinical navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Consultation" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "patient scene" })).toHaveClass("clinical-profile-switch");
    expect(screen.getByRole("button", { name: "Reset Scene" })).toBeInTheDocument();
    expect(screen.getByLabelText("场景操作")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "patient scene" }));
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
        profileAriaLabel="patient scene"
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
});
