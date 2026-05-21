import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { WorkspaceLayout } from "../layout/workspace-layout";
import { AppShell, PanelGrid, TopNav, type TopNavItem } from ".";

const navItems: TopNavItem[] = [
  { key: "consultation", label: "Consultation" },
  { key: "database", label: "Patient Database" },
  { key: "multimodal", label: "Multimodal", disabled: true },
];

describe("AppShell, TopNav, and PanelGrid", () => {
  it("renders an app shell with top navigation and body content", () => {
    render(
      <AppShell
        data-testid="app-shell"
        topNav={
          <TopNav
            id="primary-top-nav"
            data-owner="clinical"
            brandLabel="LangGraph Clinical Assistant"
            brandIcon={<span data-testid="brand-icon" />}
            navLabel="Clinical navigation"
            items={navItems}
            activeKey="consultation"
            onSelect={vi.fn()}
            actions={<button type="button">Reset Scene</button>}
            statusLabel="SSE Connected"
            profileLabel="Doctor"
            profileAriaLabel="患者场景"
            profileIcon={<span data-testid="profile-icon" />}
          />
        }
      >
        <section>Workspace body</section>
      </AppShell>,
    );

    expect(screen.getByTestId("app-shell")).toHaveClass("ui-app-shell");
    expect(screen.getByTestId("workspace-toolbar")).toHaveClass("ui-top-nav");
    expect(screen.getByTestId("workspace-toolbar")).toHaveAttribute("id", "primary-top-nav");
    expect(screen.getByTestId("workspace-toolbar")).toHaveAttribute("data-owner", "clinical");
    expect(screen.getByTestId("brand-icon")).toBeInTheDocument();
    expect(screen.getByText("LangGraph Clinical Assistant")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Clinical navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset Scene" })).toBeInTheDocument();
    expect(screen.getByText("SSE Connected")).toBeInTheDocument();
    expect(screen.getByTestId("profile-icon")).toBeInTheDocument();
    expect(screen.getByText("Workspace body")).toBeInTheDocument();
  });

  it("uses active and disabled nav state and only selects enabled items", () => {
    const onSelect = vi.fn();

    render(
      <TopNav
        brandLabel="LangGraph Clinical Assistant"
        navLabel="Clinical navigation"
        items={navItems}
        activeKey="database"
        onSelect={onSelect}
        statusLabel="Safe Mode"
        statusTone="safe"
        profileLabel="Doctor"
        profileAriaLabel="患者场景"
      />,
    );

    const activeNav = screen.getByRole("button", { name: "Patient Database" });
    expect(activeNav).toHaveAttribute("aria-current", "page");
    expect(activeNav).toHaveAttribute("aria-pressed", "true");

    const disabledNav = screen.getByRole("button", { name: "Multimodal" });
    expect(disabledNav).toBeDisabled();
    expect(disabledNav).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(disabledNav);
    expect(onSelect).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Consultation" }));
    expect(onSelect).toHaveBeenCalledWith("consultation");

    expect(screen.getByRole("button", { name: "患者场景" })).toHaveClass(
      "ui-profile-switch",
      "clinical-profile-switch",
    );
  });

  it("computes panel layout mode and hides collapsed panels", () => {
    render(
      <PanelGrid
        leftRail={<div>Left rail</div>}
        centerWorkspace={<div>Center workspace</div>}
        rightInspector={<div>Right inspector</div>}
        leftRailOpen={false}
        rightInspectorOpen={true}
      />,
    );

    expect(screen.getByTestId("workspace-layout-grid")).toHaveAttribute("data-layout-mode", "no-left");
    expect(screen.getByTestId("left-rail")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByTestId("left-rail")).toHaveClass("ui-panel-collapsed");
    expect(screen.getByTestId("center-workspace")).toHaveTextContent("Center workspace");
    expect(screen.getByTestId("right-inspector")).not.toHaveAttribute("aria-hidden");
  });

  it("uses center-only layout when both side panels are collapsed", () => {
    render(
      <PanelGrid
        leftRail={<div>Left rail</div>}
        centerWorkspace={<div>Center workspace</div>}
        rightInspector={<div>Right inspector</div>}
        leftRailOpen={false}
        rightInspectorOpen={false}
      />,
    );

    expect(screen.getByTestId("workspace-layout-grid")).toHaveAttribute("data-layout-mode", "center-only");
    expect(screen.getByTestId("left-rail")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByTestId("right-inspector")).toHaveAttribute("aria-hidden", "true");
  });

  it("keeps the workspace toolbar wrapper without injecting legacy brand content", () => {
    const { container } = render(
      <WorkspaceLayout
        toolbar={<span>Toolbar content</span>}
        leftRail={<div>Left rail</div>}
        centerWorkspace={<div>Center workspace</div>}
        rightInspector={<div>Right inspector</div>}
      />,
    );

    const toolbar = container.querySelector(".workspace-toolbar");
    expect(toolbar).not.toBeNull();
    expect(toolbar).toHaveTextContent("Toolbar content");
    expect(toolbar?.querySelector(".workspace-brand")).toBeNull();
  });
});
