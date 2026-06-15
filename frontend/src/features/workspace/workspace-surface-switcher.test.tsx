import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { WorkspaceSurfaceSwitcher, type WorkspaceSurface } from "./workspace-surface-switcher";

const surfaces: WorkspaceSurface[] = ["patient", "doctor", "agent-admin"];

describe("WorkspaceSurfaceSwitcher", () => {
  it("opens a collapsible surface menu and selects the agent admin surface", () => {
    const onSelect = vi.fn();

    render(
      <WorkspaceSurfaceSwitcher
        activeSurface="doctor"
        surfaces={surfaces}
        onSelect={onSelect}
      />,
    );

    const trigger = screen.getByRole("button", { name: "切换工作台，当前为医生" });
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(trigger);

    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("menu", { name: "工作台切换" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /患者/ })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /医生/ })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("menuitem", { name: /后台/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("menuitem", { name: /后台/ }));

    expect(onSelect).toHaveBeenCalledWith("agent-admin");
    expect(screen.queryByRole("menu", { name: "工作台切换" })).not.toBeInTheDocument();
  });

  it("closes the menu with Escape without changing surfaces", () => {
    const onSelect = vi.fn();

    render(
      <WorkspaceSurfaceSwitcher
        activeSurface="agent-admin"
        surfaces={surfaces}
        onSelect={onSelect}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "切换工作台，当前为后台" }));
    expect(screen.getByRole("menu", { name: "工作台切换" })).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("menu", { name: "工作台切换" })).not.toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
  });
});
