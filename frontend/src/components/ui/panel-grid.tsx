import type { HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export type PanelGridLayoutMode = "full" | "no-right" | "no-left" | "center-only";

export interface PanelGridProps extends HTMLAttributes<HTMLDivElement> {
  leftRail: ReactNode;
  centerWorkspace: ReactNode;
  rightInspector: ReactNode;
  leftRailOpen?: boolean;
  rightInspectorOpen?: boolean;
}

export function panelGridLayoutMode(leftRailOpen: boolean, rightInspectorOpen: boolean): PanelGridLayoutMode {
  if (leftRailOpen && rightInspectorOpen) {
    return "full";
  }

  if (leftRailOpen) {
    return "no-right";
  }

  if (rightInspectorOpen) {
    return "no-left";
  }

  return "center-only";
}

function collapsedPanelAttributes(isOpen: boolean): Record<string, string> {
  return isOpen ? {} : { "aria-hidden": "true", inert: "" };
}

export function PanelGrid({
  leftRail,
  centerWorkspace,
  rightInspector,
  leftRailOpen = true,
  rightInspectorOpen = true,
  className,
  ...props
}: PanelGridProps) {
  const layoutMode = panelGridLayoutMode(leftRailOpen, rightInspectorOpen);

  return (
    <div
      {...props}
      className={classNames([
        "ui-panel-grid",
        `ui-panel-grid-${layoutMode}`,
        className,
      ])}
      data-testid="panel-grid"
      data-layout-mode={layoutMode}
    >
      <aside
        className={classNames([
          "ui-panel",
          "ui-panel-left",
          !leftRailOpen && "ui-panel-collapsed",
        ])}
        data-testid="left-rail"
        data-panel-state={leftRailOpen ? "open" : "closed"}
        {...collapsedPanelAttributes(leftRailOpen)}
      >
        {leftRail}
      </aside>
      <section
        className="ui-panel ui-panel-center"
        data-testid="center-workspace"
      >
        {centerWorkspace}
      </section>
      <aside
        className={classNames([
          "ui-panel",
          "ui-panel-right",
          !rightInspectorOpen && "ui-panel-collapsed",
        ])}
        data-testid="right-inspector"
        data-panel-state={rightInspectorOpen ? "open" : "closed"}
        {...collapsedPanelAttributes(rightInspectorOpen)}
      >
        {rightInspector}
      </aside>
    </div>
  );
}
