import type { ReactNode } from "react";

export interface WorkspaceLayoutProps {
  leftRail: ReactNode;
  centerWorkspace: ReactNode;
  rightInspector: ReactNode;
  toolbar?: ReactNode;
  leftRailOpen?: boolean;
  rightInspectorOpen?: boolean;
}

export function WorkspaceLayout({
  leftRail,
  centerWorkspace,
  rightInspector,
  toolbar,
  leftRailOpen = true,
  rightInspectorOpen = true,
}: WorkspaceLayoutProps) {
  const layoutMode = leftRailOpen
    ? rightInspectorOpen
      ? "full"
      : "no-right"
    : rightInspectorOpen
      ? "no-left"
      : "center-only";

  return (
    <main className="workspace-shell">
      {toolbar ? (
        <div className="workspace-toolbar" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {toolbar}
          <div className="workspace-brand" style={{ marginLeft: "16px" }}>亿铸科技</div>
        </div>
      ) : null}
      <div
        className={`workspace-layout workspace-layout-${layoutMode}`}
        data-testid="workspace-layout-grid"
        data-layout-mode={layoutMode}
      >
        <aside
          className={
            leftRailOpen
              ? "workspace-panel workspace-panel-rail"
              : "workspace-panel workspace-panel-rail workspace-panel-collapsed"
          }
          data-testid="left-rail"
          data-panel-state={leftRailOpen ? "open" : "closed"}
          aria-hidden={leftRailOpen ? undefined : "true"}
        >
          {leftRail}
        </aside>
        <section className="workspace-panel workspace-panel-center" data-testid="center-workspace">
          {centerWorkspace}
        </section>
        <aside
          className={
            rightInspectorOpen
              ? "workspace-panel workspace-panel-inspector"
              : "workspace-panel workspace-panel-inspector workspace-panel-collapsed"
          }
          data-testid="right-inspector"
          data-panel-state={rightInspectorOpen ? "open" : "closed"}
          aria-hidden={rightInspectorOpen ? undefined : "true"}
        >
          {rightInspector}
        </aside>
      </div>
    </main>
  );
}
