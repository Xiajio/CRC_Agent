import type { ReactNode } from "react";

import { AppShell, PanelGrid } from "../ui";

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
  return (
    <AppShell
      className="workspace-shell"
      topNav={toolbar ? <div className="workspace-toolbar">{toolbar}</div> : undefined}
      bodyClassName="ui-app-body-flush"
    >
      <PanelGrid
        leftRail={leftRail}
        centerWorkspace={centerWorkspace}
        rightInspector={rightInspector}
        leftRailOpen={leftRailOpen}
        rightInspectorOpen={rightInspectorOpen}
      />
    </AppShell>
  );
}
