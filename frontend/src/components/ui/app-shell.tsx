import type { HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export interface AppShellProps extends HTMLAttributes<HTMLElement> {
  topNav?: ReactNode;
  bodyClassName?: string;
  children: ReactNode;
}

export function AppShell({ topNav, bodyClassName, children, className, ...props }: AppShellProps) {
  return (
    <main className={classNames(["ui-app-shell", className])} {...props}>
      {topNav}
      <div className={classNames(["ui-app-body", bodyClassName])}>{children}</div>
    </main>
  );
}
