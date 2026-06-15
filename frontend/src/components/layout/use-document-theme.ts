import { useLayoutEffect } from "react";

export type WorkspaceTheme = "doctor-cockpit" | "patient-care" | "agent-admin";

/**
 * 把场景主题挂到 <html data-theme="...">。
 * 主题令牌覆盖定义在 tokens.css 的 :root[data-theme=...] 块里；挂在根元素上
 * 可以让 :root 中的派生令牌（var() 别名）与覆盖值在同一元素上求值，
 * 从而整套设计令牌自动跟随主题。
 */
export function useDocumentTheme(theme: WorkspaceTheme) {
  useLayoutEffect(() => {
    const root = document.documentElement;
    const previous = root.getAttribute("data-theme");
    root.setAttribute("data-theme", theme);
    return () => {
      if (previous === null) {
        root.removeAttribute("data-theme");
      } else {
        root.setAttribute("data-theme", previous);
      }
    };
  }, [theme]);
}
