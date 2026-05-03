import type { HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export interface MessageBubbleProps extends HTMLAttributes<HTMLLIElement> {
  author: "assistant" | "user";
  children: ReactNode;
  label: ReactNode;
}

export function MessageBubble({
  author,
  children,
  className,
  label,
  ...props
}: MessageBubbleProps) {
  return (
    <li
      className={classNames([
        "ui-message-bubble",
        `ui-message-${author}`,
        `ui-message-bubble-${author}`,
        className,
      ])}
      {...props}
    >
      <span className="ui-message-avatar" aria-hidden="true">
        {author === "user" ? "U" : "AI"}
      </span>
      <div className="ui-message-content">
        <div className="ui-message-header">
          <strong>{label}</strong>
        </div>
        {children}
      </div>
    </li>
  );
}
