import type { HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export type CardPadding = "none" | "sm" | "md";
export type CardTone = "surface" | "soft" | "warning" | "danger";

export interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: "article" | "aside" | "div" | "section";
  children: ReactNode;
  footer?: ReactNode;
  header?: ReactNode;
  padding?: CardPadding;
  selected?: boolean;
  tone?: CardTone;
}

export function Card({
  as: Element = "div",
  children,
  className,
  footer,
  header,
  padding = "md",
  selected = false,
  tone = "surface",
  ...props
}: CardProps) {
  return (
    <Element
      className={classNames([
        "ui-card",
        `ui-card-padding-${padding}`,
        tone !== "surface" && `ui-card-${tone}`,
        selected && "ui-card-selected",
        className,
      ])}
      {...props}
    >
      {header ? <div className="ui-card-header">{header}</div> : null}
      <div className="ui-card-body">{children}</div>
      {footer ? <div className="ui-card-footer">{footer}</div> : null}
    </Element>
  );
}
