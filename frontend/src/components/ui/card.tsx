import { forwardRef } from "react";
import type { ForwardedRef, HTMLAttributes, ReactNode } from "react";

import { classNames } from "./class-names";

export type CardPadding = "none" | "sm" | "md";
export type CardTone = "surface" | "soft" | "warning" | "danger";
export type CardVariant = "default" | "clinical-panel";

export interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: "article" | "aside" | "div" | "section";
  children: ReactNode;
  footer?: ReactNode;
  header?: ReactNode;
  padding?: CardPadding;
  selected?: boolean;
  tone?: CardTone;
  variant?: CardVariant;
}

function assignCardRef(ref: ForwardedRef<HTMLElement>, node: HTMLElement | null) {
  if (typeof ref === "function") {
    ref(node);
    return;
  }
  if (ref) {
    ref.current = node;
  }
}

export const Card = forwardRef<HTMLElement, CardProps>(function Card(
  {
    as: Element = "div",
    children,
    className,
    footer,
    header,
    padding = "md",
    selected = false,
    tone = "surface",
    variant = "default",
    ...props
  },
  ref,
) {
  return (
    <Element
      ref={(node) => assignCardRef(ref, node)}
      className={classNames([
        "ui-card",
        variant !== "default" && `ui-card-${variant}`,
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
});
