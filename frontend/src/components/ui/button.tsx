import type { ButtonHTMLAttributes } from "react";

import { classNames } from "./class-names";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  size?: ButtonSize;
  variant?: ButtonVariant;
}

export function Button({
  children,
  className,
  size = "md",
  type = "button",
  variant = "secondary",
  ...props
}: ButtonProps) {
  return (
    <button
      className={classNames(["ui-button", `ui-button-${variant}`, `ui-button-${size}`, className])}
      type={type}
      {...props}
    >
      {children}
    </button>
  );
}
