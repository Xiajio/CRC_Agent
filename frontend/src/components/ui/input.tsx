import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

import { classNames } from "./class-names";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: ReactNode;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: ReactNode;
}

export function Input({ className, label, ...props }: InputProps) {
  const input = <input className={classNames(["ui-input", className])} {...props} />;

  if (!label) {
    return input;
  }

  return (
    <label className="ui-field">
      <span className="ui-field-label">{label}</span>
      {input}
    </label>
  );
}

export function Select({
  children,
  className,
  label,
  ...props
}: SelectProps) {
  const select = (
    <select className={classNames(["ui-select", className])} {...props}>
      {children}
    </select>
  );

  if (!label) {
    return select;
  }

  return (
    <label className="ui-field">
      <span className="ui-field-label">{label}</span>
      {select}
    </label>
  );
}
