import type { TextareaHTMLAttributes } from "react";

import { classNames } from "./class-names";

export function Textarea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={classNames(["ui-textarea", className])} {...props} />;
}
