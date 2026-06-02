import type { RefObject } from "react";

import { useHighlightPulse } from "./use-highlight-pulse";

export function useHighlightFlash(target: RefObject<HTMLElement>, trigger: unknown) {
  useHighlightPulse(target, trigger);
}
