import { useEffect, useRef, type RefObject } from "react";

import { motionTokens } from "./motion-tokens";
import { usePrefersReducedMotion } from "./use-prefers-reduced-motion";

const HIGHLIGHT_CLASS_NAME = "motion-highlight-pulse";
const HIGHLIGHT_ACTIVE_ATTRIBUTE = "data-motion-highlight-pulse";

export function useHighlightPulse(target: RefObject<HTMLElement>, trigger: unknown) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const lastTrigger = useRef<unknown>(trigger);

  useEffect(() => {
    const element = target.current;
    if (!element) {
      lastTrigger.current = trigger;
      return;
    }

    element.classList.add(HIGHLIGHT_CLASS_NAME);

    if (prefersReducedMotion) {
      lastTrigger.current = trigger;
      element.removeAttribute(HIGHLIGHT_ACTIVE_ATTRIBUTE);
      return;
    }

    if (lastTrigger.current === trigger) {
      return;
    }

    lastTrigger.current = trigger;
    element.removeAttribute(HIGHLIGHT_ACTIVE_ATTRIBUTE);
    const requestFrame =
      typeof window.requestAnimationFrame === "function"
        ? window.requestAnimationFrame.bind(window)
        : (callback: FrameRequestCallback) => window.setTimeout(callback, 16);
    const cancelFrame =
      typeof window.cancelAnimationFrame === "function"
        ? window.cancelAnimationFrame.bind(window)
        : window.clearTimeout.bind(window);
    const animationFrameId = requestFrame(() => {
      element.setAttribute(HIGHLIGHT_ACTIVE_ATTRIBUTE, "active");
    });

    const timeoutId = window.setTimeout(() => {
      element.removeAttribute(HIGHLIGHT_ACTIVE_ATTRIBUTE);
    }, motionTokens.duration.highlight * 1000);

    return () => {
      cancelFrame(animationFrameId);
      window.clearTimeout(timeoutId);
      element.removeAttribute(HIGHLIGHT_ACTIVE_ATTRIBUTE);
    };
  }, [prefersReducedMotion, target, trigger]);
}
