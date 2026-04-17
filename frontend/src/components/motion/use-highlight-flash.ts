import { useEffect, useRef, type RefObject } from "react";
import { gsap } from "gsap";

import { usePrefersReducedMotion } from "./use-prefers-reduced-motion";

export function useHighlightFlash(target: RefObject<HTMLElement>, trigger: unknown) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const lastTrigger = useRef<unknown>(trigger);

  useEffect(() => {
    if (prefersReducedMotion || !target.current) {
      lastTrigger.current = trigger;
      return;
    }

    if (lastTrigger.current === trigger) {
      return;
    }

    lastTrigger.current = trigger;
    const animation = gsap.fromTo(
      target.current,
      { boxShadow: "0 0 0 0 rgba(15, 76, 129, 0.28)" },
      {
        boxShadow: "0 0 0 10px rgba(15, 76, 129, 0)",
        duration: 0.42,
        ease: "power2.out",
      },
    );

    return () => {
      animation.kill();
    };
  }, [prefersReducedMotion, target, trigger]);
}
