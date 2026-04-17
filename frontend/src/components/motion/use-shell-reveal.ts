import { useEffect, type RefObject } from "react";
import { gsap } from "gsap";

import { usePrefersReducedMotion } from "./use-prefers-reduced-motion";

export function useShellReveal(target: RefObject<HTMLElement>, deps: ReadonlyArray<unknown> = []) {
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion || !target.current) {
      return;
    }

    const animation = gsap.fromTo(
      target.current,
      { autoAlpha: 0, y: 14 },
      { autoAlpha: 1, y: 0, duration: 0.32, ease: "power2.out" },
    );

    return () => {
      animation.kill();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefersReducedMotion, target, ...deps]);
}
