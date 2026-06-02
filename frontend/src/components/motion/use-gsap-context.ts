import { useEffect, type RefObject } from "react";
import { gsap } from "gsap";

import { usePrefersReducedMotion } from "./use-prefers-reduced-motion";

type GsapScopeElement = HTMLElement | SVGElement;

export function useGsapContext(
  scope: RefObject<GsapScopeElement>,
  setup: (element: GsapScopeElement) => void | (() => void),
  deps: ReadonlyArray<unknown> = [],
) {
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion || typeof window === "undefined" || !scope.current) {
      return;
    }

    let cleanup: void | (() => void);
    const element = scope.current;
    const context = gsap.context(() => {
      cleanup = setup(element);
    }, element);

    return () => {
      cleanup?.();
      context.revert();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefersReducedMotion, scope, ...deps]);

  return prefersReducedMotion;
}
