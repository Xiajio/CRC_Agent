import type { RefObject } from "react";
import { gsap } from "gsap";

import { motionTokens } from "./motion-tokens";
import { useGsapContext } from "./use-gsap-context";

export function useViewTransition(target: RefObject<HTMLElement>, viewKey: unknown) {
  useGsapContext(target, (element) => {
    gsap.set(element, { willChange: "opacity" });
    gsap.fromTo(
      element,
      { opacity: 0 },
      {
        opacity: 1,
        duration: motionTokens.duration.transition,
        ease: motionTokens.ease.out,
        clearProps: "opacity,willChange",
      },
    );
  }, [viewKey]);
}
