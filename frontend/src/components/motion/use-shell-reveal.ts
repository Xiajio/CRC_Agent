import type { RefObject } from "react";
import { gsap } from "gsap";

import { motionTokens } from "./motion-tokens";
import { useGsapContext } from "./use-gsap-context";

export function useShellReveal(target: RefObject<HTMLElement>, deps: ReadonlyArray<unknown> = []) {
  useGsapContext(target, (element) => {
    gsap.set(element, { willChange: "transform, opacity" });
    gsap.fromTo(
      element,
      { opacity: 0, y: motionTokens.enter.y },
      {
        opacity: 1,
        y: 0,
        duration: motionTokens.duration.enter,
        ease: motionTokens.ease.out,
        clearProps: "opacity,transform,willChange",
      },
    );
  }, deps);
}
