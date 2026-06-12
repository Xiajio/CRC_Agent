const motionCssTokens = {
  durationFeedback: "160ms",
  durationHighlight: "240ms",
  durationTransition: "240ms",
  durationEnter: "320ms",
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
  gsapEaseOut: "power3.out",
  enterY: "12px",
  highlightScale: "1.018",
  highlightRingOpacity: "0.26",
} as const;

function millisecondsToSeconds(value: `${number}ms`) {
  return Number(value.replace("ms", "")) / 1000;
}

function pixelsToNumber(value: `${number}px`) {
  return Number(value.replace("px", ""));
}

export const motionTokens = {
  css: motionCssTokens,
  duration: {
    feedback: millisecondsToSeconds(motionCssTokens.durationFeedback),
    highlight: millisecondsToSeconds(motionCssTokens.durationHighlight),
    transition: millisecondsToSeconds(motionCssTokens.durationTransition),
    enter: millisecondsToSeconds(motionCssTokens.durationEnter),
  },
  ease: {
    out: motionCssTokens.gsapEaseOut,
  },
  enter: {
    y: pixelsToNumber(motionCssTokens.enterY),
  },
  highlight: {
    scale: Number(motionCssTokens.highlightScale),
    ringOpacity: Number(motionCssTokens.highlightRingOpacity),
  },
} as const;
