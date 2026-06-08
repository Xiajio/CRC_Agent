import { memo, useCallback, useRef, type KeyboardEvent } from "react";
import { gsap } from "gsap";

import { motionTokens } from "../../components/motion/motion-tokens";
import { useGsapContext } from "../../components/motion/use-gsap-context";

type WholeBodyAnatomyOverviewProps = {
  active: boolean;
  disabled?: boolean;
  onRegionSelect?: () => void;
};

function isKeyboardActivation(event: KeyboardEvent<SVGPathElement>): boolean {
  return event.key === "Enter" || event.key === " ";
}

export const WholeBodyAnatomyOverview = memo(function WholeBodyAnatomyOverview({
  active,
  disabled = false,
  onRegionSelect,
}: WholeBodyAnatomyOverviewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const isInteractive = active && !disabled;

  useGsapContext(svgRef, (element) => {
    if (!active) {
      return;
    }

    const targets = element.querySelectorAll("[data-active='true']");
    if (targets.length === 0) {
      return;
    }

    gsap.set(targets, {
      transformBox: "fill-box",
      transformOrigin: "50% 55%",
      willChange: "transform, opacity",
    });
    gsap.fromTo(
      targets,
      { opacity: 0.72, scale: 0.985 },
      {
        opacity: 1,
        scale: motionTokens.highlight.scale,
        duration: motionTokens.duration.highlight,
        ease: motionTokens.ease.out,
        clearProps: "opacity,transform,willChange",
      },
    );
  }, [active]);

  const handleSelect = useCallback(() => {
    if (!isInteractive) {
      return;
    }
    onRegionSelect?.();
  }, [isInteractive, onRegionSelect]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<SVGPathElement>) => {
      if (!isKeyboardActivation(event)) {
        return;
      }
      event.preventDefault();
      handleSelect();
    },
    [handleSelect],
  );

  return (
    <div className="whole-body-anatomy-overview">
      <svg
        ref={svgRef}
        className="whole-body-anatomy-svg"
        viewBox="0 0 120 220"
        role="group"
        aria-label="人体定位总览"
      >
        <circle
          className="whole-body-anatomy-outline"
          cx="60"
          cy="22"
          r="14"
          fill="none"
          stroke="var(--color-border-strong)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={6}
          aria-hidden="true"
        />
        <path
          className="whole-body-anatomy-outline"
          d="M43 42 C45 34 75 34 77 42 L84 105 C86 123 76 139 72 158 L67 204 M77 48 C92 69 100 91 98 120 M43 48 C28 69 20 91 22 120 M43 105 C34 128 36 157 43 204 M43 42 C36 63 36 85 43 105 C48 113 72 113 77 105 C84 85 84 63 77 42"
          fill="none"
          stroke="var(--color-border-strong)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={6}
          aria-hidden="true"
        />
        <path
          className="whole-body-anatomy-region-halo"
          data-active={active}
          d="M42 107 C48 97 72 97 78 107 C82 119 77 132 68 138 L52 138 C43 132 38 119 42 107"
          fill="var(--clinical-success)"
          opacity={active ? 0.16 : 0}
          aria-hidden="true"
        />
        <path
          role="button"
          tabIndex={isInteractive ? 0 : -1}
          aria-label="腹盆腔结直肠定位区域"
          aria-pressed={active}
          aria-disabled={isInteractive ? undefined : "true"}
          className="whole-body-anatomy-region"
          data-active={active}
          d="M42 107 C48 97 72 97 78 107 C82 119 77 132 68 138 L52 138 C43 132 38 119 42 107"
          fill={
            active
              ? "color-mix(in srgb, var(--clinical-success) 28%, transparent)"
              : "var(--color-surface-muted)"
          }
          stroke={
            active
              ? "color-mix(in srgb, var(--clinical-success) 82%, var(--color-text))"
              : "var(--color-border-strong)"
          }
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          opacity={active ? 1 : 0.5}
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
        />
      </svg>
      <p className="whole-body-anatomy-caption">腹盆腔/下腹部</p>
    </div>
  );
});
