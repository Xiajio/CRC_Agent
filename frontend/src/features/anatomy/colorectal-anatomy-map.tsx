import { memo, useCallback, useMemo, useRef, type KeyboardEvent } from "react";
import { gsap } from "gsap";

import { motionTokens } from "../../components/motion/motion-tokens";
import { useGsapContext } from "../../components/motion/use-gsap-context";
import {
  ANATOMY_REGIONS,
  type AnatomyRegionCode,
  type AnatomyRegion,
} from "./anatomy-region-map";

type ColorectalAnatomyMapProps = {
  activeRegionCodes: AnatomyRegionCode[];
  disabled?: boolean;
  onRegionSelect?: (region: AnatomyRegion) => void;
};

const REGION_PATHS: Record<AnatomyRegionCode, string> = {
  cecum: "M78 184 C62 181 53 193 57 207 C61 220 78 219 83 205 C86 196 84 188 78 184",
  ascending_colon: "M82 184 C82 151 82 113 86 84",
  hepatic_flexure: "M86 84 C92 59 112 49 133 55",
  transverse_colon: "M133 55 C151 59 166 72 174 88",
  splenic_flexure: "M174 88 C188 112 179 130 164 138",
  descending_colon: "M164 138 C158 164 154 184 143 199",
  sigmoid_colon: "M143 199 C128 218 100 216 98 194 C97 180 115 176 124 188",
  rectosigmoid: "M124 188 C132 196 133 207 127 216",
  rectum: "M127 216 C123 226 122 236 122 244",
  anus: "M122 244 L122 251",
};

const REGION_PATH_ENTRIES = ANATOMY_REGIONS.map((region) => ({
  region,
  path: REGION_PATHS[region.code],
}));

function isKeyboardActivation(event: KeyboardEvent<SVGPathElement>): boolean {
  return event.key === "Enter" || event.key === " ";
}

type AnatomyRegionPathProps = {
  active: boolean;
  disabled: boolean;
  onSelect: (region: AnatomyRegion) => void;
  path: string;
  region: AnatomyRegion;
};

const AnatomyRegionPath = memo(function AnatomyRegionPath({
  active,
  disabled,
  onSelect,
  path,
  region,
}: AnatomyRegionPathProps) {
  const handleClick = useCallback(() => {
    onSelect(region);
  }, [onSelect, region]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<SVGPathElement>) => {
      if (!isKeyboardActivation(event)) {
        return;
      }
      event.preventDefault();
      onSelect(region);
    },
    [onSelect, region],
  );

  return (
    <path
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label={region.label}
      aria-pressed={active}
      aria-disabled={disabled ? "true" : undefined}
      className="anatomy-map-region"
      data-region={region.code}
      data-active={active}
      d={path}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    />
  );
});

export function ColorectalAnatomyMap({
  activeRegionCodes,
  disabled = false,
  onRegionSelect,
}: ColorectalAnatomyMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const activeKey = activeRegionCodes.join("|");
  const activeSet = useMemo(() => new Set(activeRegionCodes), [activeKey]);

  useGsapContext(svgRef, (element) => {
    if (activeRegionCodes.length === 0) {
      return;
    }

    const targets = element.querySelectorAll("[data-active='true']");
    if (targets.length === 0) {
      return;
    }

    gsap.set(targets, {
      transformBox: "fill-box",
      transformOrigin: "50% 50%",
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
  }, [activeKey, activeRegionCodes.length]);

  const handleSelect = useCallback((region: AnatomyRegion) => {
    if (disabled) {
      return;
    }
    onRegionSelect?.(region);
  }, [disabled, onRegionSelect]);

  return (
    <svg
      ref={svgRef}
      className="anatomy-map-svg"
      viewBox="0 0 220 260"
      role="group"
      aria-label="结直肠分段示意图"
    >
      <path
        className="anatomy-map-backdrop"
        d="M78 184 C62 181 53 193 57 207 C61 220 78 219 83 205 C86 196 84 188 78 184 M82 184 C82 151 82 113 86 84 C92 59 112 49 133 55 C151 59 166 72 174 88 C188 112 179 130 164 138 C158 164 154 184 143 199 C128 218 100 216 98 194 C97 180 115 176 124 188 C132 196 133 207 127 216 C123 226 122 236 122 244 L122 251"
        aria-hidden="true"
      />
      {REGION_PATH_ENTRIES.map(({ region, path }) => (
        <AnatomyRegionPath
          key={region.code}
          active={activeSet.has(region.code)}
          disabled={disabled}
          onSelect={handleSelect}
          path={path}
          region={region}
        />
      ))}
    </svg>
  );
}
