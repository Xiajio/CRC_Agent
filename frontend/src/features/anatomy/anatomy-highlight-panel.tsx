import { useCallback, useMemo, useRef, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";

import { useShellReveal } from "../../components/motion/use-shell-reveal";
import type { CardPatientContext, CardPromptHandler } from "../cards/card-renderers-extended";
import { Card } from "../../components/ui/card";
import colorectalAnatomyImage from "../../assets/anatomy/colorectal-anatomy-medical-2d.webp";
import {
  ANATOMY_REGIONS,
  resolveAnatomyRegions,
  regionByCode,
  type AnatomyPatientDetail,
  type AnatomyRegion,
  type AnatomyRegionCode,
} from "./anatomy-region-map";

type AnatomyHighlightPanelProps = {
  detail: AnatomyPatientDetail | null;
  patientContext?: CardPatientContext | null;
  onPromptRequest?: CardPromptHandler;
  disabled?: boolean;
  isStreaming?: boolean;
};

type HotZone = {
  left: string;
  top: string;
  width: string;
  height: string;
};

const OVERVIEW_HOT_ZONE: HotZone = {
  left: "8.6%",
  top: "33.4%",
  width: "26.4%",
  height: "49.8%",
};

const REGION_HOT_ZONES: Record<AnatomyRegionCode, HotZone> = {
  cecum: { left: "51.6%", top: "39.4%", width: "10.8%", height: "25.4%" },
  ascending_colon: { left: "51.0%", top: "13.4%", width: "9.8%", height: "31.4%" },
  hepatic_flexure: { left: "52.4%", top: "6.6%", width: "14.8%", height: "13.4%" },
  transverse_colon: { left: "62.4%", top: "5.8%", width: "25.0%", height: "14.0%" },
  splenic_flexure: { left: "82.6%", top: "4.6%", width: "12.6%", height: "15.8%" },
  descending_colon: { left: "86.0%", top: "16.8%", width: "10.4%", height: "40.8%" },
  sigmoid_colon: { left: "72.0%", top: "55.0%", width: "22.2%", height: "18.8%" },
  rectosigmoid: { left: "68.2%", top: "55.8%", width: "10.4%", height: "13.8%" },
  rectum: { left: "69.0%", top: "64.2%", width: "9.0%", height: "21.2%" },
  anus: { left: "69.6%", top: "83.2%", width: "7.8%", height: "7.4%" },
};

function hotZoneStyle(zone: HotZone): CSSProperties {
  return {
    left: zone.left,
    top: zone.top,
    width: zone.width,
    height: zone.height,
  };
}

function isKeyboardActivation(event: KeyboardEvent<HTMLButtonElement>): boolean {
  return event.key === "Enter" || event.key === " ";
}

function AnatomyIcon(): ReactNode {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 4c-2 3-2 6 0 9s2 5 0 7" />
      <path d="M16 4c2 3 2 6 0 9s-2 5 0 7" />
      <path d="M8 13h8" />
    </svg>
  );
}

function AnatomyImageMap({
  activeRegionCodes,
  canPrompt,
  hasResolvedRegion,
  onOverviewSelect,
  onRegionSelect,
}: {
  activeRegionCodes: AnatomyRegionCode[];
  canPrompt: boolean;
  hasResolvedRegion: boolean;
  onOverviewSelect: () => void;
  onRegionSelect: (region: AnatomyRegion) => void;
}) {
  const activeKey = activeRegionCodes.join("|");
  const activeSet = useMemo(() => new Set(activeRegionCodes), [activeKey]);
  const overviewDisabled = !canPrompt || !hasResolvedRegion;

  const handleOverviewKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>) => {
    if (!isKeyboardActivation(event)) {
      return;
    }
    event.preventDefault();
    onOverviewSelect();
  }, [onOverviewSelect]);

  const handleRegionKeyDown = useCallback((
    event: KeyboardEvent<HTMLButtonElement>,
    region: AnatomyRegion,
  ) => {
    if (!isKeyboardActivation(event)) {
      return;
    }
    event.preventDefault();
    onRegionSelect(region);
  }, [onRegionSelect]);

  return (
    <div className="anatomy-image-map">
      <img
        className="anatomy-medical-image"
        src={colorectalAnatomyImage}
        alt="结直肠解剖定位示意图"
      />
      <div className="anatomy-image-hotzones" role="group" aria-label="结直肠解剖热区">
        <button
          type="button"
          className="anatomy-image-hotspot anatomy-image-hotspot-overview"
          style={hotZoneStyle(OVERVIEW_HOT_ZONE)}
          aria-label="腹盆腔结直肠定位区域"
          aria-pressed={hasResolvedRegion}
          aria-disabled={overviewDisabled ? "true" : undefined}
          data-active={hasResolvedRegion}
          disabled={overviewDisabled}
          tabIndex={overviewDisabled ? -1 : 0}
          onClick={onOverviewSelect}
          onKeyDown={handleOverviewKeyDown}
        />
        {ANATOMY_REGIONS.map((region) => {
          const active = activeSet.has(region.code);
          const disabled = !canPrompt;
          return (
            <button
              key={region.code}
              type="button"
              className="anatomy-image-hotspot anatomy-image-hotspot-region"
              style={hotZoneStyle(REGION_HOT_ZONES[region.code])}
              aria-label={region.label}
              aria-pressed={active}
              aria-disabled={disabled ? "true" : undefined}
              data-active={active}
              data-region={region.code}
              disabled={disabled}
              tabIndex={disabled ? -1 : 0}
              onClick={() => onRegionSelect(region)}
              onKeyDown={(event) => handleRegionKeyDown(event, region)}
            />
          );
        })}
      </div>
    </div>
  );
}

function buildRegionPrompt(region: AnatomyRegion): string {
  return `请针对${region.label}病灶给出分期与下一步检查建议。`;
}

function buildPromptContext(
  region: AnatomyRegion,
  patientContext: CardPatientContext | null | undefined,
): Record<string, unknown> {
  return {
    ...(patientContext ?? {}),
    anatomy_region_code: region.code,
    anatomy_region_label: region.label,
    icd_o_topography: region.icdOTopography,
  };
}

function buildMultiSegmentOverviewPrompt(summaryLabel: string | null): string {
  const locationLabel = summaryLabel ?? "多分段结直肠定位";
  return `请结合${locationLabel}的结直肠定位总结病灶位置与下一步检查建议。`;
}

function buildMultiSegmentOverviewPromptContext(
  regions: AnatomyRegion[],
  summaryLabel: string | null,
  patientContext: CardPatientContext | null | undefined,
): Record<string, unknown> {
  return {
    ...(patientContext ?? {}),
    anatomy_region_codes: regions.map((region) => region.code),
    anatomy_region_labels: regions.map((region) => region.label),
    icd_o_topographies: regions.map((region) => region.icdOTopography),
    anatomy_region_scope: "colorectal_multi_segment",
    anatomy_region_summary: summaryLabel ?? "多分段结直肠定位",
  };
}

function sourceLabel(source: ReturnType<typeof resolveAnatomyRegions>["source"]): string {
  if (source === "structured") {
    return "结构化定位";
  }
  if (source === "text") {
    return "文本解析";
  }
  return "待确认";
}

export function AnatomyHighlightPanel({
  detail,
  patientContext,
  onPromptRequest,
  disabled = false,
  isStreaming = false,
}: AnatomyHighlightPanelProps) {
  const cardRef = useRef<HTMLElement>(null);
  const resolved = useMemo(() => resolveAnatomyRegions(detail), [detail]);
  const highlightedRegions = useMemo(
    () => resolved.regionCodes.map((code) => regionByCode(code)),
    [resolved.regionCodes],
  );
  const hasResolvedRegion = resolved.regionCodes.length > 0;
  const canPrompt = Boolean(onPromptRequest) && !disabled && !isStreaming;
  useShellReveal(cardRef);

  const handleRegionSelect = useCallback((region: AnatomyRegion) => {
    if (!canPrompt || !onPromptRequest) {
      return;
    }
    onPromptRequest(buildRegionPrompt(region), buildPromptContext(region, patientContext));
  }, [canPrompt, onPromptRequest, patientContext]);

  const handleOverviewSelect = useCallback(() => {
    if (!canPrompt || !onPromptRequest || !hasResolvedRegion) {
      return;
    }
    const overviewRegions = resolved.regionCodes.map((code) => regionByCode(code));
    if (overviewRegions.length === 1) {
      const overviewRegion = overviewRegions[0];
      onPromptRequest(buildRegionPrompt(overviewRegion), buildPromptContext(overviewRegion, patientContext));
      return;
    }
    onPromptRequest(
      buildMultiSegmentOverviewPrompt(resolved.summaryLabel),
      buildMultiSegmentOverviewPromptContext(overviewRegions, resolved.summaryLabel, patientContext),
    );
  }, [canPrompt, hasResolvedRegion, onPromptRequest, patientContext, resolved.regionCodes, resolved.summaryLabel]);

  return (
    <Card ref={cardRef} as="section" padding="none" className="clinical-card anatomy-highlight-card" aria-label="解剖定位">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon">{AnatomyIcon()}</span>
        <h2>解剖定位</h2>
      </div>
      <div className="anatomy-highlight-body">
        <AnatomyImageMap
          activeRegionCodes={resolved.regionCodes}
          canPrompt={canPrompt}
          hasResolvedRegion={hasResolvedRegion}
          onOverviewSelect={handleOverviewSelect}
          onRegionSelect={handleRegionSelect}
        />
        <div className="anatomy-highlight-summary">
          <p className="anatomy-highlight-kicker">{sourceLabel(resolved.source)}</p>
          {resolved.summaryLabel ? (
            <p className="anatomy-highlight-location">{resolved.summaryLabel}</p>
          ) : (
            <p className="anatomy-highlight-location anatomy-highlight-muted">暂未定位肿瘤分段</p>
          )}
          {highlightedRegions.length > 0 ? (
            <ul className="anatomy-highlight-legend" aria-label="已高亮区域">
              {highlightedRegions.map((region) => {
                return (
                  <li key={region.code}>
                    <span aria-hidden="true" />
                    {region.label}
                  </li>
                );
              })}
            </ul>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
