import { useCallback, useMemo, useRef, type ReactNode } from "react";

import { useShellReveal } from "../../components/motion/use-shell-reveal";
import type { CardPatientContext, CardPromptHandler } from "../cards/card-renderers-extended";
import { Card } from "../../components/ui/card";
import {
  resolveAnatomyRegions,
  regionByCode,
  type AnatomyPatientDetail,
  type AnatomyRegion,
} from "./anatomy-region-map";
import { ColorectalAnatomyMap } from "./colorectal-anatomy-map";
import { WholeBodyAnatomyOverview } from "./whole-body-anatomy-overview";

type AnatomyHighlightPanelProps = {
  detail: AnatomyPatientDetail | null;
  patientContext?: CardPatientContext | null;
  onPromptRequest?: CardPromptHandler;
  disabled?: boolean;
  isStreaming?: boolean;
};

function AnatomyIcon(): ReactNode {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 4c-2 3-2 6 0 9s2 5 0 7" />
      <path d="M16 4c2 3 2 6 0 9s-2 5 0 7" />
      <path d="M8 13h8" />
    </svg>
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
        <div className="anatomy-highlight-visuals">
          <WholeBodyAnatomyOverview
            active={hasResolvedRegion}
            disabled={!canPrompt}
            onRegionSelect={handleOverviewSelect}
          />
          <ColorectalAnatomyMap
            activeRegionCodes={resolved.regionCodes}
            disabled={!canPrompt}
            onRegionSelect={handleRegionSelect}
          />
        </div>
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
