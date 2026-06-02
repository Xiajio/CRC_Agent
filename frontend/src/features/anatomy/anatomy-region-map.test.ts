import { describe, expect, it } from "vitest";

import {
  COLON_SEGMENT_REGION_CODES,
  resolveAnatomyRegions,
  type AnatomyPatientDetail,
} from "./anatomy-region-map";

describe("resolveAnatomyRegions", () => {
  it("uses structured region codes before falling back to tumor location text", () => {
    const detail: AnatomyPatientDetail = {
      tumor_region_codes: ["sigmoid_colon", "rectum", "sigmoid_colon"],
      tumor_location: "rectum",
    };

    expect(resolveAnatomyRegions(detail)).toMatchObject({
      regionCodes: ["sigmoid_colon", "rectum"],
      summaryLabel: "乙状结肠、直肠",
      source: "structured",
    });
  });

  it("matches precise Chinese and English colorectal subsites from text", () => {
    expect(resolveAnatomyRegions({ tumor_location: "乙状结肠 cT4bN1cM0" })).toMatchObject({
      regionCodes: ["sigmoid_colon"],
      summaryLabel: "乙状结肠",
      source: "text",
    });
    expect(resolveAnatomyRegions({ tumor_location: "rectosigmoid junction mass" })).toMatchObject({
      regionCodes: ["rectosigmoid"],
      summaryLabel: "直乙交界",
      source: "text",
    });
  });

  it("handles broad colon text without inventing a precise subsite", () => {
    expect(resolveAnatomyRegions({ tumor_location: "colon" })).toMatchObject({
      regionCodes: COLON_SEGMENT_REGION_CODES,
      summaryLabel: "结肠（未细分）",
      source: "text",
    });
  });

  it("returns an empty result when no location signal is available", () => {
    expect(resolveAnatomyRegions({ tumor_location: null })).toMatchObject({
      regionCodes: [],
      summaryLabel: null,
      source: "none",
    });
  });
});
