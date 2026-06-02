export type AnatomyRegionCode =
  | "cecum"
  | "ascending_colon"
  | "hepatic_flexure"
  | "transverse_colon"
  | "splenic_flexure"
  | "descending_colon"
  | "sigmoid_colon"
  | "rectosigmoid"
  | "rectum"
  | "anus";

export type AnatomyRegion = {
  code: AnatomyRegionCode;
  label: string;
  icdOTopography: string;
  keywords: string[];
};

export type AnatomyPatientDetail = {
  patient_id?: number | null;
  tumor_location?: string | null;
  tumor_region_code?: unknown;
  tumor_region_codes?: unknown;
};

export type ResolvedAnatomyRegions = {
  regionCodes: AnatomyRegionCode[];
  summaryLabel: string | null;
  source: "structured" | "text" | "none";
};

export const ANATOMY_REGIONS: readonly AnatomyRegion[] = [
  {
    code: "cecum",
    label: "盲肠",
    icdOTopography: "C18.0",
    keywords: ["盲肠", "盲腸", "回盲部", "cecum", "caecum", "ileocecal", "c18.0"],
  },
  {
    code: "ascending_colon",
    label: "升结肠",
    icdOTopography: "C18.2",
    keywords: ["升结肠", "升結腸", "ascending colon", "ascending_colon", "c18.2"],
  },
  {
    code: "hepatic_flexure",
    label: "肝曲",
    icdOTopography: "C18.3",
    keywords: ["肝曲", "结肠肝曲", "結腸肝曲", "hepatic flexure", "hepatic_flexure", "c18.3"],
  },
  {
    code: "transverse_colon",
    label: "横结肠",
    icdOTopography: "C18.4",
    keywords: ["横结肠", "橫結腸", "横", "transverse colon", "transverse_colon", "c18.4"],
  },
  {
    code: "splenic_flexure",
    label: "脾曲",
    icdOTopography: "C18.5",
    keywords: ["脾曲", "结肠脾曲", "結腸脾曲", "splenic flexure", "splenic_flexure", "c18.5"],
  },
  {
    code: "descending_colon",
    label: "降结肠",
    icdOTopography: "C18.6",
    keywords: ["降结肠", "降結腸", "descending colon", "descending_colon", "c18.6"],
  },
  {
    code: "sigmoid_colon",
    label: "乙状结肠",
    icdOTopography: "C18.7",
    keywords: ["乙状结肠", "乙狀結腸", "乙状", "乙狀", "sigmoid colon", "sigmoid_colon", "sigmoid", "c18.7"],
  },
  {
    code: "rectosigmoid",
    label: "直乙交界",
    icdOTopography: "C19",
    keywords: [
      "直肠乙状结肠交界",
      "直腸乙狀結腸交界",
      "直乙交界",
      "直乙",
      "rectosigmoid junction",
      "rectosigmoid",
      "c19",
    ],
  },
  {
    code: "rectum",
    label: "直肠",
    icdOTopography: "C20",
    keywords: ["直肠", "直腸", "rectum", "rectal", "c20"],
  },
  {
    code: "anus",
    label: "肛管",
    icdOTopography: "C21",
    keywords: ["肛管", "肛门管", "肛門管", "anus", "anal canal", "c21"],
  },
] as const;

export const COLON_SEGMENT_REGION_CODES: AnatomyRegionCode[] = [
  "cecum",
  "ascending_colon",
  "hepatic_flexure",
  "transverse_colon",
  "splenic_flexure",
  "descending_colon",
  "sigmoid_colon",
];

const COLORECTAL_REGION_CODES: AnatomyRegionCode[] = [
  ...COLON_SEGMENT_REGION_CODES,
  "rectosigmoid",
  "rectum",
];

const REGION_BY_CODE = new Map(ANATOMY_REGIONS.map((region) => [region.code, region]));
const BROAD_COLON_KEYWORDS = ["colon", "colonic", "结肠", "結腸", "结肠癌", "結腸癌"];
const BROAD_COLORECTAL_KEYWORDS = ["crc", "colorectal", "结直肠", "結直腸", "结直肠癌", "結直腸癌"];

function uniqueRegionCodes(regionCodes: AnatomyRegionCode[]): AnatomyRegionCode[] {
  const seen = new Set<AnatomyRegionCode>();
  const unique: AnatomyRegionCode[] = [];
  for (const code of regionCodes) {
    if (seen.has(code)) {
      continue;
    }
    seen.add(code);
    unique.push(code);
  }
  return unique;
}

function asRegionCode(value: unknown): AnatomyRegionCode | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase().replace(/-/g, "_");
  return REGION_BY_CODE.has(normalized as AnatomyRegionCode) ? (normalized as AnatomyRegionCode) : null;
}

function structuredRegionCodes(detail: AnatomyPatientDetail | null | undefined): AnatomyRegionCode[] {
  if (!detail) {
    return [];
  }

  const values = Array.isArray(detail.tumor_region_codes)
    ? detail.tumor_region_codes
    : typeof detail.tumor_region_codes === "string"
      ? detail.tumor_region_codes.split(/[,\s，、]+/)
      : [];
  const regionCodes = [...values, detail.tumor_region_code]
    .map((value) => asRegionCode(value))
    .filter((code): code is AnatomyRegionCode => code !== null);
  return uniqueRegionCodes(regionCodes);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isAsciiKeyword(value: string): boolean {
  return /^[\w.\s-]+$/.test(value);
}

function matchesKeyword(text: string, compactText: string, keyword: string): boolean {
  const normalizedKeyword = keyword.toLowerCase();
  if (normalizedKeyword === "直肠" || normalizedKeyword === "直腸") {
    return compactText.includes(normalizedKeyword) && !compactText.includes("结直肠") && !compactText.includes("結直腸");
  }
  if (normalizedKeyword === "rectal") {
    return /\brectal\b/i.test(text) && !/\bcolorectal\b/i.test(text);
  }
  if (normalizedKeyword === "rectum") {
    return /\brectum\b/i.test(text);
  }
  if (isAsciiKeyword(normalizedKeyword)) {
    const pattern = normalizedKeyword.includes(".")
      ? escapeRegExp(normalizedKeyword)
      : `\\b${escapeRegExp(normalizedKeyword).replace(/\\s+/g, "\\s+")}\\b`;
    return new RegExp(pattern, "i").test(text);
  }
  return compactText.includes(normalizedKeyword);
}

function textRegionCodes(value: string | null | undefined): ResolvedAnatomyRegions {
  if (!value || !value.trim()) {
    return { regionCodes: [], summaryLabel: null, source: "none" };
  }

  const text = value.trim().toLowerCase();
  const compactText = text.replace(/\s+/g, "");
  const preciseMatches = ANATOMY_REGIONS
    .filter((region) => region.keywords.some((keyword) => matchesKeyword(text, compactText, keyword)))
    .map((region) => region.code);
  const uniquePreciseMatches = uniqueRegionCodes(preciseMatches);
  if (uniquePreciseMatches.length > 0) {
    return {
      regionCodes: uniquePreciseMatches,
      summaryLabel: uniquePreciseMatches.map((code) => REGION_BY_CODE.get(code)?.label).filter(Boolean).join("、"),
      source: "text",
    };
  }

  if (BROAD_COLORECTAL_KEYWORDS.some((keyword) => matchesKeyword(text, compactText, keyword))) {
    return {
      regionCodes: COLORECTAL_REGION_CODES,
      summaryLabel: "结直肠（未细分）",
      source: "text",
    };
  }

  if (BROAD_COLON_KEYWORDS.some((keyword) => matchesKeyword(text, compactText, keyword))) {
    return {
      regionCodes: COLON_SEGMENT_REGION_CODES,
      summaryLabel: "结肠（未细分）",
      source: "text",
    };
  }

  return { regionCodes: [], summaryLabel: null, source: "none" };
}

export function regionByCode(code: AnatomyRegionCode): AnatomyRegion {
  const region = REGION_BY_CODE.get(code);
  if (!region) {
    throw new Error(`Unknown anatomy region: ${code}`);
  }
  return region;
}

export function resolveAnatomyRegions(detail: AnatomyPatientDetail | null | undefined): ResolvedAnatomyRegions {
  const structured = structuredRegionCodes(detail);
  if (structured.length > 0) {
    return {
      regionCodes: structured,
      summaryLabel: structured.map((code) => regionByCode(code).label).join("、"),
      source: "structured",
    };
  }

  return textRegionCodes(detail?.tumor_location);
}
