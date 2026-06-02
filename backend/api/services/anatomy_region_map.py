from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


AnatomyRegionCode = str


@dataclass(frozen=True)
class AnatomyRegion:
    code: AnatomyRegionCode
    label: str
    icd_o_topography: str
    keywords: tuple[str, ...]


ANATOMY_REGIONS: tuple[AnatomyRegion, ...] = (
    AnatomyRegion(
        code="cecum",
        label="\u76f2\u80a0",
        icd_o_topography="C18.0",
        keywords=(
            "\u76f2\u80a0",
            "\u76f2\u8178",
            "\u56de\u76f2\u90e8",
            "cecum",
            "caecum",
            "ileocecal",
            "c18.0",
        ),
    ),
    AnatomyRegion(
        code="ascending_colon",
        label="\u5347\u7ed3\u80a0",
        icd_o_topography="C18.2",
        keywords=(
            "\u5347\u7ed3\u80a0",
            "\u5347\u7d50\u8178",
            "ascending colon",
            "ascending_colon",
            "c18.2",
        ),
    ),
    AnatomyRegion(
        code="hepatic_flexure",
        label="\u809d\u66f2",
        icd_o_topography="C18.3",
        keywords=(
            "\u809d\u66f2",
            "\u7ed3\u80a0\u809d\u66f2",
            "\u7d50\u8178\u809d\u66f2",
            "hepatic flexure",
            "hepatic_flexure",
            "c18.3",
        ),
    ),
    AnatomyRegion(
        code="transverse_colon",
        label="\u6a2a\u7ed3\u80a0",
        icd_o_topography="C18.4",
        keywords=(
            "\u6a2a\u7ed3\u80a0",
            "\u6a6b\u7d50\u8178",
            "transverse colon",
            "transverse_colon",
            "c18.4",
        ),
    ),
    AnatomyRegion(
        code="splenic_flexure",
        label="\u813e\u66f2",
        icd_o_topography="C18.5",
        keywords=(
            "\u813e\u66f2",
            "\u7ed3\u80a0\u813e\u66f2",
            "\u7d50\u8178\u813e\u66f2",
            "splenic flexure",
            "splenic_flexure",
            "c18.5",
        ),
    ),
    AnatomyRegion(
        code="descending_colon",
        label="\u964d\u7ed3\u80a0",
        icd_o_topography="C18.6",
        keywords=(
            "\u964d\u7ed3\u80a0",
            "\u964d\u7d50\u8178",
            "descending colon",
            "descending_colon",
            "c18.6",
        ),
    ),
    AnatomyRegion(
        code="sigmoid_colon",
        label="\u4e59\u72b6\u7ed3\u80a0",
        icd_o_topography="C18.7",
        keywords=(
            "\u4e59\u72b6\u7ed3\u80a0",
            "\u4e59\u72c0\u7d50\u8178",
            "\u4e59\u72b6",
            "\u4e59\u72c0",
            "sigmoid colon",
            "sigmoid_colon",
            "sigmoid",
            "c18.7",
        ),
    ),
    AnatomyRegion(
        code="rectosigmoid",
        label="\u76f4\u4e59\u4ea4\u754c",
        icd_o_topography="C19",
        keywords=(
            "\u76f4\u80a0\u4e59\u72b6\u7ed3\u80a0\u4ea4\u754c",
            "\u76f4\u8178\u4e59\u72c0\u7d50\u8178\u4ea4\u754c",
            "\u76f4\u4e59\u4ea4\u754c",
            "\u76f4\u4e59",
            "rectosigmoid junction",
            "rectosigmoid",
            "c19",
        ),
    ),
    AnatomyRegion(
        code="rectum",
        label="\u76f4\u80a0",
        icd_o_topography="C20",
        keywords=(
            "\u76f4\u80a0",
            "\u76f4\u8178",
            "rectum",
            "rectal",
            "c20",
        ),
    ),
    AnatomyRegion(
        code="anus",
        label="\u809b\u7ba1",
        icd_o_topography="C21",
        keywords=(
            "\u809b\u7ba1",
            "\u809b\u95e8\u7ba1",
            "\u809b\u9580\u7ba1",
            "anus",
            "anal canal",
            "c21",
        ),
    ),
)

COLON_SEGMENT_REGION_CODES: tuple[AnatomyRegionCode, ...] = (
    "cecum",
    "ascending_colon",
    "hepatic_flexure",
    "transverse_colon",
    "splenic_flexure",
    "descending_colon",
    "sigmoid_colon",
)
COLORECTAL_REGION_CODES: tuple[AnatomyRegionCode, ...] = (
    *COLON_SEGMENT_REGION_CODES,
    "rectosigmoid",
    "rectum",
)
REGION_CODES: frozenset[AnatomyRegionCode] = frozenset(region.code for region in ANATOMY_REGIONS)

BROAD_COLON_KEYWORDS: tuple[str, ...] = (
    "colon",
    "colonic",
    "\u7ed3\u80a0",
    "\u7d50\u8178",
    "\u7ed3\u80a0\u764c",
    "\u7d50\u8178\u764c",
)
BROAD_COLORECTAL_KEYWORDS: tuple[str, ...] = (
    "crc",
    "colorectal",
    "\u7ed3\u76f4\u80a0",
    "\u7d50\u76f4\u8178",
    "\u7ed3\u76f4\u80a0\u764c",
    "\u7d50\u76f4\u8178\u764c",
)
PLACEHOLDER_LOCATION_VALUES: frozenset[str] = frozenset(
    {
        "",
        "not_provided",
        "unknown",
        "pending_assessment",
        "pending_evaluation",
        "parse_failed_text",
    }
)


def is_region_code(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().replace("-", "_") in REGION_CODES


def normalize_region_code(value: Any) -> AnatomyRegionCode | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower().replace("-", "_")
    return candidate if candidate in REGION_CODES else None


def unique_region_codes(region_codes: list[AnatomyRegionCode]) -> list[AnatomyRegionCode]:
    seen: set[AnatomyRegionCode] = set()
    unique: list[AnatomyRegionCode] = []
    for code in region_codes:
        if code in seen:
            continue
        seen.add(code)
        unique.append(code)
    return unique


def _escape_regexp(value: str) -> str:
    return re.escape(value)


def _is_ascii_keyword(value: str) -> bool:
    return re.fullmatch(r"[\w.\s-]+", value) is not None


def _matches_keyword(text: str, compact_text: str, keyword: str) -> bool:
    normalized_keyword = keyword.lower()
    if normalized_keyword in {"\u76f4\u80a0", "\u76f4\u8178"}:
        return (
            normalized_keyword in compact_text
            and "\u7ed3\u76f4\u80a0" not in compact_text
            and "\u7d50\u76f4\u8178" not in compact_text
            and "\u76f4\u80a0\u4e59\u72b6" not in compact_text
            and "\u76f4\u8178\u4e59\u72c0" not in compact_text
            and "\u76f4\u4e59" not in compact_text
        )
    if normalized_keyword == "rectal":
        return re.search(r"\brectal\b", text, re.IGNORECASE) is not None and re.search(
            r"\bcolorectal\b",
            text,
            re.IGNORECASE,
        ) is None
    if normalized_keyword == "rectum":
        return re.search(r"\brectum\b", text, re.IGNORECASE) is not None
    if _is_ascii_keyword(normalized_keyword):
        if "." in normalized_keyword:
            pattern = _escape_regexp(normalized_keyword)
        else:
            pattern = r"\b" + re.sub(r"\\\s+", r"\\s+", _escape_regexp(normalized_keyword)) + r"\b"
        return re.search(pattern, text, re.IGNORECASE) is not None
    return normalized_keyword in compact_text


def resolve_region_codes(tumor_location: Any) -> list[AnatomyRegionCode]:
    if tumor_location is None:
        return []
    text = str(tumor_location).strip().lower()
    if text in PLACEHOLDER_LOCATION_VALUES:
        return []

    compact_text = re.sub(r"\s+", "", text)
    precise_matches = [
        region.code
        for region in ANATOMY_REGIONS
        if any(_matches_keyword(text, compact_text, keyword) for keyword in region.keywords)
    ]
    precise_codes = unique_region_codes(precise_matches)
    if precise_codes:
        return precise_codes

    if any(_matches_keyword(text, compact_text, keyword) for keyword in BROAD_COLORECTAL_KEYWORDS):
        return list(COLORECTAL_REGION_CODES)
    if any(_matches_keyword(text, compact_text, keyword) for keyword in BROAD_COLON_KEYWORDS):
        return list(COLON_SEGMENT_REGION_CODES)
    return []
