from __future__ import annotations

from backend.api.services.anatomy_region_map import resolve_region_codes


def test_resolve_region_codes_matches_precise_crc_subsites() -> None:
    assert resolve_region_codes("\u4e59\u72b6\u7ed3\u80a0 cT4bN1cM0") == ["sigmoid_colon"]
    assert resolve_region_codes("rectosigmoid junction mass") == ["rectosigmoid"]
    assert resolve_region_codes("C20 rectal lesion") == ["rectum"]


def test_resolve_region_codes_disambiguates_rectum_from_colorectal() -> None:
    assert resolve_region_codes("\u7ed3\u76f4\u80a0\u764c") == [
        "cecum",
        "ascending_colon",
        "hepatic_flexure",
        "transverse_colon",
        "splenic_flexure",
        "descending_colon",
        "sigmoid_colon",
        "rectosigmoid",
        "rectum",
    ]
    assert resolve_region_codes("colorectal cancer")[-1] == "rectum"


def test_resolve_region_codes_handles_broad_colon_without_rectum() -> None:
    assert resolve_region_codes("colon") == [
        "cecum",
        "ascending_colon",
        "hepatic_flexure",
        "transverse_colon",
        "splenic_flexure",
        "descending_colon",
        "sigmoid_colon",
    ]


def test_resolve_region_codes_ignores_missing_or_unknown_locations() -> None:
    assert resolve_region_codes(None) == []
    assert resolve_region_codes("not_provided") == []
