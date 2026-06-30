from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.contracts.evidence_claim import (
    APPLICABILITY_TO_CRC_CONTEXTS,
    EFFECT_DIRECTIONS,
    EVIDENCE_GRADES,
    LOCAL_GUIDELINE_CONFLICTS,
    RISK_OF_BIAS_LEVELS,
    EvidenceClaim,
    SourceQuality,
    SourceSpan,
    make_claim_id,
)


FIXTURE_PATH = Path("tests/fixtures/literature_claim_pack_v0.json")
NEGATIVE_OR_CONFLICTING_DIRECTIONS = {
    "neutral",
    "harm",
    "conflicting",
    "inconclusive",
}
REQUIRED_EXTRACTED_CLAIM_FIELDS = {
    "claim_text",
    "population",
    "intervention",
    "comparator",
    "outcome",
    "effect_direction",
    "effect_size",
    "uncertainty",
    "evidence_grade",
    "study_design",
    "sample_size",
    "risk_of_bias",
    "local_guideline_conflict",
    "applicability_to_crc_context",
    "source_span",
}


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _all_claims(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        claim
        for candidate in pack["paper_candidates"]
        for claim in candidate["extracted_claims"]
    ]


def _source_quality(payload: dict[str, Any]) -> SourceQuality:
    return SourceQuality(
        is_guideline=payload["is_guideline"],
        is_systematic_review=payload["is_systematic_review"],
        is_preprint=payload["is_preprint"],
        is_retracted=payload["is_retracted"],
    )


def _source_span(payload: dict[str, Any]) -> SourceSpan:
    return SourceSpan(
        page=payload["page"],
        section=payload["section"],
        quote=payload["quote"],
    )


def _evidence_claim_from_fixture(
    pack: dict[str, Any],
    candidate: dict[str, Any],
    claim: dict[str, Any],
) -> EvidenceClaim:
    source_span = _source_span(claim["source_span"])
    claim_id = make_claim_id(
        source_id=candidate["source_id"],
        claim_text=claim["claim_text"],
        population=claim["population"],
        intervention=claim["intervention"],
        comparator=claim["comparator"],
        outcome=claim["outcome"],
        effect_direction=claim["effect_direction"],
        source_span=source_span,
    )
    return EvidenceClaim(
        claim_id=claim_id,
        source_id=candidate["source_id"],
        claim_text=claim["claim_text"],
        population=claim["population"],
        intervention=claim["intervention"],
        comparator=claim["comparator"],
        outcome=claim["outcome"],
        effect_direction=claim["effect_direction"],
        effect_size=claim["effect_size"],
        uncertainty=claim["uncertainty"],
        evidence_grade=claim["evidence_grade"],
        study_design=claim["study_design"],
        sample_size=claim["sample_size"],
        risk_of_bias=claim["risk_of_bias"],
        source_quality=_source_quality(candidate["source_quality"]),
        local_guideline_conflict=claim["local_guideline_conflict"],
        applicability_to_crc_context=claim["applicability_to_crc_context"],
        source_span=source_span,
        review_status="candidate",
        created_from=pack["claim_pack_id"],
    )


def test_literature_claim_pack_has_required_shadow_cases() -> None:
    pack = _load_fixture()
    candidates = pack["paper_candidates"]
    effect_directions = {
        claim["effect_direction"]
        for candidate in candidates
        for claim in candidate["extracted_claims"]
    }

    assert pack["claim_pack_id"] == "literature_claim_pack_v0"
    assert pack["evidence_index_version"] == "rag_crc_guideline_20260620"
    assert pack["expected_min_negative_or_conflicting"] == 2
    assert len(candidates) == 3
    assert "benefit" in effect_directions
    assert "neutral" in effect_directions
    assert "harm" in effect_directions
    assert pack["isolation_inputs"] == {
        "clinical_rag_claim_ids": [],
        "patient_default_claim_ids": [],
        "doctor_default_claim_ids": [],
    }

    negative_or_conflicting_claims = [
        claim
        for claim in _all_claims(pack)
        if claim["effect_direction"] in NEGATIVE_OR_CONFLICTING_DIRECTIONS
    ]
    assert (
        len(negative_or_conflicting_claims)
        >= pack["expected_min_negative_or_conflicting"]
    )


def test_literature_claim_pack_marks_safety_signal_as_preprint() -> None:
    pack = _load_fixture()
    safety_candidate = next(
        candidate
        for candidate in pack["paper_candidates"]
        if candidate["source_id"] == "paper_crc_2026_safety_signal"
    )

    assert safety_candidate["source_quality"]["is_preprint"] is True


def test_literature_claim_pack_claims_are_contract_compatible() -> None:
    pack = _load_fixture()

    for candidate in pack["paper_candidates"]:
        for claim in candidate["extracted_claims"]:
            assert REQUIRED_EXTRACTED_CLAIM_FIELDS <= set(claim)
            assert claim["effect_direction"] in EFFECT_DIRECTIONS
            assert claim["evidence_grade"] in EVIDENCE_GRADES
            assert claim["risk_of_bias"] in RISK_OF_BIAS_LEVELS
            assert claim["local_guideline_conflict"] in LOCAL_GUIDELINE_CONFLICTS
            assert (
                claim["applicability_to_crc_context"]
                in APPLICABILITY_TO_CRC_CONTEXTS
            )

            evidence_claim = _evidence_claim_from_fixture(pack, candidate, claim)
            assert evidence_claim.source_id == candidate["source_id"]
