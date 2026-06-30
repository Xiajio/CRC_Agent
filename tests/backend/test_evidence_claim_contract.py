from __future__ import annotations

import pytest

from src.contracts.evidence_claim import (
    EvidenceClaim,
    EvidenceDelta,
    PaperCandidate,
    SourceQuality,
    SourceSpan,
    make_claim_id,
    make_delta_id,
)


def _span() -> SourceSpan:
    return SourceSpan(page=4, section="Results", quote="short extracted span")


def _quality(**overrides: bool) -> SourceQuality:
    payload = {
        "is_guideline": False,
        "is_systematic_review": False,
        "is_preprint": False,
        "is_retracted": False,
    }
    payload.update(overrides)
    return SourceQuality(**payload)


def test_make_claim_id_is_stable_and_content_addressed() -> None:
    first = make_claim_id(
        source_id="paper_crc_2026_001",
        claim_text="Intervention X improved overall survival.",
        population="adults with colorectal cancer",
        intervention="Intervention X",
        comparator="standard of care",
        outcome="overall_survival",
        effect_direction="benefit",
        source_span=_span(),
    )
    second = make_claim_id(
        source_id="paper_crc_2026_001",
        claim_text="Intervention X improved overall survival.",
        population="adults with colorectal cancer",
        intervention="Intervention X",
        comparator="standard of care",
        outcome="overall_survival",
        effect_direction="benefit",
        source_span=_span(),
    )

    assert first == second
    assert first.startswith("claim_paper_crc_2026_001_overall_survival_")
    assert len(first.rsplit("_", 1)[-1]) == 8


def test_evidence_claim_serializes_to_json_safe_dict() -> None:
    claim_id = make_claim_id(
        source_id="paper_crc_2026_001",
        claim_text="Intervention X improved outcome Y in adults with colorectal cancer.",
        population="adults with colorectal cancer",
        intervention="Intervention X",
        comparator="standard of care",
        outcome="overall_survival",
        effect_direction="benefit",
        source_span=_span(),
    )
    claim = EvidenceClaim(
        claim_id=claim_id,
        source_id="paper_crc_2026_001",
        claim_text="Intervention X improved outcome Y in adults with colorectal cancer.",
        population="adults with colorectal cancer",
        intervention="Intervention X",
        comparator="standard of care",
        outcome="overall_survival",
        effect_direction="benefit",
        effect_size="HR 0.82",
        uncertainty="95% CI 0.70-0.96",
        evidence_grade="rct",
        study_design="randomized_controlled_trial",
        sample_size=820,
        risk_of_bias="moderate",
        source_quality=_quality(),
        local_guideline_conflict="none",
        applicability_to_crc_context="partial",
        source_span=_span(),
        review_status="candidate",
        created_from="literature_claim_pack_v0",
    )

    payload = claim.to_dict()

    assert payload["claim_id"] == claim_id
    assert payload["effect_direction"] == "benefit"
    assert payload["source_quality"] == {
        "is_guideline": False,
        "is_systematic_review": False,
        "is_preprint": False,
        "is_retracted": False,
    }
    assert payload["source_span"] == {
        "page": 4,
        "section": "Results",
        "quote": "short extracted span",
    }
    assert payload["review_status"] == "candidate"


def test_evidence_claim_rejects_clinical_rag_approval_in_step10_helpers() -> None:
    with pytest.raises(ValueError, match="review_status"):
        EvidenceClaim(
            claim_id="claim_bad",
            source_id="paper_bad",
            claim_text="Unsupported promotion.",
            population="adults with colorectal cancer",
            intervention=None,
            comparator=None,
            outcome="overall_survival",
            effect_direction="benefit",
            effect_size=None,
            uncertainty=None,
            evidence_grade="rct",
            study_design="randomized_controlled_trial",
            sample_size=100,
            risk_of_bias="low",
            source_quality=_quality(),
            local_guideline_conflict="none",
            applicability_to_crc_context="partial",
            source_span=_span(),
            review_status="approved_for_clinical_rag",
            created_from="literature_claim_pack_v0",
        )


def test_evidence_claim_rejects_invalid_enum_and_sample_size() -> None:
    with pytest.raises(ValueError, match="effect_direction"):
        EvidenceClaim(
            claim_id="claim_bad_direction",
            source_id="paper_bad",
            claim_text="Invalid direction.",
            population="adults with colorectal cancer",
            intervention=None,
            comparator=None,
            outcome="overall_survival",
            effect_direction="maybe",
            effect_size=None,
            uncertainty=None,
            evidence_grade="rct",
            study_design="randomized_controlled_trial",
            sample_size=100,
            risk_of_bias="low",
            source_quality=_quality(),
            local_guideline_conflict="none",
            applicability_to_crc_context="partial",
            source_span=_span(),
            review_status="candidate",
            created_from="literature_claim_pack_v0",
        )

    with pytest.raises(ValueError, match="sample_size"):
        EvidenceClaim(
            claim_id="claim_bad_sample",
            source_id="paper_bad",
            claim_text="Invalid sample size.",
            population="adults with colorectal cancer",
            intervention=None,
            comparator=None,
            outcome="overall_survival",
            effect_direction="neutral",
            effect_size=None,
            uncertainty=None,
            evidence_grade="observational",
            study_design="cohort",
            sample_size=0,
            risk_of_bias="high",
            source_quality=_quality(),
            local_guideline_conflict="none",
            applicability_to_crc_context="partial",
            source_span=_span(),
            review_status="needs_review",
            created_from="literature_claim_pack_v0",
        )


def test_paper_candidate_and_delta_serialize() -> None:
    candidate = PaperCandidate(
        source_id="paper_crc_2026_001",
        title="Trial of Intervention X in metastatic colorectal cancer",
        url="https://example.org/paper_crc_2026_001",
        publication_year=2026,
        venue="Example Oncology Journal",
        candidate_summary="Reports improved overall survival.",
        retrieval_query="crc intervention x overall survival",
        retrieval_timestamp="2026-06-30T00:00:00+08:00",
        source_quality=_quality(),
        extracted_claims=[
            {
                "claim_text": "Intervention X improved overall survival.",
                "population": "adults with colorectal cancer",
                "outcome": "overall_survival",
                "effect_direction": "benefit",
                "evidence_grade": "rct",
                "study_design": "randomized_controlled_trial",
                "risk_of_bias": "moderate",
                "local_guideline_conflict": "none",
                "applicability_to_crc_context": "partial",
                "source_span": {"page": 4, "section": "Results"},
            }
        ],
    )
    delta_id = make_delta_id(
        claim_id="claim_1",
        related_claim_id="claim_2",
        delta_type="conflict",
    )
    delta = EvidenceDelta(
        delta_id=delta_id,
        claim_id="claim_1",
        related_claim_id="claim_2",
        delta_type="conflict",
        summary="Benefit and neutral claims disagree on overall survival.",
        severity="review_required",
        recommended_action="human_evidence_review",
    )

    assert candidate.to_dict()["source_id"] == "paper_crc_2026_001"
    assert candidate.to_dict()["extracted_claims"][0]["effect_direction"] == "benefit"
    assert delta.to_dict()["delta_type"] == "conflict"
    assert delta.to_dict()["severity"] == "review_required"
