from __future__ import annotations

import pytest

from src.contracts.evidence_claim import (
    EvidenceClaim,
    EvidenceDelta,
    IsolationCheck,
    LiteratureHarnessRun,
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


def _claim(**overrides: object) -> EvidenceClaim:
    payload = {
        "claim_id": "claim_test",
        "source_id": "paper_crc_2026_001",
        "claim_text": "Intervention X changed overall survival.",
        "population": "adults with colorectal cancer",
        "intervention": "Intervention X",
        "comparator": "standard of care",
        "outcome": "overall_survival",
        "effect_direction": "benefit",
        "effect_size": "HR 0.82",
        "uncertainty": "95% CI 0.70-0.96",
        "evidence_grade": "rct",
        "study_design": "randomized_controlled_trial",
        "sample_size": 820,
        "risk_of_bias": "moderate",
        "source_quality": _quality(),
        "local_guideline_conflict": "none",
        "applicability_to_crc_context": "partial",
        "source_span": _span(),
        "review_status": "candidate",
        "created_from": "literature_claim_pack_v0",
    }
    payload.update(overrides)
    return EvidenceClaim(**payload)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> PaperCandidate:
    payload = {
        "source_id": "paper_crc_2026_001",
        "title": "Trial of Intervention X in metastatic colorectal cancer",
        "url": "https://example.org/paper_crc_2026_001",
        "publication_year": 2026,
        "venue": "Example Oncology Journal",
        "candidate_summary": "Reports changed overall survival.",
        "retrieval_query": "crc intervention x overall survival",
        "retrieval_timestamp": "2026-06-30T00:00:00+08:00",
        "source_quality": _quality(),
        "extracted_claims": [],
    }
    payload.update(overrides)
    return PaperCandidate(**payload)  # type: ignore[arg-type]


def _run(**overrides: object) -> LiteratureHarnessRun:
    payload = {
        "run_id": "lit_run_20260630_001",
        "run_level": "L0_literature_contract",
        "claim_pack_version": "literature_claim_pack_v0",
        "evidence_index_version": "rag_crc_guideline_20260620",
        "summary": {"claim_count": 0},
        "claims": [],
        "deltas": [],
        "isolation_checks": [],
        "release_decision": "shadow_only",
        "validation_errors": [],
    }
    payload.update(overrides)
    return LiteratureHarnessRun(**payload)  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    "review_status",
    ["reviewed", "approved_for_project_evidence_pool"],
)
def test_evidence_claim_rejects_post_candidate_review_statuses(
    review_status: str,
) -> None:
    with pytest.raises(ValueError, match="review_status"):
        EvidenceClaim(
            claim_id=f"claim_bad_{review_status}",
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
            review_status=review_status,
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


def test_literature_harness_run_and_isolation_check_serialize_release_gate() -> None:
    claim = _claim()
    delta = EvidenceDelta(
        delta_id="delta_claim_1_claim_2_conflict",
        claim_id=claim.claim_id,
        related_claim_id=None,
        delta_type="conflict",
        summary="Benefit and neutral claims disagree on overall survival.",
        severity="review_required",
        recommended_action="human_evidence_review",
    )
    isolation_check = IsolationCheck(
        check_id="iso_no_clinical_rag_promotion",
        passed=True,
        details={
            "zone": "external_literature_search",
            "forbidden_behavior": "clinical_rag_ingest",
            "promotion_gate": "human_evidence_review",
        },
    )
    run = LiteratureHarnessRun(
        run_id="lit_run_20260630_001",
        run_level="L0_literature_contract",
        claim_pack_version="literature_claim_pack_v0",
        evidence_index_version="rag_crc_guideline_20260620",
        summary={"claim_count": 1, "delta_count": 1},
        claims=[claim],
        deltas=[delta],
        isolation_checks=[isolation_check],
        release_decision="shadow_only",
        validation_errors=[],
    )

    payload = run.to_dict()

    assert payload["run_level"] == "L0_literature_contract"
    assert payload["claim_pack_version"] == "literature_claim_pack_v0"
    assert payload["evidence_index_version"] == "rag_crc_guideline_20260620"
    assert payload["release_decision"] == "shadow_only"
    assert payload["validation_errors"] == []
    assert payload["isolation_checks"] == [
        {
            "check_id": "iso_no_clinical_rag_promotion",
            "passed": True,
            "details": {
                "zone": "external_literature_search",
                "forbidden_behavior": "clinical_rag_ingest",
                "promotion_gate": "human_evidence_review",
            },
        }
    ]


def test_literature_harness_run_rejects_invalid_release_decision() -> None:
    with pytest.raises(ValueError, match="release_decision"):
        LiteratureHarnessRun(
            run_id="lit_run_bad_release_decision",
            run_level="L0_literature_contract",
            claim_pack_version="literature_claim_pack_v0",
            evidence_index_version="rag_crc_guideline_20260620",
            summary={},
            claims=[],
            deltas=[],
            isolation_checks=[],
            release_decision="pass",
            validation_errors=[],
        )


def test_numeric_fields_reject_bool_values() -> None:
    with pytest.raises(ValueError, match="source_span.page"):
        SourceSpan(page=True)

    with pytest.raises(ValueError, match="publication_year"):
        _candidate(publication_year=True)

    with pytest.raises(ValueError, match="sample_size"):
        _claim(sample_size=True)


def test_step10_contract_enums_accept_plan_values() -> None:
    inconclusive_claim = _claim(
        claim_id="claim_inconclusive_case_series",
        effect_direction="inconclusive",
        evidence_grade="case_series",
    )
    conflicting_claim = _claim(
        claim_id="claim_conflicting_expert_opinion",
        effect_direction="conflicting",
        evidence_grade="expert_opinion",
    )
    deltas = [
        EvidenceDelta(
            delta_id="delta_new_claim",
            claim_id=inconclusive_claim.claim_id,
            related_claim_id=None,
            delta_type="new_claim",
            summary="A new claim was extracted.",
            severity="info",
            recommended_action="human_evidence_review",
        ),
        EvidenceDelta(
            delta_id="delta_supporting",
            claim_id=inconclusive_claim.claim_id,
            related_claim_id=conflicting_claim.claim_id,
            delta_type="supporting",
            summary="A related claim supports this signal.",
            severity="review_required",
            recommended_action="compare_claims",
        ),
        EvidenceDelta(
            delta_id="delta_negative_evidence",
            claim_id=inconclusive_claim.claim_id,
            related_claim_id=conflicting_claim.claim_id,
            delta_type="negative_evidence",
            summary="Negative evidence should remain visible.",
            severity="review_required",
            recommended_action="compare_claims",
        ),
        EvidenceDelta(
            delta_id="delta_retraction_or_quality_warning",
            claim_id=conflicting_claim.claim_id,
            related_claim_id=None,
            delta_type="retraction_or_quality_warning",
            summary="A quality warning blocks promotion.",
            severity="block_promotion",
            recommended_action="remove_from_promotion_queue",
        ),
    ]

    assert inconclusive_claim.to_dict()["effect_direction"] == "inconclusive"
    assert inconclusive_claim.to_dict()["evidence_grade"] == "case_series"
    assert conflicting_claim.to_dict()["effect_direction"] == "conflicting"
    assert conflicting_claim.to_dict()["evidence_grade"] == "expert_opinion"
    assert [delta.to_dict()["delta_type"] for delta in deltas] == [
        "new_claim",
        "supporting",
        "negative_evidence",
        "retraction_or_quality_warning",
    ]
    assert deltas[0].to_dict()["related_claim_id"] is None
    assert deltas[-1].to_dict()["severity"] == "block_promotion"


@pytest.mark.parametrize("bad_value", [{}, "not-a-list", None])
def test_paper_candidate_rejects_non_list_extracted_claims(
    bad_value: object,
) -> None:
    with pytest.raises(TypeError, match="extracted_claims"):
        _candidate(extracted_claims=bad_value)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("claims", {}),
        ("claims", "not-a-list"),
        ("claims", None),
        ("deltas", {}),
        ("deltas", "not-a-list"),
        ("deltas", None),
        ("isolation_checks", {}),
        ("isolation_checks", "not-a-list"),
        ("isolation_checks", None),
        ("validation_errors", {}),
        ("validation_errors", "not-a-list"),
        ("validation_errors", None),
    ],
)
def test_literature_harness_run_rejects_non_list_containers(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _run(**{field_name: bad_value})


def test_contract_rejects_invalid_text_field_types() -> None:
    with pytest.raises(TypeError, match="section"):
        SourceSpan(section=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="candidate_summary"):
        _candidate(candidate_summary=123)

    with pytest.raises(TypeError, match="intervention"):
        _claim(intervention=123)

    with pytest.raises(TypeError, match="related_claim_id"):
        EvidenceDelta(
            delta_id="delta_bad_related_claim_id",
            claim_id="claim_1",
            related_claim_id=123,  # type: ignore[arg-type]
            delta_type="conflict",
            summary="Invalid related claim id type.",
            severity="review_required",
            recommended_action="human_evidence_review",
        )


@pytest.mark.parametrize("bad_details", [[], "not-a-dict", None])
def test_isolation_check_rejects_non_dict_details(bad_details: object) -> None:
    with pytest.raises(TypeError, match="details"):
        IsolationCheck(
            check_id="iso_bad_details",
            passed=False,
            details=bad_details,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_summary", [[], "not-a-dict", None])
def test_literature_harness_run_rejects_non_dict_summary(
    bad_summary: object,
) -> None:
    with pytest.raises(TypeError, match="summary"):
        _run(summary=bad_summary)


def test_literature_harness_run_rejects_non_string_validation_errors() -> None:
    with pytest.raises(TypeError, match="validation_errors"):
        _run(validation_errors=[{"message": "must be a string"}])
