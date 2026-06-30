from __future__ import annotations

from copy import deepcopy
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
from src.services.literature_harness import build_literature_harness_run


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


def test_literature_claim_pack_binds_safety_signal_to_harm_claim() -> None:
    pack = _load_fixture()
    safety_candidate = next(
        candidate
        for candidate in pack["paper_candidates"]
        if candidate["source_id"] == "paper_crc_2026_safety_signal"
    )
    safety_claim = safety_candidate["extracted_claims"][0]

    assert safety_candidate["source_quality"]["is_preprint"] is True
    assert safety_claim["effect_direction"] == "harm"
    assert safety_claim["outcome"] == "serious_adverse_events"
    assert safety_claim["risk_of_bias"] == "high"
    assert safety_claim["evidence_grade"] == "observational"
    assert safety_claim["applicability_to_crc_context"] == "indirect"


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


def test_literature_harness_outputs_shadow_only_claim_cards_and_deltas() -> None:
    harness = build_literature_harness_run(
        run_id="literature_harness_test",
        claim_pack=_load_fixture(),
    )

    assert harness["run_id"] == "literature_harness_test"
    assert harness["run_level"] == "L0_shadow"
    assert harness["claim_pack_version"] == "literature_claim_pack_v0"
    assert harness["evidence_index_version"] == "rag_crc_guideline_20260620"
    assert harness["summary"]["paper_candidates"] == 3
    assert harness["summary"]["claims"] == 3
    assert harness["summary"]["negative_or_conflicting_claims"] == 2
    assert harness["summary"]["isolation_violations"] == 0
    assert harness["release_decision"] == "shadow_only"
    assert {claim["review_status"] for claim in harness["claims"]} <= {
        "candidate",
        "needs_review",
        "rejected",
    }
    assert "approved_for_clinical_rag" not in {
        claim["review_status"] for claim in harness["claims"]
    }
    assert any(delta["delta_type"] == "negative_evidence" for delta in harness["deltas"])
    assert any(delta["delta_type"] == "conflict" for delta in harness["deltas"])
    assert any(
        delta["delta_type"] == "retraction_or_quality_warning"
        for delta in harness["deltas"]
    )
    assert all(check["passed"] for check in harness["isolation_checks"])


def test_literature_harness_is_deterministic_for_same_pack() -> None:
    first = build_literature_harness_run(
        run_id="literature_harness_test",
        claim_pack=_load_fixture(),
    )
    second = build_literature_harness_run(
        run_id="literature_harness_test",
        claim_pack=_load_fixture(),
    )
    assert first == second


def test_literature_harness_blocks_when_candidate_reaches_clinical_rag() -> None:
    pack = _load_fixture()
    probe = build_literature_harness_run(run_id="probe_ids", claim_pack=pack)
    leaked_claim_id = probe["claims"][0]["claim_id"]
    pack["isolation_inputs"]["clinical_rag_claim_ids"] = [leaked_claim_id]

    harness = build_literature_harness_run(
        run_id="literature_harness_isolation_failure",
        claim_pack=pack,
    )

    assert harness["release_decision"] == "block"
    failed_checks = [
        check for check in harness["isolation_checks"] if check["passed"] is False
    ]
    assert failed_checks == [
        {
            "check_id": "no_candidate_in_clinical_rag",
            "passed": False,
            "details": {"leaked_claim_ids": [leaked_claim_id]},
        }
    ]


def test_literature_harness_rejects_retracted_sources_and_blocks() -> None:
    pack = _load_fixture()
    candidate = pack["paper_candidates"][0]
    candidate["source_id"] = "paper_crc_2026_retracted"
    candidate["source_quality"]["is_retracted"] = True

    harness = build_literature_harness_run(
        run_id="literature_harness_retracted",
        claim_pack=pack,
    )

    retracted_claims = [
        claim
        for claim in harness["claims"]
        if claim["source_id"] == "paper_crc_2026_retracted"
    ]
    assert retracted_claims
    assert {claim["review_status"] for claim in retracted_claims} == {"rejected"}
    assert harness["release_decision"] == "block"
    assert any(
        delta["severity"] == "block_promotion"
        and delta["delta_type"] == "retraction_or_quality_warning"
        for delta in harness["deltas"]
    )


def test_literature_harness_blocks_malformed_pack_and_missing_source_span() -> None:
    empty_harness = build_literature_harness_run(run_id="bad", claim_pack={})

    assert empty_harness["release_decision"] == "block"
    assert empty_harness["validation_errors"]

    non_dict_harness = build_literature_harness_run(
        run_id="bad_non_dict",
        claim_pack="not-a-dict",  # type: ignore[arg-type]
    )

    assert non_dict_harness["release_decision"] == "block"
    assert non_dict_harness["validation_errors"]

    pack = _load_fixture()
    del pack["paper_candidates"][0]["extracted_claims"][0]["source_span"]

    missing_span_harness = build_literature_harness_run(
        run_id="missing_source_span",
        claim_pack=pack,
    )

    assert missing_span_harness["release_decision"] == "block"
    assert any(
        "source_span" in error
        for error in missing_span_harness["validation_errors"]
    )


def test_literature_harness_blocks_invalid_isolation_inputs() -> None:
    pack = _load_fixture()
    pack["isolation_inputs"] = "not-a-dict"

    string_inputs_harness = build_literature_harness_run(
        run_id="bad_isolation_inputs",
        claim_pack=pack,
    )

    assert string_inputs_harness["release_decision"] == "block"
    assert string_inputs_harness["validation_errors"]

    pack = _load_fixture()
    pack["isolation_inputs"]["clinical_rag_claim_ids"] = "not-a-list"

    string_claim_ids_harness = build_literature_harness_run(
        run_id="bad_clinical_rag_claim_ids",
        claim_pack=pack,
    )

    assert string_claim_ids_harness["release_decision"] == "block"
    assert string_claim_ids_harness["validation_errors"]

    pack = _load_fixture()
    pack["isolation_inputs"]["clinical_rag_claim_ids"] = ["claim_ok", 123]

    non_string_claim_ids_harness = build_literature_harness_run(
        run_id="bad_clinical_rag_claim_id_item",
        claim_pack=pack,
    )

    assert non_string_claim_ids_harness["release_decision"] == "block"
    assert non_string_claim_ids_harness["validation_errors"]


def test_literature_harness_deduplicates_duplicate_delta_ids() -> None:
    pack = _load_fixture()
    neutral_claim = pack["paper_candidates"][1]["extracted_claims"][0]
    pack["paper_candidates"][1]["extracted_claims"].append(deepcopy(neutral_claim))

    harness = build_literature_harness_run(
        run_id="duplicate_delta_ids",
        claim_pack=pack,
    )

    delta_ids = [delta["delta_id"] for delta in harness["deltas"]]
    assert len(delta_ids) == len(set(delta_ids))


def test_literature_harness_blocks_when_negative_evidence_minimum_is_not_met() -> None:
    pack = _load_fixture()
    pack["expected_min_negative_or_conflicting"] = 99

    harness = build_literature_harness_run(
        run_id="negative_evidence_minimum_not_met",
        claim_pack=pack,
    )

    assert harness["summary"]["negative_or_conflicting_claims"] == 2
    assert harness["release_decision"] == "block"
    assert (
        any(
            "expected_min_negative_or_conflicting" in error
            for error in harness["validation_errors"]
        )
        or any(
            check["check_id"] == "negative_evidence_preserved"
            and check["passed"] is False
            for check in harness["isolation_checks"]
        )
    )


def test_literature_harness_pair_conflicts_match_intervention_and_comparator() -> None:
    pack = _load_fixture()
    unrelated_neutral_claim = deepcopy(pack["paper_candidates"][1]["extracted_claims"][0])
    unrelated_neutral_claim.update(
        {
            "claim_text": "Intervention Y did not significantly improve overall survival in a real-world colorectal cancer cohort.",
            "intervention": "Intervention Y",
            "local_guideline_conflict": "none",
            "source_span": {
                "page": 8,
                "section": "Subgroup analysis",
                "quote": "No significant survival association was observed for Intervention Y.",
            },
        }
    )
    pack["paper_candidates"][1]["extracted_claims"].append(unrelated_neutral_claim)

    harness = build_literature_harness_run(
        run_id="unrelated_therapy_conflict",
        claim_pack=pack,
    )
    benefit_claim_id = next(
        claim["claim_id"]
        for claim in harness["claims"]
        if claim["effect_direction"] == "benefit"
    )
    unrelated_neutral_claim_id = next(
        claim["claim_id"]
        for claim in harness["claims"]
        if claim["intervention"] == "Intervention Y"
    )

    unrelated_cross_pair_conflicts = [
        delta
        for delta in harness["deltas"]
        if delta["claim_id"] == unrelated_neutral_claim_id
        and delta["related_claim_id"] == benefit_claim_id
        and delta["delta_type"] == "conflict"
        and "conflicts with benefit evidence" in delta["summary"]
    ]
    assert unrelated_cross_pair_conflicts == []


from scripts.run_literature_harness import run_literature_harness


def test_literature_harness_replay_writes_shadow_report(tmp_path) -> None:
    report_path = run_literature_harness(output_root=tmp_path)

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == "literature_harness_20260630_001"
    assert report["claim_pack_version"] == "literature_claim_pack_v0"
    assert report["release_decision"] == "shadow_only"
    assert report["summary"]["claims"] == 3
    assert all(
        claim["review_status"] in {"candidate", "needs_review", "rejected"}
        for claim in report["claims"]
    )
