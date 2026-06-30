from __future__ import annotations

import json
from pathlib import Path


FIXTURE_PATH = Path("tests/fixtures/literature_claim_pack_v0.json")


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


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
    assert len(candidates) == 3
    assert "benefit" in effect_directions
    assert "neutral" in effect_directions
    assert "harm" in effect_directions
    assert pack["isolation_inputs"] == {
        "clinical_rag_claim_ids": [],
        "patient_default_claim_ids": [],
        "doctor_default_claim_ids": [],
    }
