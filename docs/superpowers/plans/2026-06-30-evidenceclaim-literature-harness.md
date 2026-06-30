# EvidenceClaim Literature Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Step 10 shadow-only EvidenceClaim literature harness so fixed literature candidates become deterministic claim-level evidence cards, deltas, and isolation-gated reports without entering clinical paths.

**Architecture:** Add a pure evidence-claim contract first, then a fixture-backed harness service, then a replay script that writes a committed report. Keep all outputs shadow-only; do not call live search, models, RAG indexes, patient APIs, or doctor default UI.

**Tech Stack:** Python 3.10+, dataclasses, pytest, JSON fixtures, existing `reports/*` release evidence pattern.

---

## File Structure

Create:

- `src/contracts/evidence_claim.py`: dataclass contracts for `SourceQuality`, `SourceSpan`, `PaperCandidate`, `EvidenceClaim`, `EvidenceDelta`, `IsolationCheck`, `LiteratureHarnessRun`, plus deterministic ID helpers and validation.
- `src/services/literature_harness.py`: pure fixture-to-report harness builder with claim extraction, delta creation, isolation checks, and release decision.
- `tests/backend/test_evidence_claim_contract.py`: contract serialization, validation, and stable ID tests.
- `tests/backend/test_literature_harness.py`: fixture replay, negative/conflicting evidence preservation, deterministic output, isolation failure, and report writing tests.
- `tests/fixtures/literature_claim_pack_v0.json`: fixed local literature candidate pack with benefit, neutral/negative, and conflicting/quality-warning claims.
- `scripts/run_literature_harness.py`: local replay script that writes `reports/literature/literature_harness_20260630_001.json`.
- `reports/literature/README.md`: explains the report directory and shadow-only boundary.
- `reports/literature/literature_harness_20260630_001.json`: generated report from the replay script.

Do not create or modify:

- `CRC-client/`
- `src/services/web_search_service.py`
- `src/tools/web_search_tools.py`
- `src/tools/manifest.py`
- `config/safety_policy.yaml`
- `chroma_db/`
- `bm25_index/`
- Patient workspace UI files
- Doctor default flow UI files
- RAG ingest files

No frontend file is required for this Step 10 implementation. Step 11 can consume the JSON report and add Agent Admin display later.

## Execution Setup

- [ ] **Step 0.1: Confirm the clean starting point**

Run:

```powershell
git status --short --branch
```

Expected: branch is `main`; the only expected pre-existing state may be documentation commits already ahead of `origin/main`. Do not edit ignored `CRC-client/`.

- [ ] **Step 0.2: Create an isolated worktree before implementation**

Use the worktree skill at execution time. Recommended command:

```powershell
git worktree add .worktrees/step10-evidenceclaim-literature-harness -b step10-evidenceclaim-literature-harness main
```

Expected: a new worktree exists at `.worktrees/step10-evidenceclaim-literature-harness`.

- [ ] **Step 0.3: Re-open the spec inside the worktree**

Run:

```powershell
Get-Content -Raw docs\superpowers\specs\2026-06-30-evidenceclaim-literature-harness-design.md
```

Expected: the spec confirms Step 10 is shadow-only and excludes live search, model calls, RAG ingest, patient UI, and doctor default UI.

## Task 1: EvidenceClaim Contract

**Files:**

- Create: `src/contracts/evidence_claim.py`
- Create: `tests/backend/test_evidence_claim_contract.py`

- [ ] **Step 1.1: Write the failing contract tests**

Create `tests/backend/test_evidence_claim_contract.py`:

```python
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
```

- [ ] **Step 1.2: Run the contract tests to verify they fail**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.contracts.evidence_claim'`.

- [ ] **Step 1.3: Implement the evidence claim contract**

Create `src/contracts/evidence_claim.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Literal, TypeAlias


EffectDirection = Literal["benefit", "harm", "neutral", "inconclusive", "conflicting"]
EvidenceGrade = Literal[
    "guideline",
    "systematic_review",
    "rct",
    "observational",
    "case_series",
    "preclinical",
    "expert_opinion",
    "unknown",
]
ReviewStatus = Literal["candidate", "needs_review", "rejected"]
DeltaType = Literal[
    "new_claim",
    "supporting",
    "conflict",
    "negative_evidence",
    "safety_signal",
    "retraction_or_quality_warning",
]
DeltaSeverity = Literal["info", "review_required", "block_promotion"]
ReleaseDecision = Literal["block", "shadow_only", "candidate_ready_for_human_review"]
JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

ALLOWED_EFFECT_DIRECTIONS = {"benefit", "harm", "neutral", "inconclusive", "conflicting"}
ALLOWED_EVIDENCE_GRADES = {
    "guideline",
    "systematic_review",
    "rct",
    "observational",
    "case_series",
    "preclinical",
    "expert_opinion",
    "unknown",
}
ALLOWED_REVIEW_STATUSES = {"candidate", "needs_review", "rejected"}
ALLOWED_DELTA_TYPES = {
    "new_claim",
    "supporting",
    "conflict",
    "negative_evidence",
    "safety_signal",
    "retraction_or_quality_warning",
}
ALLOWED_DELTA_SEVERITIES = {"info", "review_required", "block_promotion"}
ALLOWED_RELEASE_DECISIONS = {"block", "shadow_only", "candidate_ready_for_human_review"}


@dataclass(frozen=True)
class SourceQuality:
    is_guideline: bool
    is_systematic_review: bool
    is_preprint: bool
    is_retracted: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SourceQuality":
        payload = payload or {}
        return cls(
            is_guideline=bool(payload.get("is_guideline", False)),
            is_systematic_review=bool(payload.get("is_systematic_review", False)),
            is_preprint=bool(payload.get("is_preprint", False)),
            is_retracted=bool(payload.get("is_retracted", False)),
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "is_guideline": self.is_guideline,
            "is_systematic_review": self.is_systematic_review,
            "is_preprint": self.is_preprint,
            "is_retracted": self.is_retracted,
        }


@dataclass(frozen=True)
class SourceSpan:
    page: int | None = None
    section: str | None = None
    quote: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SourceSpan":
        payload = payload or {}
        page = payload.get("page")
        return cls(
            page=int(page) if isinstance(page, int) else None,
            section=str(payload["section"]) if payload.get("section") else None,
            quote=str(payload["quote"]) if payload.get("quote") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "page": self.page,
                "section": self.section,
                "quote": self.quote,
            }
        )


@dataclass(frozen=True)
class PaperCandidate:
    source_id: str
    title: str
    url: str
    publication_year: int | None
    venue: str | None
    candidate_summary: str
    retrieval_query: str
    retrieval_timestamp: str
    source_quality: SourceQuality
    extracted_claims: list[dict[str, JsonValue]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperCandidate":
        _require_string(payload, "source_id")
        _require_string(payload, "title")
        _require_string(payload, "url")
        _require_string(payload, "candidate_summary")
        _require_string(payload, "retrieval_query")
        _require_string(payload, "retrieval_timestamp")
        extracted_claims = payload.get("extracted_claims", [])
        if not isinstance(extracted_claims, list):
            raise ValueError("extracted_claims must be a list")
        for item in extracted_claims:
            _validate_json_value(item)
        publication_year = payload.get("publication_year")
        return cls(
            source_id=str(payload["source_id"]),
            title=str(payload["title"]),
            url=str(payload["url"]),
            publication_year=int(publication_year)
            if isinstance(publication_year, int)
            else None,
            venue=str(payload["venue"]) if payload.get("venue") else None,
            candidate_summary=str(payload["candidate_summary"]),
            retrieval_query=str(payload["retrieval_query"]),
            retrieval_timestamp=str(payload["retrieval_timestamp"]),
            source_quality=SourceQuality.from_dict(payload.get("source_quality")),
            extracted_claims=list(extracted_claims),
        )

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "source_id": self.source_id,
                "title": self.title,
                "url": self.url,
                "publication_year": self.publication_year,
                "venue": self.venue,
                "candidate_summary": self.candidate_summary,
                "retrieval_query": self.retrieval_query,
                "retrieval_timestamp": self.retrieval_timestamp,
                "source_quality": self.source_quality.to_dict(),
                "extracted_claims": self.extracted_claims,
            }
        )


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    source_id: str
    claim_text: str
    population: str
    intervention: str | None
    comparator: str | None
    outcome: str
    effect_direction: str
    effect_size: str | None
    uncertainty: str | None
    evidence_grade: str
    study_design: str
    sample_size: int | None
    risk_of_bias: str
    source_quality: SourceQuality
    local_guideline_conflict: str
    applicability_to_crc_context: str
    source_span: SourceSpan
    review_status: str
    created_from: str

    def __post_init__(self) -> None:
        _require_non_empty_value(self.claim_id, "claim_id")
        _require_non_empty_value(self.source_id, "source_id")
        _require_non_empty_value(self.claim_text, "claim_text")
        _require_non_empty_value(self.population, "population")
        _require_non_empty_value(self.outcome, "outcome")
        _validate_choice("effect_direction", self.effect_direction, ALLOWED_EFFECT_DIRECTIONS)
        _validate_choice("evidence_grade", self.evidence_grade, ALLOWED_EVIDENCE_GRADES)
        _validate_choice("review_status", self.review_status, ALLOWED_REVIEW_STATUSES)
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "claim_id": self.claim_id,
                "source_id": self.source_id,
                "claim_text": self.claim_text,
                "population": self.population,
                "intervention": self.intervention,
                "comparator": self.comparator,
                "outcome": self.outcome,
                "effect_direction": self.effect_direction,
                "effect_size": self.effect_size,
                "uncertainty": self.uncertainty,
                "evidence_grade": self.evidence_grade,
                "study_design": self.study_design,
                "sample_size": self.sample_size,
                "risk_of_bias": self.risk_of_bias,
                "source_quality": self.source_quality.to_dict(),
                "local_guideline_conflict": self.local_guideline_conflict,
                "applicability_to_crc_context": self.applicability_to_crc_context,
                "source_span": self.source_span.to_dict(),
                "review_status": self.review_status,
                "created_from": self.created_from,
            }
        )


@dataclass(frozen=True)
class EvidenceDelta:
    delta_id: str
    claim_id: str
    related_claim_id: str | None
    delta_type: str
    summary: str
    severity: str
    recommended_action: str

    def __post_init__(self) -> None:
        _validate_choice("delta_type", self.delta_type, ALLOWED_DELTA_TYPES)
        _validate_choice("severity", self.severity, ALLOWED_DELTA_SEVERITIES)

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "delta_id": self.delta_id,
                "claim_id": self.claim_id,
                "related_claim_id": self.related_claim_id,
                "delta_type": self.delta_type,
                "summary": self.summary,
                "severity": self.severity,
                "recommended_action": self.recommended_action,
            }
        )


@dataclass(frozen=True)
class IsolationCheck:
    check_id: str
    passed: bool
    details: dict[str, JsonValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass(frozen=True)
class LiteratureHarnessRun:
    run_id: str
    run_level: str
    claim_pack_version: str
    evidence_index_version: str
    summary: dict[str, JsonValue]
    claims: list[EvidenceClaim]
    deltas: list[EvidenceDelta]
    isolation_checks: list[IsolationCheck]
    release_decision: str
    validation_errors: list[str]

    def __post_init__(self) -> None:
        _validate_choice("release_decision", self.release_decision, ALLOWED_RELEASE_DECISIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_level": self.run_level,
            "claim_pack_version": self.claim_pack_version,
            "evidence_index_version": self.evidence_index_version,
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "deltas": [delta.to_dict() for delta in self.deltas],
            "isolation_checks": [check.to_dict() for check in self.isolation_checks],
            "release_decision": self.release_decision,
            "validation_errors": self.validation_errors,
        }


def make_claim_id(
    *,
    source_id: str,
    claim_text: str,
    population: str,
    intervention: str | None,
    comparator: str | None,
    outcome: str,
    effect_direction: str,
    source_span: SourceSpan,
) -> str:
    fingerprint = {
        "source_id": source_id,
        "claim_text": " ".join(claim_text.split()),
        "population": " ".join(population.split()),
        "intervention": intervention,
        "comparator": comparator,
        "outcome": outcome,
        "effect_direction": effect_direction,
        "source_span": source_span.to_dict(),
    }
    digest = _stable_hash(fingerprint)
    return f"claim_{_slug(source_id)}_{_slug(outcome)}_{digest}"


def make_delta_id(
    *,
    claim_id: str,
    related_claim_id: str | None,
    delta_type: str,
) -> str:
    digest = _stable_hash(
        {
            "claim_id": claim_id,
            "related_claim_id": related_claim_id,
            "delta_type": delta_type,
        }
    )
    related = related_claim_id or "none"
    return f"delta_{_slug(claim_id)}_{_slug(related)}_{_slug(delta_type)}_{digest}"


def _stable_hash(payload: dict[str, Any]) -> str:
    stable_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(stable_json.encode("utf-8")).hexdigest()[:8]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


def _validate_choice(field: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")


def _require_string(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")


def _require_non_empty_value(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")


def _omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _validate_json_value(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("payload must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("payload dict keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError("payload must be JSON-safe")


__all__ = [
    "DeltaSeverity",
    "DeltaType",
    "EffectDirection",
    "EvidenceClaim",
    "EvidenceDelta",
    "EvidenceGrade",
    "IsolationCheck",
    "LiteratureHarnessRun",
    "PaperCandidate",
    "ReleaseDecision",
    "ReviewStatus",
    "SourceQuality",
    "SourceSpan",
    "make_claim_id",
    "make_delta_id",
]
```

- [ ] **Step 1.4: Run the contract tests to verify they pass**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py -q
```

Expected: `5 passed`.

- [ ] **Step 1.5: Commit Task 1**

Run:

```powershell
git add src/contracts/evidence_claim.py tests/backend/test_evidence_claim_contract.py
git commit -m "feat: add evidence claim contract"
```

Expected: commit succeeds with only these two files.

## Task 2: Literature Claim Pack Fixture

**Files:**

- Create: `tests/fixtures/literature_claim_pack_v0.json`
- Modify: `tests/backend/test_literature_harness.py`

- [ ] **Step 2.1: Create the fixed literature claim pack**

Create `tests/fixtures/literature_claim_pack_v0.json`:

```json
{
  "claim_pack_id": "literature_claim_pack_v0",
  "evidence_index_version": "rag_crc_guideline_20260620",
  "expected_min_negative_or_conflicting": 2,
  "paper_candidates": [
    {
      "source_id": "paper_crc_2026_benefit",
      "title": "Trial of Intervention X in metastatic colorectal cancer",
      "url": "https://example.org/paper_crc_2026_benefit",
      "publication_year": 2026,
      "venue": "Example Oncology Journal",
      "candidate_summary": "The trial reports improved overall survival with Intervention X.",
      "retrieval_query": "metastatic colorectal cancer intervention x overall survival",
      "retrieval_timestamp": "2026-06-30T00:00:00+08:00",
      "source_quality": {
        "is_guideline": false,
        "is_systematic_review": false,
        "is_preprint": false,
        "is_retracted": false
      },
      "extracted_claims": [
        {
          "claim_text": "Intervention X improved overall survival compared with standard of care in adults with metastatic colorectal cancer.",
          "population": "adults with metastatic colorectal cancer",
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
          "local_guideline_conflict": "none",
          "applicability_to_crc_context": "partial",
          "source_span": {
            "page": 4,
            "section": "Results",
            "quote": "Overall survival favored Intervention X."
          }
        }
      ]
    },
    {
      "source_id": "paper_crc_2026_neutral",
      "title": "Real-world comparison of Intervention X in colorectal cancer",
      "url": "https://example.org/paper_crc_2026_neutral",
      "publication_year": 2026,
      "venue": "Example Registry Reports",
      "candidate_summary": "The registry analysis reports no statistically significant survival improvement.",
      "retrieval_query": "colorectal cancer intervention x real world survival neutral",
      "retrieval_timestamp": "2026-06-30T00:00:00+08:00",
      "source_quality": {
        "is_guideline": false,
        "is_systematic_review": false,
        "is_preprint": false,
        "is_retracted": false
      },
      "extracted_claims": [
        {
          "claim_text": "Intervention X did not significantly improve overall survival in a real-world colorectal cancer cohort.",
          "population": "adults with metastatic colorectal cancer",
          "intervention": "Intervention X",
          "comparator": "standard of care",
          "outcome": "overall_survival",
          "effect_direction": "neutral",
          "effect_size": "HR 0.98",
          "uncertainty": "95% CI 0.84-1.14",
          "evidence_grade": "observational",
          "study_design": "retrospective_cohort",
          "sample_size": 410,
          "risk_of_bias": "high",
          "local_guideline_conflict": "possible",
          "applicability_to_crc_context": "partial",
          "source_span": {
            "page": 7,
            "section": "Adjusted analysis",
            "quote": "No significant survival association was observed."
          }
        }
      ]
    },
    {
      "source_id": "paper_crc_2026_safety_signal",
      "title": "Early safety signal for Intervention X combinations",
      "url": "https://example.org/paper_crc_2026_safety_signal",
      "publication_year": 2026,
      "venue": "Preprint Oncology Archive",
      "candidate_summary": "A preprint reports increased serious adverse events in a small cohort.",
      "retrieval_query": "intervention x colorectal cancer adverse events preprint",
      "retrieval_timestamp": "2026-06-30T00:00:00+08:00",
      "source_quality": {
        "is_guideline": false,
        "is_systematic_review": false,
        "is_preprint": true,
        "is_retracted": false
      },
      "extracted_claims": [
        {
          "claim_text": "Intervention X combinations were associated with increased grade 3 or higher adverse events in a small colorectal cancer cohort.",
          "population": "adults with metastatic colorectal cancer",
          "intervention": "Intervention X combination therapy",
          "comparator": "standard of care",
          "outcome": "serious_adverse_events",
          "effect_direction": "harm",
          "effect_size": "RR 1.45",
          "uncertainty": "95% CI 1.05-2.10",
          "evidence_grade": "observational",
          "study_design": "small_cohort_preprint",
          "sample_size": 96,
          "risk_of_bias": "high",
          "local_guideline_conflict": "possible",
          "applicability_to_crc_context": "limited",
          "source_span": {
            "page": 5,
            "section": "Safety",
            "quote": "Grade 3 or higher events were more frequent."
          }
        }
      ]
    }
  ],
  "isolation_inputs": {
    "clinical_rag_claim_ids": [],
    "patient_default_claim_ids": [],
    "doctor_default_claim_ids": []
  }
}
```

- [ ] **Step 2.2: Write the failing fixture test**

Create `tests/backend/test_literature_harness.py`:

```python
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
```

- [ ] **Step 2.3: Run the fixture test**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_literature_harness.py -q
```

Expected: `1 passed`.

- [ ] **Step 2.4: Commit Task 2**

Run:

```powershell
git add tests/fixtures/literature_claim_pack_v0.json tests/backend/test_literature_harness.py
git commit -m "test: add literature claim pack fixture"
```

Expected: commit succeeds with only the fixture and harness test file.

## Task 3: Literature Harness Service

**Files:**

- Create: `src/services/literature_harness.py`
- Modify: `tests/backend/test_literature_harness.py`

- [ ] **Step 3.1: Add failing harness behavior tests**

Append to `tests/backend/test_literature_harness.py`:

```python
from src.services.literature_harness import build_literature_harness_run


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
    assert any(delta["delta_type"] == "retraction_or_quality_warning" for delta in harness["deltas"])
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
    probe = build_literature_harness_run(
        run_id="probe_ids",
        claim_pack=pack,
    )
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
```

- [ ] **Step 3.2: Run the harness tests to verify they fail**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_literature_harness.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.literature_harness'`.

- [ ] **Step 3.3: Implement the literature harness service**

Create `src/services/literature_harness.py`:

```python
from __future__ import annotations

from itertools import combinations
from typing import Any

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


RUN_LEVEL = "L0_shadow"
DEFAULT_EVIDENCE_INDEX_VERSION = "rag_crc_guideline_20260620"
NEGATIVE_OR_CONFLICTING_DIRECTIONS = {"harm", "neutral", "inconclusive", "conflicting"}


def build_literature_harness_run(
    *,
    run_id: str,
    claim_pack: dict[str, Any],
) -> dict[str, Any]:
    candidates, validation_errors = _load_candidates(claim_pack)
    claims, claim_errors = _claims_from_candidates(
        candidates,
        created_from=str(claim_pack.get("claim_pack_id", "unknown_claim_pack")),
    )
    validation_errors.extend(claim_errors)
    deltas = _build_deltas(claims)
    isolation_checks = _build_isolation_checks(claim_pack, claims)
    negative_count = _negative_or_conflicting_count(claims)
    isolation_violations = len([check for check in isolation_checks if not check.passed])
    release_decision = _release_decision(
        validation_errors=validation_errors,
        deltas=deltas,
        isolation_checks=isolation_checks,
    )
    run = LiteratureHarnessRun(
        run_id=run_id,
        run_level=RUN_LEVEL,
        claim_pack_version=str(claim_pack.get("claim_pack_id", "unknown_claim_pack")),
        evidence_index_version=str(
            claim_pack.get("evidence_index_version", DEFAULT_EVIDENCE_INDEX_VERSION)
        ),
        summary={
            "paper_candidates": len(candidates),
            "claims": len(claims),
            "deltas": len(deltas),
            "negative_or_conflicting_claims": negative_count,
            "isolation_violations": isolation_violations,
        },
        claims=claims,
        deltas=deltas,
        isolation_checks=isolation_checks,
        release_decision=release_decision,
        validation_errors=validation_errors,
    )
    return run.to_dict()


def _load_candidates(
    claim_pack: dict[str, Any],
) -> tuple[list[PaperCandidate], list[str]]:
    candidates: list[PaperCandidate] = []
    errors: list[str] = []
    for index, payload in enumerate(claim_pack.get("paper_candidates", [])):
        try:
            candidates.append(PaperCandidate.from_dict(payload))
        except (TypeError, ValueError) as exc:
            errors.append(f"paper_candidates[{index}]: {exc}")
    return candidates, errors


def _claims_from_candidates(
    candidates: list[PaperCandidate],
    *,
    created_from: str,
) -> tuple[list[EvidenceClaim], list[str]]:
    claims: list[EvidenceClaim] = []
    errors: list[str] = []
    for candidate in candidates:
        for index, claim_payload in enumerate(candidate.extracted_claims):
            try:
                claims.append(
                    _claim_from_candidate(
                        candidate,
                        claim_payload,
                        created_from=created_from,
                    )
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{candidate.source_id}.extracted_claims[{index}]: {exc}")
    return claims, errors


def _claim_from_candidate(
    candidate: PaperCandidate,
    claim_payload: dict[str, Any],
    *,
    created_from: str,
) -> EvidenceClaim:
    claim_text = _required_text(claim_payload, "claim_text")
    population = _required_text(claim_payload, "population")
    intervention = _optional_text(claim_payload, "intervention")
    comparator = _optional_text(claim_payload, "comparator")
    outcome = _required_text(claim_payload, "outcome")
    effect_direction = _required_text(claim_payload, "effect_direction")
    source_span = SourceSpan.from_dict(_mapping(claim_payload.get("source_span")))
    source_quality = _source_quality(candidate, claim_payload)
    review_status = _review_status(
        source_quality=source_quality,
        risk_of_bias=_required_text(claim_payload, "risk_of_bias"),
        local_guideline_conflict=_required_text(
            claim_payload,
            "local_guideline_conflict",
        ),
    )
    return EvidenceClaim(
        claim_id=make_claim_id(
            source_id=candidate.source_id,
            claim_text=claim_text,
            population=population,
            intervention=intervention,
            comparator=comparator,
            outcome=outcome,
            effect_direction=effect_direction,
            source_span=source_span,
        ),
        source_id=candidate.source_id,
        claim_text=claim_text,
        population=population,
        intervention=intervention,
        comparator=comparator,
        outcome=outcome,
        effect_direction=effect_direction,
        effect_size=_optional_text(claim_payload, "effect_size"),
        uncertainty=_optional_text(claim_payload, "uncertainty"),
        evidence_grade=_required_text(claim_payload, "evidence_grade"),
        study_design=_required_text(claim_payload, "study_design"),
        sample_size=_optional_positive_int(claim_payload, "sample_size"),
        risk_of_bias=_required_text(claim_payload, "risk_of_bias"),
        source_quality=source_quality,
        local_guideline_conflict=_required_text(
            claim_payload,
            "local_guideline_conflict",
        ),
        applicability_to_crc_context=_required_text(
            claim_payload,
            "applicability_to_crc_context",
        ),
        source_span=source_span,
        review_status=review_status,
        created_from=created_from,
    )


def _build_deltas(claims: list[EvidenceClaim]) -> list[EvidenceDelta]:
    deltas: list[EvidenceDelta] = []
    for claim in claims:
        deltas.extend(_single_claim_deltas(claim))
    deltas.extend(_cross_claim_conflict_deltas(claims))
    deduped: dict[str, EvidenceDelta] = {}
    for delta in deltas:
        deduped[delta.delta_id] = delta
    return list(deduped.values())


def _single_claim_deltas(claim: EvidenceClaim) -> list[EvidenceDelta]:
    deltas: list[EvidenceDelta] = []
    if claim.effect_direction in NEGATIVE_OR_CONFLICTING_DIRECTIONS:
        deltas.append(
            _delta(
                claim=claim,
                related_claim_id=None,
                delta_type="negative_evidence",
                severity="review_required",
                summary=f"{claim.claim_id} reports {claim.effect_direction} evidence for {claim.outcome}.",
            )
        )
    if claim.local_guideline_conflict != "none":
        deltas.append(
            _delta(
                claim=claim,
                related_claim_id=None,
                delta_type="conflict",
                severity="review_required",
                summary=f"{claim.claim_id} has local guideline conflict: {claim.local_guideline_conflict}.",
            )
        )
    if claim.effect_direction == "harm":
        deltas.append(
            _delta(
                claim=claim,
                related_claim_id=None,
                delta_type="safety_signal",
                severity="review_required",
                summary=f"{claim.claim_id} reports a harm signal for {claim.outcome}.",
            )
        )
    if claim.source_quality.is_retracted:
        deltas.append(
            _delta(
                claim=claim,
                related_claim_id=None,
                delta_type="retraction_or_quality_warning",
                severity="block_promotion",
                summary=f"{claim.claim_id} comes from a retracted source.",
            )
        )
    elif claim.source_quality.is_preprint or claim.risk_of_bias == "high":
        deltas.append(
            _delta(
                claim=claim,
                related_claim_id=None,
                delta_type="retraction_or_quality_warning",
                severity="review_required",
                summary=f"{claim.claim_id} requires quality review before promotion.",
            )
        )
    return deltas


def _cross_claim_conflict_deltas(claims: list[EvidenceClaim]) -> list[EvidenceDelta]:
    deltas: list[EvidenceDelta] = []
    for left, right in combinations(claims, 2):
        if left.population != right.population or left.outcome != right.outcome:
            continue
        if left.effect_direction == right.effect_direction:
            continue
        if "benefit" not in {left.effect_direction, right.effect_direction}:
            continue
        deltas.append(
            _delta(
                claim=left,
                related_claim_id=right.claim_id,
                delta_type="conflict",
                severity="review_required",
                summary=(
                    f"{left.claim_id} and {right.claim_id} disagree on "
                    f"{left.outcome} for {left.population}."
                ),
            )
        )
    return deltas


def _delta(
    *,
    claim: EvidenceClaim,
    related_claim_id: str | None,
    delta_type: str,
    severity: str,
    summary: str,
) -> EvidenceDelta:
    return EvidenceDelta(
        delta_id=make_delta_id(
            claim_id=claim.claim_id,
            related_claim_id=related_claim_id,
            delta_type=delta_type,
        ),
        claim_id=claim.claim_id,
        related_claim_id=related_claim_id,
        delta_type=delta_type,
        summary=summary,
        severity=severity,
        recommended_action="human_evidence_review",
    )


def _build_isolation_checks(
    claim_pack: dict[str, Any],
    claims: list[EvidenceClaim],
) -> list[IsolationCheck]:
    claim_ids = {claim.claim_id for claim in claims}
    isolation_inputs = claim_pack.get("isolation_inputs", {})
    checks = [
        _no_leak_check(
            check_id="no_candidate_in_clinical_rag",
            claim_ids=claim_ids,
            observed_ids=_string_set(isolation_inputs.get("clinical_rag_claim_ids")),
        ),
        _no_leak_check(
            check_id="no_candidate_in_patient_default_path",
            claim_ids=claim_ids,
            observed_ids=_string_set(isolation_inputs.get("patient_default_claim_ids")),
        ),
        _no_leak_check(
            check_id="no_candidate_in_doctor_default_path",
            claim_ids=claim_ids,
            observed_ids=_string_set(isolation_inputs.get("doctor_default_claim_ids")),
        ),
    ]
    expected_negative = int(claim_pack.get("expected_min_negative_or_conflicting", 0))
    actual_negative = _negative_or_conflicting_count(claims)
    checks.append(
        IsolationCheck(
            check_id="negative_evidence_preserved",
            passed=actual_negative >= expected_negative,
            details={
                "expected_min": expected_negative,
                "actual": actual_negative,
            },
        )
    )
    return checks


def _no_leak_check(
    *,
    check_id: str,
    claim_ids: set[str],
    observed_ids: set[str],
) -> IsolationCheck:
    leaked = sorted(claim_ids.intersection(observed_ids))
    return IsolationCheck(
        check_id=check_id,
        passed=not leaked,
        details={"leaked_claim_ids": leaked},
    )


def _release_decision(
    *,
    validation_errors: list[str],
    deltas: list[EvidenceDelta],
    isolation_checks: list[IsolationCheck],
) -> str:
    if validation_errors:
        return "block"
    if any(not check.passed for check in isolation_checks):
        return "block"
    if any(delta.severity == "block_promotion" for delta in deltas):
        return "block"
    return "shadow_only"


def _negative_or_conflicting_count(claims: list[EvidenceClaim]) -> int:
    return len(
        [
            claim
            for claim in claims
            if claim.effect_direction in NEGATIVE_OR_CONFLICTING_DIRECTIONS
            or claim.local_guideline_conflict != "none"
        ]
    )


def _review_status(
    *,
    source_quality: SourceQuality,
    risk_of_bias: str,
    local_guideline_conflict: str,
) -> str:
    if source_quality.is_retracted:
        return "rejected"
    if source_quality.is_preprint or risk_of_bias == "high" or local_guideline_conflict != "none":
        return "needs_review"
    return "candidate"


def _source_quality(
    candidate: PaperCandidate,
    claim_payload: dict[str, Any],
) -> SourceQuality:
    if isinstance(claim_payload.get("source_quality"), dict):
        return SourceQuality.from_dict(claim_payload.get("source_quality"))
    return candidate.source_quality


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def _optional_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("source_span is required")
    return value


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}
```

- [ ] **Step 3.4: Run harness tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
```

Expected: all tests pass.

- [ ] **Step 3.5: Commit Task 3**

Run:

```powershell
git add src/services/literature_harness.py tests/backend/test_literature_harness.py
git commit -m "feat: add literature evidence harness"
```

Expected: commit succeeds with only service and test changes.

## Task 4: Literature Harness Replay Script And Report

**Files:**

- Create: `scripts/run_literature_harness.py`
- Create: `reports/literature/README.md`
- Create: `reports/literature/literature_harness_20260630_001.json`
- Modify: `tests/backend/test_literature_harness.py`

- [ ] **Step 4.1: Add failing replay test**

Append to `tests/backend/test_literature_harness.py`:

```python
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
```

- [ ] **Step 4.2: Run replay test to verify it fails**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_literature_harness.py::test_literature_harness_replay_writes_shadow_report -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_literature_harness'`.

- [ ] **Step 4.3: Implement the replay script**

Create `scripts/run_literature_harness.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.literature_harness import build_literature_harness_run


LITERATURE_HARNESS_RUN_ID = "literature_harness_20260630_001"
CLAIM_PACK_PATH = ROOT / "tests" / "fixtures" / "literature_claim_pack_v0.json"


def run_literature_harness(
    *,
    output_root: str | Path = ROOT / "reports",
) -> Path:
    output_base = Path(output_root)
    literature_dir = output_base / "literature"
    literature_dir.mkdir(parents=True, exist_ok=True)

    claim_pack = _read_json(CLAIM_PACK_PATH)
    harness_run = build_literature_harness_run(
        run_id=LITERATURE_HARNESS_RUN_ID,
        claim_pack=claim_pack,
    )

    report_path = literature_dir / f"{LITERATURE_HARNESS_RUN_ID}.json"
    _write_json(report_path, harness_run)
    return report_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    report_file = run_literature_harness()
    print(f"Wrote {report_file}")
```

- [ ] **Step 4.4: Add report directory README**

Create `reports/literature/README.md`:

```markdown
# Literature Harness Reports

This directory stores deterministic Step 10 literature harness outputs.

The reports are shadow-only evidence artifacts. They may be shown in future Agent Admin or research review surfaces, but they must not be used as patient advice, doctor default-flow clinical facts, prompt patches, training data, or clinical RAG index content without later human sign-off and release gates.
```

- [ ] **Step 4.5: Run replay tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_literature_harness.py -q
```

Expected: all literature harness tests pass.

- [ ] **Step 4.6: Generate the committed report**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe scripts\run_literature_harness.py
```

Expected: output includes `Wrote D:\YiZhu_Agnet\LangG\reports\literature\literature_harness_20260630_001.json`.

- [ ] **Step 4.7: Inspect generated report**

Run:

```powershell
Get-Content -Raw reports\literature\literature_harness_20260630_001.json
```

Expected:

- `release_decision` is `shadow_only`.
- `summary.claims` is `3`.
- `summary.negative_or_conflicting_claims` is `2`.
- `summary.isolation_violations` is `0`.
- No claim has `review_status: approved_for_clinical_rag`.

- [ ] **Step 4.8: Commit Task 4**

Run:

```powershell
git add scripts/run_literature_harness.py reports/literature/README.md reports/literature/literature_harness_20260630_001.json tests/backend/test_literature_harness.py
git commit -m "feat: add literature harness replay report"
```

Expected: commit succeeds with script, report README, generated report, and replay test changes.

## Task 5: Step 10 Boundary Regressions

**Files:**

- Modify: `tests/backend/test_literature_harness.py`

- [ ] **Step 5.1: Add forbidden-promotion regression test**

Append to `tests/backend/test_literature_harness.py`:

```python
def test_literature_harness_never_promotes_candidates_to_clinical_paths() -> None:
    harness = build_literature_harness_run(
        run_id="literature_harness_boundary",
        claim_pack=_load_fixture(),
    )

    forbidden_statuses = {
        "approved_for_project_pool",
        "approved_for_clinical_rag",
    }
    assert forbidden_statuses.isdisjoint(
        {claim["review_status"] for claim in harness["claims"]}
    )
    assert {
        check["check_id"]: check["passed"]
        for check in harness["isolation_checks"]
    } == {
        "no_candidate_in_clinical_rag": True,
        "no_candidate_in_patient_default_path": True,
        "no_candidate_in_doctor_default_path": True,
        "negative_evidence_preserved": True,
    }
```

- [ ] **Step 5.2: Run boundary tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_literature_harness.py::test_literature_harness_never_promotes_candidates_to_clinical_paths -q
```

Expected: test passes.

- [ ] **Step 5.3: Run P0/P1 backend regressions**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
```

Expected: P1 backend tests pass.

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_harness_replay.py -q
```

Expected: P0 safety and harness tests pass.

- [ ] **Step 5.4: Confirm forbidden paths were not touched**

Run:

```powershell
git diff --name-only main...HEAD
```

Expected output paths are limited to:

```text
src/contracts/evidence_claim.py
src/services/literature_harness.py
tests/backend/test_evidence_claim_contract.py
tests/backend/test_literature_harness.py
tests/fixtures/literature_claim_pack_v0.json
scripts/run_literature_harness.py
reports/literature/README.md
reports/literature/literature_harness_20260630_001.json
```

Forbidden paths must not appear:

```text
CRC-client
src/services/web_search_service.py
src/tools/web_search_tools.py
src/tools/manifest.py
config/safety_policy.yaml
frontend/src/pages/workspace-page.tsx
frontend/src/features/doctor/doctor-scene-shell.tsx
frontend/src/features/patient-crc-triage
chroma_db
bm25_index
```

- [ ] **Step 5.5: Commit Task 5 if the boundary test was added after Task 4**

Run:

```powershell
git add tests/backend/test_literature_harness.py
git commit -m "test: lock literature harness shadow boundary"
```

Expected: commit succeeds if Task 5 changed the test file. If Step 5.1 was folded into Task 4 before its commit, skip this commit and record that the boundary test was already committed.

## Task 6: Full Verification And Handoff

**Files:**

- Modify implementation files only if verification exposes a real Step 10 defect.

- [ ] **Step 6.1: Run focused Step 10 verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
```

Expected: all Step 10 tests pass.

- [ ] **Step 6.2: Run P1 regression verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
```

Expected: P1 tests pass.

- [ ] **Step 6.3: Run P0 regression verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
```

Expected: P0 tests pass.

- [ ] **Step 6.4: Run frontend regression verification**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

Expected: frontend tests pass. This Step 10 plan does not modify frontend files; if this broad command fails for an unrelated pre-existing frontend issue, capture the failing file and rerun the focused backend commands from Steps 6.1-6.3 before handoff.

- [ ] **Step 6.5: Run generated report determinism check**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe scripts\run_literature_harness.py
git diff -- reports/literature/literature_harness_20260630_001.json
```

Expected: `git diff` prints no changes for the generated report.

- [ ] **Step 6.6: Inspect final diff**

Run:

```powershell
git diff --stat main...HEAD
git diff --name-status main...HEAD
```

Expected: only Step 10 contract, harness, fixture, script, tests, and literature report paths appear.

- [ ] **Step 6.7: Commit final verification fixes if needed**

If verification required fixes, commit only the concrete fixed files:

```powershell
git add src/contracts/evidence_claim.py src/services/literature_harness.py tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py tests/fixtures/literature_claim_pack_v0.json scripts/run_literature_harness.py reports/literature/README.md reports/literature/literature_harness_20260630_001.json
git commit -m "fix: stabilize literature evidence harness"
```

Expected: commit succeeds only if verification produced changes. If there were no changes, skip this commit.

## Self-Review

Spec coverage:

- `EvidenceClaim`, `PaperCandidate`, `EvidenceDelta`, and `LiteratureHarnessRun` contracts are implemented by Task 1.
- Deterministic claim IDs are implemented and tested by Task 1.
- Fixture-based claim pack is implemented by Task 2.
- Claim extraction, negative/conflicting evidence preservation, delta creation, isolation checks, and release decision are implemented by Task 3.
- Replay script and committed shadow report are implemented by Task 4.
- Shadow-only boundary and no clinical-path promotion are locked by Task 5.
- P0/P1 regression and determinism checks are covered by Task 6.

Placeholder scan:

- Every task contains concrete file paths, commands, expected results, and code for created or modified implementation files.
- No task asks the implementer to invent missing contracts, statuses, paths, or command names.

Type consistency:

- `effect_direction`, `evidence_grade`, `review_status`, `delta_type`, `severity`, and `release_decision` values match the Step 10 spec.
- Backend tests, contract dataclasses, harness service, and report JSON use the same snake_case field names.
- `approved_for_project_pool` and `approved_for_clinical_rag` are intentionally rejected during Step 10.

Scope check:

- This plan deliberately omits frontend UI, live search, model summarization, clinical RAG ingest, Agent Admin dashboard, research cohort feasibility, and LearningJob. Those belong to Step 11 or later.

## Execution Handoff

Plan complete when this file is saved. Use one of these execution modes:

1. **Subagent-Driven (recommended)**: dispatch a fresh subagent per task, then run spec compliance and code quality review between tasks.
2. **Inline Execution**: execute tasks in this session with review checkpoints after each backend layer and after report generation.
