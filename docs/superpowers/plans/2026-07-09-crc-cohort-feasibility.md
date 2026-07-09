# CRC Cohort Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P2 Step 12 as a shadow-only, aggregate-only CRC cohort feasibility loop that estimates research data readiness without exporting patient-level rows or producing clinical recommendations.

**Architecture:** Add a small research contract layer, a deterministic cohort feasibility service over existing patient registry records and `ClinicalAssertion` projections, and an admin-only read endpoint for aggregate previews. Keep all patient-level data inside the registry/projection boundary and return only counts, coverage, warnings, and review queue metadata.

**Tech Stack:** Python 3.10, dataclasses, FastAPI, Pydantic v2, pytest, existing patient registry service, existing `ClinicalAssertion` projection.

---

## Global Constraints

- Step 12 is read-only. It must not write patient records, patient events, safety policy, prompts, rubrics, routes, templates, RAG indexes, release reports, literature reports, learning-job artifacts, or `CRC-client/`.
- API responses must not include `patient_id`, `session_id`, `record_id`, patient names, raw record payloads, transcripts, doctor notes, or patient-level rows.
- `patient_level_export_requested: true` must return a `blocked_by_governance` aggregate result without reading records.
- Every non-blocked feasibility response that inspects patient-level source records must include a `research_ethics_review` queue item.
- The first implementation intentionally excludes a frontend research panel. The backend API is the acceptance surface.

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-08-crc-cohort-feasibility-design.md`
- `docs/superpowers/plans/2026-06-30-p0-p1-p15-to-p2-readiness-handoff.md`
- `src/contracts/clinical_assertion.py`
- `src/services/clinical_assertion_projection.py`
- `backend/api/services/patient_registry_service.py`
- `backend/api/routes/patient_registry.py`
- `backend/app.py`
- `tests/backend/test_auth_security.py`

## File Structure

Backend contracts:

- Create `src/contracts/research_asset.py`
  - Dataclass contracts for `ResearchAsset`, `CohortFeasibilityRequest`, `VariableCoverage`, `ReviewQueueItem`, and `CohortFeasibilityResult`.
  - Stable ID helpers and JSON-safe `to_dict()` methods.
  - Construction-time guards for forbidden statuses, malformed data scope, patient identifiers in criteria, and patient-level response rows.
- Create `tests/backend/test_research_asset_contract.py`
  - Contract serialization, export request preservation for governance blocking, review item, forbidden status, and patient-row response guards.

Backend read model and service:

- Modify `backend/api/services/patient_registry_service.py`
  - Add `list_research_projection_records(limit=1000)` as a read-only method that returns registry record rows needed for projection.
- Create `src/services/cohort_feasibility_service.py`
  - Project existing CRC triage assertions through `project_clinical_assertions_from_records`.
  - Add narrow document-fact extraction for `colonoscopy_status` and `pathology_result` from existing normalized record payloads.
  - Map required features to assertions, compute aggregate coverage, create a review item, and return `CohortFeasibilityResult`.
- Create `tests/backend/test_cohort_feasibility_service.py`
  - Aggregate count, coverage, missing variables, unmapped variables, export block, no patient rows, and read-only registry tests.

Backend API:

- Create `backend/api/schemas/research.py`
  - Pydantic request schema for admin cohort feasibility preview.
- Create `backend/api/routes/research.py`
  - `POST /api/admin/research/cohort-feasibility`.
  - Pull patient registry service from `request.app.state.runtime.patient_registry_service`.
  - Return aggregate feasibility only.
- Modify `backend/app.py`
  - Include the research router.
  - Require admin token for `POST /api/admin/research/cohort-feasibility`.
- Create `tests/backend/test_research_api.py`
  - Admin auth, response shape, validation mapping, unavailable registry, and no patient-level fields.
- Modify `tests/backend/test_auth_security.py`
  - Add the new research endpoint to the admin-token matrix.

Docs and git visibility:

- Modify `.gitignore`
  - Add new Step 12 backend test whitelist entries when implementation begins.

---

### Task 1: Research Contracts

**Files:**
- Modify: `.gitignore`
- Create: `src/contracts/research_asset.py`
- Create: `tests/backend/test_research_asset_contract.py`

**Interfaces:**
- Produces: `ResearchAsset`, `CohortFeasibilityRequest`, `VariableCoverage`, `ReviewQueueItem`, `CohortFeasibilityResult`.
- Produces: `make_research_asset_id(prefix, seed) -> str`.
- Produces: `make_review_item_id(request_id, review_type) -> str`.

- [ ] **Step 1: Add test whitelist entries**

Modify `.gitignore` near the backend test whitelist:

```gitignore
!tests/backend/test_research_asset_contract.py
!tests/backend/test_cohort_feasibility_service.py
!tests/backend/test_research_api.py
```

- [ ] **Step 2: Write failing contract tests**

Create `tests/backend/test_research_asset_contract.py`:

```python
from __future__ import annotations

import pytest

from src.contracts.research_asset import (
    CohortFeasibilityRequest,
    CohortFeasibilityResult,
    ResearchAsset,
    ReviewQueueItem,
    VariableCoverage,
    make_research_asset_id,
    make_review_item_id,
)


def make_request(**overrides: object) -> CohortFeasibilityRequest:
    payload = {
        "request_id": "cohort_request_crc_001",
        "project_id": "research_crc_001",
        "question": "Is there enough structured CRC triage data to study rectal bleeding escalation?",
        "cohort_criteria": {
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "age_min": 50,
            "required_features": ["rectal_bleeding", "pathology_result"],
        },
        "data_scope": {
            "source": "patient_record_projection",
            "patient_level_export_requested": False,
            "deidentified_only": True,
        },
        "version_refs": {
            "projection_version": "patient_record_projection_v0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
        },
    }
    payload.update(overrides)
    return CohortFeasibilityRequest(**payload)


def test_research_contracts_round_trip_to_dict() -> None:
    request = make_request()
    review_item = ReviewQueueItem(
        review_item_id=make_review_item_id(request.request_id, "research_ethics_review"),
        review_type="research_ethics_review",
        status="pending",
        trigger="patient_level_data_used_for_cohort_feasibility",
        scope={
            "project_id": request.project_id,
            "request_id": request.request_id,
            "data_minimization": "aggregate_only",
            "patient_level_export_requested": False,
        },
        required_checks=[
            "authorization_basis",
            "deidentification_strategy",
            "data_minimization",
            "irb_or_local_policy_need",
        ],
    )
    result = CohortFeasibilityResult(
        result_id=make_research_asset_id("cohort_feasibility", request.request_id),
        request_id=request.request_id,
        project_id=request.project_id,
        status="needs_review",
        estimated_count=2,
        variable_coverage={
            "rectal_bleeding": VariableCoverage(
                covered_count=2,
                coverage_ratio=1.0,
                source_fact_types=["condition_signal"],
                reviewed_status_mix={"unreviewed": 2},
            ),
            "pathology_result": VariableCoverage(
                covered_count=1,
                coverage_ratio=0.5,
                source_fact_types=["document_fact"],
                reviewed_status_mix={"unreviewed": 1},
            ),
        },
        missing_key_variables=[],
        unmapped_required_features=[],
        bias_warnings=[],
        requires_review=True,
        review_queue_items=[review_item],
        patient_level_rows_returned=False,
    )
    asset = ResearchAsset(
        asset_id=make_research_asset_id("research_asset", request.request_id),
        asset_type="cohort_feasibility",
        title="CRC triage cohort feasibility",
        status="candidate",
        created_by="research_workspace",
        created_at="2026-07-09T10:00:00+08:00",
        source_refs=[{"kind": "clinical_assertion_projection", "id": "patient_record_projection_v0"}],
        governance_refs=[review_item.review_item_id],
    )

    assert request.to_dict()["data_scope"]["patient_level_export_requested"] is False
    assert result.to_dict()["patient_level_rows_returned"] is False
    assert result.to_dict()["review_queue_items"][0]["review_type"] == "research_ethics_review"
    assert asset.to_dict()["asset_type"] == "cohort_feasibility"


def test_request_preserves_export_request_for_governance_block() -> None:
    request = make_request(
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": True,
            "deidentified_only": True,
        }
    )

    assert request.patient_level_export_requested is True
    assert (
        request.to_dict()["data_scope"]["patient_level_export_requested"]
        is True
    )
        )


def test_request_rejects_patient_identifiers_in_criteria() -> None:
    with pytest.raises(ValueError, match="patient identifiers are not allowed"):
        make_request(
            cohort_criteria={
                "condition": "crc",
                "required_features": ["rectal_bleeding"],
                "patient_ids": ["patient-1"],
            }
        )


def test_result_rejects_patient_level_rows() -> None:
    with pytest.raises(ValueError, match="patient_level_rows_returned must be false"):
        CohortFeasibilityResult(
            result_id="cohort_feasibility_bad",
            request_id="cohort_request_crc_001",
            project_id="research_crc_001",
            status="needs_review",
            estimated_count=1,
            variable_coverage={},
            missing_key_variables=[],
            unmapped_required_features=[],
            bias_warnings=[],
            requires_review=True,
            review_queue_items=[],
            patient_level_rows_returned=True,
        )


@pytest.mark.parametrize("status", ["approved_dataset", "published", "clinical_rag_ready"])
def test_research_asset_rejects_out_of_scope_status(status: str) -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        ResearchAsset(
            asset_id="research_asset_bad",
            asset_type="cohort_feasibility",
            title="Bad asset",
            status=status,
            created_by="research_workspace",
            created_at="2026-07-09T10:00:00+08:00",
            source_refs=[],
            governance_refs=[],
        )
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.contracts.research_asset'`.

- [ ] **Step 3: Implement research contracts**

Create `src/contracts/research_asset.py` with the contracts exercised by the tests:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, TypeAlias


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

ResearchAssetType = Literal["cohort_feasibility", "ethics_review_item"]
ResearchAssetStatus = Literal["candidate", "needs_review", "blocked", "reviewed"]
CohortFeasibilityStatus = Literal[
    "feasible_for_review",
    "needs_review",
    "insufficient_data",
    "blocked_by_governance",
]
ReviewType = Literal["research_ethics_review", "pi_review", "data_governance_review"]
ReviewItemStatus = Literal["pending", "blocked", "reviewed"]

RESEARCH_ASSET_TYPES = ("cohort_feasibility", "ethics_review_item")
RESEARCH_ASSET_STATUSES = ("candidate", "needs_review", "blocked", "reviewed")
COHORT_FEASIBILITY_STATUSES = (
    "feasible_for_review",
    "needs_review",
    "insufficient_data",
    "blocked_by_governance",
)
REVIEW_TYPES = ("research_ethics_review", "pi_review", "data_governance_review")
REVIEW_ITEM_STATUSES = ("pending", "blocked", "reviewed")
FORBIDDEN_CRITERIA_KEYS = {
    "patient_id",
    "patient_ids",
    "patient_identifier",
    "patient_identifiers",
    "patient_name",
    "patient_number",
    "medical_record_number",
    "mrn",
}


@dataclass(frozen=True)
class ResearchAsset:
    asset_id: str
    asset_type: ResearchAssetType
    title: str
    status: ResearchAssetStatus
    created_by: str
    created_at: str
    source_refs: list[dict[str, JsonValue]]
    governance_refs: list[str]

    def __post_init__(self) -> None:
        _require_non_empty("asset_id", self.asset_id)
        _validate_choice("asset_type", self.asset_type, RESEARCH_ASSET_TYPES)
        _validate_choice("status", self.status, RESEARCH_ASSET_STATUSES)
        _require_non_empty("title", self.title)
        _require_non_empty("created_by", self.created_by)
        _require_non_empty("created_at", self.created_at)
        _validate_json_value(self.source_refs)
        _require_string_list("governance_refs", self.governance_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "title": self.title,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "source_refs": list(self.source_refs),
            "governance_refs": list(self.governance_refs),
        }


@dataclass(frozen=True)
class CohortFeasibilityRequest:
    request_id: str
    project_id: str
    question: str
    cohort_criteria: dict[str, JsonValue]
    data_scope: dict[str, JsonValue]
    version_refs: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty("request_id", self.request_id)
        _require_non_empty("project_id", self.project_id)
        _require_non_empty("question", self.question)
        _validate_json_value(self.cohort_criteria)
        _validate_json_value(self.data_scope)
        _validate_json_value(self.version_refs)
        required_features = self.cohort_criteria.get("required_features")
        if not isinstance(required_features, list) or not required_features:
            raise ValueError("required_features must be a non-empty list")
        _require_string_list("required_features", required_features)
        if self.data_scope.get("source") != "patient_record_projection":
            raise ValueError("data_scope.source must be patient_record_projection")
        if not isinstance(self.data_scope.get("patient_level_export_requested"), bool):
            raise ValueError("patient_level_export_requested must be a boolean")
        if self.data_scope.get("deidentified_only") is not True:
            raise ValueError("deidentified_only must be true")
        _reject_patient_identifier_keys(self.cohort_criteria)

    @property
    def required_features(self) -> list[str]:
        return [str(item) for item in self.cohort_criteria["required_features"]]

    @property
    def patient_level_export_requested(self) -> bool:
        return bool(self.data_scope["patient_level_export_requested"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "project_id": self.project_id,
            "question": self.question,
            "cohort_criteria": dict(self.cohort_criteria),
            "data_scope": dict(self.data_scope),
            "version_refs": dict(self.version_refs),
        }


@dataclass(frozen=True)
class VariableCoverage:
    covered_count: int
    coverage_ratio: float
    source_fact_types: list[str]
    reviewed_status_mix: dict[str, int]

    def __post_init__(self) -> None:
        if self.covered_count < 0:
            raise ValueError("covered_count must be non-negative")
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError("coverage_ratio must be between 0 and 1")
        _require_string_list("source_fact_types", self.source_fact_types)
        for key, value in self.reviewed_status_mix.items():
            _require_non_empty("reviewed_status_mix key", key)
            if value < 0:
                raise ValueError("reviewed_status_mix counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_count": self.covered_count,
            "coverage_ratio": self.coverage_ratio,
            "source_fact_types": list(self.source_fact_types),
            "reviewed_status_mix": dict(self.reviewed_status_mix),
        }


@dataclass(frozen=True)
class ReviewQueueItem:
    review_item_id: str
    review_type: ReviewType
    status: ReviewItemStatus
    trigger: str
    scope: dict[str, JsonValue]
    required_checks: list[str]

    def __post_init__(self) -> None:
        _require_non_empty("review_item_id", self.review_item_id)
        _validate_choice("review_type", self.review_type, REVIEW_TYPES)
        _validate_choice("status", self.status, REVIEW_ITEM_STATUSES)
        _require_non_empty("trigger", self.trigger)
        _validate_json_value(self.scope)
        _require_string_list("required_checks", self.required_checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_item_id": self.review_item_id,
            "review_type": self.review_type,
            "status": self.status,
            "trigger": self.trigger,
            "scope": dict(self.scope),
            "required_checks": list(self.required_checks),
        }


@dataclass(frozen=True)
class CohortFeasibilityResult:
    result_id: str
    request_id: str
    project_id: str
    status: CohortFeasibilityStatus
    estimated_count: int
    variable_coverage: dict[str, VariableCoverage]
    missing_key_variables: list[str]
    unmapped_required_features: list[str]
    bias_warnings: list[str]
    requires_review: bool
    review_queue_items: list[ReviewQueueItem]
    patient_level_rows_returned: bool

    def __post_init__(self) -> None:
        _require_non_empty("result_id", self.result_id)
        _require_non_empty("request_id", self.request_id)
        _require_non_empty("project_id", self.project_id)
        _validate_choice("status", self.status, COHORT_FEASIBILITY_STATUSES)
        if self.estimated_count < 0:
            raise ValueError("estimated_count must be non-negative")
        if self.patient_level_rows_returned is not False:
            raise ValueError("patient_level_rows_returned must be false")
        _require_string_list("missing_key_variables", self.missing_key_variables)
        _require_string_list("unmapped_required_features", self.unmapped_required_features)
        _require_string_list("bias_warnings", self.bias_warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "request_id": self.request_id,
            "project_id": self.project_id,
            "status": self.status,
            "estimated_count": self.estimated_count,
            "variable_coverage": {
                key: value.to_dict() for key, value in self.variable_coverage.items()
            },
            "missing_key_variables": list(self.missing_key_variables),
            "unmapped_required_features": list(self.unmapped_required_features),
            "bias_warnings": list(self.bias_warnings),
            "requires_review": self.requires_review,
            "review_queue_items": [item.to_dict() for item in self.review_queue_items],
            "patient_level_rows_returned": False,
        }


def make_research_asset_id(prefix: str, seed: str) -> str:
    _require_non_empty("prefix", prefix)
    _require_non_empty("seed", seed)
    digest = hashlib.sha256(f"{prefix}:{seed}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def make_review_item_id(request_id: str, review_type: str) -> str:
    return make_research_asset_id(f"review_queue_{review_type}", request_id)


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _require_string_list(name: str, values: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        result.append(value)
    return result


def _reject_patient_identifier_keys(payload: dict[str, JsonValue]) -> None:
    for key, value in payload.items():
        if key.lower() in FORBIDDEN_CRITERIA_KEYS:
            raise ValueError("patient identifiers are not allowed in cohort criteria")
        if isinstance(value, dict):
            _reject_patient_identifier_keys(value)


def _validate_json_value(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("value must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        json.dumps(value, ensure_ascii=False)
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("value must be JSON-safe")
            _validate_json_value(item)
        return
    raise TypeError("value must be JSON-safe")
```

- [ ] **Step 4: Run contract tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit contract slice**

```powershell
git add .gitignore src/contracts/research_asset.py tests/backend/test_research_asset_contract.py
git commit -m "feat: add research cohort feasibility contracts"
```

---

### Task 2: Cohort Feasibility Service

**Files:**
- Modify: `backend/api/services/patient_registry_service.py`
- Create: `src/services/cohort_feasibility_service.py`
- Create: `tests/backend/test_cohort_feasibility_service.py`

**Interfaces:**
- Produces: `PatientRegistryService.list_research_projection_records(limit=1000) -> list[dict[str, Any]]`.
- Produces: `CohortFeasibilityService.evaluate(request, records) -> CohortFeasibilityResult`.

- [ ] **Step 1: Write failing service tests**

Create `tests/backend/test_cohort_feasibility_service.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.patient_registry_service import PatientRegistryService
from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


def _request(required_features: list[str]) -> CohortFeasibilityRequest:
    return CohortFeasibilityRequest(
        request_id="cohort_request_crc_001",
        project_id="research_crc_001",
        question="Is there enough structured CRC triage data to study rectal bleeding escalation?",
        cohort_criteria={
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "age_min": 50,
            "required_features": required_features,
        },
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": False,
            "deidentified_only": True,
        },
        version_refs={
            "projection_version": "patient_record_projection_v0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
        },
    )


def _triage_record(patient_id: int, record_id: int, *, rectal_bleeding: bool) -> dict[str, object]:
    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "record_type": "crc_triage_assessment",
        "normalized_payload_json": json.dumps(
            {
                "assessment_id": f"assessment_{record_id}",
                "known_crc_signals": {"rectal_bleeding": rectal_bleeding},
                "red_flags": ["weight_loss"] if rectal_bleeding else [],
                "disposition": "urgent_gi_clinic",
                "matched_rules": ["rectal_bleeding_age_escalation"] if rectal_bleeding else [],
                "source_session_id": f"session_{record_id}",
                "safety_policy_version": "crc_safety_policy_v0",
            },
            ensure_ascii=False,
        ),
    }


def _document_record(patient_id: int, record_id: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "record_type": "medical_card",
        "normalized_payload_json": json.dumps(payload, ensure_ascii=False),
    }


def test_service_returns_aggregate_coverage_without_patient_rows() -> None:
    records = [
        _triage_record(1, 101, rectal_bleeding=True),
        _document_record(1, 102, {"document_type": "pathology_report", "pathology_result": "adenocarcinoma"}),
        _triage_record(2, 201, rectal_bleeding=True),
        _document_record(2, 202, {"document_type": "colonoscopy_report", "colonoscopy_status": "completed"}),
    ]

    result = CohortFeasibilityService().evaluate(
        request=_request(["rectal_bleeding", "pathology_result", "colonoscopy_status"]),
        records=records,
    )
    payload = result.to_dict()

    assert payload["estimated_count"] == 2
    assert payload["variable_coverage"]["rectal_bleeding"]["covered_count"] == 2
    assert payload["variable_coverage"]["pathology_result"]["covered_count"] == 1
    assert payload["variable_coverage"]["colonoscopy_status"]["covered_count"] == 1
    assert payload["missing_key_variables"] == ["pathology_result", "colonoscopy_status"]
    assert payload["patient_level_rows_returned"] is False
    assert "patient_id" not in json.dumps(payload)
    assert payload["review_queue_items"][0]["review_type"] == "research_ethics_review"


def test_service_marks_unmapped_features_without_crashing() -> None:
    result = CohortFeasibilityService().evaluate(
        request=_request(["unknown_feature"]),
        records=[_triage_record(1, 101, rectal_bleeding=True)],
    )

    assert result.status == "needs_review"
    assert result.unmapped_required_features == ["unknown_feature"]
    assert result.variable_coverage["unknown_feature"].covered_count == 0


def test_service_returns_insufficient_data_for_empty_registry() -> None:
    result = CohortFeasibilityService().evaluate(
        request=_request(["rectal_bleeding"]),
        records=[],
    )

    assert result.status == "insufficient_data"
    assert result.estimated_count == 0
    assert result.variable_coverage["rectal_bleeding"].coverage_ratio == 0


def test_service_blocks_export_request_before_reading_records() -> None:
    request = CohortFeasibilityRequest(
        request_id="cohort_request_export_block",
        project_id="research_crc_001",
        question="Export a patient-level dataset.",
        cohort_criteria={
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": ["rectal_bleeding"],
        },
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": True,
            "deidentified_only": True,
        },
        version_refs={"projection_version": "patient_record_projection_v0"},
    )

    def records_that_must_not_be_read():
        raise AssertionError("export block must happen before records are read")
        yield {}

    result = CohortFeasibilityService().evaluate(
        request=request,
        records=records_that_must_not_be_read(),
    )

    assert result.status == "blocked_by_governance"
    assert result.estimated_count == 0
    assert result.patient_level_rows_returned is False


def test_registry_exposes_read_only_research_projection_records(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_1")
    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "pathology.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-pathology",
            "storage_path": "runtime/assets/pathology.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"age": 62, "tumor_location": "rectum"},
        record_payload={"document_type": "pathology_report", "pathology_result": "adenocarcinoma"},
        summary_text="pathology report",
        record_type="medical_card",
    )

    before = service.get_patient_detail(patient_id)
    rows = service.list_research_projection_records(limit=10)
    after = service.get_patient_detail(patient_id)

    assert len(rows) == 1
    assert rows[0]["patient_id"] == patient_id
    assert rows[0]["record_type"] == "medical_card"
    assert before == after
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_cohort_feasibility_service.py -q`

Expected: FAIL because `src.services.cohort_feasibility_service` and `list_research_projection_records` do not exist.

- [ ] **Step 2: Add read-only registry method**

Modify `backend/api/services/patient_registry_service.py` after `list_patient_records`:

```python
    def list_research_projection_records(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 5000))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    record_id,
                    patient_id,
                    record_type,
                    document_type,
                    normalized_payload_json,
                    summary_text,
                    source,
                    created_at
                FROM patient_records
                ORDER BY record_id ASC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]
```

- [ ] **Step 3: Implement aggregate service**

Create `src/services/cohort_feasibility_service.py`:

```python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.contracts.clinical_assertion import ClinicalAssertion, EvidenceRef, NormalizedFact
from src.contracts.research_asset import (
    CohortFeasibilityRequest,
    CohortFeasibilityResult,
    ReviewQueueItem,
    VariableCoverage,
    make_research_asset_id,
    make_review_item_id,
)
from src.services.clinical_assertion_projection import project_clinical_assertions_from_records


LOW_COVERAGE_THRESHOLD = 0.5
FEATURE_FACT_MAP: dict[str, set[tuple[str, str]]] = {
    "rectal_bleeding": {("condition_signal", "rectal_bleeding"), ("symptom", "rectal_bleeding")},
    "weight_loss": {("condition_signal", "weight_loss"), ("symptom", "weight_loss")},
    "disposition": {("risk_disposition", "disposition")},
    "matched_safety_rule": {("safety_rule_match", "*")},
    "colonoscopy_status": {("test_status", "colonoscopy_status"), ("document_fact", "colonoscopy_status")},
    "pathology_result": {("document_fact", "pathology_result")},
}


class CohortFeasibilityService:
    def evaluate(
        self,
        *,
        request: CohortFeasibilityRequest,
        records: Iterable[Mapping[str, Any]],
    ) -> CohortFeasibilityResult:
        if request.patient_level_export_requested:
            return _blocked_by_governance_result(request)

        record_list = [dict(record) for record in records]
        assertions = project_clinical_assertions_from_records(record_list)
        assertions.extend(_project_document_fact_assertions(record_list))
        cohort_patient_ids = _cohort_patient_ids(record_list, assertions)
        estimated_count = len(cohort_patient_ids)
        variable_coverage: dict[str, VariableCoverage] = {}
        missing_key_variables: list[str] = []
        unmapped_required_features: list[str] = []
        bias_warnings: list[str] = []

        for feature in request.required_features:
            matches = FEATURE_FACT_MAP.get(feature)
            if matches is None:
                unmapped_required_features.append(feature)
                coverage = VariableCoverage(
                    covered_count=0,
                    coverage_ratio=0,
                    source_fact_types=[],
                    reviewed_status_mix={},
                )
                variable_coverage[feature] = coverage
                bias_warnings.append(f"{feature} has no Step 12 feature mapper")
                continue

            covered_patient_ids: set[str] = set()
            fact_types: set[str] = set()
            status_mix: Counter[str] = Counter()
            for assertion in assertions:
                if str(assertion.patient_id) not in cohort_patient_ids:
                    continue
                fact = assertion.normalized_fact
                if _matches_feature(fact.type, fact.name, matches):
                    covered_patient_ids.add(str(assertion.patient_id))
                    fact_types.add(fact.type)
                    status_mix[str(assertion.reviewed_status)] += 1
            ratio = 0 if estimated_count == 0 else round(len(covered_patient_ids) / estimated_count, 4)
            coverage = VariableCoverage(
                covered_count=len(covered_patient_ids),
                coverage_ratio=ratio,
                source_fact_types=sorted(fact_types),
                reviewed_status_mix=dict(sorted(status_mix.items())),
            )
            variable_coverage[feature] = coverage
            if estimated_count == 0 or ratio < LOW_COVERAGE_THRESHOLD:
                missing_key_variables.append(feature)
                bias_warnings.append(f"{feature} coverage is below {LOW_COVERAGE_THRESHOLD}")

        status = _status_for(
            estimated_count=estimated_count,
            missing_key_variables=missing_key_variables,
            unmapped_required_features=unmapped_required_features,
        )
        review_item = ReviewQueueItem(
            review_item_id=make_review_item_id(request.request_id, "research_ethics_review"),
            review_type="research_ethics_review",
            status="pending",
            trigger="patient_level_data_used_for_cohort_feasibility",
            scope={
                "project_id": request.project_id,
                "request_id": request.request_id,
                "data_minimization": "aggregate_only",
                "patient_level_export_requested": False,
            },
            required_checks=[
                "authorization_basis",
                "deidentification_strategy",
                "data_minimization",
                "irb_or_local_policy_need",
            ],
        )
        return CohortFeasibilityResult(
            result_id=make_research_asset_id("cohort_feasibility", request.request_id),
            request_id=request.request_id,
            project_id=request.project_id,
            status=status,
            estimated_count=estimated_count,
            variable_coverage=variable_coverage,
            missing_key_variables=missing_key_variables,
            unmapped_required_features=unmapped_required_features,
            bias_warnings=bias_warnings,
            requires_review=True,
            review_queue_items=[review_item],
            patient_level_rows_returned=False,
        )


def _status_for(
    *,
    estimated_count: int,
    missing_key_variables: list[str],
    unmapped_required_features: list[str],
) -> str:
    if estimated_count == 0:
        return "insufficient_data"
    if missing_key_variables or unmapped_required_features:
        return "needs_review"
    return "needs_review"


def _blocked_by_governance_result(
    request: CohortFeasibilityRequest,
) -> CohortFeasibilityResult:
    return CohortFeasibilityResult(
        result_id=make_research_asset_id("cohort_feasibility", request.request_id),
        request_id=request.request_id,
        project_id=request.project_id,
        status="blocked_by_governance",
        estimated_count=0,
        variable_coverage={
            feature: VariableCoverage(
                covered_count=0,
                coverage_ratio=0,
                source_fact_types=[],
                reviewed_status_mix={},
            )
            for feature in request.required_features
        },
        missing_key_variables=list(request.required_features),
        unmapped_required_features=[],
        bias_warnings=["patient-level export requests require ethics and data-governance approval"],
        requires_review=True,
        review_queue_items=[
            ReviewQueueItem(
                review_item_id=make_review_item_id(request.request_id, "research_ethics_review"),
                review_type="research_ethics_review",
                status="blocked",
                trigger="patient_level_export_requested",
                scope={
                    "project_id": request.project_id,
                    "request_id": request.request_id,
                    "data_minimization": "aggregate_only",
                    "patient_level_export_requested": True,
                },
                required_checks=[
                    "authorization_basis",
                    "deidentification_strategy",
                    "data_minimization",
                    "irb_or_local_policy_need",
                ],
            )
        ],
        patient_level_rows_returned=False,
    )


def _cohort_patient_ids(
    records: list[dict[str, Any]],
    assertions: list[ClinicalAssertion],
) -> set[str]:
    patient_ids = {str(assertion.patient_id) for assertion in assertions if str(assertion.patient_id)}
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id is not None:
            patient_ids.add(str(patient_id))
    return patient_ids


def _matches_feature(
    fact_type: str,
    fact_name: str,
    matchers: set[tuple[str, str]],
) -> bool:
    return (fact_type, fact_name) in matchers or (fact_type, "*") in matchers


def _project_document_fact_assertions(records: list[dict[str, Any]]) -> list[ClinicalAssertion]:
    assertions: list[ClinicalAssertion] = []
    for record in records:
        payload = _load_payload(record.get("normalized_payload_json"))
        record_id = str(record.get("record_id") or "")
        patient_id = str(record.get("patient_id") or "")
        for name in ("colonoscopy_status", "pathology_result"):
            value = payload.get(name)
            if value in (None, ""):
                continue
            fact = NormalizedFact(type="document_fact", name=name, value=value)
            evidence_refs = [EvidenceRef(kind="patient_record", id=record_id, field=name)]
            assertions.append(
                ClinicalAssertion(
                    assertion_id=make_research_asset_id(
                        f"assertion_document_fact_{name}",
                        f"{patient_id}:{record_id}:{value}",
                    ),
                    patient_id=patient_id,
                    session_id=None,
                    source="patient_upload",
                    source_record_id=record_id,
                    source_assessment_id=None,
                    normalized_fact=fact,
                    evidence_refs=evidence_refs,
                    confidence="projected_document_fact",
                    reviewed_status="unreviewed",
                    safety_policy_version=None,
                    created_from_projection_version="patient_record_projection_v0",
                )
            )
    return assertions


def _load_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}
```

- [ ] **Step 4: Run service tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_cohort_feasibility_service.py -q`

Expected: PASS.

- [ ] **Step 5: Run projection regression**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_patient_registry_anatomy_regions.py -q`

Expected: PASS.

- [ ] **Step 6: Commit service slice**

```powershell
git add backend/api/services/patient_registry_service.py src/services/cohort_feasibility_service.py tests/backend/test_cohort_feasibility_service.py
git commit -m "feat: add aggregate cohort feasibility service"
```

---

### Task 3: Admin Research API

**Files:**
- Create: `backend/api/schemas/research.py`
- Create: `backend/api/routes/research.py`
- Modify: `backend/app.py`
- Create: `tests/backend/test_research_api.py`
- Modify: `tests/backend/test_auth_security.py`

**Interfaces:**
- Produces: `POST /api/admin/research/cohort-feasibility`.
- Requires: admin bearer token when `AUTH_MODE=bearer`.

- [ ] **Step 1: Write failing API tests**

Create `tests/backend/test_research_api.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import research


class Registry:
    def list_research_projection_records(self, *, limit: int = 1000):
        return [
            {
                "record_id": 1,
                "patient_id": 101,
                "record_type": "crc_triage_assessment",
                "normalized_payload_json": {
                    "assessment_id": "assessment_1",
                    "known_crc_signals": {"rectal_bleeding": True},
                    "disposition": "urgent_gi_clinic",
                    "matched_rules": ["rectal_bleeding_age_escalation"],
                    "safety_policy_version": "crc_safety_policy_v0",
                },
            }
        ]


def _client(registry=Registry()) -> TestClient:
    app = FastAPI()
    app.state.runtime = SimpleNamespace(patient_registry_service=registry)
    app.include_router(research.router)
    return TestClient(app)


def _payload() -> dict[str, object]:
    return {
        "request_id": "cohort_request_crc_001",
        "project_id": "research_crc_001",
        "question": "Is there enough structured CRC triage data?",
        "cohort_criteria": {
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "age_min": 50,
            "required_features": ["rectal_bleeding"],
        },
        "data_scope": {
            "source": "patient_record_projection",
            "patient_level_export_requested": False,
            "deidentified_only": True,
        },
        "version_refs": {
            "projection_version": "patient_record_projection_v0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
        },
    }


def test_cohort_feasibility_api_returns_aggregate_response() -> None:
    response = _client().post("/api/admin/research/cohort-feasibility", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["estimated_count"] == 1
    assert body["variable_coverage"]["rectal_bleeding"]["covered_count"] == 1
    assert body["patient_level_rows_returned"] is False
    assert "patient_id" not in response.text
    assert body["runtime"] == {
        "auth": "admin",
        "source": "patient_record_projection",
        "mode": "shadow_cohort_feasibility",
    }


def test_cohort_feasibility_api_returns_blocked_result_for_export_request() -> None:
    payload = _payload()
    payload["data_scope"] = {
        "source": "patient_record_projection",
        "patient_level_export_requested": True,
        "deidentified_only": True,
    }

    response = _client().post("/api/admin/research/cohort-feasibility", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "blocked_by_governance"
    assert response.json()["estimated_count"] == 0
    assert response.json()["patient_level_rows_returned"] is False


def test_cohort_feasibility_api_returns_503_without_registry() -> None:
    app = FastAPI()
    app.state.runtime = SimpleNamespace()
    app.include_router(research.router)
    client = TestClient(app)

    response = client.post("/api/admin/research/cohort-feasibility", json=_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Patient registry is not initialized"
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_api.py -q`

Expected: FAIL because `backend.api.routes.research` does not exist.

- [ ] **Step 2: Add Pydantic schema**

Create `backend/api/schemas/research.py`:

```python
from __future__ import annotations

from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]


class CohortFeasibilityRequestPayload(BaseModel):
    request_id: NonEmptyString
    project_id: NonEmptyString
    question: NonEmptyString
    cohort_criteria: dict[str, Any]
    data_scope: dict[str, Any]
    version_refs: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
```

- [ ] **Step 3: Add admin research route**

Create `backend/api/routes/research.py`:

```python
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.research import CohortFeasibilityRequestPayload
from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


router = APIRouter(prefix="/api/admin/research", tags=["admin-research"])


def _get_registry_service(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    service = getattr(runtime, "patient_registry_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Patient registry is not initialized")
    return service


@router.post("/cohort-feasibility")
async def preview_cohort_feasibility(
    request: Request,
    payload: CohortFeasibilityRequestPayload,
) -> dict[str, Any]:
    try:
        feasibility_request = CohortFeasibilityRequest(**payload.model_dump())
        registry = _get_registry_service(request)
        records = registry.list_research_projection_records(limit=1000)
        result = CohortFeasibilityService().evaluate(
            request=feasibility_request,
            records=records,
        )
    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = result.to_dict()
    response["runtime"] = {
        "auth": "admin",
        "source": "patient_record_projection",
        "mode": "shadow_cohort_feasibility",
    }
    return response
```

- [ ] **Step 4: Wire app and admin auth**

Modify `backend/app.py` imports:

```python
from backend.api.routes import research as research_routes
```

Modify `_requires_admin_token`:

```python
    if method == "POST" and path == "/api/admin/research/cohort-feasibility":
        return True
```

Modify `create_app()` router includes near the other routers:

```python
    app.include_router(research_routes.router)
```

- [ ] **Step 5: Extend auth-security matrix**

In `tests/backend/test_auth_security.py`, add the stub route inside `_auth_client`:

```python
    @app.post("/api/admin/research/cohort-feasibility")
    async def admin_research_cohort_feasibility() -> dict[str, object]:
        return {"runtime": {"auth": "admin", "mode": "shadow_cohort_feasibility"}}
```

Add `("post", "/api/admin/research/cohort-feasibility")` to each admin endpoint parameter list in:

- `test_admin_endpoints_reject_user_token_when_admin_token_is_distinct`
- `test_admin_endpoints_accept_admin_token`
- `test_admin_endpoints_use_user_token_when_no_separate_admin_token`

- [ ] **Step 6: Run API and auth tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_api.py tests/backend/test_auth_security.py -q`

Expected: PASS.

- [ ] **Step 7: Commit API slice**

```powershell
git add backend/api/schemas/research.py backend/api/routes/research.py backend/app.py tests/backend/test_research_api.py tests/backend/test_auth_security.py
git commit -m "feat: expose admin cohort feasibility preview"
```

---

### Task 4: Non-Mutation And Regression Verification

**Files:**
- Create: `tests/backend/test_cohort_feasibility_non_mutation.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add non-mutation test whitelist**

Modify `.gitignore` near the backend test whitelist:

```gitignore
!tests/backend/test_cohort_feasibility_non_mutation.py
```

- [ ] **Step 2: Write non-mutation test**

Create `tests/backend/test_cohort_feasibility_non_mutation.py`:

```python
from __future__ import annotations

from pathlib import Path

from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


def _snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return snapshot


def test_cohort_feasibility_does_not_mutate_runtime_artifacts(tmp_path: Path) -> None:
    protected = [
        tmp_path / "config" / "safety_policy.yaml",
        tmp_path / "reports" / "literature" / "literature_harness.json",
        tmp_path / "reports" / "learning_jobs" / "sentinel.json",
        tmp_path / "src" / "prompts" / "decision_prompts.py",
        tmp_path / "src" / "routes" / "router.py",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}: original\n", encoding="utf-8")
    before = _snapshot(tmp_path)

    request = CohortFeasibilityRequest(
        request_id="cohort_request_crc_001",
        project_id="research_crc_001",
        question="Is there enough structured CRC triage data?",
        cohort_criteria={
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": ["rectal_bleeding"],
        },
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": False,
            "deidentified_only": True,
        },
        version_refs={"projection_version": "patient_record_projection_v0"},
    )

    CohortFeasibilityService().evaluate(request=request, records=[])

    assert _snapshot(tmp_path) == before
```

- [ ] **Step 3: Run focused verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py tests/backend/test_research_api.py tests/backend/test_cohort_feasibility_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 4: Run inherited regression set**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_action_trace.py tests/backend/test_crc_harness_replay.py tests/backend/test_evidence_claim_contract.py tests/backend/test_admin_release_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit verification slice**

```powershell
git add .gitignore tests/backend/test_cohort_feasibility_non_mutation.py
git commit -m "test: guard cohort feasibility read-only boundary"
```

---

## Acceptance Verification

Run the complete Step 12 verification set:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py tests/backend/test_research_api.py tests/backend/test_cohort_feasibility_non_mutation.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_action_trace.py tests/backend/test_crc_harness_replay.py tests/backend/test_evidence_claim_contract.py tests/backend/test_admin_release_dashboard.py -q
```

Step 12 is complete when:

- The admin API returns aggregate-only feasibility results.
- `patient_level_rows_returned` is always `false`.
- Export requests return `blocked_by_governance` and patient identifier criteria are rejected.
- Every non-blocked result includes a `research_ethics_review` queue item.
- The service reads through the patient registry/projection boundary.
- Non-mutation tests prove no active runtime artifact is changed.

## Self-Review

Spec coverage:

- Research contracts: Task 1.
- Deterministic variable mapping and aggregate coverage: Task 2.
- Missing and unmapped variable detection: Task 2.
- Ethics review queue item: Task 2.
- Admin-only read API: Task 3.
- No patient-level export and no runtime mutation: Tasks 1, 2, and 4.
- Frontend panel: intentionally outside first implementation because the spec marks it optional and the API is sufficient for Step 12 acceptance.

Marker scan:

- No unresolved implementation markers are present.
- Every new file path has a task and verification command.
- Status names and endpoint paths are consistent across tasks.
