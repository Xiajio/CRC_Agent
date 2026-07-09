# LearningJob Candidate Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P2 Step 13 as a shadow-only `LearningJob` pipeline that turns reviewed or reviewable signals into auditable candidate patches without changing production prompts, rubrics, routes, templates, RAG indexes, safety policy, feature flags, or training data.

**Architecture:** Add contracts for learning signals, candidate patches, and learning jobs; add an append-only file store under `reports/learning_jobs/`; add a deterministic service that maps explicit deidentified signals to candidate patch artifacts; expose an admin-only read/create API. The pipeline creates reviewable artifacts and required harness metadata only.

**Tech Stack:** Python 3.10, dataclasses, FastAPI, Pydantic v2, pytest, JSON artifact store, existing admin auth middleware, existing release-governance and evidence-safety boundaries.

---

## Global Constraints

- Step 13 may write only under `reports/learning_jobs/`.
- Step 13 must not edit prompts, rubrics, routes, templates, safety policy, RAG indexes, model files, tool manifests, feature flags, release-governance artifacts, release-execution artifacts, patient records, doctor state, literature reports, or `CRC-client/`.
- Candidate patches must keep `applies_automatically: false`.
- Signals must be deidentified and must not include patient-level rows, hidden reasoning, prompts with secrets, API keys, bearer tokens, credentials, or training data.
- The first implementation intentionally excludes an Agent Admin frontend page. The backend API and artifact store are the acceptance surface.

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-08-learningjob-candidate-pipeline-design.md`
- `docs/superpowers/specs/2026-07-08-crc-cohort-feasibility-design.md`
- `docs/superpowers/plans/2026-07-09-crc-cohort-feasibility.md`
- `backend/api/schemas/doctor_action_trace.py`
- `src/contracts/evidence_claim.py`
- `src/contracts/harness.py`
- `src/contracts/release_safety_report.py`
- `src/services/release_governance.py`
- `src/services/release_execution.py`
- `backend/app.py`
- `tests/backend/test_auth_security.py`

## File Structure

Backend contracts:

- Create `src/contracts/learning_job.py`
  - Dataclass contracts for `LearningSignal`, `CandidatePatch`, `HarnessRequirement`, `HumanReviewRequirement`, and `LearningJob`.
  - Stable ID helpers, canonical hash helper, and forbidden payload guards.
  - Construction-time guards against active/applied/released/trained statuses and automatic application.
- Create `tests/backend/test_learning_job_contract.py`
  - Signal validation, candidate validation, job serialization, deterministic IDs, and forbidden payload tests.

Backend artifact store:

- Create `backend/api/services/learning_job_store.py`
  - File-backed append-only store under `reports/learning_jobs/`.
  - Write-once job and candidate artifacts.
  - Read state with integrity warnings for malformed artifacts.
- Create `tests/backend/test_learning_job_store.py`
  - Write-once behavior, read state, duplicate rejection, malformed artifact warning, and path containment tests.
- Create `reports/learning_jobs/README.md`
  - Documents local shadow artifacts and states that generated job JSON is review material only.

Backend service:

- Create `src/services/learning_job_service.py`
  - Validate explicit signals.
  - Map signals to candidate patch types.
  - Attach required harness metadata and human review roles.
  - Write artifacts via `LearningJobStore`.
- Create `tests/backend/test_learning_job_service.py`
  - Doctor action, evidence delta, harness failure, cohort gap, and monitoring alert signals.
  - Candidate creation, zero candidate behavior for weak signals, write-only learning root, and no runtime mutation.

Backend API:

- Create `backend/api/schemas/learning_jobs.py`
  - Pydantic request schemas for admin learning-job create.
- Create `backend/api/routes/learning_jobs.py`
  - `GET /api/admin/learning-jobs`
  - `POST /api/admin/learning-jobs`
- Modify `backend/app.py`
  - Include the learning jobs router.
  - Require admin token for both learning-job endpoints.
- Create `tests/backend/test_learning_jobs_api.py`
  - Admin auth, list, create, validation mapping, store unavailable, and no apply/release routes.
- Modify `tests/backend/test_auth_security.py`
  - Add learning-job endpoints to the admin-token matrix.

Regression:

- Create `tests/backend/test_learning_job_non_mutation.py`
  - Proves candidate creation writes only to `reports/learning_jobs/`.

---

### Task 1: LearningJob Contracts

**Files:**
- Modify: `.gitignore`
- Create: `src/contracts/learning_job.py`
- Create: `tests/backend/test_learning_job_contract.py`

**Interfaces:**
- Produces: `LearningSignal`, `CandidatePatch`, `HarnessRequirement`, `HumanReviewRequirement`, `LearningJob`.
- Produces: `canonical_learning_payload_hash(payload) -> str`.
- Produces: `make_learning_signal_id(source_ref) -> str`.
- Produces: `make_candidate_patch_id(patch_type, seed) -> str`.
- Produces: `make_learning_job_id(source_signal_ids, idempotency_key) -> str`.

- [ ] **Step 1: Add test whitelist entries**

Modify `.gitignore` near the backend test whitelist:

```gitignore
!tests/backend/test_learning_job_contract.py
!tests/backend/test_learning_job_store.py
!tests/backend/test_learning_job_service.py
!tests/backend/test_learning_jobs_api.py
!tests/backend/test_learning_job_non_mutation.py
```

- [ ] **Step 2: Write failing contract tests**

Create `tests/backend/test_learning_job_contract.py`:

```python
from __future__ import annotations

import pytest

from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    LearningSignal,
    canonical_learning_payload_hash,
    make_candidate_patch_id,
    make_learning_job_id,
    make_learning_signal_id,
)


def make_signal(**overrides: object) -> LearningSignal:
    payload = {
        "signal_id": make_learning_signal_id({"kind": "doctor_action_trace", "id": "doctor_trace_001"}),
        "signal_type": "doctor_action_trace",
        "source_ref": {"kind": "doctor_action_trace", "id": "doctor_trace_001"},
        "reason_code": "unsafe_disposition",
        "target_area": "prompt",
        "severity": "review_required",
        "summary": "Doctor marked a CRC disposition as unsafe because the escalation was too low.",
        "deidentified": True,
        "created_at": "2026-07-09T10:00:00+08:00",
    }
    payload.update(overrides)
    return LearningSignal(**payload)


def make_patch(signal: LearningSignal) -> CandidatePatch:
    return CandidatePatch(
        patch_id=make_candidate_patch_id("prompt", signal.signal_id),
        patch_type="prompt",
        target_ref={"kind": "prompt", "id": "assessment_prompt_crc_triage"},
        change_summary="Add escalation language for rectal bleeding risk.",
        proposed_diff={
            "format": "unified_diff",
            "content": "--- current\n+++ candidate\n@@ -1 +1 @@\n-Review risk.\n+Escalate older rectal bleeding risk for clinician review.",
        },
        source_signal_ids=[signal.signal_id],
        status="candidate",
        applies_automatically=False,
    )


def test_learning_job_contracts_round_trip() -> None:
    signal = make_signal()
    patch = make_patch(signal)
    job = LearningJob(
        job_id=make_learning_job_id([signal.signal_id], "learning-1"),
        job_type="candidate_patch_generation",
        status="shadow_only",
        created_at="2026-07-09T10:00:00+08:00",
        source_signal_ids=[signal.signal_id],
        candidate_patch_ids=[patch.patch_id],
        required_harness=HarnessRequirement(
            case_pack_version="crc_mutation_pack_v0",
            required_levels=["L0_L1"],
            hard_fail_policy="block_on_any_hard_fail",
        ),
        human_review=HumanReviewRequirement(
            required=True,
            required_roles=["clinical_safety_reviewer", "release_manager"],
            status="pending",
        ),
        release_governance_ref=None,
        idempotency_key="learning-1",
    )

    assert signal.to_dict()["deidentified"] is True
    assert patch.to_dict()["applies_automatically"] is False
    assert job.to_dict()["status"] == "shadow_only"
    assert job.to_dict()["required_harness"]["case_pack_version"] == "crc_mutation_pack_v0"


def test_signal_rejects_non_deidentified_input() -> None:
    with pytest.raises(ValueError, match="deidentified must be true"):
        make_signal(deidentified=False)


def test_signal_rejects_patient_level_rows() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        make_signal(source_ref={"kind": "cohort_rows", "patient_id": "p-1"})


def test_candidate_rejects_automatic_application() -> None:
    signal = make_signal()
    payload = make_patch(signal).to_dict()
    payload["applies_automatically"] = True

    with pytest.raises(ValueError, match="applies_automatically must be false"):
        CandidatePatch(**payload)


@pytest.mark.parametrize("status", ["applied", "released", "trained", "clinical_rag_active"])
def test_job_rejects_active_statuses(status: str) -> None:
    signal = make_signal()
    with pytest.raises(ValueError, match="status must be one of"):
        LearningJob(
            job_id="learning_job_bad",
            job_type="candidate_patch_generation",
            status=status,
            created_at="2026-07-09T10:00:00+08:00",
            source_signal_ids=[signal.signal_id],
            candidate_patch_ids=[],
            required_harness=HarnessRequirement(
                case_pack_version="crc_mutation_pack_v0",
                required_levels=["L0_L1"],
                hard_fail_policy="block_on_any_hard_fail",
            ),
            human_review=HumanReviewRequirement(
                required=True,
                required_roles=["clinical_safety_reviewer"],
                status="pending",
            ),
            release_governance_ref=None,
            idempotency_key="learning-1",
        )


def test_canonical_hash_rejects_secret_content() -> None:
    with pytest.raises(ValueError, match="forbidden content"):
        canonical_learning_payload_hash({"note": "Bearer abcdef123456"})
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.contracts.learning_job'`.

- [ ] **Step 3: Implement learning contracts**

Create `src/contracts/learning_job.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
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

SignalType = Literal[
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
]
TargetArea = Literal["prompt", "rubric", "route", "template", "evidence_ingest", "test_case"]
CandidatePatchType = Literal["prompt", "rubric", "route", "template", "evidence_ingest", "test_case"]
CandidateStatus = Literal[
    "candidate",
    "needs_harness",
    "needs_human_review",
    "rejected",
    "approved_for_release_intent",
]
LearningJobStatus = Literal[
    "draft",
    "shadow_only",
    "ready_for_harness",
    "harness_failed",
    "awaiting_human_review",
    "rejected",
    "approved_for_release_intent",
    "archived",
]
LearningJobType = Literal[
    "candidate_patch_generation",
    "candidate_evidence_ingest",
    "candidate_test_case_generation",
]

SIGNAL_TYPES = (
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
)
TARGET_AREAS = ("prompt", "rubric", "route", "template", "evidence_ingest", "test_case")
CANDIDATE_STATUSES = (
    "candidate",
    "needs_harness",
    "needs_human_review",
    "rejected",
    "approved_for_release_intent",
)
LEARNING_JOB_STATUSES = (
    "draft",
    "shadow_only",
    "ready_for_harness",
    "harness_failed",
    "awaiting_human_review",
    "rejected",
    "approved_for_release_intent",
    "archived",
)
LEARNING_JOB_TYPES = (
    "candidate_patch_generation",
    "candidate_evidence_ingest",
    "candidate_test_case_generation",
)
FORBIDDEN_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "bearer_token",
    "chain_of_thought",
    "credential",
    "credentials",
    "hidden_reasoning",
    "medical_record_number",
    "mrn",
    "patient_id",
    "patient_ids",
    "patient_name",
    "patient_record",
    "patient_records",
    "prompt_secret",
    "secret",
    "token",
    "training_rows",
}
FORBIDDEN_CONTENT_RE = re.compile(
    r"(bearer\s+[a-z0-9._-]{6,}|api[_-]?key\s*[:=]\s*[a-z0-9._-]{5,}|password\s*[:=])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LearningSignal:
    signal_id: str
    signal_type: SignalType
    source_ref: dict[str, JsonValue]
    reason_code: str
    target_area: TargetArea
    severity: str
    summary: str
    deidentified: bool
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty("signal_id", self.signal_id)
        _validate_choice("signal_type", self.signal_type, SIGNAL_TYPES)
        _validate_choice("target_area", self.target_area, TARGET_AREAS)
        _require_non_empty("reason_code", self.reason_code)
        _require_non_empty("severity", self.severity)
        _require_non_empty("summary", self.summary)
        _require_non_empty("created_at", self.created_at)
        if self.deidentified is not True:
            raise ValueError("deidentified must be true")
        _validate_safe_payload(self.source_ref)
        _validate_safe_payload({"summary": self.summary})

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_ref": dict(self.source_ref),
            "reason_code": self.reason_code,
            "target_area": self.target_area,
            "severity": self.severity,
            "summary": self.summary,
            "deidentified": True,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CandidatePatch:
    patch_id: str
    patch_type: CandidatePatchType
    target_ref: dict[str, JsonValue]
    change_summary: str
    proposed_diff: dict[str, JsonValue]
    source_signal_ids: list[str]
    status: CandidateStatus
    applies_automatically: bool

    def __post_init__(self) -> None:
        _require_non_empty("patch_id", self.patch_id)
        _validate_choice("patch_type", self.patch_type, TARGET_AREAS)
        _validate_choice("status", self.status, CANDIDATE_STATUSES)
        _validate_safe_payload(self.target_ref)
        _validate_safe_payload(self.proposed_diff)
        _validate_safe_payload({"change_summary": self.change_summary})
        _require_string_list("source_signal_ids", self.source_signal_ids)
        if self.applies_automatically is not False:
            raise ValueError("applies_automatically must be false")
        target_kind = str(self.target_ref.get("kind", ""))
        target_id = str(self.target_ref.get("id", ""))
        if "clinical_safety_policy" in {target_kind, target_id}:
            raise ValueError("ClinicalSafetyPolicyVersion targets are outside Step 13")

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "patch_type": self.patch_type,
            "target_ref": dict(self.target_ref),
            "change_summary": self.change_summary,
            "proposed_diff": dict(self.proposed_diff),
            "source_signal_ids": list(self.source_signal_ids),
            "status": self.status,
            "applies_automatically": False,
        }


@dataclass(frozen=True)
class HarnessRequirement:
    case_pack_version: str
    required_levels: list[str]
    hard_fail_policy: str

    def __post_init__(self) -> None:
        _require_non_empty("case_pack_version", self.case_pack_version)
        _require_string_list("required_levels", self.required_levels)
        _require_non_empty("hard_fail_policy", self.hard_fail_policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_pack_version": self.case_pack_version,
            "required_levels": list(self.required_levels),
            "hard_fail_policy": self.hard_fail_policy,
        }


@dataclass(frozen=True)
class HumanReviewRequirement:
    required: bool
    required_roles: list[str]
    status: str

    def __post_init__(self) -> None:
        if self.required is not True:
            raise ValueError("human review is required")
        _require_string_list("required_roles", self.required_roles)
        _require_non_empty("status", self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": True,
            "required_roles": list(self.required_roles),
            "status": self.status,
        }


@dataclass(frozen=True)
class LearningJob:
    job_id: str
    job_type: LearningJobType
    status: LearningJobStatus
    created_at: str
    source_signal_ids: list[str]
    candidate_patch_ids: list[str]
    required_harness: HarnessRequirement
    human_review: HumanReviewRequirement
    release_governance_ref: str | None
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_non_empty("job_id", self.job_id)
        _validate_choice("job_type", self.job_type, LEARNING_JOB_TYPES)
        _validate_choice("status", self.status, LEARNING_JOB_STATUSES)
        _require_non_empty("created_at", self.created_at)
        _require_string_list("source_signal_ids", self.source_signal_ids)
        _require_string_list(
            "candidate_patch_ids",
            self.candidate_patch_ids,
            allow_empty=True,
        )
        _require_non_empty("idempotency_key", self.idempotency_key)
        if self.release_governance_ref is not None:
            _require_non_empty("release_governance_ref", self.release_governance_ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "source_signal_ids": list(self.source_signal_ids),
            "candidate_patch_ids": list(self.candidate_patch_ids),
            "required_harness": self.required_harness.to_dict(),
            "human_review": self.human_review.to_dict(),
            "release_governance_ref": self.release_governance_ref,
            "idempotency_key": self.idempotency_key,
        }


def canonical_learning_payload_hash(payload: dict[str, JsonValue]) -> str:
    _validate_safe_payload(payload)
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def make_learning_signal_id(source_ref: dict[str, JsonValue]) -> str:
    return "learning_signal_" + canonical_learning_payload_hash(source_ref).split(":", 1)[1][:12]


def make_candidate_patch_id(patch_type: str, seed: str) -> str:
    digest = hashlib.sha256(f"{patch_type}:{seed}".encode("utf-8")).hexdigest()[:12]
    return f"candidate_{patch_type}_patch_{digest}"


def make_learning_job_id(source_signal_ids: list[str], idempotency_key: str) -> str:
    seed = json.dumps(
        {"source_signal_ids": sorted(source_signal_ids), "idempotency_key": idempotency_key},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "learning_job_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def _require_string_list(
    name: str,
    values: list[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a non-empty list")
    if not values and not allow_empty:
        raise ValueError(f"{name} must be a non-empty list")
    for value in values:
        _require_non_empty(name, value)


def _validate_safe_payload(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("payload must be JSON-safe")
    if value is None or isinstance(value, (int, bool)):
        return
    if isinstance(value, str):
        if FORBIDDEN_CONTENT_RE.search(value):
            raise ValueError("forbidden content in learning payload")
        return
    if isinstance(value, list):
        for item in value:
            _validate_safe_payload(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("payload keys must be strings")
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key in learning payload: {key}")
            _validate_safe_payload(item)
        return
    raise TypeError("payload must be JSON-safe")
```

- [ ] **Step 4: Run contract tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit contract slice**

```powershell
git add .gitignore src/contracts/learning_job.py tests/backend/test_learning_job_contract.py
git commit -m "feat: add learning job contracts"
```

---

### Task 2: Append-Only LearningJob Store

**Files:**
- Create: `backend/api/services/learning_job_store.py`
- Create: `tests/backend/test_learning_job_store.py`
- Create: `reports/learning_jobs/README.md`

**Interfaces:**
- Produces: `LearningJobStore(root).read_state() -> LearningJobState`.
- Produces: `LearningJobStore(root).write_job(job, candidates) -> None`.

- [ ] **Step 1: Write failing store tests**

Create `tests/backend/test_learning_job_store.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.learning_job_store import LearningJobIntegrityError, LearningJobStore
from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    LearningSignal,
    make_candidate_patch_id,
    make_learning_job_id,
    make_learning_signal_id,
)


def _signal() -> LearningSignal:
    return LearningSignal(
        signal_id=make_learning_signal_id({"kind": "doctor_action_trace", "id": "doctor_trace_001"}),
        signal_type="doctor_action_trace",
        source_ref={"kind": "doctor_action_trace", "id": "doctor_trace_001"},
        reason_code="unsafe_disposition",
        target_area="prompt",
        severity="review_required",
        summary="Doctor marked a CRC disposition as unsafe.",
        deidentified=True,
        created_at="2026-07-09T10:00:00+08:00",
    )


def _candidate(signal: LearningSignal) -> CandidatePatch:
    return CandidatePatch(
        patch_id=make_candidate_patch_id("prompt", signal.signal_id),
        patch_type="prompt",
        target_ref={"kind": "prompt", "id": "assessment_prompt_crc_triage"},
        change_summary="Add explicit escalation language.",
        proposed_diff={"format": "unified_diff", "content": "--- current\n+++ candidate"},
        source_signal_ids=[signal.signal_id],
        status="candidate",
        applies_automatically=False,
    )


def _job(signal: LearningSignal, candidate: CandidatePatch) -> LearningJob:
    return LearningJob(
        job_id=make_learning_job_id([signal.signal_id], "learning-1"),
        job_type="candidate_patch_generation",
        status="shadow_only",
        created_at="2026-07-09T10:00:00+08:00",
        source_signal_ids=[signal.signal_id],
        candidate_patch_ids=[candidate.patch_id],
        required_harness=HarnessRequirement(
            case_pack_version="crc_mutation_pack_v0",
            required_levels=["L0_L1"],
            hard_fail_policy="block_on_any_hard_fail",
        ),
        human_review=HumanReviewRequirement(
            required=True,
            required_roles=["clinical_safety_reviewer", "release_manager"],
            status="pending",
        ),
        release_governance_ref=None,
        idempotency_key="learning-1",
    )


def test_store_writes_and_reads_learning_job(tmp_path: Path) -> None:
    signal = _signal()
    candidate = _candidate(signal)
    job = _job(signal, candidate)
    store = LearningJobStore(tmp_path / "reports" / "learning_jobs")

    store.write_job(job, [candidate])
    state = store.read_state()

    assert [item.job_id for item in state.jobs] == [job.job_id]
    assert [item.patch_id for item in state.candidates] == [candidate.patch_id]
    assert state.integrity == {"status": "verified", "warnings": []}
    assert (tmp_path / "reports" / "learning_jobs" / "jobs" / f"{job.job_id}.json").exists()
    assert (tmp_path / "reports" / "learning_jobs" / "candidates" / f"{candidate.patch_id}.json").exists()


def test_store_rejects_duplicate_job_write(tmp_path: Path) -> None:
    signal = _signal()
    candidate = _candidate(signal)
    job = _job(signal, candidate)
    store = LearningJobStore(tmp_path / "reports" / "learning_jobs")

    store.write_job(job, [candidate])

    with pytest.raises(FileExistsError, match="learning job already exists"):
        store.write_job(job, [candidate])


def test_store_reports_malformed_artifact_warning(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    (root / "jobs").mkdir(parents=True)
    (root / "jobs" / "bad.json").write_text("{not json", encoding="utf-8")
    store = LearningJobStore(root)

    state = store.read_state()

    assert state.jobs == []
    assert state.integrity["status"] == "warning"
    assert "bad.json" in state.integrity["warnings"][0]


def test_store_keeps_paths_inside_root(tmp_path: Path) -> None:
    store = LearningJobStore(tmp_path / "reports" / "learning_jobs")

    with pytest.raises(LearningJobIntegrityError, match="unsafe artifact id"):
        store._artifact_path(store.jobs_dir, "../escape")
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_store.py -q`

Expected: FAIL because `backend.api.services.learning_job_store` does not exist.

- [ ] **Step 2: Implement store**

Create `backend/api/services/learning_job_store.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from src.contracts.learning_job import CandidatePatch, LearningJob


class LearningJobIntegrityError(RuntimeError):
    """Raised when learning-job artifacts are unsafe to read or write."""


@dataclass(frozen=True)
class LearningJobState:
    jobs: list[LearningJob]
    candidates: list[CandidatePatch]
    integrity: dict[str, Any]


_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class LearningJobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.candidates_dir = self.root / "candidates"

    def read_state(self) -> LearningJobState:
        job_result = self._read_artifacts(self.jobs_dir, LearningJob)
        candidate_result = self._read_artifacts(self.candidates_dir, CandidatePatch)
        warnings = job_result["warnings"] + candidate_result["warnings"]
        return LearningJobState(
            jobs=job_result["items"],
            candidates=candidate_result["items"],
            integrity={
                "status": "verified" if not warnings else "warning",
                "warnings": warnings,
            },
        )

    def write_job(self, job: LearningJob, candidates: list[CandidatePatch]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        job_path = self._artifact_path(self.jobs_dir, job.job_id)
        candidate_paths = [
            self._artifact_path(self.candidates_dir, candidate.patch_id)
            for candidate in candidates
        ]
        if job_path.exists():
            raise FileExistsError("learning job already exists")
        for path in candidate_paths:
            if path.exists():
                raise FileExistsError("candidate patch already exists")

        written: list[Path] = []
        try:
            for candidate, path in zip(candidates, candidate_paths, strict=True):
                self._write_json_once(path, candidate.to_dict())
                written.append(path)
            self._write_json_once(job_path, job.to_dict())
            written.append(job_path)
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise

    def _artifact_path(self, directory: Path, artifact_id: str) -> Path:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise LearningJobIntegrityError("unsafe artifact id")
        path = directory / f"{artifact_id}.json"
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise LearningJobIntegrityError("unsafe artifact path") from exc
        return path

    def _write_json_once(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_artifacts(self, directory: Path, cls):
        items: list[Any] = []
        warnings: list[str] = []
        if not directory.exists():
            return {"items": items, "warnings": warnings}
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("artifact must be a JSON object")
                items.append(cls(**payload))
            except Exception as exc:
                warnings.append(f"{path.name}: {exc}")
        return {"items": items, "warnings": warnings}
```

- [ ] **Step 3: Add artifact README**

Create `reports/learning_jobs/README.md`:

```markdown
# LearningJob Shadow Artifacts

This directory stores Step 13 shadow-only learning job artifacts.

- `jobs/*.json` records candidate-only learning jobs.
- `candidates/*.json` records candidate patch payloads.

Artifacts in this directory are review material. They do not apply prompts, rubrics, routes, templates, RAG indexes, safety policy, feature flags, or model training data.
```

- [ ] **Step 4: Run store tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit store slice**

```powershell
git add backend/api/services/learning_job_store.py tests/backend/test_learning_job_store.py reports/learning_jobs/README.md
git commit -m "feat: add append-only learning job store"
```

---

### Task 3: LearningJob Service

**Files:**
- Create: `src/services/learning_job_service.py`
- Create: `tests/backend/test_learning_job_service.py`

**Interfaces:**
- Produces: `LearningJobService.read_jobs() -> dict[str, object]`.
- Produces: `LearningJobService.create_job(signals, requested_by, idempotency_key) -> dict[str, object]`.

- [ ] **Step 1: Write failing service tests**

Create `tests/backend/test_learning_job_service.py`:

```python
from __future__ import annotations

from pathlib import Path

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import LearningJobService


def _signal(reason_code: str, target_area: str, signal_type: str = "doctor_action_trace") -> LearningSignal:
    source_ref = {"kind": signal_type, "id": f"{signal_type}_{reason_code}"}
    return LearningSignal(
        signal_id=make_learning_signal_id(source_ref),
        signal_type=signal_type,
        source_ref=source_ref,
        reason_code=reason_code,
        target_area=target_area,
        severity="review_required",
        summary=f"{reason_code} requires candidate review.",
        deidentified=True,
        created_at="2026-07-09T10:00:00+08:00",
    )


def _service(tmp_path: Path) -> LearningJobService:
    return LearningJobService(
        store=LearningJobStore(tmp_path / "reports" / "learning_jobs"),
        now=lambda: "2026-07-09T10:00:00+08:00",
    )


def test_create_prompt_candidate_from_doctor_signal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    response = service.create_job(
        signals=[_signal("unsafe_disposition", "prompt")],
        requested_by="release_manager",
        idempotency_key="learning-1",
    )

    assert response["job"]["status"] == "shadow_only"
    assert response["candidates"][0]["patch_type"] == "prompt"
    assert response["candidates"][0]["applies_automatically"] is False
    assert response["job"]["required_harness"]["case_pack_version"] == "crc_mutation_pack_v0"
    assert "clinical_safety_reviewer" in response["job"]["human_review"]["required_roles"]


def test_create_evidence_ingest_candidate_from_evidence_delta(tmp_path: Path) -> None:
    service = _service(tmp_path)
    response = service.create_job(
        signals=[_signal("citation_not_traceable", "evidence_ingest", "evidence_delta")],
        requested_by="evidence_reviewer",
        idempotency_key="learning-2",
    )

    assert response["candidates"][0]["patch_type"] == "evidence_ingest"
    assert "evidence_reviewer" in response["job"]["human_review"]["required_roles"]


def test_weak_signal_creates_shadow_job_with_no_candidates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    response = service.create_job(
        signals=[_signal("documentation_note", "template")],
        requested_by="release_manager",
        idempotency_key="learning-3",
    )

    assert response["job"]["status"] == "shadow_only"
    assert response["candidates"] == []


def test_read_jobs_returns_store_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_job(
        signals=[_signal("unsafe_disposition", "prompt")],
        requested_by="release_manager",
        idempotency_key="learning-1",
    )

    response = service.read_jobs()

    assert len(response["jobs"]) == 1
    assert len(response["candidates"]) == 1
    assert response["runtime"] == {
        "auth": "admin",
        "source": "reports/learning_jobs",
        "mode": "shadow_learning_jobs",
    }
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_service.py -q`

Expected: FAIL because `src.services.learning_job_service` does not exist.

- [ ] **Step 2: Implement service**

Create `src/services/learning_job_service.py`:

```python
from __future__ import annotations

from typing import Callable

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    LearningSignal,
    make_candidate_patch_id,
    make_learning_job_id,
)


class LearningJobValidationError(ValueError):
    """Raised when a learning job request is outside Step 13 boundaries."""


class LearningJobService:
    def __init__(self, *, store: LearningJobStore, now: Callable[[], str]) -> None:
        self._store = store
        self._now = now

    def read_jobs(self) -> dict[str, object]:
        state = self._store.read_state()
        return {
            "jobs": [job.to_dict() for job in state.jobs],
            "candidates": [candidate.to_dict() for candidate in state.candidates],
            "integrity": state.integrity,
            "disabled_actions": [
                {
                    "id": "apply_candidate_patch",
                    "label": "Apply candidate patch",
                    "disabled": True,
                    "reason": "Step 13 is shadow-only; release governance must approve later changes.",
                },
                {
                    "id": "train_from_feedback",
                    "label": "Train from feedback",
                    "disabled": True,
                    "reason": "Doctor feedback and cohort gaps are not training data.",
                },
            ],
            "runtime": {
                "auth": "admin",
                "source": "reports/learning_jobs",
                "mode": "shadow_learning_jobs",
            },
        }

    def create_job(
        self,
        *,
        signals: list[LearningSignal],
        requested_by: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if not signals:
            raise LearningJobValidationError("at least one learning signal is required")
        if not requested_by.strip():
            raise LearningJobValidationError("requested_by is required")
        if not idempotency_key.strip():
            raise LearningJobValidationError("idempotency_key is required")

        candidates = self._build_candidates(signals)
        source_signal_ids = [signal.signal_id for signal in signals]
        job_type = "candidate_evidence_ingest" if _only_evidence(candidates) else "candidate_patch_generation"
        job = LearningJob(
            job_id=make_learning_job_id(source_signal_ids, idempotency_key),
            job_type=job_type,
            status="shadow_only",
            created_at=self._now(),
            source_signal_ids=source_signal_ids,
            candidate_patch_ids=[candidate.patch_id for candidate in candidates],
            required_harness=self._harness_for(candidates),
            human_review=self._human_review_for(candidates),
            release_governance_ref=None,
            idempotency_key=idempotency_key,
        )
        self._store.write_job(job, candidates)
        return {
            "job": job.to_dict(),
            "signals": [signal.to_dict() for signal in signals],
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    def _build_candidates(self, signals: list[LearningSignal]) -> list[CandidatePatch]:
        candidates: list[CandidatePatch] = []
        for signal in signals:
            patch_type = _patch_type_for(signal)
            if patch_type is None:
                continue
            candidates.append(
                CandidatePatch(
                    patch_id=make_candidate_patch_id(patch_type, signal.signal_id),
                    patch_type=patch_type,
                    target_ref=_target_ref_for(patch_type),
                    change_summary=_change_summary_for(signal, patch_type),
                    proposed_diff={
                        "format": "structured_candidate",
                        "content": {
                            "signal_id": signal.signal_id,
                            "reason_code": signal.reason_code,
                            "target_area": signal.target_area,
                            "summary": signal.summary,
                        },
                    },
                    source_signal_ids=[signal.signal_id],
                    status="candidate",
                    applies_automatically=False,
                )
            )
        return candidates

    def _harness_for(self, candidates: list[CandidatePatch]) -> HarnessRequirement:
        levels = ["L0_L1"]
        if any(candidate.patch_type == "evidence_ingest" for candidate in candidates):
            levels.append("literature_shadow")
        return HarnessRequirement(
            case_pack_version="crc_mutation_pack_v0",
            required_levels=sorted(set(levels)),
            hard_fail_policy="block_on_any_hard_fail",
        )

    def _human_review_for(self, candidates: list[CandidatePatch]) -> HumanReviewRequirement:
        roles = {"release_manager"}
        for candidate in candidates:
            if candidate.patch_type in {"prompt", "rubric", "route", "template", "test_case"}:
                roles.add("clinical_safety_reviewer")
            if candidate.patch_type == "evidence_ingest":
                roles.add("evidence_reviewer")
        if not candidates:
            roles.add("release_manager")
        return HumanReviewRequirement(
            required=True,
            required_roles=sorted(roles),
            status="pending",
        )


def _patch_type_for(signal: LearningSignal) -> str | None:
    strong_reason_codes = {
        "unsafe_disposition",
        "citation_not_traceable",
        "evidence_conflict",
        "safety_signal",
        "harness_hard_fail",
        "missing_variable",
        "monitoring_alert",
    }
    if signal.reason_code not in strong_reason_codes:
        return None
    if signal.target_area in {"prompt", "rubric", "route", "template", "evidence_ingest", "test_case"}:
        return signal.target_area
    return None


def _target_ref_for(patch_type: str) -> dict[str, str]:
    targets = {
        "prompt": {"kind": "prompt", "id": "assessment_prompt_crc_triage"},
        "rubric": {"kind": "rubric", "id": "crc_safety_judge_rubric"},
        "route": {"kind": "route", "id": "crc_triage_route"},
        "template": {"kind": "template", "id": "crc_patient_or_doctor_copy"},
        "evidence_ingest": {"kind": "evidence_ingest_candidate", "id": "literature_review_queue"},
        "test_case": {"kind": "harness_case", "id": "crc_mutation_pack_candidate"},
    }
    return targets[patch_type]


def _change_summary_for(signal: LearningSignal, patch_type: str) -> str:
    return f"Create {patch_type} candidate from {signal.signal_type}:{signal.reason_code}."


def _only_evidence(candidates: list[CandidatePatch]) -> bool:
    return bool(candidates) and all(candidate.patch_type == "evidence_ingest" for candidate in candidates)
```

- [ ] **Step 3: Run service tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_service.py -q`

Expected: PASS.

- [ ] **Step 4: Commit service slice**

```powershell
git add src/services/learning_job_service.py tests/backend/test_learning_job_service.py
git commit -m "feat: add shadow learning job service"
```

---

### Task 4: Admin LearningJob API

**Files:**
- Create: `backend/api/schemas/learning_jobs.py`
- Create: `backend/api/routes/learning_jobs.py`
- Modify: `backend/app.py`
- Create: `tests/backend/test_learning_jobs_api.py`
- Modify: `tests/backend/test_auth_security.py`

**Interfaces:**
- Produces: `GET /api/admin/learning-jobs`.
- Produces: `POST /api/admin/learning-jobs`.
- Does not produce any apply, release, train, or RAG-ingest endpoint.

- [ ] **Step 1: Write failing API tests**

Create `tests/backend/test_learning_jobs_api.py`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
import backend.api.routes.learning_jobs as learning_job_routes


def _signal_payload() -> dict[str, object]:
    return {
        "signal_type": "doctor_action_trace",
        "source_ref": {"kind": "doctor_action_trace", "id": "doctor_trace_001"},
        "reason_code": "unsafe_disposition",
        "target_area": "prompt",
        "severity": "review_required",
        "summary": "Doctor marked disposition as unsafe.",
        "deidentified": True,
        "created_at": "2026-07-09T10:00:00+08:00",
    }


def test_get_learning_jobs_returns_state(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")

    class Service:
        def read_jobs(self):
            return {"jobs": [], "candidates": [], "integrity": {"status": "verified", "warnings": []}}

    monkeypatch.setattr(learning_job_routes, "_learning_job_service", lambda: Service())
    client = TestClient(create_app())

    response = client.get(
        "/api/admin/learning-jobs",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["jobs"] == []


def test_create_learning_job_returns_shadow_candidate(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")

    class Service:
        def create_job(self, **kwargs):
            return {
                "job": {"job_id": "learning_job_1", "status": "shadow_only"},
                "signals": [],
                "candidates": [{"patch_id": "candidate_prompt_patch_1", "applies_automatically": False}],
            }

    monkeypatch.setattr(learning_job_routes, "_learning_job_service", lambda: Service())
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/learning-jobs",
        headers={"Authorization": "Bearer admin-token"},
        json={
            "signals": [_signal_payload()],
            "requested_by": "release_manager",
            "idempotency_key": "learning-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["job"]["status"] == "shadow_only"
    assert response.json()["candidates"][0]["applies_automatically"] is False


def test_create_learning_job_rejects_non_deidentified_signal(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")
    client = TestClient(create_app())
    payload = _signal_payload()
    payload["deidentified"] = False

    response = client.post(
        "/api/admin/learning-jobs",
        headers={"Authorization": "Bearer admin-token"},
        json={
            "signals": [payload],
            "requested_by": "release_manager",
            "idempotency_key": "learning-1",
        },
    )

    assert response.status_code == 422


def test_learning_job_api_has_no_apply_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/learning-jobs/learning_job_1/apply",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert response.status_code == 404
```

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_jobs_api.py -q`

Expected: FAIL because `backend.api.routes.learning_jobs` does not exist.

- [ ] **Step 2: Add API schemas**

Create `backend/api/schemas/learning_jobs.py`:

```python
from __future__ import annotations

from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]
SignalType = Literal[
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
]
TargetArea = Literal["prompt", "rubric", "route", "template", "evidence_ingest", "test_case"]


class LearningSignalPayload(BaseModel):
    signal_type: SignalType
    source_ref: dict[str, Any]
    reason_code: NonEmptyString
    target_area: TargetArea
    severity: NonEmptyString
    summary: NonEmptyString
    deidentified: bool = True
    created_at: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class LearningJobCreateRequest(BaseModel):
    signals: list[LearningSignalPayload] = Field(min_length=1)
    requested_by: NonEmptyString
    idempotency_key: NonEmptyString

    model_config = ConfigDict(extra="forbid")
```

- [ ] **Step 3: Add API route**

Create `backend/api/routes/learning_jobs.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.schemas.learning_jobs import LearningJobCreateRequest
from backend.api.services.admin_release_dashboard import REPO_ROOT
from backend.api.services.learning_job_store import LearningJobIntegrityError, LearningJobStore
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import LearningJobService, LearningJobValidationError


router = APIRouter(prefix="/api/admin/learning-jobs", tags=["admin-learning-jobs"])
_LEARNING_JOB_STORE_ROOT = REPO_ROOT / "reports" / "learning_jobs"


def _timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _learning_job_service() -> LearningJobService:
    return LearningJobService(
        store=LearningJobStore(_LEARNING_JOB_STORE_ROOT),
        now=_timestamp,
    )


@router.get("")
async def get_admin_learning_jobs() -> dict[str, Any]:
    try:
        return _learning_job_service().read_jobs()
    except LearningJobIntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("")
async def create_admin_learning_job(payload: LearningJobCreateRequest) -> dict[str, Any]:
    try:
        signals = [
            LearningSignal(
                signal_id=make_learning_signal_id(item.source_ref),
                **item.model_dump(),
            )
            for item in payload.signals
        ]
        return _learning_job_service().create_job(
            signals=signals,
            requested_by=payload.requested_by,
            idempotency_key=payload.idempotency_key,
        )
    except (LearningJobValidationError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LearningJobIntegrityError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

- [ ] **Step 4: Wire app and admin auth**

Modify `backend/app.py` imports:

```python
from backend.api.routes import learning_jobs as learning_job_routes
```

Modify `_requires_admin_token`:

```python
    if path == "/api/admin/learning-jobs" and method in {"GET", "POST"}:
        return True
```

Modify `create_app()` router includes:

```python
    app.include_router(learning_job_routes.router)
```

- [ ] **Step 5: Extend auth-security matrix**

In `tests/backend/test_auth_security.py`, add the stub routes inside `_auth_client`:

```python
    @app.get("/api/admin/learning-jobs")
    async def admin_learning_jobs() -> dict[str, object]:
        return {"runtime": {"auth": "admin", "mode": "shadow_learning_jobs"}}

    @app.post("/api/admin/learning-jobs")
    async def admin_create_learning_job() -> dict[str, object]:
        return {"job": {"status": "shadow_only"}}
```

Add these endpoint tuples to each admin endpoint parameter list:

```python
("get", "/api/admin/learning-jobs"),
("post", "/api/admin/learning-jobs"),
```

- [ ] **Step 6: Run API and auth tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_jobs_api.py tests/backend/test_auth_security.py -q`

Expected: PASS.

- [ ] **Step 7: Commit API slice**

```powershell
git add backend/api/schemas/learning_jobs.py backend/api/routes/learning_jobs.py backend/app.py tests/backend/test_learning_jobs_api.py tests/backend/test_auth_security.py
git commit -m "feat: expose admin learning job API"
```

---

### Task 5: Non-Mutation And Regression Verification

**Files:**
- Create: `tests/backend/test_learning_job_non_mutation.py`

- [ ] **Step 1: Write non-mutation test**

Create `tests/backend/test_learning_job_non_mutation.py`:

```python
from __future__ import annotations

from pathlib import Path

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import LearningJobService


def _snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path.startswith("reports/learning_jobs/"):
            continue
        snapshot[relative_path] = path.read_text(encoding="utf-8")
    return snapshot


def test_learning_job_writes_only_to_learning_job_root(tmp_path: Path) -> None:
    protected_paths = [
        tmp_path / "config" / "safety_policy.yaml",
        tmp_path / "src" / "prompts" / "decision_prompts.py",
        tmp_path / "src" / "routes" / "router.py",
        tmp_path / "reports" / "release_governance" / "intent.json",
        tmp_path / "reports" / "release_execution" / "feature_flags" / "current.json",
        tmp_path / "reports" / "literature" / "literature_harness.json",
        tmp_path / "reports" / "harness" / "harness.json",
    ]
    for path in protected_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}: original\n", encoding="utf-8")
    before = _snapshot(tmp_path)

    signal = LearningSignal(
        signal_id=make_learning_signal_id({"kind": "doctor_action_trace", "id": "doctor_trace_001"}),
        signal_type="doctor_action_trace",
        source_ref={"kind": "doctor_action_trace", "id": "doctor_trace_001"},
        reason_code="unsafe_disposition",
        target_area="prompt",
        severity="review_required",
        summary="Doctor marked disposition as unsafe.",
        deidentified=True,
        created_at="2026-07-09T10:00:00+08:00",
    )
    service = LearningJobService(
        store=LearningJobStore(tmp_path / "reports" / "learning_jobs"),
        now=lambda: "2026-07-09T10:00:00+08:00",
    )

    service.create_job(
        signals=[signal],
        requested_by="release_manager",
        idempotency_key="learning-1",
    )

    assert _snapshot(tmp_path) == before
    written = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "reports" / "learning_jobs").rglob("*")
        if path.is_file()
    )
    assert written
    assert all(path.startswith("reports/learning_jobs/") for path in written)
```

- [ ] **Step 2: Run focused verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_store.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py tests/backend/test_learning_job_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run inherited regression set**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_action_trace.py tests/backend/test_literature_harness.py tests/backend/test_crc_harness_replay.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit verification slice**

```powershell
git add tests/backend/test_learning_job_non_mutation.py
git commit -m "test: guard learning job shadow-only boundary"
```

---

## Acceptance Verification

Run the complete Step 13 verification set:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_store.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py tests/backend/test_learning_job_non_mutation.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_action_trace.py tests/backend/test_literature_harness.py tests/backend/test_crc_harness_replay.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py -q
```

Step 13 is complete when:

- Learning jobs are created as `shadow_only`.
- Candidate patches keep `applies_automatically: false`.
- Learning artifacts are append-only under `reports/learning_jobs/`.
- Admin API can list and create jobs, but no apply/release/train/RAG-ingest endpoint exists.
- Tests prove active prompt, rubric, route, template, RAG, policy, release, feature-flag, patient, and doctor artifacts are unchanged.

## Self-Review

Spec coverage:

- `LearningSignal`, `LearningJob`, and candidate patch contracts: Task 1.
- Append-only artifact store: Task 2.
- Signal-to-candidate mapping: Task 3.
- Harness and human review metadata: Task 3.
- Admin read/create API: Task 4.
- No runtime patch application or training data creation: Tasks 1, 3, 4, and 5.
- Frontend learning-job page: intentionally outside first implementation because the spec marks it optional and backend artifacts are sufficient for Step 13 acceptance.

Marker scan:

- No unresolved implementation markers are present.
- Every created or modified file path has a task and verification command.
- Status names, endpoint paths, and artifact roots are consistent across tasks.
