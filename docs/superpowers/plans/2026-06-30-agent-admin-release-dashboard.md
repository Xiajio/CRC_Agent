# Agent Admin Release Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P1.5 Step 11: a read-only Agent Admin release dashboard backed by committed P0/P1.5 release reports.

**Architecture:** Backend owns report interpretation through a pure release-dashboard normalizer and exposes one admin-protected read-only endpoint. Frontend adds typed API access and a new Agent Admin task page that renders version chain, harness runs, hard fails, rollback target, sign-off readiness, Step 10 shadow status, and disabled mutation controls.

**Tech Stack:** Python 3.10, FastAPI, pytest, TypeScript, React, Vitest, Testing Library, existing Agent Admin UI components.

---

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-06-22-agent-tool-manifest-admin-api-design.md`

Step 11 must remain read-only. Do not add release, sign-off, rollback, feature flag, RAG ingest, policy editing, prompt editing, training, LearningJob, or `CRC-client/` changes.

## File Structure

Backend:

- Create `backend/api/services/admin_release_dashboard.py`
  - Pure report reader and normalizer.
  - Owns all release dashboard status derivation.
  - Reads committed static reports by default.
  - Accepts explicit paths for tests.
- Create `tests/backend/test_admin_release_dashboard.py`
  - Unit tests for normalizer success, missing report, malformed report, and read-only behavior.
- Create `tests/backend/test_admin_release_dashboard_api.py`
  - Route tests for `/api/admin/release-dashboard`.
- Modify `backend/api/routes/admin.py`
  - Add read-only endpoint.
- Modify `backend/app.py`
  - Add `/api/admin/release-dashboard` to admin-token guard.
- Modify `tests/backend/test_auth_security.py`
  - Include the new admin endpoint in existing auth matrix.

Frontend:

- Modify `frontend/src/app/api/types.ts`
  - Add release dashboard response interfaces and status unions.
- Modify `frontend/src/app/api/client.ts`
  - Add `getAdminReleaseDashboard()`.
- Modify `frontend/src/app/api/client.test.ts`
  - Add API client test for the new endpoint and headers.
- Modify `frontend/src/test/test-utils.tsx`
  - Add default API mock method for tests that construct a full client.
- Modify `frontend/src/features/agent-admin/agent-admin-model.ts`
  - Add `release` task and nav item.
- Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`
  - Load release dashboard data only when `release` task is active.
- Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`
  - Add Release Dashboard page body.
- Modify `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
  - Add task/nav, loading, success, error, and disabled-control tests.

Do not create a standalone route, new admin app, new auth channel, browser token store, or release mutation route.

---

### Task 1: Backend Release Dashboard Normalizer

**Files:**
- Create: `backend/api/services/admin_release_dashboard.py`
- Create: `tests/backend/test_admin_release_dashboard.py`

- [ ] **Step 1: Write failing normalizer tests**

Create `tests/backend/test_admin_release_dashboard.py` with these tests:

```python
from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.admin_release_dashboard import (
    ReleaseDashboardPaths,
    build_release_dashboard,
)


def test_build_release_dashboard_from_committed_reports() -> None:
    dashboard = build_release_dashboard()

    assert dashboard["version_chain"] == {
        "agent_policy_version": "agent_policy_20260629_0",
        "clinical_safety_policy_version": "crc_safety_policy_v0",
        "evidence_index_version": "rag_crc_guideline_20260620",
        "judge_rubric_version": "crc_rubric_v0",
    }
    assert dashboard["release_decision"] == "feature_flag_or_pass"
    assert dashboard["rollback_target"] == "agent_policy_20260624_0"
    assert dashboard["human_signoff"] == {
        "required": True,
        "status": "missing",
        "reason": "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    }
    assert dashboard["summary"]["hard_fail_count"] == 0
    assert dashboard["summary"]["p0_cases_total"] == 5
    assert dashboard["summary"]["p0_cases_passed"] == 5
    assert dashboard["summary"]["literature_claims"] == 3
    assert dashboard["summary"]["literature_isolation_violations"] == 0
    assert dashboard["summary"]["clinical_rag_ingest_enabled"] is False
    assert [run["run_id"] for run in dashboard["runs"]] == [
        "harness_20260629_001",
        "release_safety_20260629_001",
        "literature_harness_20260630_001",
    ]
    assert [run["status"] for run in dashboard["runs"]] == ["pass", "pass", "shadow_only"]
    assert {action["id"] for action in dashboard["disabled_actions"]} == {
        "record_human_signoff",
        "publish_feature_flag",
        "rollback_release",
    }
    assert dashboard["runtime"] == {
        "auth": "admin",
        "source": "reports/static_release_artifacts",
        "mode": "read_only",
    }


def test_missing_harness_report_marks_missing_without_hiding_literature() -> None:
    paths = ReleaseDashboardPaths(
        harness_report=Path("reports/harness/missing_harness.json"),
        release_safety_report=Path("reports/release_safety/release_safety_20260629_001.json"),
        literature_report=Path("reports/literature/literature_harness_20260630_001.json"),
    )

    dashboard = build_release_dashboard(paths=paths)

    harness_run = dashboard["runs"][0]
    literature_run = dashboard["runs"][2]
    assert harness_run["kind"] == "p0_crc_harness"
    assert harness_run["status"] == "missing"
    assert harness_run["run_id"] == "missing"
    assert dashboard["summary"]["p0_cases_total"] == 0
    assert dashboard["summary"]["p0_cases_passed"] == 0
    assert literature_run["run_id"] == "literature_harness_20260630_001"
    assert literature_run["status"] == "shadow_only"
    assert dashboard["summary"]["literature_claims"] == 3


def test_malformed_literature_report_marks_invalid_and_blocks_promotion(tmp_path: Path) -> None:
    malformed = tmp_path / "literature.json"
    malformed.write_text("{not valid json", encoding="utf-8")
    paths = ReleaseDashboardPaths(
        harness_report=Path("reports/harness/harness_20260629_001.json"),
        release_safety_report=Path("reports/release_safety/release_safety_20260629_001.json"),
        literature_report=malformed,
    )

    dashboard = build_release_dashboard(paths=paths)

    literature_run = dashboard["runs"][2]
    assert literature_run["kind"] == "literature_shadow_harness"
    assert literature_run["status"] == "invalid"
    assert literature_run["run_id"] == "invalid"
    assert dashboard["summary"]["literature_claims"] == 0
    assert dashboard["summary"]["literature_isolation_violations"] == 1
    assert any(gate["id"] == "no_literature_clinical_rag" for gate in dashboard["blocking_gates"])


def test_build_release_dashboard_does_not_write_report_files(tmp_path: Path) -> None:
    harness = tmp_path / "harness.json"
    release = tmp_path / "release.json"
    literature = tmp_path / "literature.json"
    harness_payload = {
        "run_id": "harness_test",
        "summary": {"total_cases": 1, "passed": 1, "hard_fail_count": 0},
    }
    release_payload = {
        "report_id": "release_test",
        "version_chain": {
            "agent_policy_version": "agent_policy_test",
            "clinical_safety_policy_version": "safety_test",
            "evidence_index_version": "evidence_test",
            "judge_rubric_version": "rubric_test",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_previous",
        "hard_fail_summary": {"count": 0, "types": []},
    }
    literature_payload = {
        "run_id": "literature_test",
        "run_level": "L0_shadow",
        "summary": {"claims": 2, "negative_or_conflicting_claims": 1, "isolation_violations": 0},
    }
    harness.write_text(json.dumps(harness_payload), encoding="utf-8")
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    literature.write_text(json.dumps(literature_payload), encoding="utf-8")
    before = {
        harness: harness.read_text(encoding="utf-8"),
        release: release.read_text(encoding="utf-8"),
        literature: literature.read_text(encoding="utf-8"),
    }

    build_release_dashboard(
        paths=ReleaseDashboardPaths(
            harness_report=harness,
            release_safety_report=release,
            literature_report=literature,
        )
    )

    assert {
        harness: harness.read_text(encoding="utf-8"),
        release: release.read_text(encoding="utf-8"),
        literature: literature.read_text(encoding="utf-8"),
    } == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py -q
```

Expected: FAIL because `backend.api.services.admin_release_dashboard` does not exist.

- [ ] **Step 3: Implement the normalizer**

Create `backend/api/services/admin_release_dashboard.py` with this structure:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ReleaseDashboardPaths:
    harness_report: Path
    release_safety_report: Path
    literature_report: Path


def default_release_dashboard_paths(repo_root: Path | None = None) -> ReleaseDashboardPaths:
    root = repo_root or REPO_ROOT
    return ReleaseDashboardPaths(
        harness_report=root / "reports" / "harness" / "harness_20260629_001.json",
        release_safety_report=root / "reports" / "release_safety" / "release_safety_20260629_001.json",
        literature_report=root / "reports" / "literature" / "literature_harness_20260630_001.json",
    )


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_report(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "missing", {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return "invalid", {}
    if not isinstance(payload, dict):
        return "invalid", {}
    return "ok", payload


def _int_value(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _version_chain(release_payload: dict[str, Any]) -> dict[str, str | None]:
    raw = release_payload.get("version_chain")
    chain = raw if isinstance(raw, dict) else {}
    return {
        "agent_policy_version": chain.get("agent_policy_version") if isinstance(chain.get("agent_policy_version"), str) else None,
        "clinical_safety_policy_version": chain.get("clinical_safety_policy_version") if isinstance(chain.get("clinical_safety_policy_version"), str) else None,
        "evidence_index_version": chain.get("evidence_index_version") if isinstance(chain.get("evidence_index_version"), str) else None,
        "judge_rubric_version": chain.get("judge_rubric_version") if isinstance(chain.get("judge_rubric_version"), str) else None,
    }


def _hard_fail_count(release_payload: dict[str, Any], harness_payload: dict[str, Any]) -> int:
    release_summary = release_payload.get("hard_fail_summary")
    if isinstance(release_summary, dict):
        count = release_summary.get("count")
        if isinstance(count, int):
            return count
    harness_summary = harness_payload.get("summary")
    if isinstance(harness_summary, dict):
        return _int_value(harness_summary.get("hard_fail_count"))
    return 0


def build_release_dashboard(paths: ReleaseDashboardPaths | None = None) -> dict[str, Any]:
    resolved_paths = paths or default_release_dashboard_paths()
    harness_state, harness_payload = _read_report(resolved_paths.harness_report)
    release_state, release_payload = _read_report(resolved_paths.release_safety_report)
    literature_state, literature_payload = _read_report(resolved_paths.literature_report)

    harness_summary = harness_payload.get("summary") if isinstance(harness_payload.get("summary"), dict) else {}
    literature_summary = literature_payload.get("summary") if isinstance(literature_payload.get("summary"), dict) else {}
    hard_fail_count = _hard_fail_count(release_payload, harness_payload)
    literature_isolation_violations = (
        _int_value(literature_summary.get("isolation_violations"), 1)
        if literature_state == "ok"
        else 1
    )
    literature_status = "shadow_only" if literature_state == "ok" and literature_isolation_violations == 0 else literature_state
    release_decision = (
        release_payload.get("release_decision")
        if release_state == "ok" and isinstance(release_payload.get("release_decision"), str)
        else "missing"
    )
    rollback_target = (
        release_payload.get("rollback_target")
        if release_state == "ok" and isinstance(release_payload.get("rollback_target"), str)
        else None
    )

    return {
        "version_chain": _version_chain(release_payload) if release_state == "ok" else {
            "agent_policy_version": None,
            "clinical_safety_policy_version": None,
            "evidence_index_version": None,
            "judge_rubric_version": None,
        },
        "release_decision": release_decision,
        "rollback_target": rollback_target,
        "human_signoff": {
            "required": True,
            "status": "missing",
            "reason": "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
        },
        "summary": {
            "hard_fail_count": hard_fail_count,
            "p0_cases_total": _int_value(harness_summary.get("total_cases")) if harness_state == "ok" else 0,
            "p0_cases_passed": _int_value(harness_summary.get("passed")) if harness_state == "ok" else 0,
            "literature_claims": _int_value(literature_summary.get("claims")) if literature_state == "ok" else 0,
            "literature_isolation_violations": literature_isolation_violations,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [
            {
                "run_id": harness_payload.get("run_id", harness_state) if harness_state == "ok" else harness_state,
                "kind": "p0_crc_harness",
                "status": "pass" if harness_state == "ok" and hard_fail_count == 0 else harness_state,
                "source_path": _repo_relative(resolved_paths.harness_report),
                "hard_fail_count": _int_value(harness_summary.get("hard_fail_count")) if harness_state == "ok" else 0,
            },
            {
                "run_id": release_payload.get("report_id", release_state) if release_state == "ok" else release_state,
                "kind": "release_safety",
                "status": "pass" if release_state == "ok" and hard_fail_count == 0 else release_state,
                "source_path": _repo_relative(resolved_paths.release_safety_report),
                "hard_fail_count": hard_fail_count,
            },
            {
                "run_id": literature_payload.get("run_id", literature_state) if literature_state == "ok" else literature_state,
                "kind": "literature_shadow_harness",
                "status": literature_status,
                "source_path": _repo_relative(resolved_paths.literature_report),
                "hard_fail_count": 0,
            },
        ],
        "blocking_gates": [
            {
                "id": "no_literature_patient_default",
                "label": "Unreviewed literature stays out of patient default path",
                "state": "locked" if literature_isolation_violations == 0 else "blocked",
                "reason": f"Step 10 report has {literature_isolation_violations} isolation violations.",
            },
            {
                "id": "no_literature_clinical_rag",
                "label": "Unreviewed literature stays out of clinical RAG",
                "state": "locked" if literature_isolation_violations == 0 else "blocked",
                "reason": "Clinical RAG ingest is disabled in Step 11.",
            },
        ],
        "disabled_actions": [
            {
                "id": "record_human_signoff",
                "label": "Record human sign-off",
                "reason": "Requires a later audited write-path design.",
            },
            {
                "id": "publish_feature_flag",
                "label": "Publish feature flag release",
                "reason": "Step 11 observes readiness only.",
            },
            {
                "id": "rollback_release",
                "label": "Rollback release",
                "reason": "Rollback execution is outside this read-only slice.",
            },
        ],
        "runtime": {
            "auth": "admin",
            "source": "reports/static_release_artifacts",
            "mode": "read_only",
        },
    }
```

- [ ] **Step 4: Run normalizer tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add backend/api/services/admin_release_dashboard.py tests/backend/test_admin_release_dashboard.py
git commit -m "feat: add admin release dashboard normalizer"
```

---

### Task 2: Admin API Endpoint And Auth

**Files:**
- Modify: `backend/api/routes/admin.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_auth_security.py`
- Create: `tests/backend/test_admin_release_dashboard_api.py`

- [ ] **Step 1: Write failing route and auth tests**

Add this route test file:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes


def test_admin_release_dashboard_route_returns_service_payload(monkeypatch) -> None:
    payload = {
        "version_chain": {
            "agent_policy_version": "agent_policy_test",
            "clinical_safety_policy_version": "safety_test",
            "evidence_index_version": "evidence_test",
            "judge_rubric_version": "rubric_test",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_previous",
        "human_signoff": {"required": True, "status": "missing", "reason": "read-only"},
        "summary": {
            "hard_fail_count": 0,
            "p0_cases_total": 1,
            "p0_cases_passed": 1,
            "literature_claims": 2,
            "literature_isolation_violations": 0,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [],
        "blocking_gates": [],
        "disabled_actions": [],
        "runtime": {"auth": "admin", "source": "reports/static_release_artifacts", "mode": "read_only"},
    }
    calls = {"count": 0}

    def fake_build_release_dashboard():
        calls["count"] += 1
        return payload

    monkeypatch.setattr(admin_routes, "build_release_dashboard", fake_build_release_dashboard)
    app = FastAPI()
    app.include_router(admin_routes.router)
    client = TestClient(app)

    try:
        response = client.get("/api/admin/release-dashboard")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == payload
    assert calls["count"] == 1
```

Update `tests/backend/test_auth_security.py`:

```python
    @app.get("/api/admin/release-dashboard")
    async def admin_release_dashboard() -> dict[str, object]:
        return {"runtime": {"auth": "admin"}}
```

Add `("get", "/api/admin/release-dashboard")` to each admin endpoint parameter list in:

- `test_admin_endpoints_reject_user_token_when_admin_token_is_distinct`
- `test_admin_endpoints_accept_admin_token`
- `test_admin_endpoints_use_user_token_when_no_separate_admin_token`

- [ ] **Step 2: Run route/auth tests to verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py -q
```

Expected:

- `test_admin_release_dashboard_route_returns_service_payload` fails with `404` or missing route.
- auth tests fail for `/api/admin/release-dashboard` because `_requires_admin_token()` does not yet guard it.

- [ ] **Step 3: Implement admin route**

Modify `backend/api/routes/admin.py`:

```python
from backend.api.services.admin_release_dashboard import build_release_dashboard
```

Add this route below `get_admin_tools()`:

```python
@router.get("/release-dashboard")
async def get_admin_release_dashboard() -> dict[str, Any]:
    return build_release_dashboard()
```

- [ ] **Step 4: Protect the endpoint**

Modify `backend/app.py::_requires_admin_token()`:

```python
    if method == "GET" and path == "/api/admin/release-dashboard":
        return True
```

Keep the existing `/api/admin/tools` guard.

- [ ] **Step 5: Run backend admin tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add backend/api/routes/admin.py backend/app.py tests/backend/test_auth_security.py tests/backend/test_admin_release_dashboard_api.py
git commit -m "feat: expose admin release dashboard API"
```

---

### Task 3: Frontend API Types And Client

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/test/test-utils.tsx`

- [ ] **Step 1: Write failing client test**

Add this test near the existing admin tools test in `frontend/src/app/api/client.test.ts`:

```ts
  it("loads admin release dashboard with configured Authorization headers", async () => {
    const payload = {
      version_chain: {
        agent_policy_version: "agent_policy_20260629_0",
        clinical_safety_policy_version: "crc_safety_policy_v0",
        evidence_index_version: "rag_crc_guideline_20260620",
        judge_rubric_version: "crc_rubric_v0",
      },
      release_decision: "feature_flag_or_pass",
      rollback_target: "agent_policy_20260624_0",
      human_signoff: {
        required: true,
        status: "missing",
        reason: "Step 11 is read-only",
      },
      summary: {
        hard_fail_count: 0,
        p0_cases_total: 5,
        p0_cases_passed: 5,
        literature_claims: 3,
        literature_isolation_violations: 0,
        clinical_rag_ingest_enabled: false,
      },
      runs: [
        {
          run_id: "harness_20260629_001",
          kind: "p0_crc_harness",
          status: "pass",
          source_path: "reports/harness/harness_20260629_001.json",
          hard_fail_count: 0,
        },
      ],
      blocking_gates: [],
      disabled_actions: [],
      runtime: {
        auth: "admin",
        source: "reports/static_release_artifacts",
        mode: "read_only",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminReleaseDashboard()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-dashboard",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });
```

- [ ] **Step 2: Run client test to verify failure**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
```

Expected: FAIL because `getAdminReleaseDashboard()` does not exist.

- [ ] **Step 3: Add frontend response types**

Modify `frontend/src/app/api/types.ts`:

```ts
export type AdminReleaseRunStatus = "pass" | "fail" | "shadow_only" | "missing" | "invalid";
export type AdminReleaseGateState = "pass" | "locked" | "warning" | "blocked" | "missing";
export type AdminReleaseHumanSignoffStatus = "missing" | "recorded_elsewhere" | "not_required";

export interface AdminReleaseDashboardVersionChain {
  agent_policy_version: string | null;
  clinical_safety_policy_version: string | null;
  evidence_index_version: string | null;
  judge_rubric_version: string | null;
}

export interface AdminReleaseDashboardHumanSignoff {
  required: boolean;
  status: AdminReleaseHumanSignoffStatus;
  reason: string;
}

export interface AdminReleaseDashboardSummary {
  hard_fail_count: number;
  p0_cases_total: number;
  p0_cases_passed: number;
  literature_claims: number;
  literature_isolation_violations: number;
  clinical_rag_ingest_enabled: boolean;
}

export interface AdminReleaseDashboardRun {
  run_id: string;
  kind: "p0_crc_harness" | "release_safety" | "literature_shadow_harness";
  status: AdminReleaseRunStatus;
  source_path: string;
  hard_fail_count: number;
}

export interface AdminReleaseDashboardGate {
  id: string;
  label: string;
  state: AdminReleaseGateState;
  reason: string;
}

export interface AdminReleaseDashboardDisabledAction {
  id: "record_human_signoff" | "publish_feature_flag" | "rollback_release";
  label: string;
  reason: string;
}

export interface AdminReleaseDashboardResponse {
  version_chain: AdminReleaseDashboardVersionChain;
  release_decision: string;
  rollback_target: string | null;
  human_signoff: AdminReleaseDashboardHumanSignoff;
  summary: AdminReleaseDashboardSummary;
  runs: AdminReleaseDashboardRun[];
  blocking_gates: AdminReleaseDashboardGate[];
  disabled_actions: AdminReleaseDashboardDisabledAction[];
  runtime: {
    auth: "admin";
    source: "reports/static_release_artifacts";
    mode: "read_only";
  };
}
```

- [ ] **Step 4: Add client method**

Modify imports in `frontend/src/app/api/client.ts`:

```ts
  AdminReleaseDashboardResponse,
```

Add to `ApiClient`:

```ts
  getAdminReleaseDashboard(): Promise<AdminReleaseDashboardResponse>;
```

Add inside `createApiClient()` return object near `getAdminTools()`:

```ts
    async getAdminReleaseDashboard() {
      const response = await fetchImpl(buildUrl("/api/admin/release-dashboard", baseUrl), {
        headers: defaultHeaders,
      });
      return parseJsonResponse<AdminReleaseDashboardResponse>(response);
    },
```

- [ ] **Step 5: Add default API mock**

Modify `frontend/src/test/test-utils.tsx` imports if needed:

```ts
import type { AdminReleaseDashboardResponse, AdminToolManifestResponse } from "../app/api/types";
```

Add a mock near `getAdminTools`:

```ts
  const getAdminReleaseDashboard = vi.fn(async (): Promise<AdminReleaseDashboardResponse> => ({
    version_chain: {
      agent_policy_version: "agent_policy_20260629_0",
      clinical_safety_policy_version: "crc_safety_policy_v0",
      evidence_index_version: "rag_crc_guideline_20260620",
      judge_rubric_version: "crc_rubric_v0",
    },
    release_decision: "feature_flag_or_pass",
    rollback_target: "agent_policy_20260624_0",
    human_signoff: {
      required: true,
      status: "missing",
      reason: "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    },
    summary: {
      hard_fail_count: 0,
      p0_cases_total: 5,
      p0_cases_passed: 5,
      literature_claims: 3,
      literature_isolation_violations: 0,
      clinical_rag_ingest_enabled: false,
    },
    runs: [],
    blocking_gates: [],
    disabled_actions: [],
    runtime: {
      auth: "admin",
      source: "reports/static_release_artifacts",
      mode: "read_only",
    },
  }));
```

Include `getAdminReleaseDashboard` in the returned mock client object.

- [ ] **Step 6: Run frontend client tests**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx
git commit -m "feat: add admin release dashboard client"
```

---

### Task 4: Agent Admin Release Dashboard Page

**Files:**
- Modify: `frontend/src/features/agent-admin/agent-admin-model.ts`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`

- [ ] **Step 1: Add frontend test fixtures**

In `frontend/src/features/agent-admin/agent-admin-view.test.tsx`, import the response type:

```ts
import type { AdminReleaseDashboardResponse, AdminToolManifestResponse } from "../../app/api/types";
```

Add fixture and helper:

```ts
function makeAdminReleaseDashboard(): AdminReleaseDashboardResponse {
  return {
    version_chain: {
      agent_policy_version: "agent_policy_20260629_0",
      clinical_safety_policy_version: "crc_safety_policy_v0",
      evidence_index_version: "rag_crc_guideline_20260620",
      judge_rubric_version: "crc_rubric_v0",
    },
    release_decision: "feature_flag_or_pass",
    rollback_target: "agent_policy_20260624_0",
    human_signoff: {
      required: true,
      status: "missing",
      reason: "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    },
    summary: {
      hard_fail_count: 0,
      p0_cases_total: 5,
      p0_cases_passed: 5,
      literature_claims: 3,
      literature_isolation_violations: 0,
      clinical_rag_ingest_enabled: false,
    },
    runs: [
      {
        run_id: "harness_20260629_001",
        kind: "p0_crc_harness",
        status: "pass",
        source_path: "reports/harness/harness_20260629_001.json",
        hard_fail_count: 0,
      },
      {
        run_id: "literature_harness_20260630_001",
        kind: "literature_shadow_harness",
        status: "shadow_only",
        source_path: "reports/literature/literature_harness_20260630_001.json",
        hard_fail_count: 0,
      },
    ],
    blocking_gates: [
      {
        id: "no_literature_clinical_rag",
        label: "Unreviewed literature stays out of clinical RAG",
        state: "locked",
        reason: "Clinical RAG ingest is disabled in Step 11.",
      },
    ],
    disabled_actions: [
      {
        id: "record_human_signoff",
        label: "Record human sign-off",
        reason: "Requires a later audited write-path design.",
      },
      {
        id: "publish_feature_flag",
        label: "Publish feature flag release",
        reason: "Step 11 observes readiness only.",
      },
      {
        id: "rollback_release",
        label: "Rollback release",
        reason: "Rollback execution is outside this read-only slice.",
      },
    ],
    runtime: {
      auth: "admin",
      source: "reports/static_release_artifacts",
      mode: "read_only",
    },
  };
}

function clickReleaseTask() {
  const button = screen
    .getAllByRole("button")
    .find((candidate) => candidate.textContent?.includes("version chain / harness runs"));
  expect(button).toBeDefined();
  fireEvent.click(button!);
}
```

- [ ] **Step 2: Write failing release page tests**

Add these tests:

```ts
  it("renders the release dashboard task from the admin rail", async () => {
    const releaseDashboard = makeAdminReleaseDashboard();
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => releaseDashboard),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release" })}
        doctor={makeState({ sessionId: "doctor-release" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(apiClient.getAdminReleaseDashboard).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", "release");
    expect(page).toHaveTextContent("Release Dashboard");
    expect(page).toHaveTextContent("agent_policy_20260629_0");
    expect(page).toHaveTextContent("crc_safety_policy_v0");
    expect(page).toHaveTextContent("harness_20260629_001");
    expect(page).toHaveTextContent("literature_harness_20260630_001");
    expect(page).toHaveTextContent("feature_flag_or_pass");
    expect(page).toHaveTextContent("agent_policy_20260624_0");
    expect(page).toHaveTextContent("Step 11 observes readiness only");
    expect(page).toHaveTextContent("Clinical RAG ingest is disabled in Step 11");
  });

  it("does not fetch release dashboard until the release task is selected", () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-idle" })}
        doctor={makeState({ sessionId: "doctor-release-idle" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    expect(apiClient.getAdminReleaseDashboard).not.toHaveBeenCalled();
  });

  it("shows release dashboard loading state without fallback data", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(() => new Promise<AdminReleaseDashboardResponse>(() => {})),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-loading" })}
        doctor={makeState({ sessionId: "doctor-release-loading" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(apiClient.getAdminReleaseDashboard).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("reading release dashboard");
    expect(page).not.toHaveTextContent("agent_policy_20260629_0");
  });

  it("shows release dashboard error state without breaking the admin shell", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => {
        throw new Error("Forbidden");
      }),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-error" })}
        doctor={makeState({ sessionId: "doctor-release-error" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("release dashboard unavailable"));
    expect(screen.getByTestId("agent-admin-console")).toBeInTheDocument();
  });

  it("renders release mutation controls as disabled read-only actions", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-disabled" })}
        doctor={makeState({ sessionId: "doctor-release-disabled" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("Record human sign-off"));

    const page = screen.getByTestId("agent-admin-task-page");
    for (const label of ["Record human sign-off", "Publish feature flag release", "Rollback release"]) {
      const action = within(page).getByText(label).closest("button");
      expect(action).toBeDisabled();
    }
  });
```

Update the rail count assertion from `9` to `10` and include `Release` in the task label loop. Update the selected-page matrix with:

```ts
{ label: /Release/, taskId: "release", hidden: "RAG pipeline", visible: "Release Dashboard" },
```

- [ ] **Step 3: Run Agent Admin tests to verify failure**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: FAIL because `release` task and `getAdminReleaseDashboard` loading are not implemented.

- [ ] **Step 4: Add release task model**

Modify `frontend/src/features/agent-admin/agent-admin-model.ts`:

```ts
import {
  Activity,
  BookOpenCheck,
  Brain,
  FileText,
  Gauge,
  GitBranch,
  KeyRound,
  ListChecks,
  ServerCog,
  Wrench,
  type LucideIcon,
} from "lucide-react";
```

Extend `AgentAdminTaskId`:

```ts
  | "release"
```

Add to `AGENT_ADMIN_TASKS` after `evidence` and before `read-only`:

```ts
  {
    id: "release",
    label: "Release",
    detailTitle: "Release Dashboard",
    description: "version chain / harness runs / rollback target / sign-off readiness",
    status: "read-only",
    responsibility: "version chain / harness runs",
    icon: GitBranch,
  },
```

Add to `ADMIN_NAV_ITEMS`:

```ts
  { key: "release", label: "Release" },
```

- [ ] **Step 5: Add release resource loading**

Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`.

Add type import:

```ts
import type { AdminReleaseDashboardResponse, AdminToolManifestResponse } from "../../app/api/types";
```

Update props:

```ts
  apiClient?: Pick<ApiClient, "getAdminTools" | "getAdminReleaseDashboard">;
```

Add resource type:

```ts
export type AgentAdminReleaseDashboardResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseDashboardResponse }
  | { status: "error"; error: { status?: number; message: string } };
```

Add state:

```ts
  const [releaseDashboardResource, setReleaseDashboardResource] = useState<AgentAdminReleaseDashboardResource>({ status: "idle" });
```

Add effect modeled on `toolsResource`:

```ts
  useEffect(() => {
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseDashboard !== "function") {
      setReleaseDashboardResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setReleaseDashboardResource({ status: "loading" });

    void apiClient.getAdminReleaseDashboard().then(
      (data) => {
        if (!cancelled) {
          setReleaseDashboardResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        if (error instanceof ApiClientError) {
          setReleaseDashboardResource({
            status: "error",
            error: { status: error.status, message: error.message },
          });
          return;
        }

        setReleaseDashboardResource({
          status: "error",
          error: {
            message: error instanceof Error ? error.message : "Unknown admin release dashboard error",
          },
        });
      },
    );

    return () => {
      cancelled = true;
    };
  }, [activeTaskId, apiClient]);
```

Update `navigateTask()`:

```ts
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseDashboard === "function") {
      setReleaseDashboardResource({ status: "loading" });
    }
```

Pass to `AgentAdminTaskPages`:

```tsx
          releaseDashboardResource={releaseDashboardResource}
```

- [ ] **Step 6: Render release page**

Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`.

Add imports:

```ts
import type { AdminReleaseDashboardResponse } from "../../app/api/types";
import type { AgentAdminReleaseDashboardResource, AgentAdminToolsResource } from "./agent-admin-view";
```

Add prop:

```ts
  releaseDashboardResource: AgentAdminReleaseDashboardResource;
```

Update branching:

```tsx
      ) : activeTaskId === "release" ? (
        <ReleasePage releaseDashboardResource={releaseDashboardResource} />
```

Add helper functions:

```ts
function releaseTone(status: string): "success" | "warning" | "red" | "neutral" {
  if (status === "pass" || status === "locked" || status === "shadow_only") {
    return "success";
  }
  if (status === "missing" || status === "invalid" || status === "blocked" || status === "fail") {
    return "warning";
  }
  return "neutral";
}

function ReleasePage({ releaseDashboardResource }: { releaseDashboardResource: AgentAdminReleaseDashboardResource }) {
  if (releaseDashboardResource.status === "loading") {
    return (
      <AgentAdminPanel eyebrow="release artifacts" title="reading release dashboard" icon={GitBranch}>
        <div className="agent-admin-detail-list">
          <span>reading release dashboard</span>
          <span>Waiting for /api/admin/release-dashboard</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (releaseDashboardResource.status === "error") {
    return (
      <AgentAdminPanel eyebrow="release artifacts" title="release dashboard unavailable" icon={AlertTriangle}>
        <div className="agent-admin-detail-list">
          <span>release dashboard unavailable</span>
          <span>{releaseDashboardResource.error.status ?? "network"} / {releaseDashboardResource.error.message}</span>
        </div>
      </AgentAdminPanel>
    );
  }

  if (releaseDashboardResource.status !== "success") {
    return (
      <AgentAdminPanel eyebrow="release artifacts" title="release dashboard idle" icon={GitBranch}>
        <div className="agent-admin-detail-list">
          <span>select Release to read committed release artifacts</span>
        </div>
      </AgentAdminPanel>
    );
  }

  return <ReleaseDashboard data={releaseDashboardResource.data} />;
}

function ReleaseDashboard({ data }: { data: AdminReleaseDashboardResponse }) {
  const metrics = [
    { label: "Hard fails", value: String(data.summary.hard_fail_count), detail: data.release_decision, tone: data.summary.hard_fail_count === 0 ? "success" as const : "warning" as const },
    { label: "P0 cases", value: `${data.summary.p0_cases_passed}/${data.summary.p0_cases_total}`, detail: "CRC safety harness", tone: "success" as const },
    { label: "Literature claims", value: String(data.summary.literature_claims), detail: "Step 10 shadow", tone: "warning" as const },
    { label: "Isolation violations", value: String(data.summary.literature_isolation_violations), detail: "clinical RAG ingest disabled", tone: data.summary.literature_isolation_violations === 0 ? "success" as const : "warning" as const },
  ];

  return (
    <>
      <AgentAdminMetricStrip metrics={metrics} />
      <AgentAdminSplitWorkbench
        primary={
          <>
            <AgentAdminPanel eyebrow="version chain" title="Version Chain" icon={GitBranch}>
              <div className="agent-admin-detail-list">
                <span>Agent policy / {data.version_chain.agent_policy_version ?? "missing"}</span>
                <span>Clinical safety policy / {data.version_chain.clinical_safety_policy_version ?? "missing"}</span>
                <span>Evidence index / {data.version_chain.evidence_index_version ?? "missing"}</span>
                <span>Judge rubric / {data.version_chain.judge_rubric_version ?? "missing"}</span>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="harness runs" title="Release Runs" icon={ListChecks}>
              <div className="agent-admin-list-rows">
                {data.runs.map((run) => (
                  <p key={`${run.kind}-${run.run_id}`}>
                    <span>{run.run_id}</span>
                    <strong>{run.kind}</strong>
                    <em>{run.status}</em>
                    <small>{run.source_path} / hard fails {run.hard_fail_count}</small>
                  </p>
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
        secondary={
          <>
            <AgentAdminPanel eyebrow="release decision" title="Rollback And Sign-off" icon={ShieldCheck}>
              <div className="agent-admin-detail-list">
                <span>release decision / {data.release_decision}</span>
                <span>rollback target / {data.rollback_target ?? "missing"}</span>
                <span>human sign-off / {data.human_signoff.status}</span>
                <span>{data.human_signoff.reason}</span>
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="blocking gates" title="Blocking Gates" icon={AlertTriangle}>
              <div className="agent-admin-detail-list">
                {data.blocking_gates.map((gate) => (
                  <span key={gate.id}>{gate.label} / {gate.state} / {gate.reason}</span>
                ))}
              </div>
            </AgentAdminPanel>
            <AgentAdminPanel eyebrow="disabled controls" title="Read-only Release Controls" icon={KeyRound}>
              <div className="agent-admin-detail-list">
                {data.disabled_actions.map((action) => (
                  <AgentAdminDisabledAction key={action.id} label={action.label} reason={action.reason} />
                ))}
              </div>
            </AgentAdminPanel>
          </>
        }
      />
    </>
  );
}
```

If `ShieldCheck` is not imported in this file, add it to the existing lucide import list. If `releaseTone()` is unused after implementation, omit it to keep lint clean.

- [ ] **Step 7: Run frontend release tests**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add frontend/src/features/agent-admin/agent-admin-model.ts frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx
git commit -m "feat: add agent admin release dashboard page"
```

---

### Task 5: Final Integration Verification

**Files:**
- No planned source edits.
- If verification exposes a bug, fix only the smallest relevant files and add or adjust tests before rerunning this task.

- [ ] **Step 1: Run backend Step 11 tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend Step 11 tests**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run P0/P1/Step10 backend regressions**

Run Step 10:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
```

Expected: `52 passed`.

Run P1:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
```

Expected: `34 passed`.

Run P0:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
```

Expected: `29 passed`.

- [ ] **Step 4: Run diff and scope checks**

Run:

```powershell
git diff --check
git diff --name-only main~4..HEAD
```

Expected:

- `git diff --check` has no output.
- Changed files are limited to backend admin release dashboard files, admin auth tests, frontend API files, Agent Admin files, and this plan/spec history.
- No files under `CRC-client/`, clinical RAG index write paths, prompt/rubric/route/template patch files, or safety policy config are changed by Step 11 implementation.

- [ ] **Step 5: Browser UI verification if frontend tests pass**

Use the existing app dev workflow only if the frontend dependencies are available. Verify:

- Agent Admin rail shows `Release`.
- Selecting `Release` loads the dashboard.
- Version chain, release decision, rollback target, hard fail count, Step 10 shadow run, and disabled controls are visible.
- Text does not overlap at desktop width and a mobile viewport.
- No release, rollback, sign-off, or feature flag button is enabled.

If the local frontend test environment lacks dependencies, record the exact missing dependency error and do not claim browser UI verification passed.

- [ ] **Step 6: Commit verification fix if needed**

If Task 5 required fixes, commit them:

```powershell
git add backend/api/services/admin_release_dashboard.py backend/api/routes/admin.py backend/app.py tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx frontend/src/features/agent-admin/agent-admin-model.ts frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx
git commit -m "fix: finalize admin release dashboard integration"
```

If Task 5 required no fixes, do not create an empty commit.

## Final Handoff

When all tasks pass, report:

- backend Step 11 test results;
- frontend Step 11 test results;
- P0/P1/Step10 regression results;
- final changed file list;
- whether browser UI verification was completed or blocked by environment;
- current branch and whether it has been pushed.

Do not push to `origin/main` unless the user explicitly asks.

## Self-Review

Spec coverage: Task 1 covers backend normalization and report state handling. Task 2 covers API and admin auth. Task 3 covers frontend types/client. Task 4 covers Agent Admin page rendering, loading/error states, and disabled controls. Task 5 covers final P0/P1/Step10 regression and UI verification.

Marker scan: the plan contains no unresolved work markers and no unspecified file paths.

Type consistency: backend response keys match frontend interfaces: `version_chain`, `release_decision`, `rollback_target`, `human_signoff`, `summary`, `runs`, `blocking_gates`, `disabled_actions`, and `runtime`. The frontend client method name is consistently `getAdminReleaseDashboard()`.

Scope check: this plan implements only the read-only Step 11 release dashboard. It does not add release execution, sign-off writes, rollback writes, policy mutation, clinical RAG ingest, research cohort feasibility, LearningJob automation, or `CRC-client/` edits.
