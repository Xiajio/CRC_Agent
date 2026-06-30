from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable


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
        release_safety_report=root
        / "reports"
        / "release_safety"
        / "release_safety_20260629_001.json",
        literature_report=root
        / "reports"
        / "literature"
        / "literature_harness_20260630_001.json",
    )


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_required_strings(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(isinstance(payload.get(key), str) for key in keys)


def _valid_harness_report(payload: dict[str, Any]) -> bool:
    summary = payload.get("summary")
    return (
        isinstance(payload.get("run_id"), str)
        and isinstance(summary, dict)
        and _is_int(summary.get("total_cases"))
        and _is_int(summary.get("passed"))
        and _is_int(summary.get("hard_fail_count"))
    )


def _valid_release_safety_report(payload: dict[str, Any]) -> bool:
    version_chain = payload.get("version_chain")
    hard_fail_summary = payload.get("hard_fail_summary")
    rollback_target = payload.get("rollback_target")
    return (
        isinstance(payload.get("report_id"), str)
        and isinstance(version_chain, dict)
        and _has_required_strings(
            version_chain,
            (
                "agent_policy_version",
                "clinical_safety_policy_version",
                "evidence_index_version",
                "judge_rubric_version",
            ),
        )
        and isinstance(payload.get("release_decision"), str)
        and (
            "rollback_target" not in payload
            or rollback_target is None
            or isinstance(rollback_target, str)
        )
        and isinstance(hard_fail_summary, dict)
        and _is_int(hard_fail_summary.get("count"))
    )


def _valid_literature_report(payload: dict[str, Any]) -> bool:
    summary = payload.get("summary")
    validation_errors = payload.get("validation_errors")
    return (
        isinstance(payload.get("run_id"), str)
        and isinstance(payload.get("release_decision"), str)
        and isinstance(summary, dict)
        and _is_int(summary.get("claims"))
        and _is_int(summary.get("isolation_violations"))
        and (
            "validation_errors" not in payload
            or isinstance(validation_errors, list)
        )
    )


def _read_report(
    path: Path, validator: Callable[[dict[str, Any]], bool]
) -> tuple[str, dict[str, Any]]:
    try:
        if not path.exists():
            return "missing", {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError, UnicodeDecodeError):
        return "invalid", {}
    if not isinstance(payload, dict):
        return "invalid", {}
    if not validator(payload):
        return "invalid", {}
    return "ok", payload


def _int_value(value: object, default: int = 0) -> int:
    return value if _is_int(value) else default


def _version_chain(release_payload: dict[str, Any]) -> dict[str, str | None]:
    raw = release_payload.get("version_chain")
    chain = raw if isinstance(raw, dict) else {}
    return {
        "agent_policy_version": chain.get("agent_policy_version")
        if isinstance(chain.get("agent_policy_version"), str)
        else None,
        "clinical_safety_policy_version": chain.get("clinical_safety_policy_version")
        if isinstance(chain.get("clinical_safety_policy_version"), str)
        else None,
        "evidence_index_version": chain.get("evidence_index_version")
        if isinstance(chain.get("evidence_index_version"), str)
        else None,
        "judge_rubric_version": chain.get("judge_rubric_version")
        if isinstance(chain.get("judge_rubric_version"), str)
        else None,
    }


def _hard_fail_count(
    release_payload: dict[str, Any], harness_payload: dict[str, Any]
) -> int:
    release_summary = release_payload.get("hard_fail_summary")
    if isinstance(release_summary, dict):
        count = release_summary.get("count")
        if isinstance(count, int):
            return count
    harness_summary = harness_payload.get("summary")
    if isinstance(harness_summary, dict):
        return _int_value(harness_summary.get("hard_fail_count"))
    return 0


def _run_status_for_hard_fails(report_state: str, hard_fail_count: int) -> str:
    if report_state != "ok":
        return report_state
    return "pass" if hard_fail_count == 0 else "fail"


def build_release_dashboard(paths: ReleaseDashboardPaths | None = None) -> dict[str, Any]:
    resolved_paths = paths or default_release_dashboard_paths()
    harness_state, harness_payload = _read_report(
        resolved_paths.harness_report, _valid_harness_report
    )
    release_state, release_payload = _read_report(
        resolved_paths.release_safety_report, _valid_release_safety_report
    )
    literature_state, literature_payload = _read_report(
        resolved_paths.literature_report, _valid_literature_report
    )

    harness_summary = (
        harness_payload.get("summary")
        if isinstance(harness_payload.get("summary"), dict)
        else {}
    )
    literature_summary = (
        literature_payload.get("summary")
        if isinstance(literature_payload.get("summary"), dict)
        else {}
    )
    hard_fail_count = _hard_fail_count(release_payload, harness_payload)
    literature_isolation_violations = (
        _int_value(literature_summary.get("isolation_violations"), 1)
        if literature_state == "ok"
        else 1
    )
    literature_validation_errors = literature_payload.get("validation_errors", [])
    literature_release_decision = literature_payload.get("release_decision")
    literature_can_shadow = (
        literature_state == "ok"
        and literature_release_decision == "shadow_only"
        and literature_validation_errors == []
        and literature_isolation_violations == 0
    )
    literature_status = (
        "shadow_only" if literature_can_shadow else "fail"
    ) if literature_state == "ok" else literature_state
    literature_gate_state = "locked" if literature_can_shadow else "blocked"
    literature_gate_reason = (
        f"Step 10 report has {literature_isolation_violations} isolation violations."
    )
    if literature_state != "ok":
        literature_gate_reason = f"Literature report is {literature_state}."
    elif literature_release_decision != "shadow_only":
        literature_gate_reason = (
            f"Step 10 release decision is {literature_release_decision}."
        )
    elif literature_validation_errors != []:
        literature_gate_reason = (
            f"Step 10 report has {len(literature_validation_errors)} validation errors."
        )
    elif literature_isolation_violations != 0:
        literature_gate_reason = (
            f"Step 10 report has {literature_isolation_violations} "
            "isolation violations."
        )
    harness_run_status = _run_status_for_hard_fails(harness_state, hard_fail_count)
    release_run_status = _run_status_for_hard_fails(
        release_state, hard_fail_count
    )
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
        "version_chain": _version_chain(release_payload)
        if release_state == "ok"
        else {
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
            "p0_cases_total": _int_value(harness_summary.get("total_cases"))
            if harness_state == "ok"
            else 0,
            "p0_cases_passed": _int_value(harness_summary.get("passed"))
            if harness_state == "ok"
            else 0,
            "literature_claims": _int_value(literature_summary.get("claims"))
            if literature_state == "ok"
            else 0,
            "literature_isolation_violations": literature_isolation_violations,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [
            {
                "run_id": harness_payload.get("run_id", harness_state)
                if harness_state == "ok"
                else harness_state,
                "kind": "p0_crc_harness",
                "status": harness_run_status,
                "source_path": _repo_relative(resolved_paths.harness_report),
                "hard_fail_count": _int_value(harness_summary.get("hard_fail_count"))
                if harness_state == "ok"
                else 0,
            },
            {
                "run_id": release_payload.get("report_id", release_state)
                if release_state == "ok"
                else release_state,
                "kind": "release_safety",
                "status": release_run_status,
                "source_path": _repo_relative(resolved_paths.release_safety_report),
                "hard_fail_count": hard_fail_count,
            },
            {
                "run_id": literature_payload.get("run_id", literature_state)
                if literature_state == "ok"
                else literature_state,
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
                "state": literature_gate_state,
                "reason": literature_gate_reason,
            },
            {
                "id": "no_literature_clinical_rag",
                "label": "Unreviewed literature stays out of clinical RAG",
                "state": literature_gate_state,
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
