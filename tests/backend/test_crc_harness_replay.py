from __future__ import annotations

import json

from scripts.run_crc_harness_replay import run_crc_harness_replay
from src.contracts.harness import build_harness_run
from src.contracts.release_safety_report import build_release_safety_report
from src.services.clinical_safety_policy import load_clinical_safety_policy


def test_crc_harness_replay_writes_harness_and_release_reports(tmp_path):
    harness_path, release_path = run_crc_harness_replay(output_root=tmp_path)

    assert harness_path.exists()
    assert release_path.exists()

    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))

    assert harness["case_pack_version"] == "crc_mutation_pack_v0"
    assert harness["agent_policy_version"] == "agent_policy_20260629_0"
    assert harness["clinical_safety_policy_version"] == "crc_safety_policy_v0"
    assert harness["evidence_index_version"] == "rag_crc_guideline_20260620"
    assert harness["judge_rubric_version"] == "crc_rubric_v0"
    assert harness["summary"]["total_cases"] == 5
    assert "total" not in harness["summary"]
    assert harness["summary"]["hard_fail_count"] == 0
    assert harness["release_decision"] in {"pass", "shadow_only", "block"}
    assert harness["release_decision"] == "pass"

    assert release["harness_runs"] == [harness["run_id"]]
    assert (
        release["version_chain"]["clinical_safety_policy_version"]
        == "crc_safety_policy_v0"
    )
    assert release["rollback_target"] == "agent_policy_20260624_0"
    assert release["release_decision"] == "feature_flag_or_pass"
    assert release["hard_fail_summary"] == {"count": 0, "types": []}


def test_crc_harness_synthesizes_metadata_only_actuals_for_any_case_id():
    policy = load_clinical_safety_policy()
    harness = build_harness_run(
        run_id="metadata_only_generic",
        mutation_pack={
            "case_pack_id": "metadata_only_pack",
            "cases": [
                {
                    "case_id": "metadata_only_state_guard",
                    "base_input": {},
                    "mutation": {},
                    "expected": {
                        "crc_state_persisted": True,
                        "patient_assistant_not_polluted": True,
                    },
                }
            ],
        },
        policy=policy,
    )

    assert harness["cases"][0]["actual"] == {
        "crc_state_persisted": True,
        "patient_assistant_not_polluted": True,
    }
    assert harness["cases"][0]["passed"] is True


def test_crc_harness_mixed_metadata_and_policy_expected_uses_policy_actual():
    policy = load_clinical_safety_policy()
    harness = build_harness_run(
        run_id="mixed_metadata_policy",
        mutation_pack={
            "case_pack_id": "mixed_metadata_policy_pack",
            "cases": [
                {
                    "case_id": "mixed_state_guard_with_policy_escalation",
                    "base_input": {
                        "age": 25,
                        "rectal_bleeding": True,
                    },
                    "mutation": {
                        "age": 62,
                    },
                    "expected": {
                        "crc_state_persisted": True,
                        "disposition_minimum": "urgent",
                    },
                }
            ],
        },
        policy=policy,
    )

    case = harness["cases"][0]

    assert case["actual"]["disposition"] == "urgent"
    assert case["actual"]["matched_rules"] == ["rectal_bleeding_age_escalation"]
    assert case["passed"] is True


def test_release_safety_report_uses_hard_fail_types_not_case_ids():
    policy = load_clinical_safety_policy()
    harness = build_harness_run(
        run_id="hard_fail_type_probe",
        mutation_pack={
            "case_pack_id": "hard_fail_type_pack",
            "cases": [
                {
                    "case_id": "case_missing_urgent_escalation",
                    "base_input": {
                        "age": 25,
                        "rectal_bleeding": True,
                    },
                    "mutation": {},
                    "expected": {
                        "disposition_minimum": "urgent",
                        "hard_fail_if_below": "urgent",
                    },
                }
            ],
        },
        policy=policy,
    )

    report = build_release_safety_report(
        report_id="release_safety_hard_fail_probe",
        harness_run=harness,
    )

    assert harness["hard_fails"] == [
        {
            "case_id": "case_missing_urgent_escalation",
            "type": "below_urgent",
        }
    ]
    assert report["hard_fail_summary"] == {
        "count": 1,
        "types": ["below_urgent"],
    }
