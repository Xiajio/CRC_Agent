from __future__ import annotations

import json

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


def make_review_item(**overrides: object) -> ReviewQueueItem:
    payload = {
        "review_item_id": "review_item_crc_001",
        "review_type": "research_ethics_review",
        "status": "pending",
        "trigger": "patient_level_data_used_for_cohort_feasibility",
        "scope": {
            "project_id": "research_crc_001",
            "request_id": "cohort_request_crc_001",
            "data_minimization": "aggregate_only",
            "patient_level_export_requested": False,
        },
        "required_checks": [
            "authorization_basis",
            "deidentification_strategy",
            "data_minimization",
            "irb_or_local_policy_need",
        ],
    }
    payload.update(overrides)
    return ReviewQueueItem(**payload)


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


def make_result(**overrides: object) -> CohortFeasibilityResult:
    payload = {
        "result_id": "cohort_feasibility_crc_001",
        "request_id": "cohort_request_crc_001",
        "project_id": "research_crc_001",
        "status": "needs_review",
        "estimated_count": 2,
        "variable_coverage": {
            "rectal_bleeding": VariableCoverage(
                covered_count=2,
                coverage_ratio=1.0,
                source_fact_types=["condition_signal"],
                reviewed_status_mix={"unreviewed": 2},
            )
        },
        "missing_key_variables": [],
        "unmapped_required_features": [],
        "bias_warnings": [],
        "requires_review": True,
        "review_queue_items": [make_review_item()],
        "patient_level_rows_returned": False,
    }
    payload.update(overrides)
    return CohortFeasibilityResult(**payload)


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


def test_request_rejects_patient_identifiers_in_criteria() -> None:
    with pytest.raises(ValueError, match="patient identifiers are not allowed"):
        make_request(
            cohort_criteria={
                "condition": "crc",
                "required_features": ["rectal_bleeding"],
                "patient_ids": ["patient-1"],
            }
        )


@pytest.mark.parametrize(
    "cohort_criteria",
    [
        {
            "condition": "crc",
            "required_features": ["rectal_bleeding"],
            "patient_name": "Jane Doe",
        },
        {
            "condition": "crc",
            "required_features": ["rectal_bleeding"],
            "nested": {"session_id": "session-1"},
        },
        {
            "condition": "crc",
            "required_features": ["rectal_bleeding"],
            "records": [{"record_id": "record-1"}],
        },
    ],
)
def test_request_rejects_direct_or_nested_patient_identifier_keys(
    cohort_criteria: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="patient identifiers are not allowed"):
        make_request(cohort_criteria=cohort_criteria)


def test_request_rejects_malformed_data_scope_cleanly() -> None:
    with pytest.raises(ValueError, match="data_scope must be a dict"):
        make_request(data_scope=[])


def test_contracts_reject_malformed_container_types_cleanly() -> None:
    with pytest.raises(ValueError, match="source_refs must be a list of dicts"):
        ResearchAsset(
            asset_id="research_asset_bad",
            asset_type="cohort_feasibility",
            title="Bad asset",
            status="candidate",
            created_by="research_workspace",
            created_at="2026-07-09T10:00:00+08:00",
            source_refs=["patient_record_projection_v0"],
            governance_refs=[],
        )
    with pytest.raises(ValueError, match="governance_refs must be a list of non-empty strings"):
        ResearchAsset(
            asset_id="research_asset_bad",
            asset_type="cohort_feasibility",
            title="Bad asset",
            status="candidate",
            created_by="research_workspace",
            created_at="2026-07-09T10:00:00+08:00",
            source_refs=[],
            governance_refs="review_item_001",
        )
    with pytest.raises(ValueError, match="variable_coverage must be a dict"):
        make_result(variable_coverage=[])
    with pytest.raises(ValueError, match="review_queue_items must be a list"):
        make_result(review_queue_items={})


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


@pytest.mark.parametrize("value", [1, "true", None])
def test_result_rejects_non_false_patient_level_rows(value: object) -> None:
    with pytest.raises(ValueError, match="patient_level_rows_returned must be false"):
        make_result(patient_level_rows_returned=value)


def test_review_queue_item_rejects_bad_status() -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        make_review_item(status="done")


@pytest.mark.parametrize("required_checks", [["authorization_basis", ""], "check"])
def test_review_queue_item_rejects_bad_required_checks(
    required_checks: object,
) -> None:
    with pytest.raises(ValueError, match="required_checks must be a list of non-empty strings"):
        make_review_item(required_checks=required_checks)


def test_result_to_dict_is_json_serializable() -> None:
    json.dumps(make_result().to_dict())


def test_to_dict_returns_defensive_nested_copies() -> None:
    result = make_result()

    first_payload = result.to_dict()
    first_payload["variable_coverage"]["rectal_bleeding"]["source_fact_types"].append(
        {"not": "json-safe"}
    )
    first_payload["review_queue_items"][0]["scope"]["project_id"] = {"not": object()}

    second_payload = result.to_dict()

    assert second_payload["variable_coverage"]["rectal_bleeding"]["source_fact_types"] == [
        "condition_signal"
    ]
    assert second_payload["review_queue_items"][0]["scope"]["project_id"] == "research_crc_001"
    json.dumps(second_payload)


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
