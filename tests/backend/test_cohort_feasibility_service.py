from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.api.services.patient_registry_service import PatientRegistryService
from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


def _request(
    required_features: list[str],
    *,
    patient_level_export_requested: bool = False,
) -> CohortFeasibilityRequest:
    return CohortFeasibilityRequest(
        request_id="cohort_request_crc_001",
        project_id="research_crc_001",
        question="Is there enough aggregate CRC data for feasibility review?",
        cohort_criteria={
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": required_features,
        },
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": patient_level_export_requested,
            "deidentified_only": True,
        },
        version_refs={
            "projection_version": "patient_record_projection_v0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
        },
    )


def _triage_record(patient_id: int, *, rectal_bleeding: bool) -> dict[str, Any]:
    return {
        "record_id": patient_id * 10,
        "patient_id": patient_id,
        "record_type": "crc_triage_assessment",
        "document_type": "crc_triage_assessment",
        "normalized_payload_json": {
            "record_type": "crc_triage_assessment",
            "assessment_id": f"crc_assessment_{patient_id}",
            "known_crc_signals": {"rectal_bleeding": rectal_bleeding},
            "safety_policy_version": "crc_safety_policy_v0",
        },
    }


def _document_record(
    patient_id: int,
    *,
    document_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": patient_id * 10 + 1,
        "patient_id": patient_id,
        "record_type": "medical_card",
        "document_type": document_type,
        "normalized_payload_json": payload,
        "summary_text": document_type,
        "source": "patient_generated",
    }


def test_service_returns_aggregate_coverage_without_patient_rows() -> None:
    records = [
        _triage_record(1, rectal_bleeding=True),
        _document_record(
            1,
            document_type="pathology_report",
            payload={"pathology_result": "adenocarcinoma"},
        ),
        _triage_record(2, rectal_bleeding=True),
        _document_record(
            2,
            document_type="colonoscopy_report",
            payload={"colonoscopy_status": "completed"},
        ),
    ]

    result = CohortFeasibilityService().evaluate(
        request=_request(
            ["rectal_bleeding", "pathology_result", "colonoscopy_status"]
        ),
        records=records,
    )
    payload = result.to_dict()

    assert result.estimated_count == 2
    assert result.variable_coverage["rectal_bleeding"].covered_count == 2
    assert result.variable_coverage["pathology_result"].covered_count == 1
    assert result.variable_coverage["colonoscopy_status"].covered_count == 1
    assert result.missing_key_variables == [
        "pathology_result",
        "colonoscopy_status",
    ]
    assert result.patient_level_rows_returned is False
    assert "patient_id" not in json.dumps(payload, sort_keys=True)
    assert payload["review_queue_items"][0]["review_type"] == "research_ethics_review"


def test_service_marks_unmapped_features_without_crashing() -> None:
    result = CohortFeasibilityService().evaluate(
        request=_request(["unknown_feature"]),
        records=[_triage_record(1, rectal_bleeding=True)],
    )

    assert result.status == "needs_review"
    assert result.unmapped_required_features == ["unknown_feature"]
    assert result.variable_coverage["unknown_feature"].covered_count == 0


def test_service_accepts_positional_request_and_records() -> None:
    result = CohortFeasibilityService().evaluate(
        _request(["rectal_bleeding"]),
        [_triage_record(1, rectal_bleeding=True)],
    )

    assert result.status == "needs_review"
    assert result.estimated_count == 1
    assert result.variable_coverage["rectal_bleeding"].covered_count == 1


def test_service_returns_insufficient_data_for_empty_registry() -> None:
    result = CohortFeasibilityService().evaluate(
        request=_request(["rectal_bleeding"]),
        records=[],
    )

    assert result.status == "insufficient_data"
    assert result.estimated_count == 0
    assert result.variable_coverage["rectal_bleeding"].coverage_ratio == 0


def test_service_blocks_export_request_before_reading_records() -> None:
    def records() -> Any:
        raise AssertionError("records should not be iterated")
        yield {}

    result = CohortFeasibilityService().evaluate(
        request=_request(["rectal_bleeding"], patient_level_export_requested=True),
        records=records(),
    )

    assert result.status == "blocked_by_governance"
    assert result.estimated_count == 0
    assert result.patient_level_rows_returned is False


def test_registry_exposes_read_only_research_projection_records(tmp_path: Path) -> None:
    registry = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    registry.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "pathology.pdf",
            "content_type": "application/pdf",
            "sha256": "pathology-sha",
            "storage_path": str(tmp_path / "assets" / "pathology.pdf"),
            "source": "patient_generated",
        },
        patient_snapshot={},
        record_payload={
            "document_type": "pathology_report",
            "pathology_result": "adenocarcinoma",
        },
        summary_text="Pathology shows adenocarcinoma.",
        record_type="medical_card",
    )
    before = registry.get_patient_detail(patient_id)

    rows = registry.list_research_projection_records(10)

    after = registry.get_patient_detail(patient_id)
    assert len(rows) == 1
    assert rows[0]["patient_id"] == patient_id
    assert rows[0]["record_type"] == "medical_card"
    assert before == after
