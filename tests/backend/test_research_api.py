from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import research


def _request_payload(*, patient_level_export_requested: bool = False) -> dict[str, Any]:
    return {
        "request_id": "cohort_request_crc_001",
        "project_id": "research_crc_001",
        "question": "Is there enough aggregate CRC data for feasibility review?",
        "cohort_criteria": {
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": ["rectal_bleeding"],
        },
        "data_scope": {
            "source": "patient_record_projection",
            "patient_level_export_requested": patient_level_export_requested,
            "deidentified_only": True,
        },
        "version_refs": {
            "projection_version": "patient_record_projection_v0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
        },
    }


def _triage_record() -> dict[str, Any]:
    return {
        "record_id": 10,
        "patient_id": 1,
        "record_type": "crc_triage_assessment",
        "normalized_payload_json": {
            "record_type": "crc_triage_assessment",
            "assessment_id": "crc_assessment_1",
            "known_crc_signals": {"rectal_bleeding": True},
            "safety_policy_version": "crc_safety_policy_v0",
        },
    }


def _client_with_registry(registry: object | None) -> TestClient:
    app = FastAPI()
    if registry is None:
        app.state.runtime = SimpleNamespace()
    else:
        app.state.runtime = SimpleNamespace(patient_registry_service=registry)
    app.include_router(research.router)
    return TestClient(app)


def test_cohort_feasibility_api_returns_aggregate_response() -> None:
    class RegistryStub:
        def list_research_projection_records(
            self,
            limit: int = 1000,
        ) -> list[dict[str, Any]]:
            assert limit == 1000
            return [_triage_record()]

    client = _client_with_registry(RegistryStub())

    try:
        response = client.post(
            "/api/admin/research/cohort-feasibility",
            json=_request_payload(),
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["estimated_count"] == 1
        assert payload["variable_coverage"]["rectal_bleeding"]["covered_count"] == 1
        assert payload["patient_level_rows_returned"] is False
        assert "patient_id" not in response.text
        assert payload["runtime"] == {
            "auth": "admin",
            "source": "patient_record_projection",
            "mode": "shadow_cohort_feasibility",
        }
    finally:
        client.close()


def test_cohort_feasibility_api_returns_blocked_result_for_export_request() -> None:
    class RaisingRegistryStub:
        def list_research_projection_records(
            self,
            limit: int = 1000,
        ) -> list[dict[str, Any]]:
            raise AssertionError("registry must not be called for export requests")

    client = _client_with_registry(RaisingRegistryStub())

    try:
        response = client.post(
            "/api/admin/research/cohort-feasibility",
            json=_request_payload(patient_level_export_requested=True),
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "blocked_by_governance"
        assert payload["estimated_count"] == 0
        assert payload["patient_level_rows_returned"] is False
    finally:
        client.close()


def test_cohort_feasibility_api_returns_503_without_registry() -> None:
    client = _client_with_registry(None)

    try:
        response = client.post(
            "/api/admin/research/cohort-feasibility",
            json=_request_payload(),
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Patient registry is not initialized"
    finally:
        client.close()
