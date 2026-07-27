from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import research
from backend.api.services.auto_research_store import (
    AutoResearchRunNotFoundError,
    AutoResearchRunStore,
)
from backend.api.services.settings import RuntimeSettings
from backend.app import BearerAuthMiddleware
from src.contracts.auto_research import (
    AutoResearchRequest,
    HypothesisReview,
    ResearchSource,
)
from src.services.auto_research_service import (
    AutoResearchConflictError,
    AutoResearchService,
    HypothesisDraft,
    StudyPlanDraft,
)


def _request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "request_api_001",
        "project_id": "project_crc_auto",
        "question": "Which biomarkers predict colorectal cancer recurrence?",
        "requested_by": "pi_operator",
        "idempotency_key": "auto-research-api-001",
        "max_sources": 6,
        "max_hypotheses": 2,
        "max_iterations": 2,
        "deidentified": True,
    }
    payload.update(overrides)
    return payload


def _envelope(kind: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "integrity": {"status": "verified", "warnings": []},
        "runtime": {
            "auth": "admin",
            "source": "reports/auto_research",
            "mode": "shadow_auto_research",
        },
    }
    if kind == "list":
        payload["runs"] = [{"run_id": "auto_research_run_api", "status": "completed_shadow"}]
    else:
        payload["run"] = {
            "run_id": "auto_research_run_api",
            "status": "completed_shadow",
            "applies_automatically": False,
            "clinical_default_path_mutated": False,
            "patient_level_rows_returned": False,
        }
        if kind == "create":
            payload["reused"] = False
    return payload


class FakeService:
    def __init__(self) -> None:
        self.created_request: AutoResearchRequest | None = None

    def list_runs(self):
        return _envelope("list")

    def get_run(self, run_id: str):
        assert run_id == "auto_research_run_api"
        return _envelope("detail")

    def create_run(self, request: AutoResearchRequest):
        self.created_request = request
        return _envelope("create")


class IntegrationRetriever:
    provider_name = "fake_pubmed_integration"

    def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]:
        return [
            ResearchSource(
                source_id="research_source_api_integration",
                title="Verified API integration source",
                url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                abstract="A deidentified aggregate association requires validation.",
                journal="Integration Journal",
                publication_year="2026",
                source_type="Journal Article",
                query=question,
                retrieved_at="2026-07-21T08:00:00+00:00",
                pmid="123456",
            )
        ][:max_sources]


class IntegrationReasoner:
    provider_name = "fake_reasoner_integration"

    def generate_hypotheses(self, *, sources: list[ResearchSource], **kwargs: object):
        return [
            HypothesisDraft(
                statement="The aggregate marker predicts recurrence.",
                rationale="The verified abstract reports an association.",
                testable_prediction="A locked validation meets its calibration target.",
                supporting_source_ids=[sources[0].source_id],
                counterevidence_source_ids=[],
            )
        ]

    def review_hypotheses(self, *, drafts: list[HypothesisDraft], **kwargs: object):
        return [
            HypothesisReview(
                verdict="advance",
                evidence_support_score=0.7,
                novelty_score=0.5,
                testability_score=0.9,
                safety_risk="shadow-only interpretation required",
                critique="Independent validation remains necessary.",
                revision_instructions="",
            )
            for _draft in drafts
        ]

    def design_studies(self, *, hypotheses: list, **kwargs: object):
        return [
            StudyPlanDraft(
                hypothesis_id=hypothesis.hypothesis_id,
                study_type="external validation",
                objective="Validate a locked aggregate model.",
                required_data=["deidentified aggregate features"],
                analysis_steps=["Evaluate held-out calibration"],
                success_criteria=["Meet a preregistered threshold"],
                safety_constraints=["No patient-level export"],
            )
            for hypothesis in hypotheses
        ]

    def synthesize_report(self, *, sources: list[ResearchSource], **kwargs: object):
        return f"A human must review this source-grounded candidate [{sources[0].source_id}]."


def _client(monkeypatch: pytest.MonkeyPatch, service: object) -> TestClient:
    monkeypatch.setattr(research, "_auto_research_service", lambda: service)
    app = FastAPI()
    app.include_router(research.router)
    return TestClient(app)


def test_auto_research_list_detail_and_create_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    client = _client(monkeypatch, service)

    try:
        list_response = client.get("/api/admin/research/runs")
        detail_response = client.get(
            "/api/admin/research/runs/auto_research_run_api"
        )
        create_response = client.post(
            "/api/admin/research/runs",
            json=_request_payload(),
        )

        assert list_response.status_code == 200
        assert list_response.json()["runs"][0]["run_id"] == "auto_research_run_api"
        assert detail_response.status_code == 200
        assert detail_response.json()["run"]["applies_automatically"] is False
        assert create_response.status_code == 200
        assert create_response.json()["runtime"]["mode"] == "shadow_auto_research"
        assert isinstance(service.created_request, AutoResearchRequest)
        assert service.created_request.max_sources == 6
        assert service.created_request.deidentified is True
    finally:
        client.close()


def test_auto_research_api_round_trip_persists_and_reuses_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "auto_research"
    service = AutoResearchService(
        retriever=IntegrationRetriever(),
        reasoner=IntegrationReasoner(),
        store=AutoResearchRunStore(root),
    )
    client = _client(monkeypatch, service)

    try:
        created = client.post("/api/admin/research/runs", json=_request_payload())
        reused = client.post("/api/admin/research/runs", json=_request_payload())
        run_id = created.json()["run"]["run_id"]
        listed = client.get("/api/admin/research/runs")
        detailed = client.get(f"/api/admin/research/runs/{run_id}")

        assert created.status_code == 200
        assert created.json()["reused"] is False
        assert created.json()["run"]["status"] == "completed_shadow"
        assert created.json()["run"]["human_review_status"] == "needs_human_review"
        assert reused.status_code == 200
        assert reused.json()["reused"] is True
        assert reused.json()["run"] == created.json()["run"]
        assert listed.json()["runs"] == [created.json()["run"]]
        assert detailed.json()["run"] == created.json()["run"]
        assert (root / "runs" / f"{run_id}.json").is_file()
    finally:
        client.close()


def test_create_auto_research_api_rejects_extra_and_mutating_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, FakeService())
    payload = _request_payload(run_id="client_controlled", applies_automatically=True)

    try:
        response = client.post("/api/admin/research/runs", json=payload)

        assert response.status_code == 422
        detail = response.json()["detail"]
        locations = {tuple(item["loc"]) for item in detail}
        assert ("body", "run_id") in locations
        assert ("body", "applies_automatically") in locations
    finally:
        client.close()


def test_create_auto_research_api_requires_explicit_deidentification_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, FakeService())
    payload = _request_payload()
    payload.pop("deidentified")

    try:
        response = client.post("/api/admin/research/runs", json=payload)

        assert response.status_code == 422
        assert any(item["loc"] == ["body", "deidentified"] for item in response.json()["detail"])
    finally:
        client.close()


@pytest.mark.parametrize("deidentified", [False, 1, "true"])
def test_create_auto_research_api_requires_json_boolean_true(
    monkeypatch: pytest.MonkeyPatch,
    deidentified: object,
) -> None:
    service = FakeService()
    client = _client(monkeypatch, service)

    try:
        response = client.post(
            "/api/admin/research/runs",
            json=_request_payload(deidentified=deidentified),
        )

        assert response.status_code == 422
        assert service.created_request is None
    finally:
        client.close()


def test_read_service_does_not_initialize_the_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(research, "_AUTO_RESEARCH_STORE_ROOT", tmp_path / "auto_research")

    def fail_settings():
        raise RuntimeError("LLM configuration unavailable")

    monkeypatch.setattr(research, "load_settings", fail_settings)

    result = research._auto_research_service().list_runs()

    assert result["runs"] == []
    assert result["integrity"]["status"] == "verified"


def test_list_api_exposes_mismatched_artifact_without_listing_it_as_a_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "auto_research"
    seed_service = AutoResearchService(
        retriever=IntegrationRetriever(),
        reasoner=IntegrationReasoner(),
        store=AutoResearchRunStore(root),
    )
    created = seed_service.create_run(AutoResearchRequest(**_request_payload()))
    persisted_run_id = created["run"]["run_id"]
    valid_path = root / "runs" / f"{persisted_run_id}.json"
    filename_run_id = "auto_research_run_validation_copy"
    mismatched_path = root / "runs" / f"{filename_run_id}.json"
    mismatched_path.write_bytes(valid_path.read_bytes())
    valid_path.unlink()

    monkeypatch.setattr(research, "_AUTO_RESEARCH_STORE_ROOT", root)
    app = FastAPI()
    app.include_router(research.router)
    client = TestClient(app)

    try:
        response = client.get("/api/admin/research/runs")

        assert response.status_code == 200
        payload = response.json()
        assert payload["runs"] == []
        assert payload["integrity"]["status"] == "warning"
        assert payload["integrity"]["affected_artifacts"] == [
            {
                "code": "filename_run_id_mismatch",
                "artifact_path": f"runs/{filename_run_id}.json",
                "filename_run_id": filename_run_id,
                "persisted_run_id": persisted_run_id,
                "message": payload["integrity"]["warnings"][0],
                "excluded_from_runs": True,
            }
        ]
        assert [
            action["code"]
            for action in payload["integrity"]["recovery_actions"]
        ] == ["rerun_with_new_idempotency_key", "manual_quarantine"]
        assert mismatched_path.is_file()
    finally:
        client.close()


def test_create_returns_actionable_503_before_pubmed_or_persistence_when_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "auto_research"
    retrieval_calls = 0

    class MustNotRetrieve:
        provider_name = "must_not_retrieve"

        def retrieve(self, question: str, max_sources: int):
            nonlocal retrieval_calls
            retrieval_calls += 1
            raise AssertionError("PubMed must not be called without a reasoning model")

    def fail_settings():
        raise RuntimeError("LLM configuration unavailable")

    monkeypatch.setattr(research, "_AUTO_RESEARCH_STORE_ROOT", root)
    monkeypatch.setattr(research, "PubMedEvidenceRetriever", MustNotRetrieve)
    monkeypatch.setattr(research, "load_settings", fail_settings)
    app = FastAPI()
    app.include_router(research.router)
    client = TestClient(app)

    try:
        response = client.post("/api/admin/research/runs", json=_request_payload())

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "LLM_MODE=API" in detail
        assert "LLM_API_KEY" in detail
        assert "function-calling" in detail
        assert retrieval_calls == 0
        assert not root.exists()
    finally:
        client.close()


def test_create_auto_research_api_rejects_in_process_local_model_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "auto_research"
    retrieval_calls = 0

    class MustNotRetrieve:
        provider_name = "must_not_retrieve"

        def retrieve(self, question: str, max_sources: int):
            nonlocal retrieval_calls
            retrieval_calls += 1
            raise AssertionError("PubMed must not be called in unsupported Local mode")

    monkeypatch.setattr(research, "_AUTO_RESEARCH_STORE_ROOT", root)
    monkeypatch.setattr(research, "PubMedEvidenceRetriever", MustNotRetrieve)
    monkeypatch.setattr(
        research,
        "load_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(mode="Local")),
    )
    app = FastAPI()
    app.include_router(research.router)
    client = TestClient(app)

    try:
        response = client.post("/api/admin/research/runs", json=_request_payload())

        assert response.status_code == 503
        assert "requires LLM_MODE=API" in response.json()["detail"]
        assert retrieval_calls == 0
        assert not root.exists()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (AutoResearchRunNotFoundError("missing"), 404),
        (AutoResearchConflictError("conflict"), 409),
        (OSError("disk unavailable"), 500),
    ],
)
def test_auto_research_api_maps_service_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    class FailingService(FakeService):
        def get_run(self, run_id: str):
            raise error

    client = _client(monkeypatch, FailingService())

    try:
        response = client.get("/api/admin/research/runs/auto_research_run_api")

        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)
    finally:
        client.close()


def test_auto_research_real_router_requires_admin_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(research, "_auto_research_service", lambda: FakeService())
    app = FastAPI()
    app.include_router(research.router)
    app.add_middleware(
        BearerAuthMiddleware,
        settings=RuntimeSettings(
            auth_mode="bearer",
            api_bearer_token="user-token",
            api_admin_bearer_token="admin-token",
        ),
    )
    client = TestClient(app)

    try:
        user_response = client.get(
            "/api/admin/research/runs",
            headers={"Authorization": "Bearer user-token"},
        )
        admin_response = client.get(
            "/api/admin/research/runs",
            headers={"Authorization": "Bearer admin-token"},
        )

        assert user_response.status_code == 403
        assert admin_response.status_code == 200
    finally:
        client.close()
