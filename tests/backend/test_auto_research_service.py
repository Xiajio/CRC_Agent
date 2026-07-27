from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.api.services.auto_research_store import AutoResearchRunStore
from src.contracts.auto_research import (
    AutoResearchRequest,
    HypothesisReview,
    ResearchHypothesis,
    ResearchSource,
)
from src.services.auto_research_service import (
    AutoResearchConflictError,
    AutoResearchService,
    AutoResearchServiceUnavailableError,
    DeferredResearchReasoner,
    HypothesisDraft,
    LLMResearchReasoner,
    ReviewedDraft,
    StudyPlanDraft,
)


def _request(**overrides: object) -> AutoResearchRequest:
    payload: dict[str, object] = {
        "request_id": "request_crc_auto_001",
        "project_id": "project_crc_auto",
        "question": "Which biomarkers predict colorectal cancer recurrence?",
        "requested_by": "pi_operator",
        "idempotency_key": "auto-research-service-001",
        "max_sources": 4,
        "max_hypotheses": 2,
        "max_iterations": 2,
        "deidentified": True,
    }
    payload.update(overrides)
    return AutoResearchRequest(**payload)  # type: ignore[arg-type]


def _source() -> ResearchSource:
    return ResearchSource(
        source_id="research_source_pubmed_verified",
        title="Verified biomarker article",
        url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        abstract="Biomarker A was associated with recurrence in a retrospective cohort.",
        journal="Verified Journal",
        publication_year="2026",
        source_type="Journal Article",
        query="colorectal cancer recurrence biomarker",
        retrieved_at="2026-07-19T08:00:00+00:00",
        pmid="123456",
    )


@dataclass
class FakeRetriever:
    sources: list[ResearchSource]
    calls: int = 0

    @property
    def provider_name(self) -> str:
        return "fake_pubmed"

    def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]:
        self.calls += 1
        assert question
        return list(self.sources[:max_sources])


class FakeReasoner:
    def __init__(
        self,
        verdicts: list[str] | None = None,
        report_source_id: str = "research_source_pubmed_verified",
    ) -> None:
        self.verdicts = verdicts or ["advance"]
        self.report_source_id = report_source_id
        self.generation_calls = 0
        self.review_calls = 0
        self.plan_calls = 0
        self.report_calls = 0
        self.previous_review_counts: list[int] = []

    @property
    def provider_name(self) -> str:
        return "fake_reasoner"

    def generate_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        max_hypotheses: int,
        iteration: int,
        previous_reviews: list[ReviewedDraft],
    ) -> list[HypothesisDraft]:
        self.generation_calls += 1
        self.previous_review_counts.append(len(previous_reviews))
        return [
            HypothesisDraft(
                statement=f"Iteration {iteration}: biomarker A predicts recurrence.",
                rationale="The verified source reports an association.",
                testable_prediction="A locked model improves held-out calibration.",
                supporting_source_ids=[sources[0].source_id],
                counterevidence_source_ids=[],
            )
        ][:max_hypotheses]

    def review_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        drafts: list[HypothesisDraft],
    ) -> list[HypothesisReview]:
        verdict = self.verdicts[min(self.review_calls, len(self.verdicts) - 1)]
        self.review_calls += 1
        return [
            HypothesisReview(
                verdict=verdict,  # type: ignore[arg-type]
                evidence_support_score=0.7,
                novelty_score=0.5,
                testability_score=0.9,
                safety_risk="low while shadow-only",
                critique="External validation remains necessary.",
                revision_instructions=(
                    "Narrow the prediction and add a held-out cohort."
                    if verdict == "revise"
                    else ""
                ),
            )
            for _draft in drafts
        ]

    def design_studies(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list,
    ) -> list[StudyPlanDraft]:
        self.plan_calls += 1
        return [
            StudyPlanDraft(
                hypothesis_id=hypotheses[0].hypothesis_id,
                study_type="external validation",
                objective="Evaluate a locked prediction in held-out data.",
                required_data=["deidentified aggregate biomarker features"],
                analysis_steps=["Lock model", "Evaluate calibration"],
                success_criteria=["Meet preregistered calibration threshold"],
                safety_constraints=["No patient-level export"],
            )
        ]

    def synthesize_report(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list,
        study_plans: list,
    ) -> str:
        self.report_calls += 1
        return f"The evidence supports a testable candidate [{self.report_source_id}]."


def test_llm_reasoner_uses_function_calling_for_structured_provider_compatibility() -> None:
    class StructuredRunnable:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def invoke(self, messages: object) -> dict[str, object]:
            assert messages
            return self.payload

    class RecordingModel:
        model_name = "recording-model"

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def with_structured_output(self, schema: type, **kwargs: object):
            self.calls.append((schema.__name__, kwargs))
            payloads: dict[str, dict[str, object]] = {
                "_HypothesisBatch": {
                    "hypotheses": [
                        {
                            "statement": "A testable source-grounded hypothesis.",
                            "rationale": "The verified source reports an association.",
                            "testable_prediction": "A held-out validation meets its target.",
                            "supporting_source_ids": [_source().source_id],
                            "counterevidence_source_ids": [],
                        }
                    ]
                },
                "_ReviewBatch": {
                    "reviews": [
                        {
                            "hypothesis_index": 0,
                            "verdict": "advance",
                            "evidence_support_score": 0.7,
                            "novelty_score": 0.5,
                            "testability_score": 0.9,
                            "safety_risk": "shadow-only",
                            "critique": "External validation is required.",
                            "revision_instructions": "",
                        }
                    ]
                },
                "_StudyPlanBatch": {
                    "plans": [
                        {
                            "hypothesis_id": "research_hypothesis_recording",
                            "study_type": "external validation",
                            "objective": "Validate the locked candidate.",
                            "required_data": ["deidentified aggregate data"],
                            "analysis_steps": ["Evaluate calibration"],
                            "success_criteria": ["Meet the locked threshold"],
                            "safety_constraints": ["No patient-level export"],
                        }
                    ]
                },
                "_ReportPayload": {
                    "sections": [
                        {
                            "heading": "Evidence overview",
                            "claims": [
                                {
                                    "text": "The source supports a reviewable candidate.",
                                    "source_ids": [_source().source_id],
                                }
                            ],
                        }
                    ]
                },
            }
            return StructuredRunnable(payloads[schema.__name__])

    model = RecordingModel()
    reasoner = LLMResearchReasoner(model)  # type: ignore[arg-type]
    source = _source()
    drafts = reasoner.generate_hypotheses(
        question="A public research question",
        sources=[source],
        max_hypotheses=1,
        iteration=1,
        previous_reviews=[],
    )
    reviews = reasoner.review_hypotheses(
        question="A public research question",
        sources=[source],
        drafts=drafts,
    )
    hypothesis = ResearchHypothesis(
        hypothesis_id="research_hypothesis_recording",
        statement=drafts[0].statement,
        rationale=drafts[0].rationale,
        testable_prediction=drafts[0].testable_prediction,
        supporting_source_ids=drafts[0].supporting_source_ids,
        counterevidence_source_ids=drafts[0].counterevidence_source_ids,
        iteration=1,
        review=reviews[0],
    )
    plans = reasoner.design_studies(
        question="A public research question",
        sources=[source],
        hypotheses=[hypothesis],
    )
    report = reasoner.synthesize_report(
        question="A public research question",
        sources=[source],
        hypotheses=[hypothesis],
        study_plans=[],
    )

    assert len(plans) == 1
    assert report == (
        "## Evidence overview\n"
        "- The source supports a reviewable candidate. "
        "[research_source_pubmed_verified]"
    )
    assert [kwargs for _schema, kwargs in model.calls] == [
        {"method": "function_calling"},
        {"method": "function_calling"},
        {"method": "function_calling"},
        {"method": "function_calling"},
    ]


def _service(
    root: Path,
    retriever: FakeRetriever,
    reasoner: FakeReasoner,
) -> AutoResearchService:
    ticks = iter(
        f"2026-07-19T08:00:{index:02d}+00:00" for index in range(40)
    )
    return AutoResearchService(
        retriever=retriever,
        reasoner=reasoner,
        store=AutoResearchRunStore(root),
        now=lambda: next(ticks),
    )


def test_create_run_executes_full_shadow_research_loop_and_persists(tmp_path: Path) -> None:
    retriever = FakeRetriever([_source()])
    reasoner = FakeReasoner()
    service = _service(tmp_path / "reports" / "auto_research", retriever, reasoner)

    result = service.create_run(_request())

    run = result["run"]
    assert result["reused"] is False
    assert run["status"] == "completed_shadow"
    assert run["iteration_count"] == 1
    assert len(run["sources"]) == 1
    assert len(run["hypotheses"]) == 1
    assert run["hypotheses"][0]["review"]["verdict"] == "advance"
    assert len(run["study_plans"]) == 1
    assert run["study_plans"][0]["execution_status"] == "not_executed"
    assert "自动科研影子报告" in run["report_markdown"]
    assert run["applies_automatically"] is False
    assert run["clinical_default_path_mutated"] is False
    assert run["patient_level_rows_returned"] is False
    assert result["runtime"]["mode"] == "shadow_auto_research"
    assert service.get_run(run["run_id"])["run"] == run
    assert service.list_runs()["runs"] == [run]


def test_research_loop_uses_review_feedback_and_preserves_iteration_history(
    tmp_path: Path,
) -> None:
    reasoner = FakeReasoner(verdicts=["revise", "advance"])
    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        reasoner,
    )

    run = service.create_run(_request())["run"]

    assert run["status"] == "completed_shadow"
    assert run["iteration_count"] == 2
    assert [item["iteration"] for item in run["hypotheses"]] == [1, 2]
    assert [item["review"]["verdict"] for item in run["hypotheses"]] == [
        "revise",
        "advance",
    ]
    assert reasoner.previous_review_counts == [0, 1]


def test_same_idempotent_request_reuses_run_without_calling_providers(tmp_path: Path) -> None:
    retriever = FakeRetriever([_source()])
    reasoner = FakeReasoner()
    service = _service(tmp_path / "reports" / "auto_research", retriever, reasoner)
    request = _request()

    first = service.create_run(request)
    second = service.create_run(request)

    assert first["run"] == second["run"]
    assert second["reused"] is True
    assert retriever.calls == 1
    assert reasoner.generation_calls == 1


def test_existing_idempotent_run_is_reused_without_initializing_missing_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    request = _request()
    first_service = _service(root, FakeRetriever([_source()]), FakeReasoner())
    first = first_service.create_run(request)
    factory_calls = 0
    retriever = FakeRetriever([_source()])

    def unavailable_reasoner():
        nonlocal factory_calls
        factory_calls += 1
        raise AutoResearchServiceUnavailableError("model unavailable")

    reuse_service = AutoResearchService(
        retriever=retriever,
        reasoner=DeferredResearchReasoner(unavailable_reasoner),
        store=AutoResearchRunStore(root),
    )

    reused = reuse_service.create_run(request)

    assert reused["reused"] is True
    assert reused["run"] == first["run"]
    assert factory_calls == 0
    assert retriever.calls == 0


def test_idempotency_key_conflicts_when_request_content_changes(tmp_path: Path) -> None:
    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        FakeReasoner(),
    )
    service.create_run(_request())

    with pytest.raises(AutoResearchConflictError, match="different research request"):
        service.create_run(_request(question="A different research question"))


def test_no_verified_sources_creates_inspectable_failed_run(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "auto_research"
    service = _service(root, FakeRetriever([]), FakeReasoner())

    result = service.create_run(_request())

    run = result["run"]
    assert run["status"] == "failed_shadow"
    assert run["sources"] == []
    assert run["hypotheses"] == []
    assert run["study_plans"] == []
    assert run["report_markdown"] == ""
    assert run["stages"][-1]["name"] == "literature_search"
    assert run["stages"][-1]["status"] == "failed"
    assert "no PubMed abstracts" in run["stages"][-1]["error"]
    assert list((root / "runs").glob("*.json"))


def test_provider_errors_are_redacted_before_response_and_persistence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"

    class LeakyRetriever(FakeRetriever):
        def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]:
            raise RuntimeError(
                "Authorization: Bearer SYNTHETIC_BEARER_VALUE; "
                "headers={'Authorization': 'Bearer SYNTHETIC_NESTED_BEARER_VALUE'}, "
                "api_key=SYNTHETIC_API_KEY_VALUE&email=synthetic-operator@example.invalid, "
                "token='SYNTHETIC_TOKEN_VALUE'"
            )

    service = _service(root, LeakyRetriever([]), FakeReasoner())

    run = service.create_run(_request())["run"]

    error = run["stages"][-1]["error"]
    persisted = next((root / "runs").glob("*.json")).read_text(encoding="utf-8")
    for secret in (
        "SYNTHETIC_BEARER_VALUE",
        "SYNTHETIC_NESTED_BEARER_VALUE",
        "SYNTHETIC_API_KEY_VALUE",
        "SYNTHETIC_TOKEN_VALUE",
        "synthetic-operator@example.invalid",
    ):
        assert secret not in error
        assert secret not in persisted
    assert "[REDACTED]" in error


def test_unknown_report_citation_fails_closed_and_keeps_partial_artifacts(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        FakeReasoner(report_source_id="research_source_fabricated"),
    )

    run = service.create_run(_request())["run"]

    assert run["status"] == "partial_shadow"
    assert run["report_markdown"] == ""
    assert run["stages"][-1]["name"] == "report_synthesis"
    assert run["stages"][-1]["status"] == "failed"
    assert "unknown sources" in run["stages"][-1]["error"]


def test_same_idempotent_request_is_serialized_across_service_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    retriever = FakeRetriever([_source()])
    reasoner = FakeReasoner()
    services = [
        _service(root, retriever, reasoner),
        _service(root, retriever, reasoner),
    ]
    request = _request()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda service: service.create_run(request), services))

    assert sorted(result["reused"] for result in results) == [False, True]
    assert results[0]["run"] == results[1]["run"]
    assert retriever.calls == 1
    assert reasoner.generation_calls == 1


def test_second_iteration_failure_is_attributed_to_the_active_stage(
    tmp_path: Path,
) -> None:
    class FailingSecondGenerationReasoner(FakeReasoner):
        def generate_hypotheses(self, **kwargs):
            if kwargs["iteration"] == 2:
                raise RuntimeError("second generation failed")
            return super().generate_hypotheses(**kwargs)

    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        FailingSecondGenerationReasoner(verdicts=["revise"]),
    )

    run = service.create_run(_request())[
        "run"
    ]

    assert run["status"] == "partial_shadow"
    assert run["stages"][-1]["name"] == "hypothesis_generation_2"
    assert run["stages"][-1]["status"] == "failed"
    assert "second generation failed" in run["stages"][-1]["error"]


def test_all_hypotheses_rejected_is_a_completed_negative_result(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        FakeReasoner(verdicts=["reject"]),
    )

    run = service.create_run(_request())["run"]

    assert run["status"] == "completed_shadow"
    assert run["study_plans"] == []
    assert next(stage for stage in run["stages"] if stage["name"] == "study_design")[
        "status"
    ] == "skipped"


def test_missing_plan_for_an_advanced_hypothesis_fails_study_design(
    tmp_path: Path,
) -> None:
    class IncompletePlanReasoner(FakeReasoner):
        def generate_hypotheses(self, **kwargs):
            sources = kwargs["sources"]
            return [
                HypothesisDraft(
                    statement=f"Candidate {index} predicts recurrence.",
                    rationale="The verified source reports an association.",
                    testable_prediction=f"Prediction {index} is falsifiable.",
                    supporting_source_ids=[sources[0].source_id],
                    counterevidence_source_ids=[],
                )
                for index in (1, 2)
            ]

    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        IncompletePlanReasoner(),
    )

    run = service.create_run(_request())["run"]

    assert run["status"] == "partial_shadow"
    assert run["study_plans"] == []
    assert run["stages"][-1]["name"] == "study_design"
    assert "exactly one plan" in run["stages"][-1]["error"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("A fabricated numeric citation [1].", "unknown sources"),
        (
            "A supported statement [research_source_pubmed_verified].\nAn uncited claim.",
            "uncited content lines",
        ),
    ],
)
def test_report_rejects_unknown_or_uncited_content_lines(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    class InvalidReportReasoner(FakeReasoner):
        def synthesize_report(self, **kwargs) -> str:
            return body

    service = _service(
        tmp_path / "reports" / "auto_research",
        FakeRetriever([_source()]),
        InvalidReportReasoner(),
    )

    run = service.create_run(_request())["run"]

    assert run["status"] == "partial_shadow"
    assert run["report_markdown"] == ""
    assert message in run["stages"][-1]["error"]
