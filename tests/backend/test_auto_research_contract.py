from __future__ import annotations

from collections.abc import Callable

import pytest

from src.contracts.auto_research import (
    AutoResearchRequest,
    AutoResearchRun,
    HypothesisReview,
    ResearchHypothesis,
    ResearchSource,
    ResearchStage,
    ResearchStudyPlan,
    make_auto_research_request_hash,
    make_auto_research_run_id,
    make_research_plan_id,
)


def _request(**overrides: object) -> AutoResearchRequest:
    payload: dict[str, object] = {
        "request_id": "request_crc_001",
        "project_id": "project_crc_001",
        "question": "Which biomarkers predict colorectal cancer recurrence?",
        "requested_by": "pi_operator",
        "idempotency_key": "auto-research-001",
        "max_sources": 5,
        "max_hypotheses": 2,
        "max_iterations": 2,
        "deidentified": True,
    }
    payload.update(overrides)
    return AutoResearchRequest(**payload)  # type: ignore[arg-type]


def _source() -> ResearchSource:
    return ResearchSource(
        source_id="research_source_pubmed_001",
        title="A colorectal cancer biomarker study",
        url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        abstract="The study reports a testable association.",
        journal="Example Journal",
        publication_year="2026",
        source_type="Journal Article",
        query="colorectal cancer biomarkers",
        retrieved_at="2026-07-19T08:00:00+00:00",
        pmid="123456",
    )


def _hypothesis(source_id: str = "research_source_pubmed_001") -> ResearchHypothesis:
    return ResearchHypothesis(
        hypothesis_id="research_hypothesis_001",
        statement="Biomarker A is associated with recurrence risk.",
        rationale="The source reports an association that needs independent validation.",
        testable_prediction="A preregistered model containing A improves held-out calibration.",
        supporting_source_ids=[source_id],
        counterevidence_source_ids=[],
        iteration=1,
        review=HypothesisReview(
            verdict="advance",
            evidence_support_score=0.7,
            novelty_score=0.5,
            testability_score=0.9,
            safety_risk="low if kept outside clinical use",
            critique="Single-study support requires external validation.",
            revision_instructions="",
        ),
    )


def _run(**overrides: object) -> AutoResearchRun:
    request = _request()
    source = _source()
    hypothesis = _hypothesis()
    run_id = make_auto_research_run_id(
        request.project_id,
        request.idempotency_key,
    )
    payload: dict[str, object] = {
        "run_id": run_id,
        "request_hash": make_auto_research_request_hash(request),
        "request": request,
        "status": "completed_shadow",
        "created_at": "2026-07-19T08:00:00+00:00",
        "completed_at": "2026-07-19T08:01:00+00:00",
        "stages": [
            ResearchStage(
                name="literature_search",
                status="completed",
                started_at="2026-07-19T08:00:00+00:00",
                completed_at="2026-07-19T08:00:10+00:00",
                summary="Retrieved one verified abstract.",
            ),
            ResearchStage(
                name="hypothesis_generation_1",
                status="completed",
                started_at="2026-07-19T08:00:10+00:00",
                completed_at="2026-07-19T08:00:20+00:00",
                summary="Generated one hypothesis.",
            ),
            ResearchStage(
                name="hypothesis_review_1",
                status="completed",
                started_at="2026-07-19T08:00:20+00:00",
                completed_at="2026-07-19T08:00:30+00:00",
                summary="Advanced one hypothesis.",
            ),
            ResearchStage(
                name="study_design",
                status="completed",
                started_at="2026-07-19T08:00:30+00:00",
                completed_at="2026-07-19T08:00:40+00:00",
                summary="Designed one unexecuted study.",
            ),
            ResearchStage(
                name="report_synthesis",
                status="completed",
                started_at="2026-07-19T08:00:40+00:00",
                completed_at="2026-07-19T08:00:50+00:00",
                summary="Generated the shadow report.",
            ),
        ],
        "sources": [source],
        "hypotheses": [hypothesis],
        "study_plans": [
            ResearchStudyPlan(
                plan_id=make_research_plan_id(run_id, hypothesis.hypothesis_id),
                hypothesis_id=hypothesis.hypothesis_id,
                study_type="retrospective external validation",
                objective="Test the preregistered prediction.",
                required_data=["deidentified aggregate biomarker features"],
                analysis_steps=["Lock the model", "Evaluate held-out calibration"],
                success_criteria=["Calibration slope within preregistered range"],
                safety_constraints=["No patient-level export"],
            )
        ],
        "report_markdown": (
            "# 自动科研影子报告\n\n"
            "> 本报告由自动流程生成，仅供科研人员复核；不是临床事实、患者建议或已验证发现。\n\n"
            "Biomarker A requires external validation "
            "[research_source_pubmed_001].\n\n"
            "## 可核验来源\n\n"
            "- [research_source_pubmed_001] A colorectal cancer biomarker study "
            "— PMID 123456 — https://pubmed.ncbi.nlm.nih.gov/123456/\n"
        ),
        "iteration_count": 1,
        "provenance": {
            "pipeline_version": "shadow_auto_research_v1",
            "retriever": "ncbi_pubmed_eutilities",
            "reasoner": "fake",
        },
    }
    payload.update(overrides)
    return AutoResearchRun(**payload)  # type: ignore[arg-type]


def test_auto_research_run_round_trips_and_keeps_safety_boundaries() -> None:
    run = _run()

    restored = AutoResearchRun.from_dict(run.to_dict())

    assert restored == run
    payload = restored.to_dict()
    assert payload["mode"] == "shadow_only"
    assert payload["human_review_status"] == "needs_human_review"
    assert payload["applies_automatically"] is False
    assert payload["clinical_default_path_mutated"] is False
    assert payload["patient_level_rows_returned"] is False
    assert payload["study_plans"][0]["execution_status"] == "not_executed"


def test_request_hash_is_stable_but_changes_with_research_content() -> None:
    request = _request()

    assert make_auto_research_request_hash(request) == make_auto_research_request_hash(
        _request()
    )
    assert make_auto_research_request_hash(request) != make_auto_research_request_hash(
        _request(question="A different research question")
    )


@pytest.mark.parametrize(
    "question",
    [
        # Synthetic PII-shaped values used only to verify rejection.
        "Patient ID: SYNTHETIC-12345 should this case be included?",
        "请研究患者姓名：测试患者甲的复发风险",
        "Contact synthetic-pii@example.invalid about this patient",
        "身份证号：00000000000000000X",
        "联系电话：" + "1" + "3800000000",
    ],
)
def test_request_rejects_apparent_patient_identifiers(question: str) -> None:
    with pytest.raises(ValueError, match="apparent patient identifiers"):
        _request(question=question)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("request_id", "patient_id:12345", "apparent patient identifiers"),
        ("project_id", "研究项目张三", "opaque identifier"),
        ("idempotency_key", "x" * 129, "opaque identifier"),
        ("requested_by", "x" * 129, "must not exceed 128 characters"),
    ],
)
def test_request_rejects_unbounded_or_identifying_metadata(
    field_name: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("question", "", "question must be a non-empty string"),
        ("max_sources", 0, "max_sources must be between 1 and 20"),
        ("max_hypotheses", 6, "max_hypotheses must be between 1 and 5"),
        ("max_iterations", 4, "max_iterations must be between 1 and 3"),
        ("deidentified", False, "deidentified must be true"),
    ],
)
def test_request_rejects_invalid_boundaries(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**{field_name: value})


def test_run_rejects_unknown_hypothesis_source() -> None:
    with pytest.raises(ValueError, match="unknown source ids"):
        _run(hypotheses=[_hypothesis("research_source_unknown")])


def test_run_rejects_forged_request_hash_and_run_id() -> None:
    with pytest.raises(ValueError, match="request_hash must match"):
        _run(request_hash=f"sha256:{'0' * 64}")
    with pytest.raises(ValueError, match="run_id must match"):
        _run(run_id="auto_research_run_forged")


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("applies_automatically", True, "applies_automatically must be false"),
        (
            "clinical_default_path_mutated",
            True,
            "clinical_default_path_mutated must be false",
        ),
        (
            "patient_level_rows_returned",
            True,
            "patient_level_rows_returned must be false",
        ),
    ],
)
def test_run_rejects_mutating_boundaries(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _run(**{field_name: value})


def test_failed_stage_requires_an_error() -> None:
    with pytest.raises(ValueError, match="failed stage requires error"):
        ResearchStage(
            name="literature_search",
            status="failed",
            started_at="2026-07-19T08:00:00+00:00",
            completed_at="2026-07-19T08:00:01+00:00",
            summary="Failed closed.",
        )


def test_non_failed_stage_rejects_an_error() -> None:
    with pytest.raises(ValueError, match="non-failed stage"):
        ResearchStage(
            name="literature_search",
            status="completed",
            started_at="2026-07-19T08:00:00+00:00",
            completed_at="2026-07-19T08:00:01+00:00",
            summary="Completed.",
            error="must not be present",
        )


def test_completed_run_requires_the_complete_stage_topology() -> None:
    run = _run()

    with pytest.raises(ValueError, match="complete stage topology"):
        _run(stages=run.stages[:-1])


def test_failed_shadow_is_limited_to_an_empty_literature_failure() -> None:
    failed_stage = ResearchStage(
        name="literature_search",
        status="failed",
        started_at="2026-07-19T08:00:00+00:00",
        completed_at="2026-07-19T08:00:01+00:00",
        summary="Failed closed.",
        error="provider unavailable",
    )
    run = _run(
        status="failed_shadow",
        stages=[failed_stage],
        sources=[],
        hypotheses=[],
        study_plans=[],
        report_markdown="",
        iteration_count=0,
    )

    assert run.status == "failed_shadow"
    with pytest.raises(ValueError, match="must not contain research artifacts"):
        _run(
            status="failed_shadow",
            stages=[failed_stage],
            hypotheses=[],
            study_plans=[],
            report_markdown="",
            iteration_count=0,
        )


def test_partial_shadow_allows_only_a_terminal_failure_and_no_report() -> None:
    completed = _run()
    failed_report = ResearchStage(
        name="report_synthesis",
        status="failed",
        started_at="2026-07-19T08:00:40+00:00",
        completed_at="2026-07-19T08:00:50+00:00",
        summary="Failed closed.",
        error="report validation failed",
    )
    partial = _run(
        status="partial_shadow",
        stages=[*completed.stages[:-1], failed_report],
        report_markdown="",
    )

    assert partial.status == "partial_shadow"
    with pytest.raises(ValueError, match="must not contain a completed report"):
        _run(
            status="partial_shadow",
            stages=[*completed.stages[:-1], failed_report],
        )
    failed_study = ResearchStage(
        name="study_design",
        status="failed",
        started_at="2026-07-19T08:00:30+00:00",
        completed_at="2026-07-19T08:00:40+00:00",
        summary="Failed closed.",
        error="study design failed",
    )
    with pytest.raises(ValueError, match="terminal failed stage"):
        _run(
            status="partial_shadow",
            stages=[
                *completed.stages[:3],
                failed_study,
                completed.stages[-1],
            ],
            report_markdown="",
        )


def test_run_rejects_duplicate_or_forged_study_plans() -> None:
    plan = _run().study_plans[0]

    with pytest.raises(ValueError, match="study plan ids must be unique"):
        _run(study_plans=[plan, plan])
    forged = ResearchStudyPlan(
        plan_id="research_plan_forged",
        hypothesis_id=plan.hypothesis_id,
        study_type=plan.study_type,
        objective=plan.objective,
        required_data=plan.required_data,
        analysis_steps=plan.analysis_steps,
        success_criteria=plan.success_criteria,
        safety_constraints=plan.safety_constraints,
    )
    with pytest.raises(ValueError, match="at most one study plan"):
        _run(study_plans=[plan, forged])
    with pytest.raises(ValueError, match="plan id must match"):
        _run(study_plans=[forged])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.replace(
                "validation [research_source_pubmed_001]",
                "validation [research_source_unknown]",
            ),
            "unknown source ids",
        ),
        (
            lambda report: report.replace(
                "\n\n## 可核验来源",
                "\nAn uncited persisted claim.\n\n## 可核验来源",
            ),
            "uncited content lines",
        ),
        (
            lambda report: report.replace(
                "- [research_source_pubmed_001] A colorectal cancer biomarker study "
                "— PMID 123456 — https://pubmed.ncbi.nlm.nih.gov/123456/\n",
                "",
            ),
            "ledger must match persisted sources",
        ),
    ],
)
def test_from_dict_revalidates_persisted_report_bindings(
    mutate: Callable[[str], str],
    message: str,
) -> None:
    payload = _run().to_dict()
    payload["report_markdown"] = mutate(payload["report_markdown"])

    with pytest.raises(ValueError, match=message):
        AutoResearchRun.from_dict(payload)
