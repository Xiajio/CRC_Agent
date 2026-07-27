from __future__ import annotations

from pathlib import Path

from backend.api.services.auto_research_store import AutoResearchRunStore
from src.contracts.auto_research import (
    AutoResearchRequest,
    HypothesisReview,
    ResearchSource,
)
from src.services.auto_research_service import (
    AutoResearchService,
    HypothesisDraft,
    StudyPlanDraft,
)


class _Retriever:
    provider_name = "fake_pubmed"

    def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]:
        return [
            ResearchSource(
                source_id="research_source_non_mutation",
                title="Verified article",
                url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                abstract="A testable association was reported.",
                journal="Verified Journal",
                publication_year="2026",
                source_type="Journal Article",
                query=question,
                retrieved_at="2026-07-19T08:00:00+00:00",
                pmid="123456",
            )
        ]


class _Reasoner:
    provider_name = "fake_reasoner"

    def generate_hypotheses(self, *, sources: list[ResearchSource], **kwargs: object):
        return [
            HypothesisDraft(
                statement="Biomarker A predicts recurrence.",
                rationale="The verified source reports an association.",
                testable_prediction="A locked model improves calibration.",
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
                safety_risk="low while shadow-only",
                critique="External validation is required.",
                revision_instructions="",
            )
            for _draft in drafts
        ]

    def design_studies(self, *, hypotheses: list, **kwargs: object):
        return [
            StudyPlanDraft(
                hypothesis_id=hypotheses[0].hypothesis_id,
                study_type="external validation",
                objective="Evaluate a locked prediction.",
                required_data=["deidentified aggregate features"],
                analysis_steps=["Evaluate held-out calibration"],
                success_criteria=["Meet preregistered threshold"],
                safety_constraints=["No patient-level export"],
            )
        ]

    def synthesize_report(self, *, sources: list[ResearchSource], **kwargs: object):
        return f"Testable candidate [{sources[0].source_id}]."


def _write_sentinel(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{label}: original\n", encoding="utf-8")


def _snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {label: path.read_text(encoding="utf-8") for label, path in paths.items()}


def _all_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_auto_research_writes_only_shadow_run_artifacts(tmp_path: Path) -> None:
    protected_paths = {
        "safety_policy": tmp_path / "config" / "clinical_safety_policy.yaml",
        "prompt": tmp_path / "src" / "prompts" / "crc_prompt.md",
        "route": tmp_path / "src" / "routes" / "crc_routes.json",
        "rag": tmp_path / "reports" / "literature" / "rag_index.json",
        "patient_state": tmp_path / "runtime" / "patient_state" / "current.json",
        "doctor_state": tmp_path / "runtime" / "doctor_state" / "current.json",
        "training_data": tmp_path / "training_data" / "research.jsonl",
        "model": tmp_path / "models" / "crc.bin",
    }
    for label, path in protected_paths.items():
        _write_sentinel(path, label)
    before = _snapshot(protected_paths)
    before_files = _all_files(tmp_path)
    ticks = iter(
        f"2026-07-19T08:00:{index:02d}+00:00" for index in range(30)
    )
    service = AutoResearchService(
        retriever=_Retriever(),
        reasoner=_Reasoner(),
        store=AutoResearchRunStore(tmp_path / "reports" / "auto_research"),
        now=lambda: next(ticks),
    )
    request = AutoResearchRequest(
        request_id="request_non_mutation",
        project_id="project_non_mutation",
        question="Which biomarkers predict colorectal cancer recurrence?",
        requested_by="pi_operator",
        idempotency_key="non-mutation-001",
    )

    result = service.create_run(request)

    assert result["run"]["status"] == "completed_shadow"
    assert _snapshot(protected_paths) == before
    new_files = _all_files(tmp_path) - before_files
    assert new_files
    assert all(path.startswith("reports/auto_research/") for path in new_files)
