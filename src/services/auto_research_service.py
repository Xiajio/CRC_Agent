from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import threading
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

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
    make_research_hypothesis_id,
    make_research_plan_id,
)


class AutoResearchPipelineError(RuntimeError):
    """Raised for a deterministic or provider failure in the research pipeline."""


class AutoResearchConflictError(RuntimeError):
    """Raised when an idempotency key is reused for different content."""


class AutoResearchServiceUnavailableError(RuntimeError):
    """Raised when a new Run cannot start because its reasoner is unavailable."""


class EvidenceRetriever(Protocol):
    @property
    def provider_name(self) -> str: ...

    def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]: ...


@dataclass(frozen=True)
class HypothesisDraft:
    statement: str
    rationale: str
    testable_prediction: str
    supporting_source_ids: list[str]
    counterevidence_source_ids: list[str]


@dataclass(frozen=True)
class ReviewedDraft:
    draft: HypothesisDraft
    review: HypothesisReview


@dataclass(frozen=True)
class StudyPlanDraft:
    hypothesis_id: str
    study_type: str
    objective: str
    required_data: list[str]
    analysis_steps: list[str]
    success_criteria: list[str]
    safety_constraints: list[str]


class ResearchReasoner(Protocol):
    @property
    def provider_name(self) -> str: ...

    def generate_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        max_hypotheses: int,
        iteration: int,
        previous_reviews: list[ReviewedDraft],
    ) -> list[HypothesisDraft]: ...

    def review_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        drafts: list[HypothesisDraft],
    ) -> list[HypothesisReview]: ...

    def design_studies(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list[ResearchHypothesis],
    ) -> list[StudyPlanDraft]: ...

    def synthesize_report(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list[ResearchHypothesis],
        study_plans: list[ResearchStudyPlan],
    ) -> str: ...


class DeferredResearchReasoner:
    """Initialize the configured LLM only when a new Run reaches reasoning."""

    def __init__(
        self,
        factory: Callable[[], ResearchReasoner],
        *,
        provider_hint: str = "llm:configured",
    ) -> None:
        self._factory = factory
        self._provider_hint = provider_hint
        self._delegate: ResearchReasoner | None = None
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        delegate = self._delegate
        return delegate.provider_name if delegate is not None else self._provider_hint

    def _get(self) -> ResearchReasoner:
        if self._delegate is None:
            with self._lock:
                if self._delegate is None:
                    self._delegate = self._factory()
        return self._delegate

    def ensure_available(self) -> None:
        """Initialize the delegate without starting retrieval or a pipeline stage."""

        self._get()

    def generate_hypotheses(self, **kwargs: Any) -> list[HypothesisDraft]:
        return self._get().generate_hypotheses(**kwargs)

    def review_hypotheses(self, **kwargs: Any) -> list[HypothesisReview]:
        return self._get().review_hypotheses(**kwargs)

    def design_studies(self, **kwargs: Any) -> list[StudyPlanDraft]:
        return self._get().design_studies(**kwargs)

    def synthesize_report(self, **kwargs: Any) -> str:
        return self._get().synthesize_report(**kwargs)


class _HypothesisPayload(BaseModel):
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    testable_prediction: str = Field(min_length=1)
    supporting_source_ids: list[str] = Field(min_length=1)
    counterevidence_source_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class _HypothesisBatch(BaseModel):
    hypotheses: list[_HypothesisPayload]

    model_config = ConfigDict(extra="forbid")


class _ReviewPayload(BaseModel):
    hypothesis_index: int = Field(ge=0)
    verdict: str
    evidence_support_score: float = Field(ge=0, le=1)
    novelty_score: float = Field(ge=0, le=1)
    testability_score: float = Field(ge=0, le=1)
    safety_risk: str = Field(min_length=1)
    critique: str = Field(min_length=1)
    revision_instructions: str = ""

    model_config = ConfigDict(extra="forbid")


class _ReviewBatch(BaseModel):
    reviews: list[_ReviewPayload]

    model_config = ConfigDict(extra="forbid")


class _StudyPlanPayload(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    study_type: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    required_data: list[str] = Field(min_length=1)
    analysis_steps: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    safety_constraints: list[str] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class _StudyPlanBatch(BaseModel):
    plans: list[_StudyPlanPayload]

    model_config = ConfigDict(extra="forbid")


class _ReportClaimPayload(BaseModel):
    text: str = Field(
        min_length=1,
        description="One evidence-bound claim; put citations in source_ids.",
    )
    source_ids: list[str] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class _ReportSectionPayload(BaseModel):
    heading: str = Field(min_length=1)
    claims: list[_ReportClaimPayload] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class _ReportPayload(BaseModel):
    sections: list[_ReportSectionPayload] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class LLMResearchReasoner:
    """Structured hypothesis generation, adversarial review and study design."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    @property
    def provider_name(self) -> str:
        model_name = str(
            getattr(self.model, "model_name", "")
            or getattr(self.model, "model", "")
            or type(self.model).__name__
        )
        return f"llm:{model_name}"

    def generate_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        max_hypotheses: int,
        iteration: int,
        previous_reviews: list[ReviewedDraft],
    ) -> list[HypothesisDraft]:
        source_context = _source_context(sources)
        revision_context = _review_context(previous_reviews)
        prompt = f"""研究问题：{question}

这是第 {iteration} 轮候选假设生成。只允许使用下列已从 PubMed 取回的摘要，不能引入未列出的事实或来源。

{source_context}

上一轮评审（首轮为空）：
{revision_context}

生成最多 {max_hypotheses} 个可证伪、可由计算分析或后续受控研究检验的候选假设。
每个假设必须绑定至少一个 supporting_source_ids；若摘要中存在不一致证据，写入 counterevidence_source_ids。
不要给患者建议，不要宣称发现已被证实，不要输出隐藏思维过程。"""
        response = self.model.with_structured_output(
            _HypothesisBatch,
            method="function_calling",
        ).invoke(
            [
                SystemMessage(
                    content=(
                        "你是结直肠癌科研假设生成器。输出是待人工复核的影子候选，"
                        "必须严格受给定证据约束。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        batch = _coerce_model(response, _HypothesisBatch)
        allowed_source_ids = {source.source_id for source in sources}
        drafts: list[HypothesisDraft] = []
        for item in batch.hypotheses[:max_hypotheses]:
            support = _known_ids(item.supporting_source_ids, allowed_source_ids)
            if not support:
                continue
            drafts.append(
                HypothesisDraft(
                    statement=item.statement.strip(),
                    rationale=item.rationale.strip(),
                    testable_prediction=item.testable_prediction.strip(),
                    supporting_source_ids=support,
                    counterevidence_source_ids=_known_ids(
                        item.counterevidence_source_ids,
                        allowed_source_ids,
                    ),
                )
            )
        if not drafts:
            raise AutoResearchPipelineError(
                "hypothesis generation returned no source-grounded candidates"
            )
        return drafts

    def review_hypotheses(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        drafts: list[HypothesisDraft],
    ) -> list[HypothesisReview]:
        payload = [
            {"hypothesis_index": index, **_draft_to_dict(draft)}
            for index, draft in enumerate(drafts)
        ]
        prompt = f"""研究问题：{question}

证据摘要：
{_source_context(sources)}

候选假设：
{json.dumps(payload, ensure_ascii=False, indent=2)}

逐条进行对抗性评审。verdict 只能是 advance、revise、reject。
重点检查证据是否真的支持、是否忽略反证、是否可证伪、是否可能造成临床安全误读。
不要因为语言流畅而给高分，也不要输出隐藏思维过程。"""
        response = self.model.with_structured_output(
            _ReviewBatch,
            method="function_calling",
        ).invoke(
            [
                SystemMessage(
                    content=(
                        "你是对抗性科研评审器。你的任务是发现证据跳跃、伪新颖性、"
                        "不可检验假设和安全风险。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        batch = _coerce_model(response, _ReviewBatch)
        by_index = {item.hypothesis_index: item for item in batch.reviews}
        reviews: list[HypothesisReview] = []
        for index, _draft in enumerate(drafts):
            item = by_index.get(index)
            if item is None:
                reviews.append(_missing_review())
                continue
            verdict = item.verdict.strip().lower()
            if verdict not in {"advance", "revise", "reject"}:
                verdict = "reject"
            reviews.append(
                HypothesisReview(
                    verdict=verdict,  # type: ignore[arg-type]
                    evidence_support_score=float(item.evidence_support_score),
                    novelty_score=float(item.novelty_score),
                    testability_score=float(item.testability_score),
                    safety_risk=item.safety_risk.strip(),
                    critique=item.critique.strip(),
                    revision_instructions=item.revision_instructions.strip(),
                )
            )
        return reviews

    def design_studies(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list[ResearchHypothesis],
    ) -> list[StudyPlanDraft]:
        advanced = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.review.verdict == "advance"
        ]
        if not advanced:
            return []
        prompt = f"""研究问题：{question}

已通过影子评审的假设：
{json.dumps([item.to_dict() for item in advanced], ensure_ascii=False, indent=2)}

证据索引：
{_source_context(sources)}

为每个假设生成一个可复现的研究计划。优先计算研究、公开数据验证或前瞻性研究设计；
不得声称已经执行，不得生成患者级数据导出步骤，不得绕过伦理审批。
计划必须包含客观成功标准和安全约束。"""
        response = self.model.with_structured_output(
            _StudyPlanBatch,
            method="function_calling",
        ).invoke(
            [
                SystemMessage(
                    content=(
                        "你是科研方案设计器。只设计尚未执行的研究，不提供患者治疗建议。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        batch = _coerce_model(response, _StudyPlanBatch)
        allowed = {item.hypothesis_id for item in advanced}
        plans: list[StudyPlanDraft] = []
        seen: set[str] = set()
        for item in batch.plans:
            if item.hypothesis_id not in allowed or item.hypothesis_id in seen:
                continue
            seen.add(item.hypothesis_id)
            plans.append(
                StudyPlanDraft(
                    hypothesis_id=item.hypothesis_id,
                    study_type=item.study_type.strip(),
                    objective=item.objective.strip(),
                    required_data=_non_empty_strings(item.required_data),
                    analysis_steps=_non_empty_strings(item.analysis_steps),
                    success_criteria=_non_empty_strings(item.success_criteria),
                    safety_constraints=_non_empty_strings(item.safety_constraints),
                )
            )
        return plans

    def synthesize_report(
        self,
        *,
        question: str,
        sources: list[ResearchSource],
        hypotheses: list[ResearchHypothesis],
        study_plans: list[ResearchStudyPlan],
    ) -> str:
        prompt = f"""研究问题：{question}

证据来源：
{_source_context(sources)}

        候选假设与对抗性复核：
{json.dumps([item.to_dict() for item in hypotheses], ensure_ascii=False, indent=2)}

尚未执行的研究计划：
{json.dumps([item.to_dict() for item in study_plans], ensure_ascii=False, indent=2)}

生成结构化影子科研报告，章节应覆盖：证据概况、候选假设、反证与局限、建议验证顺序。
每条 claim 必须是可独立复核的一条陈述，并在 source_ids 中绑定至少一个给定来源；
text 不要写 Markdown 引用标记。不得创造其他引用，不得把候选表述为已证实结论，
不得提供患者治疗建议。"""
        response = self.model.with_structured_output(
            _ReportPayload,
            method="function_calling",
        ).invoke(
            [
                SystemMessage(
                    content=(
                        "你是循证科研报告撰写器。只总结提供的工件，输出待人工复核的"
                        "影子报告，不输出隐藏思维过程。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        report = _coerce_model(response, _ReportPayload)
        return _render_structured_report(report, sources)


_RUN_LOCKS_GUARD = threading.Lock()
_RUN_LOCKS: dict[str, threading.Lock] = {}


def _run_lock(store: Any, run_id: str) -> threading.Lock:
    root = getattr(store, "root", None)
    store_key = str(root) if root is not None else f"store:{id(store)}"
    key = f"{store_key}:{run_id}"
    with _RUN_LOCKS_GUARD:
        return _RUN_LOCKS.setdefault(key, threading.Lock())


class AutoResearchService:
    PIPELINE_VERSION = "shadow_auto_research_v1"

    def __init__(
        self,
        *,
        retriever: EvidenceRetriever,
        reasoner: ResearchReasoner,
        store: Any,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.reasoner = reasoner
        self.store = store
        self._now = now or (
            lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
        )

    def list_runs(self) -> dict[str, Any]:
        state = self.store.read_state()
        return {
            "runs": [run.to_dict() for run in state.runs],
            "integrity": state.integrity,
            "runtime": _runtime_metadata(),
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        return {
            "run": run.to_dict(),
            "integrity": self.store.read_state().integrity,
            "runtime": _runtime_metadata(),
        }

    def create_run(self, request: AutoResearchRequest) -> dict[str, Any]:
        run_id = make_auto_research_run_id(
            request.project_id,
            request.idempotency_key,
        )
        request_hash = make_auto_research_request_hash(request)
        with _run_lock(self.store, run_id):
            return self._create_run_locked(
                request=request,
                run_id=run_id,
                request_hash=request_hash,
            )

    def _create_run_locked(
        self,
        *,
        request: AutoResearchRequest,
        run_id: str,
        request_hash: str,
    ) -> dict[str, Any]:
        existing = self.store.find_run(run_id)
        if existing is not None:
            if existing.request_hash != request_hash:
                raise AutoResearchConflictError(
                    "idempotency key already belongs to a different research request"
                )
            return {
                "run": existing.to_dict(),
                "reused": True,
                "integrity": self.store.read_state().integrity,
                "runtime": _runtime_metadata(),
            }

        ensure_available = getattr(self.reasoner, "ensure_available", None)
        if callable(ensure_available):
            # Resolve missing model configuration before any PubMed egress. Existing
            # idempotent Runs are returned above and remain readable without a model.
            ensure_available()

        run = self._execute(run_id=run_id, request=request, request_hash=request_hash)
        try:
            self.store.write_run(run)
        except FileExistsError:
            winner = self.store.get_run(run_id)
            if winner.request_hash != request_hash:
                raise AutoResearchConflictError(
                    "idempotency key already belongs to a different research request"
                )
            return {
                "run": winner.to_dict(),
                "reused": True,
                "integrity": self.store.read_state().integrity,
                "runtime": _runtime_metadata(),
            }
        return {
            "run": run.to_dict(),
            "reused": False,
            "integrity": self.store.read_state().integrity,
            "runtime": _runtime_metadata(),
        }

    def _execute(
        self,
        *,
        run_id: str,
        request: AutoResearchRequest,
        request_hash: str,
    ) -> AutoResearchRun:
        created_at = self._now()
        stages: list[ResearchStage] = []
        sources: list[ResearchSource] = []
        hypotheses: list[ResearchHypothesis] = []
        current_hypotheses: list[ResearchHypothesis] = []
        plans: list[ResearchStudyPlan] = []
        report = ""
        iteration_count = 0
        active_stage_name = "literature_search"
        active_stage_started_at = created_at

        try:
            active_stage_started_at = self._now()
            sources = self.retriever.retrieve(
                request.question,
                request.max_sources,
            )
            if not sources:
                raise AutoResearchPipelineError(
                    "no PubMed abstracts were retrieved for the research question"
                )
            stages.append(
                _completed_stage(
                    "literature_search",
                    active_stage_started_at,
                    self._now(),
                    f"Retrieved {len(sources)} verified PubMed abstracts.",
                )
            )
            previous_reviews: list[ReviewedDraft] = []
            for iteration in range(1, request.max_iterations + 1):
                iteration_count = iteration
                active_stage_name = f"hypothesis_generation_{iteration}"
                active_stage_started_at = self._now()
                drafts = self.reasoner.generate_hypotheses(
                    question=request.question,
                    sources=sources,
                    max_hypotheses=request.max_hypotheses,
                    iteration=iteration,
                    previous_reviews=previous_reviews,
                )
                normalized_statements = [
                    " ".join(draft.statement.split()).casefold() for draft in drafts
                ]
                if len(set(normalized_statements)) != len(normalized_statements):
                    raise AutoResearchPipelineError(
                        "hypothesis generation returned duplicate candidates"
                    )
                stages.append(
                    _completed_stage(
                        active_stage_name,
                        active_stage_started_at,
                        self._now(),
                        f"Generated {len(drafts)} source-grounded hypotheses.",
                    )
                )

                active_stage_name = f"hypothesis_review_{iteration}"
                active_stage_started_at = self._now()
                reviews = self.reasoner.review_hypotheses(
                    question=request.question,
                    sources=sources,
                    drafts=drafts,
                )
                if len(reviews) != len(drafts):
                    raise AutoResearchPipelineError(
                        "hypothesis review count did not match generated candidates"
                    )
                reviewed = [
                    ReviewedDraft(draft=draft, review=review)
                    for draft, review in zip(drafts, reviews)
                ]
                previous_reviews = reviewed
                current_hypotheses = [
                    _materialize_hypothesis(run_id, iteration, item)
                    for item in reviewed
                ]
                stages.append(
                    _completed_stage(
                        active_stage_name,
                        active_stage_started_at,
                        self._now(),
                        _review_summary(reviews),
                    )
                )
                hypotheses.extend(current_hypotheses)
                if any(item.review.verdict == "advance" for item in reviewed):
                    break

            active_stage_name = "study_design"
            active_stage_started_at = self._now()
            advanced = [
                item
                for item in current_hypotheses
                if item.review.verdict == "advance"
            ]
            if advanced:
                plan_drafts = self.reasoner.design_studies(
                    question=request.question,
                    sources=sources,
                    hypotheses=advanced,
                )
                expected_hypothesis_ids = {item.hypothesis_id for item in advanced}
                planned_hypothesis_ids = {item.hypothesis_id for item in plan_drafts}
                if (
                    planned_hypothesis_ids != expected_hypothesis_ids
                    or len(plan_drafts) != len(expected_hypothesis_ids)
                ):
                    raise AutoResearchPipelineError(
                        "study design did not return exactly one plan per advanced hypothesis"
                    )
                plans = [
                    _materialize_plan(run_id, item) for item in plan_drafts
                ]
                stages.append(
                    _completed_stage(
                        "study_design",
                        active_stage_started_at,
                        self._now(),
                        f"Designed {len(plans)} unexecuted study plans.",
                    )
                )
            else:
                stages.append(
                    _skipped_stage(
                        "study_design",
                        active_stage_started_at,
                        self._now(),
                        "No hypothesis passed adversarial review.",
                    )
                )

            active_stage_name = "report_synthesis"
            active_stage_started_at = self._now()
            report_body = self.reasoner.synthesize_report(
                question=request.question,
                sources=sources,
                hypotheses=hypotheses,
                study_plans=plans,
            )
            _validate_report_citations(report_body, sources)
            report = _wrap_report(report_body, sources)
            stages.append(
                _completed_stage(
                    "report_synthesis",
                    active_stage_started_at,
                    self._now(),
                    "Generated a source-bound shadow report for human review.",
                )
            )
        except AutoResearchServiceUnavailableError:
            # Configuration failures are retryable deployment conditions, not
            # inspectable pipeline results. Do not turn them into sticky failed Runs.
            raise
        except Exception as exc:
            error = _safe_error(exc)
            failed_at = self._now()
            stages.append(
                ResearchStage(
                    name=active_stage_name,
                    status="failed",
                    started_at=active_stage_started_at,
                    completed_at=failed_at,
                    summary="The stage failed closed; no downstream mutation occurred.",
                    error=error,
                )
            )

        has_failed_stage = any(stage.status == "failed" for stage in stages)
        if not has_failed_stage and report:
            status = "completed_shadow"
        elif sources or hypotheses or report:
            status = "partial_shadow"
        else:
            status = "failed_shadow"

        return AutoResearchRun(
            run_id=run_id,
            request_hash=request_hash,
            request=request,
            status=status,  # type: ignore[arg-type]
            created_at=created_at,
            completed_at=self._now(),
            stages=stages,
            sources=sources,
            hypotheses=hypotheses,
            study_plans=plans,
            report_markdown=report,
            iteration_count=iteration_count,
            provenance={
                "pipeline_version": self.PIPELINE_VERSION,
                "retriever": self.retriever.provider_name,
                "reasoner": self.reasoner.provider_name,
            },
        )


def _runtime_metadata() -> dict[str, str]:
    return {
        "auth": "admin",
        "source": "reports/auto_research",
        "mode": "shadow_auto_research",
    }


def _coerce_model(value: Any, expected: type[BaseModel]) -> Any:
    if isinstance(value, expected):
        return value
    if isinstance(value, dict):
        return expected.model_validate(value)
    raise AutoResearchPipelineError(
        f"structured model output was not {expected.__name__}"
    )


def _source_context(sources: list[ResearchSource]) -> str:
    chunks = []
    for source in sources:
        chunks.append(
            f"SOURCE_ID: {source.source_id}\n"
            f"TITLE: {source.title}\n"
            f"YEAR/JOURNAL: {source.publication_year} / {source.journal}\n"
            f"PMID: {source.pmid or 'n/a'}\n"
            f"ABSTRACT: {source.abstract}"
        )
    return "\n\n".join(chunks)


def _review_context(reviews: list[ReviewedDraft]) -> str:
    if not reviews:
        return "无"
    return json.dumps(
        [
            {
                "hypothesis": item.draft.statement,
                "review": item.review.to_dict(),
            }
            for item in reviews
        ],
        ensure_ascii=False,
        indent=2,
    )


def _render_structured_report(
    report: _ReportPayload,
    sources: list[ResearchSource],
) -> str:
    allowed = {source.source_id for source in sources}
    lines: list[str] = []
    for section in report.sections:
        heading = " ".join(section.heading.lstrip("# ").split())
        if not heading:
            raise AutoResearchPipelineError("report section heading was empty")
        lines.append(f"## {heading}")
        for claim in section.claims:
            text = " ".join(claim.text.split())
            if not text:
                raise AutoResearchPipelineError("report claim was empty")
            source_ids = list(dict.fromkeys(claim.source_ids))
            unknown = set(source_ids) - allowed
            if unknown:
                raise AutoResearchPipelineError(
                    f"report cited unknown sources: {', '.join(sorted(unknown))}"
                )
            if not source_ids:
                raise AutoResearchPipelineError("report claim did not cite a source")
            citations = " ".join(f"[{source_id}]" for source_id in source_ids)
            lines.append(f"- {text} {citations}")
        lines.append("")
    content = "\n".join(lines).strip()
    if not content:
        raise AutoResearchPipelineError("report synthesis returned empty content")
    return content


def _known_ids(values: list[str], allowed: set[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value in allowed))


def _non_empty_strings(values: list[str]) -> list[str]:
    normalized = [value.strip() for value in values if value.strip()]
    if not normalized:
        raise AutoResearchPipelineError("study plan contains an empty required list")
    return normalized


def _draft_to_dict(draft: HypothesisDraft) -> dict[str, Any]:
    return {
        "statement": draft.statement,
        "rationale": draft.rationale,
        "testable_prediction": draft.testable_prediction,
        "supporting_source_ids": draft.supporting_source_ids,
        "counterevidence_source_ids": draft.counterevidence_source_ids,
    }


def _missing_review() -> HypothesisReview:
    return HypothesisReview(
        verdict="reject",
        evidence_support_score=0,
        novelty_score=0,
        testability_score=0,
        safety_risk="unreviewed",
        critique="The independent reviewer did not return a result for this hypothesis.",
        revision_instructions="Generate a fully reviewable candidate before reconsideration.",
    )


def _review_summary(reviews: list[HypothesisReview]) -> str:
    counts = {"advance": 0, "revise": 0, "reject": 0}
    for review in reviews:
        counts[review.verdict] += 1
    return (
        f"Adversarial review: {counts['advance']} advance, "
        f"{counts['revise']} revise, {counts['reject']} reject."
    )


def _materialize_hypothesis(
    run_id: str,
    iteration: int,
    reviewed: ReviewedDraft,
) -> ResearchHypothesis:
    draft = reviewed.draft
    return ResearchHypothesis(
        hypothesis_id=make_research_hypothesis_id(
            run_id,
            iteration,
            draft.statement,
        ),
        statement=draft.statement,
        rationale=draft.rationale,
        testable_prediction=draft.testable_prediction,
        supporting_source_ids=draft.supporting_source_ids,
        counterevidence_source_ids=draft.counterevidence_source_ids,
        iteration=iteration,
        review=reviewed.review,
    )


def _materialize_plan(run_id: str, draft: StudyPlanDraft) -> ResearchStudyPlan:
    return ResearchStudyPlan(
        plan_id=make_research_plan_id(run_id, draft.hypothesis_id),
        hypothesis_id=draft.hypothesis_id,
        study_type=draft.study_type,
        objective=draft.objective,
        required_data=draft.required_data,
        analysis_steps=draft.analysis_steps,
        success_criteria=draft.success_criteria,
        safety_constraints=draft.safety_constraints,
    )


def _completed_stage(
    name: str,
    started_at: str,
    completed_at: str,
    summary: str,
) -> ResearchStage:
    return ResearchStage(
        name=name,
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        summary=summary,
    )


def _skipped_stage(
    name: str,
    started_at: str,
    completed_at: str,
    summary: str,
) -> ResearchStage:
    return ResearchStage(
        name=name,
        status="skipped",
        started_at=started_at,
        completed_at=completed_at,
        summary=summary,
    )


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+)\]")


def _validate_report_citations(
    report: str,
    sources: list[ResearchSource],
) -> None:
    allowed = {source.source_id for source in sources}
    citations = set(_CITATION_RE.findall(report))
    unknown = citations - allowed
    if unknown:
        raise AutoResearchPipelineError(
            f"report cited unknown sources: {', '.join(sorted(unknown))}"
        )
    if not citations:
        raise AutoResearchPipelineError("report did not cite any verified source ids")
    uncited_lines: list[int] = []
    for line_number, line in enumerate(report.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in {"---", "***"}:
            continue
        line_citations = set(_CITATION_RE.findall(stripped))
        if not line_citations:
            uncited_lines.append(line_number)
    if uncited_lines:
        preview = ", ".join(str(item) for item in uncited_lines[:8])
        raise AutoResearchPipelineError(
            f"report contained uncited content lines: {preview}"
        )


def _wrap_report(body: str, sources: list[ResearchSource]) -> str:
    ledger = "\n".join(
        f"- [{source.source_id}] {source.title} — PMID {source.pmid or 'n/a'} — {source.url}"
        for source in sources
    )
    return (
        "# 自动科研影子报告\n\n"
        "> 本报告由自动流程生成，仅供科研人员复核；不是临床事实、患者建议或已验证发现。\n\n"
        f"{body.strip()}\n\n"
        "## 可核验来源\n\n"
        f"{ledger}\n"
    )


_SENSITIVE_QUOTED_ERROR_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|email|token|secret)\b[\"']?\s*[:=]\s*([\"']).*?\2"
)
_SENSITIVE_UNQUOTED_ERROR_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|email|token|secret)\b[\"']?\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;&}\"']+"
)
_BEARER_CREDENTIAL_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    message = _SENSITIVE_QUOTED_ERROR_RE.sub(r"\1=[REDACTED]", message)
    message = _SENSITIVE_UNQUOTED_ERROR_RE.sub(r"\1=[REDACTED]", message)
    message = _BEARER_CREDENTIAL_RE.sub("Bearer [REDACTED]", message)
    return message[:1000]


__all__ = [
    "AutoResearchConflictError",
    "AutoResearchPipelineError",
    "AutoResearchServiceUnavailableError",
    "AutoResearchService",
    "DeferredResearchReasoner",
    "EvidenceRetriever",
    "HypothesisDraft",
    "LLMResearchReasoner",
    "ResearchReasoner",
    "ReviewedDraft",
    "StudyPlanDraft",
]
