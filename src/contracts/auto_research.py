from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Literal

from src.contracts.deidentification import validate_deidentified_text


AutoResearchRunStatus = Literal[
    "completed_shadow",
    "partial_shadow",
    "failed_shadow",
]
ResearchStageStatus = Literal["completed", "failed", "skipped"]
HypothesisVerdict = Literal["advance", "revise", "reject"]
HumanReviewStatus = Literal["needs_human_review"]

RUN_STATUSES: tuple[AutoResearchRunStatus, ...] = (
    "completed_shadow",
    "partial_shadow",
    "failed_shadow",
)
STAGE_STATUSES: tuple[ResearchStageStatus, ...] = (
    "completed",
    "failed",
    "skipped",
)
HYPOTHESIS_VERDICTS: tuple[HypothesisVerdict, ...] = (
    "advance",
    "revise",
    "reject",
)

_REPORT_TITLE = "# 自动科研影子报告"
_REPORT_DISCLAIMER = (
    "> 本报告由自动流程生成，仅供科研人员复核；不是临床事实、患者建议或已验证发现。"
)
_REPORT_LEDGER_HEADING = "## 可核验来源"
_REPORT_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+)\]")
_REPORT_LEDGER_ENTRY_RE = re.compile(r"^- \[([A-Za-z0-9_.-]+)\]\s+.+$")
_AUTO_RESEARCH_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")

@dataclass(frozen=True)
class AutoResearchRequest:
    request_id: str
    project_id: str
    question: str
    requested_by: str
    idempotency_key: str
    max_sources: int = 8
    max_hypotheses: int = 3
    max_iterations: int = 2
    deidentified: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "project_id",
            "question",
            "requested_by",
            "idempotency_key",
        ):
            _require_non_empty(field_name, getattr(self, field_name))
        for field_name in ("request_id", "project_id", "idempotency_key"):
            value = getattr(self, field_name)
            if not _AUTO_RESEARCH_IDENTIFIER_RE.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be a 1-128 character opaque identifier "
                    "using letters, numbers, dot, underscore, colon, or hyphen"
                )
            validate_deidentified_text(field_name, value)
        if len(self.requested_by) > 128:
            raise ValueError("requested_by must not exceed 128 characters")
        if len(self.question.strip()) < 3:
            raise ValueError("question must contain at least 3 characters")
        if len(self.question) > 4000:
            raise ValueError("question must not exceed 4000 characters")
        validate_research_question_privacy_boundary(self.question)
        _require_bounded_int("max_sources", self.max_sources, minimum=1, maximum=20)
        _require_bounded_int(
            "max_hypotheses", self.max_hypotheses, minimum=1, maximum=5
        )
        _require_bounded_int(
            "max_iterations", self.max_iterations, minimum=1, maximum=3
        )
        if self.deidentified is not True:
            raise ValueError("deidentified must be true")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "project_id": self.project_id,
            "question": self.question,
            "requested_by": self.requested_by,
            "idempotency_key": self.idempotency_key,
            "max_sources": self.max_sources,
            "max_hypotheses": self.max_hypotheses,
            "max_iterations": self.max_iterations,
            "deidentified": self.deidentified,
        }


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    title: str
    url: str
    abstract: str
    journal: str
    publication_year: str
    source_type: str
    query: str
    retrieved_at: str
    pmid: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "title",
            "url",
            "abstract",
            "source_type",
            "query",
            "retrieved_at",
        ):
            _require_non_empty(field_name, getattr(self, field_name))
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("url must be an absolute HTTP(S) URL")
        if self.pmid is not None:
            _require_non_empty("pmid", self.pmid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "abstract": self.abstract,
            "journal": self.journal,
            "publication_year": self.publication_year,
            "source_type": self.source_type,
            "query": self.query,
            "retrieved_at": self.retrieved_at,
            "pmid": self.pmid,
        }


@dataclass(frozen=True)
class HypothesisReview:
    verdict: HypothesisVerdict
    evidence_support_score: float
    novelty_score: float
    testability_score: float
    safety_risk: str
    critique: str
    revision_instructions: str

    def __post_init__(self) -> None:
        _validate_choice("verdict", self.verdict, HYPOTHESIS_VERDICTS)
        for field_name in (
            "evidence_support_score",
            "novelty_score",
            "testability_score",
        ):
            _require_unit_float(field_name, getattr(self, field_name))
        for field_name in ("safety_risk", "critique"):
            _require_non_empty(field_name, getattr(self, field_name))
        if not isinstance(self.revision_instructions, str):
            raise TypeError("revision_instructions must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "evidence_support_score": self.evidence_support_score,
            "novelty_score": self.novelty_score,
            "testability_score": self.testability_score,
            "safety_risk": self.safety_risk,
            "critique": self.critique,
            "revision_instructions": self.revision_instructions,
        }


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    statement: str
    rationale: str
    testable_prediction: str
    supporting_source_ids: list[str]
    counterevidence_source_ids: list[str]
    iteration: int
    review: HypothesisReview

    def __post_init__(self) -> None:
        for field_name in (
            "hypothesis_id",
            "statement",
            "rationale",
            "testable_prediction",
        ):
            _require_non_empty(field_name, getattr(self, field_name))
        object.__setattr__(
            self,
            "supporting_source_ids",
            _require_string_list("supporting_source_ids", self.supporting_source_ids),
        )
        object.__setattr__(
            self,
            "counterevidence_source_ids",
            _require_string_list(
                "counterevidence_source_ids",
                self.counterevidence_source_ids,
                allow_empty=True,
            ),
        )
        _require_bounded_int("iteration", self.iteration, minimum=1, maximum=3)
        if not isinstance(self.review, HypothesisReview):
            raise TypeError("review must be a HypothesisReview")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "testable_prediction": self.testable_prediction,
            "supporting_source_ids": list(self.supporting_source_ids),
            "counterevidence_source_ids": list(self.counterevidence_source_ids),
            "iteration": self.iteration,
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchHypothesis":
        data = dict(payload)
        data["review"] = HypothesisReview(**data["review"])
        return cls(**data)


@dataclass(frozen=True)
class ResearchStudyPlan:
    plan_id: str
    hypothesis_id: str
    study_type: str
    objective: str
    required_data: list[str]
    analysis_steps: list[str]
    success_criteria: list[str]
    safety_constraints: list[str]
    execution_status: Literal["not_executed"] = "not_executed"

    def __post_init__(self) -> None:
        for field_name in ("plan_id", "hypothesis_id", "study_type", "objective"):
            _require_non_empty(field_name, getattr(self, field_name))
        for field_name in (
            "required_data",
            "analysis_steps",
            "success_criteria",
            "safety_constraints",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_string_list(field_name, getattr(self, field_name)),
            )
        if self.execution_status != "not_executed":
            raise ValueError("execution_status must be not_executed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "hypothesis_id": self.hypothesis_id,
            "study_type": self.study_type,
            "objective": self.objective,
            "required_data": list(self.required_data),
            "analysis_steps": list(self.analysis_steps),
            "success_criteria": list(self.success_criteria),
            "safety_constraints": list(self.safety_constraints),
            "execution_status": self.execution_status,
        }


@dataclass(frozen=True)
class ResearchStage:
    name: str
    status: ResearchStageStatus
    started_at: str
    completed_at: str
    summary: str
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "started_at", "completed_at", "summary"):
            _require_non_empty(field_name, getattr(self, field_name))
        _validate_choice("status", self.status, STAGE_STATUSES)
        if self.error is not None:
            _require_non_empty("error", self.error)
        if self.status == "failed" and self.error is None:
            raise ValueError("failed stage requires error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("non-failed stage must not contain an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass(frozen=True)
class AutoResearchRun:
    run_id: str
    request_hash: str
    request: AutoResearchRequest
    status: AutoResearchRunStatus
    created_at: str
    completed_at: str
    stages: list[ResearchStage]
    sources: list[ResearchSource]
    hypotheses: list[ResearchHypothesis]
    study_plans: list[ResearchStudyPlan]
    report_markdown: str
    iteration_count: int
    provenance: dict[str, str]
    human_review_status: HumanReviewStatus = "needs_human_review"
    mode: Literal["shadow_only"] = "shadow_only"
    applies_automatically: Literal[False] = False
    clinical_default_path_mutated: Literal[False] = False
    patient_level_rows_returned: Literal[False] = False

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        if not isinstance(self.request_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.request_hash
        ):
            raise ValueError("request_hash must be a sha256-prefixed hash")
        if not isinstance(self.request, AutoResearchRequest):
            raise TypeError("request must be an AutoResearchRequest")
        _validate_choice("status", self.status, RUN_STATUSES)
        _require_non_empty("created_at", self.created_at)
        _require_non_empty("completed_at", self.completed_at)
        _require_bounded_int("iteration_count", self.iteration_count, minimum=0, maximum=3)
        _require_instance_list("stages", self.stages, ResearchStage)
        if not self.stages:
            raise ValueError("stages must not be empty")
        _require_instance_list("sources", self.sources, ResearchSource)
        _require_instance_list("hypotheses", self.hypotheses, ResearchHypothesis)
        _require_instance_list("study_plans", self.study_plans, ResearchStudyPlan)
        if not isinstance(self.report_markdown, str):
            raise TypeError("report_markdown must be a string")
        if not isinstance(self.provenance, dict) or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and value
            for key, value in self.provenance.items()
        ):
            raise ValueError("provenance must map non-empty strings to non-empty strings")
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.human_review_status != "needs_human_review":
            raise ValueError("human_review_status must be needs_human_review")
        if self.mode != "shadow_only":
            raise ValueError("mode must be shadow_only")
        if self.applies_automatically is not False:
            raise ValueError("applies_automatically must be false")
        if self.clinical_default_path_mutated is not False:
            raise ValueError("clinical_default_path_mutated must be false")
        if self.patient_level_rows_returned is not False:
            raise ValueError("patient_level_rows_returned must be false")
        expected_run_id = make_auto_research_run_id(
            self.request.project_id,
            self.request.idempotency_key,
        )
        if self.run_id != expected_run_id:
            raise ValueError("run_id must match the request project and idempotency key")
        expected_request_hash = make_auto_research_request_hash(self.request)
        if self.request_hash != expected_request_hash:
            raise ValueError("request_hash must match the persisted request")
        stage_names = [stage.name for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("stage names must be unique")
        source_ids = {source.source_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("source ids must be unique")
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        if len(hypothesis_ids) != len(self.hypotheses):
            raise ValueError("hypothesis ids must be unique")
        for hypothesis in self.hypotheses:
            referenced = set(hypothesis.supporting_source_ids) | set(
                hypothesis.counterevidence_source_ids
            )
            if not referenced <= source_ids:
                raise ValueError("hypothesis references unknown source ids")
        plan_ids = [plan.plan_id for plan in self.study_plans]
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("study plan ids must be unique")
        planned_hypothesis_ids = [plan.hypothesis_id for plan in self.study_plans]
        if len(set(planned_hypothesis_ids)) != len(planned_hypothesis_ids):
            raise ValueError("each hypothesis may have at most one study plan")
        for plan in self.study_plans:
            if plan.hypothesis_id not in hypothesis_ids:
                raise ValueError("study plan references unknown hypothesis id")
            if plan.plan_id != make_research_plan_id(self.run_id, plan.hypothesis_id):
                raise ValueError("study plan id must match its run and hypothesis")
            hypothesis = next(
                item for item in self.hypotheses if item.hypothesis_id == plan.hypothesis_id
            )
            if hypothesis.review.verdict != "advance":
                raise ValueError("study plan may only reference an advanced hypothesis")
        if self.hypotheses and max(item.iteration for item in self.hypotheses) > self.iteration_count:
            raise ValueError("iteration_count must cover all persisted hypotheses")
        _validate_run_stage_topology(self)
        advanced_ids = {
            item.hypothesis_id
            for item in self.hypotheses
            if item.review.verdict == "advance"
        }
        study_design_stage = next(
            (stage for stage in self.stages if stage.name == "study_design"),
            None,
        )
        planned_ids = set(planned_hypothesis_ids)
        if study_design_stage is None:
            if advanced_ids:
                raise ValueError(
                    "advanced hypotheses require a persisted study_design stage"
                )
        elif study_design_stage.status == "skipped":
            if advanced_ids or self.study_plans:
                raise ValueError(
                    "skipped study_design requires no advanced hypotheses or plans"
                )
        elif study_design_stage.status == "completed":
            if planned_ids != advanced_ids or len(self.study_plans) != len(advanced_ids):
                raise ValueError(
                    "completed study_design requires one plan per advanced hypothesis"
                )
        elif self.study_plans:
            raise ValueError("failed study_design must not persist study plans")
        if self.report_markdown.strip():
            _validate_persisted_report(self.report_markdown, source_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_hash": self.request_hash,
            "request": self.request.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "stages": [stage.to_dict() for stage in self.stages],
            "sources": [source.to_dict() for source in self.sources],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "study_plans": [plan.to_dict() for plan in self.study_plans],
            "report_markdown": self.report_markdown,
            "iteration_count": self.iteration_count,
            "provenance": dict(self.provenance),
            "human_review_status": self.human_review_status,
            "mode": self.mode,
            "applies_automatically": self.applies_automatically,
            "clinical_default_path_mutated": self.clinical_default_path_mutated,
            "patient_level_rows_returned": self.patient_level_rows_returned,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AutoResearchRun":
        data = dict(payload)
        data["request"] = AutoResearchRequest(**data["request"])
        data["stages"] = [ResearchStage(**item) for item in data["stages"]]
        data["sources"] = [ResearchSource(**item) for item in data["sources"]]
        data["hypotheses"] = [
            ResearchHypothesis.from_dict(item) for item in data["hypotheses"]
        ]
        data["study_plans"] = [
            ResearchStudyPlan(**item) for item in data["study_plans"]
        ]
        return cls(**data)


def _validate_run_stage_topology(run: AutoResearchRun) -> None:
    expected_names = ["literature_search"]
    for iteration in range(1, run.iteration_count + 1):
        expected_names.extend(
            [f"hypothesis_generation_{iteration}", f"hypothesis_review_{iteration}"]
        )
    expected_names.extend(["study_design", "report_synthesis"])
    actual_names = [stage.name for stage in run.stages]
    failed_stages = [stage for stage in run.stages if stage.status == "failed"]

    if run.status == "completed_shadow":
        if run.iteration_count < 1:
            raise ValueError("completed_shadow requires at least one iteration")
        if actual_names != expected_names:
            raise ValueError("completed_shadow requires the complete stage topology")
        if failed_stages:
            raise ValueError("completed_shadow must not contain failed stages")
        if not run.sources or not run.hypotheses or not run.report_markdown.strip():
            raise ValueError(
                "completed_shadow requires sources, hypotheses, and a report"
            )
    elif run.status == "partial_shadow":
        if run.iteration_count < 1:
            raise ValueError("partial_shadow requires at least one iteration")
        if actual_names != expected_names[: len(actual_names)]:
            raise ValueError("partial_shadow stages must be a pipeline prefix")
        if len(failed_stages) != 1 or run.stages[-1].status != "failed":
            raise ValueError(
                "partial_shadow requires exactly one terminal failed stage"
            )
        if run.stages[-1].name == "literature_search" or not run.sources:
            raise ValueError("partial_shadow requires completed literature retrieval")
        if run.report_markdown.strip():
            raise ValueError("partial_shadow must not contain a completed report")
        failed_name = run.stages[-1].name
        if (
            failed_name.startswith("hypothesis_generation_")
            or failed_name.startswith("hypothesis_review_")
            or failed_name == "study_design"
        ) and run.study_plans:
            raise ValueError(
                "study plans may only survive a report_synthesis failure"
            )
    else:
        if actual_names != ["literature_search"]:
            raise ValueError(
                "failed_shadow must contain only the failed literature_search stage"
            )
        if len(failed_stages) != 1 or run.stages[0].status != "failed":
            raise ValueError("failed_shadow requires a failed literature_search stage")
        if run.iteration_count != 0:
            raise ValueError("failed_shadow requires zero iterations")
        if (
            run.sources
            or run.hypotheses
            or run.study_plans
            or run.report_markdown.strip()
        ):
            raise ValueError("failed_shadow must not contain research artifacts")

    for index, stage in enumerate(run.stages):
        is_terminal_failure = (
            index == len(run.stages) - 1 and stage.status == "failed"
        )
        if is_terminal_failure:
            continue
        if stage.name == "study_design":
            if stage.status not in {"completed", "skipped"}:
                raise ValueError("study_design must complete or be skipped")
        elif stage.status != "completed":
            raise ValueError("non-terminal pipeline stages must be completed")


def _validate_persisted_report(report: str, source_ids: set[str]) -> None:
    parts = report.split(_REPORT_LEDGER_HEADING)
    if len(parts) != 2:
        raise ValueError("report must contain exactly one source ledger")
    prefix, ledger = parts
    prefix_lines = prefix.splitlines()
    non_empty_prefix = [line.strip() for line in prefix_lines if line.strip()]
    if len(non_empty_prefix) < 3:
        raise ValueError("report must contain its title, disclaimer, and cited body")
    if non_empty_prefix[0] != _REPORT_TITLE:
        raise ValueError("report must retain the shadow report title")
    if non_empty_prefix[1] != _REPORT_DISCLAIMER:
        raise ValueError("report must retain the shadow-only disclaimer")

    body_lines = non_empty_prefix[2:]
    body_citations: set[str] = set()
    uncited_lines: list[int] = []
    for line_number, line in enumerate(body_lines, start=1):
        if line.startswith("#") or line in {"---", "***"}:
            continue
        citations = set(_REPORT_CITATION_RE.findall(line))
        body_citations.update(citations)
        if not citations:
            uncited_lines.append(line_number)
    unknown_body_citations = body_citations - source_ids
    if unknown_body_citations:
        raise ValueError(
            "report cites unknown source ids: "
            + ", ".join(sorted(unknown_body_citations))
        )
    if not body_citations:
        raise ValueError("report body must cite at least one persisted source")
    if uncited_lines:
        preview = ", ".join(str(item) for item in uncited_lines[:8])
        raise ValueError(f"report body contains uncited content lines: {preview}")

    ledger_ids: list[str] = []
    for line in ledger.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _REPORT_LEDGER_ENTRY_RE.fullmatch(stripped)
        if match is None:
            raise ValueError("report source ledger contains an invalid entry")
        ledger_ids.append(match.group(1))
    if len(set(ledger_ids)) != len(ledger_ids):
        raise ValueError("report source ledger must not contain duplicate source ids")
    if set(ledger_ids) != source_ids:
        raise ValueError("report source ledger must match persisted sources")


def make_auto_research_run_id(project_id: str, idempotency_key: str) -> str:
    return _stable_id("auto_research_run", f"{project_id}:{idempotency_key}")


def make_auto_research_request_hash(request: AutoResearchRequest) -> str:
    payload = request.to_dict()
    payload.pop("idempotency_key", None)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def make_research_source_id(source_type: str, external_id: str) -> str:
    return _stable_id("research_source", f"{source_type}:{external_id}")


def make_research_hypothesis_id(run_id: str, iteration: int, statement: str) -> str:
    return _stable_id("research_hypothesis", f"{run_id}:{iteration}:{statement}")


def make_research_plan_id(run_id: str, hypothesis_id: str) -> str:
    return _stable_id("research_plan", f"{run_id}:{hypothesis_id}")


def _stable_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"prefix": prefix, "seed": seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}_{digest}"


def validate_research_question_privacy_boundary(question: str) -> None:
    """Reject obvious patient identifiers before persistence or external egress.

    This is a conservative guardrail, not a replacement for an institutional DLP
    service or human review of research inputs.
    """

    validate_deidentified_text("question", question)


def _require_non_empty(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_bounded_int(
    field_name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")


def _require_unit_float(field_name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_choice(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {', '.join(allowed)}")


def _require_string_list(
    field_name: str,
    value: Any,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return list(value)


def _require_instance_list(field_name: str, value: Any, expected: type[Any]) -> None:
    if not isinstance(value, list) or not all(isinstance(item, expected) for item in value):
        raise TypeError(f"{field_name} must contain {expected.__name__} values")


__all__ = [
    "AutoResearchRequest",
    "AutoResearchRun",
    "AutoResearchRunStatus",
    "HypothesisReview",
    "HypothesisVerdict",
    "ResearchHypothesis",
    "ResearchSource",
    "ResearchStage",
    "ResearchStageStatus",
    "ResearchStudyPlan",
    "make_auto_research_run_id",
    "make_auto_research_request_hash",
    "make_research_hypothesis_id",
    "make_research_plan_id",
    "make_research_source_id",
    "validate_research_question_privacy_boundary",
]
