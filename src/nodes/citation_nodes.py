"""
Citation Agent Node
"""

import json
import os
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..prompts import CITATION_CHECKER_SYSTEM_PROMPT
from ..state import CRCAgentState
from .node_utils import _invoke_structured_with_recovery


class CitationReport(BaseModel):
    coverage_score: int = Field(description="0-100 citation coverage score")
    missing_claims: list[str] = Field(default_factory=list, description="Claims that still lack direct support")
    needs_more_sources: bool = Field(description="Whether additional supporting sources are required")
    notes: str = Field(default="", description="Short validation notes")


def _fast_review_enabled() -> bool:
    return os.getenv("FAST_REVIEW_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def _parse_citation_report(raw_text: str) -> Optional[dict]:
    text = (raw_text or "").replace("```json", "").replace("```", "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    coverage_match = re.search(r'"?coverage_score"?\s*[:=]\s*(\d{1,3})', text, flags=re.IGNORECASE)
    coverage_score = int(coverage_match.group(1)) if coverage_match else 0

    needs_more_match = re.search(
        r'"?needs_more_sources"?\s*[:=]\s*(true|false|yes|no)',
        text,
        flags=re.IGNORECASE,
    )
    needs_more_sources = False
    if needs_more_match:
        needs_more_sources = needs_more_match.group(1).lower() in {"true", "yes"}
    elif any(marker in text.lower() for marker in ["needs more source", "needs more evidence"]):
        needs_more_sources = True

    missing_claims: list[str] = []
    block_match = re.search(
        r'"?missing_claims"?\s*[:=]\s*\[(?P<body>.*?)\]',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block_match:
        body = block_match.group("body")
        missing_claims = [
            item.strip().strip('"').strip("'")
            for item in re.split(r",|\n", body)
            if item.strip().strip('"').strip("'")
        ]

    notes_match = re.search(
        r'"?notes"?\s*[:=]\s*"?(?P<notes>.+?)"?$',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    notes = notes_match.group("notes").strip() if notes_match else text[:500]

    return {
        "coverage_score": max(0, min(100, coverage_score)),
        "missing_claims": missing_claims,
        "needs_more_sources": needs_more_sources,
        "notes": notes,
    }


def _count_inline_reference_anchors(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return len(re.findall(r"\[\[Source:[^|\]]+\|Page:[^\]]+\]\]", text or ""))


def _fast_citation_report(state: CRCAgentState) -> CitationReport:
    decision = state.decision_json or {}
    findings = state.findings or {}
    refs = [
        (item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        for item in (state.retrieved_references or [])
    ]
    plan_items = decision.get("treatment_plan") or []
    summary = str(decision.get("summary") or "").strip()
    follow_up = decision.get("follow_up") or []
    inline_anchor_count = _count_inline_reference_anchors(decision)
    decision_strategy = str(findings.get("decision_strategy") or "")
    guideline_rag = decision_strategy == "rag_guideline"

    missing_claims: list[str] = []
    if not summary:
        missing_claims.append("summary")
    if not plan_items:
        missing_claims.append("treatment_plan")
    template_fast = findings.get("decision_strategy") == "template_fast"
    if len(refs) < 2 and not template_fast:
        missing_claims.append("insufficient_references")
    if guideline_rag and inline_anchor_count < 2:
        missing_claims.append("inline_citations")

    coverage = 35
    coverage += min(len(refs), 5) * 10
    coverage += min(inline_anchor_count, 4) * 8
    if summary:
        coverage += 10
    if plan_items:
        coverage += 15
    if len(plan_items) >= 2:
        coverage += 10
    if follow_up:
        coverage += 5
    if template_fast and not refs:
        coverage = min(coverage, 65)
    if guideline_rag and len(refs) >= 2 and inline_anchor_count >= 2:
        coverage = max(coverage, 85)
    coverage = max(0, min(95, coverage))

    needs_more_sources = (len(refs) < 2 and not template_fast) or not plan_items
    if guideline_rag and len(refs) >= 2 and plan_items:
        needs_more_sources = False
    notes = (
        f"heuristic-fast-path refs={len(refs)} plan_items={len(plan_items)} "
        f"template_fast={template_fast} guideline_rag={guideline_rag} "
        f"inline_anchors={inline_anchor_count}"
    )
    if template_fast and not refs:
        notes += " no_direct_references"
    return CitationReport(
        coverage_score=coverage,
        missing_claims=missing_claims,
        needs_more_sources=needs_more_sources,
        notes=notes,
    )


def node_citation_agent(model, streaming: bool = False, show_thinking: bool = True):
    prompt = ChatPromptTemplate.from_messages([
        ("system", CITATION_CHECKER_SYSTEM_PROMPT),
        ("human", "结论(JSON): {decision_json}\n引用来源: {references}\n并行报告: {subagent_reports}"),
    ])

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        if not state.decision_json:
            return {}

        if _fast_review_enabled():
            report = _fast_citation_report(state)
            return {"citation_report": report.model_dump()}

        refs_payload = [
            (r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
            for r in (state.retrieved_references or [])
        ]

        ctx = {
            "decision_json": json.dumps(state.decision_json, ensure_ascii=False),
            "references": json.dumps(refs_payload, ensure_ascii=False),
            "subagent_reports": json.dumps(state.subagent_reports or [], ensure_ascii=False),
        }

        try:
            report: CitationReport = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=CitationReport,
                payload=ctx,
                log_prefix="[CitationAgent]",
                raw_text_parser=_parse_citation_report,
                fallback_factory=lambda _payload, _err: CitationReport(
                    coverage_score=0,
                    missing_claims=[],
                    needs_more_sources=False,
                    notes=f"citation validation degraded: {_err}",
                ),
            )
            return {"citation_report": report.model_dump()}
        except Exception as e:
            if show_thinking:
                print(f"[CitationAgent] citation validation failed: {e}")
            return {
                "citation_report": {
                    "coverage_score": 0,
                    "missing_claims": [],
                    "needs_more_sources": False,
                    "notes": f"citation validation failed: {e}",
                }
            }

    return _run
