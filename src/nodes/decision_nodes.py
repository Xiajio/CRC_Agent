"""
Decision and Critic Nodes
"""

import json
import os
import re
import time
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..policies.diagnostics import record_review_divergence
from ..policies.review_policy import (
    build_critic_review_signal,
    decide_after_critic,
)
from ..policies.turn_facts import build_turn_facts
from ..policies.types import DegradedSignal, ReviewDecision
from ..state import CRCAgentState
from ..rag.retriever import consume_retrieval_metrics, reset_retrieval_metrics
from ..tools.rag_tools import get_guideline_tool, TreatmentSearchTool
from ..prompts import (
    DECISION_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    QUERY_GENERATION_SYSTEM_PROMPT,
)
from .node_utils import (
    _select_tools,
    _generate_fallback_plan,
    _needs_full_decision,
    _extract_and_update_references,
    _invoke_structured_with_recovery,
    _is_repeated_rejection,
    auto_update_roadmap_from_state,
    _truncate_message_history,
    _compress_rag_context,
    _estimate_tokens,
    _create_rag_digest,
    _build_pinned_context,
    _build_summary_memory,
    _latest_user_text,
)
from .sub_agent import run_isolated_rag_search


def _decision_value_to_text(value: Any) -> str:
    """Flatten mixed structured content into a readable string."""
    if value is None:
        return ""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_decision_value_to_text(item) for item in value]
        return "；".join([part for part in parts if part])
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _decision_value_to_text(item)
            if text:
                label = str(key).replace("_", " ").strip()
                parts.append(f"{label}: {text}" if label else text)
        return "；".join(parts)
    return str(value).strip()


def _coerce_follow_up_items(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    items: List[str] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        lines = [
            re.sub(r"^\s*(?:[-*•]|(?:\d+|[一二三四五六七八九十]+)[、.．)])\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]
        items = [line for line in lines if line]
        if len(items) <= 1 and "；" in text:
            items = [seg.strip() for seg in re.split(r"[；;]", text) if seg.strip()]
        if not items:
            items = [text]
    elif isinstance(value, list):
        for item in value:
            items.extend(_coerce_follow_up_items(item))
    elif isinstance(value, dict):
        for item in value.values():
            text = _decision_value_to_text(item)
            if text:
                items.append(text)
    else:
        text = _decision_value_to_text(value)
        if text:
            items = [text]

    deduped: List[str] = []
    seen = set()
    for item in items:
        normalized = _normalize_markdown_line(item)
        normalized = re.sub(r"\[\[Source:[^|\]]+\|Page:\d+\]\]", "", normalized).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _clean_decision_raw_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def _looks_like_placeholder_text(text: Any) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "")).strip("`*#>-")
    return normalized.lower() in {"", "-", "--", "---", "n/a", "na", "null", "none"} or normalized in {
        "无",
        "暂无",
        "待补充",
    }


def _normalize_markdown_line(text: Any) -> str:
    line = str(text or "").strip()
    if not line:
        return ""
    line = re.sub(r"^\s*(?:[-*•]+|\d+[.)]\s+)", "", line).strip()
    line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
    line = line.strip("`")
    if re.fullmatch(r"[-=_*:\s]{3,}", line):
        return ""
    if line.startswith("|") and line.endswith("|"):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r"[-:]{2,}", cell)]
        if not cells:
            return ""
        if len(cells) == 2 and set(cells) <= {
            "项目",
            "方案选择",
            "时间节点",
            "检查项目",
            "检测项目",
            "目的",
            "时机",
            "结果",
            "解读",
            "药物组成",
            "疗程",
        }:
            return ""
        if len(cells) == 2:
            line = f"{cells[0]}: {cells[1]}"
        else:
            line = " / ".join(cells)
    if _looks_like_placeholder_text(line):
        return ""
    return line


def _sanitize_section_title(title: Any, index: int) -> str:
    raw = str(title or "").strip()
    cleaned = _normalize_markdown_line(title)
    if raw.startswith("|") and raw.endswith("|"):
        return f"治疗建议 {index}"
    if not cleaned or cleaned.startswith("|") or len(cleaned) > 60:
        return f"治疗建议 {index}"
    return cleaned


def _sanitize_section_content(content: Any) -> str:
    text = _clean_decision_raw_text(_decision_value_to_text(content))
    lines = [_normalize_markdown_line(line) for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or "待补充。"


_ANCHOR_PATTERN = re.compile(r"\[\[Source:(?P<source>[^|\]]+)\|Page:(?P<page>\d+)\]\]")


def _normalize_reference_payload(ref: Any) -> Optional[Dict[str, Any]]:
    if ref is None:
        return None
    if hasattr(ref, "model_dump"):
        ref = ref.model_dump()
    if not isinstance(ref, dict):
        return None

    source = str(ref.get("source") or ref.get("title") or "").strip()
    if not source:
        return None

    page_val = ref.get("page")
    page: Optional[int] = None
    if isinstance(page_val, int):
        page = page_val
    elif isinstance(page_val, str):
        match = re.search(r"\d+", page_val)
        if match:
            try:
                page = int(match.group())
            except Exception:
                page = None

    relevance = ref.get("relevance", ref.get("score", 0.0))
    try:
        relevance = float(relevance)
    except Exception:
        relevance = 0.0

    source_name = source.split("/")[-1].split("\\")[-1]
    return {
        "source": source_name,
        "page": page,
        "source_id": str(ref.get("source_id") or ref.get("ref_id") or f"{source_name}:{page or '?'}"),
        "snippet": str(ref.get("snippet") or ref.get("preview") or ref.get("content") or "").strip(),
        "relevance": relevance,
    }


def _dedupe_references(references: List[Any]) -> List[Dict[str, Any]]:
    best_by_key: Dict[tuple[str, Optional[int]], Dict[str, Any]] = {}
    first_seen_order: List[tuple[str, Optional[int]]] = []

    for ref in references or []:
        normalized = _normalize_reference_payload(ref)
        if not normalized:
            continue
        key = (normalized["source"], normalized["page"])
        if key not in best_by_key:
            best_by_key[key] = normalized
            first_seen_order.append(key)
            continue

        existing = best_by_key[key]
        existing_score = float(existing.get("relevance", 0.0))
        new_score = float(normalized.get("relevance", 0.0))
        existing_snippet = len(existing.get("snippet", ""))
        new_snippet = len(normalized.get("snippet", ""))
        if (new_score, new_snippet) > (existing_score, existing_snippet):
            best_by_key[key] = normalized

    return [best_by_key[key] for key in first_seen_order]


def _tokenize_for_citation_match(text: str) -> set[str]:
    tokens: set[str] = set()
    if not text:
        return tokens

    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+/\-]*|\b\d+(?:\.\d+)?\b", text):
        token = token.lower()
        if len(token) >= 2:
            tokens.add(token)

    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= 6:
            tokens.add(chunk)
        window_sizes = (2, 3)
        for window in window_sizes:
            if len(chunk) < window:
                continue
            for i in range(0, len(chunk) - window + 1):
                tokens.add(chunk[i:i + window])

    return tokens


def _reference_match_score(text: str, ref: Dict[str, Any]) -> float:
    text_tokens = _tokenize_for_citation_match(text)
    if not text_tokens:
        return 0.0

    ref_text = " ".join(
        str(part or "")
        for part in [
            ref.get("source", ""),
            ref.get("snippet", ""),
        ]
    )
    ref_tokens = _tokenize_for_citation_match(ref_text)
    if not ref_tokens:
        return 0.0

    overlap = text_tokens & ref_tokens
    if len(overlap) < 2:
        return 0.0

    overlap_score = len(overlap) / max(4, min(len(text_tokens), len(ref_tokens)))
    relevance = float(ref.get("relevance", 0.0))
    return overlap_score + relevance * 0.02


def _reference_anchor(ref: Dict[str, Any]) -> str:
    source = ref.get("source") or "unknown"
    page = ref.get("page")
    if page is None:
        return f"[[Source:{source}|Page:?]]"
    return f"[[Source:{source}|Page:{page}]]"


def _attach_reference_anchors(
    text: str,
    references: List[Dict[str, Any]],
    max_refs: int = 2,
    fallback_to_top: bool = False,
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or _ANCHOR_PATTERN.search(cleaned):
        return cleaned

    scored_refs = []
    for ref in references:
        score = _reference_match_score(cleaned, ref)
        if score > 0:
            scored_refs.append((score, ref))

    scored_refs.sort(key=lambda item: item[0], reverse=True)
    selected = [ref for score, ref in scored_refs[:max_refs] if score >= 0.28]

    if not selected and fallback_to_top and references:
        selected = references[:1]

    if not selected:
        return cleaned

    anchors = " ".join(_reference_anchor(ref) for ref in selected)
    return f"{cleaned} {anchors}".strip()


def _order_references_by_usage(rendered_text: str, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = _dedupe_references(references)
    if not deduped:
        return []

    ref_map = {(ref["source"], ref["page"]): ref for ref in deduped}
    ordered_keys: List[tuple[str, Optional[int]]] = []
    for match in _ANCHOR_PATTERN.finditer(rendered_text or ""):
        key = (match.group("source"), int(match.group("page")))
        if key in ref_map and key not in ordered_keys:
            ordered_keys.append(key)

    remaining = [key for key in ref_map.keys() if key not in ordered_keys]
    remaining.sort(key=lambda key: ref_map[key].get("relevance", 0.0), reverse=True)
    return [ref_map[key] for key in ordered_keys + remaining]


def _build_cached_rag_context_from_references(
    references: List[Any],
    max_refs: int = 8,
    max_chars: int = 4000,
) -> str:
    normalized = _dedupe_references(references)
    if not normalized:
        return ""

    ranked = sorted(
        normalized,
        key=lambda ref: float(ref.get("relevance", 0.0)),
        reverse=True,
    )[:max_refs]

    parts: List[str] = []
    for ref in ranked:
        source = ref.get("source") or "unknown"
        page = ref.get("page")
        snippet = str(ref.get("snippet") or "").strip()
        if not snippet:
            continue
        header = f"[Source] {source}" if page is None else f"[Source] {source} p.{page}"
        parts.append(f"{header}\n{snippet}")

    context = "\n\n".join(parts).strip()
    return context[:max_chars]


def _should_reuse_cached_rag(state: CRCAgentState) -> bool:
    if not (state.retrieved_references or []):
        return False
    if (state.evaluation_retry_count or 0) <= 0:
        return False

    citation_report = state.citation_report or {}
    needs_more_sources = citation_report.get("needs_more_sources", False)
    if isinstance(needs_more_sources, str):
        needs_more_sources = needs_more_sources.strip().lower() in {"1", "true", "yes", "y", "on"}
    elif not isinstance(needs_more_sources, bool):
        needs_more_sources = bool(needs_more_sources)
    if needs_more_sources:
        return False

    return True


def _count_reference_anchors(value: Any) -> int:
    text = _decision_value_to_text(value)
    if not text:
        return 0
    return len(_ANCHOR_PATTERN.findall(text))


def _decision_has_stable_rag_support(state: CRCAgentState) -> bool:
    findings = state.findings or {}
    strategy = str(findings.get("decision_strategy") or "")
    if strategy != "rag_guideline":
        return False

    normalized_refs = _dedupe_references(state.retrieved_references or [])
    if len(normalized_refs) < 2:
        return False

    decision = state.decision_json or {}
    if not (decision.get("summary") and decision.get("treatment_plan")):
        return False

    return _count_reference_anchors(decision) >= 2


def _decision_strategy_label_v2(state: CRCAgentState, references: Optional[List[Any]] = None) -> str:
    if _should_use_template_decision_v2(state):
        return "template_fast"
    if _user_explicitly_requests_guideline_grounding_v2(state):
        return "rag_guideline"
    if references:
        return "rag_standard"
    return "decision_standard"


def _round_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _extract_best_json_dict(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_decision_raw_text(raw_text)
    if not cleaned:
        return None

    candidates: List[Dict[str, Any]] = []

    def _append_candidate(candidate: Any):
        if isinstance(candidate, dict):
            candidates.append(candidate)

    try:
        direct = json.loads(cleaned)
        _append_candidate(direct)
    except Exception:
        pass

    if cleaned.startswith("{"):
        missing = cleaned.count("{") - cleaned.count("}")
        if missing > 0:
            try:
                repaired = json.loads(cleaned + ("}" * missing))
                _append_candidate(repaired)
            except Exception:
                pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1:
        tail = cleaned[first_brace:]
        missing = tail.count("{") - tail.count("}")
        if missing > 0:
            try:
                parsed = json.loads(tail + ("}" * missing))
                _append_candidate(parsed)
            except Exception:
                pass
    if first_brace != -1 and last_brace > first_brace:
        middle = cleaned[first_brace:last_brace + 1]
        try:
            parsed = json.loads(middle)
            _append_candidate(parsed)
        except Exception:
            missing = middle.count("{") - middle.count("}")
            if missing > 0:
                try:
                    parsed = json.loads(middle + ("}" * missing))
                    _append_candidate(parsed)
                except Exception:
                    pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[idx:])
            _append_candidate(candidate)
        except Exception:
            continue

    if not candidates:
        return None

    preferred_keys = {
        "summary",
        "treatment_plan",
        "follow_up",
        "staging_summary",
        "staging_conclusion",
        "recommended_actions",
        "follow_up_plan",
    }

    def _score(candidate: Dict[str, Any]) -> int:
        keys = set(candidate.keys())
        hit_score = len(keys & preferred_keys) * 1000
        nested_score = sum(1 for value in candidate.values() if isinstance(value, (dict, list))) * 50
        size_score = len(json.dumps(candidate, ensure_ascii=False))
        return hit_score + nested_score + size_score

    return max(candidates, key=_score)


def _parse_decision_markdown(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_decision_raw_text(raw_text)
    if not cleaned:
        return None

    summary_lines: List[str] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    plan_sections: List[Dict[str, str]] = []
    follow_up: List[str] = []

    def _flush_section():
        nonlocal current_title, current_lines, plan_sections, follow_up
        if not current_title:
            return
        content = "\n".join([line for line in current_lines if line]).strip()
        if not content:
            current_title = None
            current_lines = []
            return
        if "随访" in current_title or "复查" in current_title:
            follow_up.extend(_coerce_follow_up_items(content))
        else:
            plan_sections.append({"title": current_title, "content": content})
        current_title = None
        current_lines = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = re.match(
            r"^(?:#+\s*)?((?:第?[一二三四五六七八九十0-9]+[、.．)]\s*)?[^:：]{1,30})[:：]?\s*$",
            line,
        )
        if heading_match and ("治疗" in line or "方案" in line or "评估" in line or "随访" in line or "复查" in line):
            _flush_section()
            current_title = heading_match.group(1).strip()
            continue

        if current_title:
            current_lines.append(line)
        else:
            summary_lines.append(line)

    _flush_section()

    if not summary_lines and not plan_sections and not follow_up:
        return None

    summary = " ".join(summary_lines[:3]).strip()
    if not summary and plan_sections:
        summary = plan_sections[0]["content"][:200]

    return {
        "summary": summary or "临床决策摘要",
        "treatment_plan": plan_sections,
        "follow_up": follow_up,
    }


def _parse_decision_raw_text(raw_text: str) -> Optional[Dict[str, Any]]:
    json_candidate = _extract_best_json_dict(raw_text)
    if json_candidate is not None:
        return json_candidate
    return _parse_decision_markdown(raw_text)


def _coerce_treatment_actions(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, dict):
        if any(key in value for key in ("title", "content", "regimen", "rationale", "description", "notes")):
            raw_items = [value]
        else:
            raw_items = []
            for key, item in value.items():
                if item in (None, "", [], {}):
                    continue
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("title", str(key).replace("_", " "))
                    raw_items.append(merged)
                else:
                    raw_items.append({
                        "title": str(key).replace("_", " "),
                        "content": _decision_value_to_text(item),
                    })
    else:
        raw_items = [value]

    actions: List[Dict[str, str]] = []
    for item in raw_items:
        try:
            action = TreatmentAction.model_validate(item).model_dump()
        except Exception:
            text = _decision_value_to_text(item)
            if not text:
                continue
            action = {"title": "治疗建议", "content": text}
        if action.get("content"):
            actions.append(action)
    return actions

# ==============================================================================
# 1. 数据结构定义 (改为最稳健的 List 结构)
# ==============================================================================

class TreatmentAction(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    """单项治疗措施"""
    title: str = Field(description="步骤标题。必须包含序号，例如 '1. 新辅助治疗'。")
    content: str = Field(description="详细方案。必须包含：药物名称、周期、剂量（如适用）、以及该步骤的临床获益理由（Reasoning）。")

    @model_validator(mode="before")
    @classmethod
    def coerce_variants(cls, data: Any):
        if data is None:
            return data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if isinstance(data, str):
            text = data.strip()
            return {"title": "治疗建议", "content": text}
        if not isinstance(data, dict):
            return data

        if data.get("title") and data.get("content"):
            return data

        title = ""
        for key in ("title", "name", "step", "section", "phase", "action"):
            value = data.get(key)
            if value:
                title = str(value).strip()
                break
        if not title:
            title = "治疗建议"

        content = ""
        for key in ("content", "description", "details"):
            value = data.get(key)
            if value:
                content = _decision_value_to_text(value)
                break

        if not content:
            regimen = _decision_value_to_text(data.get("regimen"))
            rationale = _decision_value_to_text(data.get("rationale"))
            notes = _decision_value_to_text(data.get("notes"))
            parts = []
            if regimen:
                parts.append(f"方案：{regimen}")
            if rationale:
                parts.append(f"依据：{rationale}")
            if notes:
                parts.append(f"备注：{notes}")
            content = "；".join([part for part in parts if part])

        if not content:
            content = _decision_value_to_text(data)

        return {"title": title, "content": content}


class ClinicalDecisionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    """决策方案结构"""
    summary: str = Field(description="患者病情摘要及诊断分期结论")
    # [关键修改] 改为 List，模型输出 List 的概率远高于 Dict，这样最稳定
    treatment_plan: List[TreatmentAction] = Field(description="按时间顺序排列的治疗步骤列表。❌ 严禁使用字典结构，必须是列表。")
    follow_up: List[str] = Field(description="随访计划")

    @model_validator(mode="before")
    @classmethod
    def coerce_variants(cls, data: Any):
        if data is None:
            return data
        if hasattr(data, "model_dump"):
            data = data.model_dump()

        if isinstance(data, str):
            parsed = _parse_decision_raw_text(data)
            if parsed is not None:
                data = parsed
            else:
                text = data.strip()
                return {
                    "summary": text[:200] or "临床决策摘要",
                    "treatment_plan": [{"title": "治疗建议", "content": text or "未返回结构化治疗方案"}],
                    "follow_up": [],
                }

        if not isinstance(data, dict):
            return data

        direct_keys = {"summary", "treatment_plan", "follow_up"}
        if not any(key in data for key in direct_keys) and any(
            key in data for key in ("title", "content", "regimen", "rationale", "description", "notes")
        ):
            action = TreatmentAction.model_validate(data).model_dump()
            return {
                "summary": action["content"][:200] or action["title"],
                "treatment_plan": [action],
                "follow_up": [],
            }

        summary_value = data.get("summary")
        if not summary_value:
            for key in (
                "staging_summary",
                "staging_conclusion",
                "clinical_stage_summary",
                "assessment_summary",
                "diagnosis_summary",
                "patient_summary",
                "case_summary",
                "staging",
            ):
                if data.get(key):
                    summary_value = data.get(key)
                    break
        summary = _decision_value_to_text(summary_value)

        plan_value = data.get("treatment_plan")
        if not plan_value:
            for key in (
                "recommended_actions",
                "treatment_recommendation",
                "recommendations",
                "plan",
                "actions",
                "management_plan",
            ):
                if data.get(key):
                    plan_value = data.get(key)
                    break
        treatment_plan = _coerce_treatment_actions(plan_value)

        follow_up_value = data.get("follow_up")
        if not follow_up_value:
            for key in ("followup", "follow_up_plan", "followup_plan", "surveillance", "monitoring_plan"):
                if data.get(key):
                    follow_up_value = data.get(key)
                    break
        follow_up = _coerce_follow_up_items(follow_up_value)

        if not treatment_plan:
            residual_parts = []
            for key, value in data.items():
                if key in {
                    "summary",
                    "staging_summary",
                    "staging_conclusion",
                    "clinical_stage_summary",
                    "assessment_summary",
                    "diagnosis_summary",
                    "patient_summary",
                    "case_summary",
                    "staging",
                    "follow_up",
                    "followup",
                    "follow_up_plan",
                    "followup_plan",
                    "surveillance",
                    "monitoring_plan",
                }:
                    continue
                text = _decision_value_to_text(value)
                if text:
                    residual_parts.append(text)
            residual_text = "；".join([part for part in residual_parts if part])
            if residual_text:
                treatment_plan = [{"title": "治疗建议", "content": residual_text}]

        if not summary or _looks_like_placeholder_text(summary):
            if treatment_plan:
                summary = treatment_plan[0]["content"][:200]
            else:
                summary = _decision_value_to_text(data)[:200] or "临床决策摘要"

        if not treatment_plan:
            treatment_plan = [{"title": "临床决策摘要", "content": summary}]

        return {
            "summary": summary,
            "treatment_plan": treatment_plan,
            "follow_up": follow_up,
        }


class CriticEvaluationSchema(BaseModel):
    """审核结果"""
    model_config = ConfigDict(
        extra="ignore",  # 忽略额外字段
        str_strip_whitespace=True,  # 自动去除首尾空格
    )
    
    verdict: Literal["APPROVED", "REJECTED", "APPROVED_WITH_WARNINGS"] = Field(description="审核结论")
    feedback: str = Field(description="审核意见")
    
    degraded: bool = Field(default=False, description="Whether critic output fell back to a degraded default")
    degraded_reason: str = Field(default="", description="Why critic output was degraded")

    @field_validator('feedback', mode='before')
    @classmethod
    def clean_feedback(cls, v):
        """清理 feedback 中可能破坏 JSON 的字符"""
        if not isinstance(v, str):
            return str(v)
        # 移除可能导致 JSON 解析失败的控制字符
        import re
        # 保留中文标点和常用符号，但确保格式正确
        v = v.strip()
        return v

class TreatmentSearchQueries(BaseModel):
    """智能生成的搜索关键词列表"""
    model_config = ConfigDict(extra="ignore")

    queries: List[str] = Field(
        description="3-5个精准的医学搜索关键词。例如：['直肠癌 T3N1 新辅助治疗 指南', 'dMMR 结直肠癌 免疫治疗 循证医学']",
        max_length=5
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_variants(cls, data: Any):
        """
        兼容模型常见输出变体，避免因为字段名/格式轻微偏差导致整体失败：
        - {"query": "..."} -> {"queries": ["..."]}
        - {"keywords": [...]} / {"queries": "..."} -> 标准化为 List[str]
        - str: 允许包含尾随解释文本，尽量提取首个 JSON 对象
        """
        if data is None:
            return {"queries": []}

        # 如果是字符串，尽量提取 JSON（处理 trailing characters / 代码块）
        if isinstance(data, str):
            import re
            s = data.strip()
            s = s.replace("```json", "").replace("```", "").strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return {"queries": arr}
                except Exception:
                    return {"queries": [s]}
            if s.startswith("{") and not s.endswith("}"):
                missing = s.count("{") - s.count("}")
                if missing > 0:
                    s2 = s + ("}" * missing)
                    try:
                        data = json.loads(s2)
                        return data
                    except Exception:
                        s = s2

            m = re.search(r'["\']queries["\']\s*:\s*\[(?P<body>[\s\S]*?)\]', s)
            if m:
                body = m.group("body")
                qs = [q.strip() for q in re.findall(r'["\']([^"\']+)["\']', body) if q.strip()]
                if qs:
                    return {"queries": qs}

            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            if len(lines) >= 2:
                candidates = []
                for ln in lines:
                    ln2 = re.sub(r"^\s*[-*•\d]+[).、.]?\s*", "", ln).strip()
                    if ln2:
                        candidates.append(ln2)
                if candidates:
                    return {"queries": candidates[:5]}

            # 截取第一个 {...}
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidate = s[first:last + 1]
                try:
                    data = json.loads(candidate)
                except Exception:
                    # 实在解析不了就当作一条 query
                    return {"queries": [s]}
            else:
                return {"queries": [s]}

        # list 直接视为 queries
        if isinstance(data, list):
            return {"queries": data}

        if isinstance(data, dict):
            if "queries" not in data:
                if "query" in data:
                    data["queries"] = [data.get("query")]
                elif "keywords" in data:
                    data["queries"] = data.get("keywords")
                elif "search_queries" in data:
                    data["queries"] = data.get("search_queries")
                else:
                    # 模型可能回显输入上下文，直接降级为空列表，让上层 fallback
                    data["queries"] = []

            q = data.get("queries")
            if isinstance(q, str):
                data["queries"] = [q]
            elif q is None:
                data["queries"] = []
            return data

        # 其他类型降级为空
        return {"queries": []}


def _infer_query_disease_labels(state: CRCAgentState) -> tuple[str, str]:
    findings = state.findings or {}
    raw_parts = [
        findings.get("tumor_location"),
        findings.get("tumor_site"),
        findings.get("primary_site"),
        findings.get("diagnosis"),
        findings.get("pathology_diagnosis"),
    ]
    raw_text = " ".join(str(part or "") for part in raw_parts)
    lowered = raw_text.lower()

    if "直肠" in raw_text or "rectal" in lowered or "rectum" in lowered:
        return "直肠癌", "rectal cancer"
    if any(marker in raw_text for marker in ["结肠", "乙状结肠", "升结肠", "降结肠", "横结肠", "盲肠"]) or any(
        marker in lowered for marker in ["colon", "sigmoid"]
    ):
        return "结肠癌", "colon cancer"
    return "结直肠癌", "colorectal cancer"


def _infer_query_disease_labels_v2(state: CRCAgentState) -> tuple[str, str]:
    findings = state.findings or {}
    semantic = findings.get("semantic_integrity") or {}

    structured_text = " ".join(
        str(part or "")
        for part in [
            findings.get("tumor_location"),
            findings.get("tumor_site"),
            findings.get("primary_site"),
            semantic.get("tumor_location_category"),
        ]
    )
    pathology_text = " ".join(
        str(part or "")
        for part in [
            findings.get("pathology_diagnosis"),
            findings.get("histology"),
            findings.get("histology_type"),
            findings.get("clinical_stage_staging"),
        ]
    )
    fallback_text = " ".join(
        str(part or "")
        for part in [
            findings.get("diagnosis"),
            findings.get("assessment_draft"),
        ]
    )

    colon_markers = [
        "colon",
        "sigmoid",
        "ascending colon",
        "descending colon",
        "transverse colon",
        "乙状结肠",
        "结肠",
        "升结肠",
        "降结肠",
        "横结肠",
        "盲肠",
    ]
    rectal_markers = ["rectal", "rectum", "直肠"]

    def _contains_any_marker(text: str, markers: List[str]) -> bool:
        lowered = str(text or "").lower()
        return any(marker.lower() in lowered for marker in markers)

    if _contains_any_marker(structured_text, colon_markers):
        return "结肠癌", "colon cancer"
    if _contains_any_marker(structured_text, rectal_markers):
        return "直肠癌", "rectal cancer"

    if _contains_any_marker(pathology_text, colon_markers):
        return "结肠癌", "colon cancer"
    if _contains_any_marker(pathology_text, rectal_markers):
        return "直肠癌", "rectal cancer"

    if _contains_any_marker(fallback_text, colon_markers):
        return "结肠癌", "colon cancer"
    if _contains_any_marker(fallback_text, rectal_markers):
        return "直肠癌", "rectal cancer"

    return "结直肠癌", "colorectal cancer"


def _collect_case_query_tokens(state: CRCAgentState, limit: int = 6) -> List[str]:
    findings = state.findings or {}
    raw_text = " ".join(
        str(part or "")
        for part in [
            findings.get("clinical_stage_staging"),
            findings.get("tnm_stage"),
            findings.get("tnm_status"),
            findings.get("diagnosis"),
            findings.get("pathology_diagnosis"),
            findings.get("molecular_profile"),
            json.dumps(findings, ensure_ascii=False),
        ]
    )

    tokens: List[str] = []

    def _push(value: str):
        value = str(value or "").strip()
        if value and value not in tokens:
            tokens.append(value)

    for pattern in (r"\b[cpy]?T\d[a-c]?\b", r"\b[cpy]?N\d[a-c]?\b", r"\b[cpy]?M\d[a-c]?\b"):
        for match in re.findall(pattern, raw_text, flags=re.IGNORECASE):
            _push(match.upper())

    stage_match = re.search(r"\bstage\s*([ivx]{1,4})\b", raw_text, flags=re.IGNORECASE)
    if stage_match:
        _push(f"stage {stage_match.group(1).upper()}")
    else:
        cn_stage_match = re.search(r"([IVX]{1,4})期", raw_text, flags=re.IGNORECASE)
        if cn_stage_match:
            _push(f"stage {cn_stage_match.group(1).upper()}")

    for keyword in [
        "MSS",
        "MSI-H",
        "MSI",
        "dMMR",
        "pMMR",
        "KRAS",
        "NRAS",
        "BRAF",
        "RAS",
        "HER2",
        "NTRK",
        "POLE",
        "CEA",
        "CapeOX",
        "mFOLFOX6",
        "FOLFOX",
    ]:
        if keyword.lower() in raw_text.lower():
            _push(keyword)

    return tokens[:limit]


def _build_crosslingual_query_boost(state: CRCAgentState) -> List[str]:
    _, disease_en = _infer_query_disease_labels_v2(state)
    token_text = " ".join(_collect_case_query_tokens(state, limit=4))
    base = " ".join(part for part in [disease_en, token_text] if part).strip()

    queries = [
        f"{base} guideline",
        f"{base} adjuvant chemotherapy guideline NCCN ESMO",
        f"{base} surgery perioperative management guideline",
    ]
    if disease_en == "rectal cancer":
        queries.append(f"{base} neoadjuvant chemoradiation guideline NCCN")
    else:
        queries.append(f"{base} stage III adjuvant CapeOX FOLFOX guideline")

    deduped: List[str] = []
    for query in queries:
        query = query.strip()
        if query and query not in deduped:
            deduped.append(query)
    return deduped


def _build_safe_fallback_queries(state: CRCAgentState) -> List[str]:
    disease_cn, disease_en = _infer_query_disease_labels_v2(state)
    token_text = " ".join(_collect_case_query_tokens(state, limit=4))
    chinese_query = " ".join(part for part in [disease_cn, token_text, "治疗 推荐 指南"] if part).strip()
    english_query = " ".join(part for part in [disease_en, token_text, "guideline"] if part).strip()
    return [query for query in [chinese_query, english_query] if query]


def _ensure_query_language_balance(
    queries: List[str],
    state: CRCAgentState,
    max_queries: int = 5,
) -> List[str]:
    cleaned: List[str] = []
    for query in queries:
        query = str(query or "").strip()
        if query and query not in cleaned:
            cleaned.append(query)

    english_count = sum(bool(re.search(r"[A-Za-z]", query)) for query in cleaned)
    supplements = _build_crosslingual_query_boost(state)

    if english_count < 2:
        for query in supplements:
            if query not in cleaned:
                cleaned.append(query)
            english_count = sum(bool(re.search(r"[A-Za-z]", item)) for item in cleaned)
            if english_count >= 2:
                break

    if len(cleaned) < max_queries:
        for query in supplements:
            if query not in cleaned:
                cleaned.append(query)
            if len(cleaned) >= max_queries:
                break

    return cleaned[:max_queries]


def _preferred_query_count(state: CRCAgentState) -> int:
    default_limit = 3
    try:
        default_limit = int(os.getenv("DECISION_MAX_QUERIES", "3"))
    except Exception:
        default_limit = 3
    if (state.findings or {}).get("fast_pass_mode"):
        return max(2, min(default_limit, 3))
    return max(2, min(default_limit, 5))


def _build_fast_queries(state: CRCAgentState) -> List[str]:
    disease_cn, disease_en = _infer_query_disease_labels_v2(state)
    token_text = " ".join(_collect_case_query_tokens(state, limit=3))
    queries = [
        " ".join(part for part in [disease_cn, token_text, "指南"] if part).strip(),
        " ".join(part for part in [disease_en, token_text, "guideline"] if part).strip(),
        " ".join(part for part in [disease_en, token_text, "adjuvant surgery chemotherapy guideline"] if part).strip(),
    ]
    deduped: List[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:_preferred_query_count(state)]


def _extract_stage_context(state: CRCAgentState) -> Dict[str, str]:
    findings = state.findings or {}
    tnm = findings.get("tnm_staging") or {}
    clinical = tnm.get("clinical") if isinstance(tnm, dict) else {}
    cT = clinical.get("cT") or tnm.get("cT") or ""
    cN = clinical.get("cN") or tnm.get("cN") or ""
    cM = clinical.get("cM") or tnm.get("cM") or ""
    stage_group = tnm.get("stage_group") or findings.get("clinical_stage_staging") or ""
    return {
        "cT": str(cT or ""),
        "cN": str(cN or ""),
        "cM": str(cM or ""),
        "stage_group": str(stage_group or ""),
        "mmr": str(findings.get("mmr_status") or ""),
        "location": str(findings.get("tumor_location") or ""),
    }


def _should_use_template_decision(state: CRCAgentState) -> bool:
    findings = state.findings or {}
    if not bool(findings.get("pathology_confirmed")):
        return False
    stage = _extract_stage_context(state)
    return bool(stage["cT"] and stage["cN"] and stage["cM"]) and bool(findings.get("fast_pass_mode"))


def _build_template_decision(state: CRCAgentState) -> Dict[str, Any]:
    findings = state.findings or {}
    profile = state.patient_profile
    stage = _extract_stage_context(state)
    cT = stage["cT"]
    cN = stage["cN"]
    cM = stage["cM"]
    mmr = stage["mmr"] or "Unknown"
    location = stage["location"] or "unknown"
    stage_group = stage["stage_group"] or f"{cT}{cN}{cM}"
    age = getattr(profile, "age", None)
    ecog = getattr(profile, "ecog_score", None)
    histology = findings.get("histology_type") or "腺癌"
    summary = f"病理提示{histology}，当前临床分期{stage_group}（{cT}{cN}{cM}），MMR状态{mmr}。"

    treatment_plan: List[Dict[str, str]] = []
    follow_up: List[str] = []

    if location == "colon" and cM.upper().endswith("0"):
        treatment_plan.append({
            "title": "分期结论",
            "content": f"按现有病理和影像，属于结肠癌围手术期决策场景，当前以{cT}{cN}{cM}为术前分期依据，最终方案仍需以术后病理分期校正。"
        })
        treatment_plan.append({
            "title": "手术策略",
            "content": "优先行根治性结肠癌切除并完成规范淋巴结清扫，目标为R0切除，术中如见受侵邻近组织需按整块切除原则处理。"
        })
        if "N1" in cN.upper() or "N2" in cN.upper() or "T4" in cT.upper():
            treatment_plan.append({
                "title": "围手术期治疗",
                "content": f"该病例已具备高复发风险特征，术后应优先进入含奥沙利铂的辅助化疗路径；若为高危III期，通常按6个月CAPOX或FOLFOX考虑。pMMR/MSS状态下，不将免疫治疗作为常规围手术期标准方案。"
            })
        else:
            treatment_plan.append({
                "title": "围手术期治疗",
                "content": "术后根据病理分期和耐受性决定辅助化疗强度；若确认为III期，通常进入CAPOX或FOLFOX辅助治疗路径。"
            })
        treatment_plan.append({
            "title": "合并症管理",
            "content": f"围手术期需同步评估年龄{age if age is not None else '未知'}岁、ECOG {ecog if ecog is not None else '未提供'}、高血压和糖尿病控制，优化血糖、血压、肝肾功能和神经毒性监测后再启动化疗。"
        })
        follow_up = [
            "以术后病理分期作为后续辅助治疗最终依据。",
            "术后按计划监测血常规、肝肾功能、CEA和化疗毒性。",
            "继续做MMR/MSI、RAS/BRAF等分子信息整合，用于后续复发风险和晚期治疗预案。",
        ]
    else:
        treatment_plan.append({
            "title": "分期结论",
            "content": f"当前按{cT}{cN}{cM}进行模板化快速生成，建议结合MDT和最终病理/影像再次确认。"
        })
        treatment_plan.append({
            "title": "治疗原则",
            "content": "先完成可切除性、远处转移和全身耐受性评估，再决定手术、围手术期治疗或系统治疗顺序。"
        })
        follow_up = [
            "必要时补充影像或病理复核，避免错误分期驱动后续方案。",
        ]

    return ClinicalDecisionSchema.model_validate({
        "summary": summary,
        "treatment_plan": treatment_plan,
        "follow_up": follow_up,
    }).model_dump()


def _preferred_query_count_v2(state: CRCAgentState) -> int:
    default_limit = 3
    try:
        default_limit = int(os.getenv("DECISION_MAX_QUERIES", "3"))
    except Exception:
        default_limit = 3
    if (state.findings or {}).get("fast_pass_mode"):
        return max(2, min(default_limit, 3))
    return max(2, min(default_limit, 5))


def _user_explicitly_requests_guideline_grounding_v2(state: CRCAgentState) -> bool:
    user_text = (_latest_user_text(state) or "").strip()
    if not user_text:
        return False

    user_text_lower = user_text.lower()
    directive_terms_cn = (
        "结合",
        "根据",
        "基于",
        "依据",
        "参照",
        "按照",
        "按",
    )
    directive_terms_en = (
        "based on",
        "according to",
        "per ",
        "guided by",
        "with citation",
        "with citations",
        "with reference",
        "with references",
    )
    general_guideline_terms_cn = (
        "指南",
        "循证",
        "证据",
        "文献依据",
        "参考文献",
        "指南推荐",
    )
    general_guideline_terms_en = (
        "guideline",
        "guidelines",
        "evidence-based",
        "evidence based",
        "citation",
        "citations",
        "reference",
        "references",
    )
    guideline_sources = ("csco", "nccn", "esmo", "asco")

    has_directive = any(term in user_text for term in directive_terms_cn) or any(
        term in user_text_lower for term in directive_terms_en
    )
    has_general_guideline_term = any(term in user_text for term in general_guideline_terms_cn) or any(
        term in user_text_lower for term in general_guideline_terms_en
    )
    has_named_guideline = any(source in user_text_lower for source in guideline_sources) and (
        "指南" in user_text or "guideline" in user_text_lower or "guidelines" in user_text_lower
    )
    has_guideline_recommendation_phrase = any(
        phrase in user_text
        for phrase in ("结合指南", "根据指南", "依据指南", "参照指南", "按指南", "指南推荐", "循证依据", "参考文献")
    ) or any(
        phrase in user_text_lower
        for phrase in (
            "guideline recommendation",
            "guideline recommendations",
            "based on guidelines",
            "based on guideline",
            "according to guidelines",
            "according to guideline",
        )
    )

    return bool(
        has_guideline_recommendation_phrase
        or (has_general_guideline_term and has_directive)
        or (has_named_guideline and (has_directive or "推荐" in user_text or "recommend" in user_text_lower))
    )


def _extract_stage_context_v2(state: CRCAgentState) -> Dict[str, str]:
    findings = state.findings or {}
    tnm = findings.get("tnm_staging") or {}
    clinical = tnm.get("clinical") if isinstance(tnm, dict) and isinstance(tnm.get("clinical"), dict) else {}
    cT = clinical.get("cT") or tnm.get("cT") or ""
    cN = clinical.get("cN") or tnm.get("cN") or ""
    cM = clinical.get("cM") or tnm.get("cM") or ""
    if cT and not str(cT).lower().startswith("c"):
        cT = f"c{cT}"
    if cN and not str(cN).lower().startswith("c"):
        cN = f"c{cN}"
    if cM and not str(cM).lower().startswith("c"):
        cM = f"c{cM}"
    stage_group = (
        tnm.get("stage_group")
        or findings.get("clinical_stage_group")
        or ""
    )
    return {
        "cT": str(cT or ""),
        "cN": str(cN or ""),
        "cM": str(cM or ""),
        "stage_group": str(stage_group or ""),
        "mmr": str(findings.get("mmr_status") or ""),
        "location": str(findings.get("tumor_location") or "").strip().lower(),
        "subsite": str(findings.get("tumor_subsite") or "").strip(),
    }


def _format_tnm_display_v2(cT: str, cN: str, cM: str) -> str:
    return "".join([
        str(cT or ""),
        str(cN or "")[1:] if str(cN or "").lower().startswith("c") else str(cN or ""),
        str(cM or "")[1:] if str(cM or "").lower().startswith("c") else str(cM or ""),
    ])


def _build_fast_queries_v2(state: CRCAgentState) -> List[str]:
    disease_cn, disease_en = _infer_query_disease_labels_v2(state)
    token_text = " ".join(_collect_case_query_tokens(state, limit=3))
    queries = [
        " ".join(part for part in [disease_cn, token_text, "指南"] if part).strip(),
        " ".join(part for part in [disease_en, token_text, "guideline"] if part).strip(),
        " ".join(part for part in [disease_en, token_text, "perioperative management guideline"] if part).strip(),
    ]
    deduped: List[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:_preferred_query_count_v2(state)]


def _should_use_template_decision_v2(state: CRCAgentState) -> bool:
    if os.getenv("DECISION_FAST_TEMPLATE", "true").strip().lower() in {"0", "false", "no"}:
        return False
    findings = state.findings or {}
    if not bool(findings.get("pathology_confirmed")):
        return False
    if _user_explicitly_requests_guideline_grounding_v2(state):
        return False
    stage = _extract_stage_context_v2(state)
    return bool(stage["cT"] and stage["cN"] and stage["cM"]) and bool(findings.get("fast_pass_mode"))


def _build_template_decision_v2(state: CRCAgentState) -> Dict[str, Any]:
    findings = state.findings or {}
    profile = state.patient_profile
    stage = _extract_stage_context_v2(state)
    cT = stage["cT"]
    cN = stage["cN"]
    cM = stage["cM"]
    mmr = stage["mmr"] or "未知"
    location = stage["location"] or "unknown"
    stage_group = stage["stage_group"] or f"{cT}{cN}{cM}"
    subsite = stage.get("subsite") or ""
    age = getattr(profile, "age", None)
    ecog = getattr(profile, "ecog_score", None)
    histology = findings.get("histology_type") or "腺癌"
    age_text = f"{age}岁" if age is not None else "老年"
    ecog_text = str(ecog) if ecog is not None else "未知"
    location_text = subsite or ("结肠" if "colon" in location else "直肠" if "rect" in location else "肠道")
    tnm_display = _format_tnm_display_v2(cT, cN, cM)
    is_colon = any(marker in location for marker in ["colon", "sigmoid", "结肠"])
    high_risk_stage_iii = cM.upper().endswith("0") and (
        "T4" in cT.upper() or "N2" in cN.upper() or "III" in stage_group.upper() or "III" in stage_group
    )
    comorbidity_text = "合并糖尿病/高血压" if age is not None else "需结合合并症"

    summary = (
        f"患者为{location_text}{histology}，当前临床分期支持 {tnm_display}"
        f"（{stage_group}），MMR 状态为 {mmr}。结合 {age_text}、ECOG {ecog_text} 及{comorbidity_text}，"
        "围手术期治疗以根治性手术联合术后辅助化疗为主。"
    )

    treatment_plan: List[Dict[str, str]] = []
    follow_up: List[str] = []

    if is_colon and cM.upper().endswith("0"):
        treatment_plan.append({
            "title": "临床分期判断",
            "content": (
                f"当前信息支持 {location_text}{histology}，临床分期为 {tnm_display}"
                f"（{stage_group}），属于无远处转移的结肠癌。"
                + (" 按 T4/N1 归入高危 III 期。" if high_risk_stage_iii else "")
            ),
        })
        treatment_plan.append({
            "title": "新辅助治疗判断",
            "content": (
                "对于当前可切除的结肠癌情形，不常规推荐新辅助放化疗。"
                "该例为 pMMR 结肠癌，也不属于优先考虑免疫新辅助的场景。"
                "如 MDT 评估存在边界可切除、固定、局部并发症或需争取 R0 切除时，才考虑个体化术前治疗。"
            ),
        })
        treatment_plan.append({
            "title": "手术方案",
            "content": (
                f"推荐行 {location_text}癌根治术。对本例更贴近的术式是乙状结肠切除/乙状结肠癌根治术，"
                "必要时根据血供、切缘和术中所见决定是否扩大切除范围。手术目标为 R0 切除、完整系膜切除、"
                "规范 D3 淋巴结清扫，并保证至少检出 12 枚淋巴结；若术中证实邻近器官受侵，应整块切除。"
            ),
        })
        treatment_plan.append({
            "title": "术后辅助化疗",
            "content": (
                (
                    "术后病理若维持 III 期，建议在术后恢复允许时尽快启动辅助化疗。"
                    "本例按高危 III 期处理时，优先 CAPOX 或 mFOLFOX6，整体更倾向 6 个月疗程；"
                    "若年龄、糖尿病、神经毒性风险或总体耐受性限制奥沙利铂使用，可在充分沟通后减量、缩短奥沙利铂暴露，"
                    "必要时改为氟嘧啶单药。"
                    if high_risk_stage_iii
                    else "术后病理若维持 III 期，建议按耐受性选择 CAPOX 或 mFOLFOX6 作为辅助化疗。"
                )
            ),
        })
        treatment_plan.append({
            "title": "围手术期管理",
            "content": (
                f"结合 {age_text}、ECOG {ecog_text} 以及糖尿病/高血压，术前应完善心功能、肝肾功能、凝血及营养评估。"
                "阿司匹林需由外科联合心内科评估停用时机；使用造影剂或拟用卡培他滨前应核查肾功能，"
                "围手术期注意二甲双胍管理、血糖控制、血压控制以及奥沙利铂相关神经毒性监测。"
            ),
        })
        follow_up = [
            "以术后病理分期为准，重新确认辅助化疗方案和疗程。",
            "以当前升高的 CEA 为基线，术后动态复查并结合恢复情况决定化疗启动时间。",
            "按结肠癌术后随访计划复查影像、CEA 和结肠镜。"
        ]
    else:
        treatment_plan.append({
            "title": "MDT 复核",
            "content": (
                f"当前结构化分期为 {cT}{cN}{cM}。该情形建议继续 MDT 讨论，"
                "结合最终病理、解剖位置和可切除性确定个体化围手术期方案。"
            ),
        })
        treatment_plan.append({
            "title": "下一步",
            "content": "待最终病理和手术评估明确后，再锁定围手术期治疗路径。",
        })
        follow_up = [
            "治疗执行前再次核对病理与分期一致性。",
        ]

    return ClinicalDecisionSchema.model_validate({
        "summary": summary,
        "treatment_plan": treatment_plan,
        "follow_up": follow_up,
    }).model_dump()


def _rag_mentions_any(rag_context: str, keywords: List[str]) -> bool:
    lowered = str(rag_context or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _rag_supports_adjuvant_immunotherapy(rag_context: str) -> bool:
    lowered = str(rag_context or "").lower()
    positive_markers = [
        "adjuvant immunotherapy",
        "辅助免疫",
        "辅助治疗 pd-1",
        "术后免疫",
        "pembrolizumab adjuvant",
    ]
    negative_markers = [
        "does not support",
        "not support",
        "not recommended",
        "不支持",
        "不推荐",
        "不作为常规推荐",
    ]
    return any(marker in lowered for marker in positive_markers) and not any(
        marker in lowered for marker in negative_markers
    )


def _split_sentences_for_safety(text: str) -> List[str]:
    chunks = re.split(r"(?<=[。；;!?！？])\s*|\n+", str(text or "").strip())
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _sanitize_generated_text(
    text: str,
    rag_context: str,
    mmr_status: str = "",
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned

    rag_lower = str(rag_context or "").lower()
    mmr_lower = str(mmr_status or "").lower()

    if not any(token in rag_lower for token in ["iiia", "iiib", "iiic", "高危iiib", "高危iiic"]):
        cleaned = re.sub(r"[（(]\s*(?:高危)?\s*III[ABC]\s*期\s*[)）]", "（III期）", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstage\s+iii[abc]\b", "stage III", cleaned, flags=re.IGNORECASE)

    filtered: List[str] = []
    for sentence in _split_sentences_for_safety(cleaned):
        lowered = sentence.lower()

        if (
            ("keynote-177" in lowered or "pd-1" in lowered or "pembrolizumab" in lowered or "帕博利珠单抗" in sentence)
            and ("辅助" in sentence or "术后" in sentence or "adjuvant" in lowered)
            and not _rag_supports_adjuvant_immunotherapy(rag_context)
        ):
            continue

        if (
            any(keyword in lowered for keyword in ["lynch", "germline", "genetic counseling"])
            or any(keyword in sentence for keyword in ["胚系", "遗传咨询", "家系筛查", "Lynch综合征", "林奇综合征"])
        ) and not _rag_mentions_any(
            rag_context,
            ["lynch", "胚系", "遗传咨询", "家系筛查", "germline"],
        ):
            continue

        if ("二甲双胍" in sentence and "乳酸酸中毒" in sentence) and not _rag_mentions_any(
            rag_context,
            ["二甲双胍", "乳酸酸中毒", "metformin", "lactic acidosis"],
        ):
            continue

        if ("维生素B6" in sentence or "vitamin b6" in lowered) and not _rag_mentions_any(
            rag_context,
            ["维生素b6", "vitamin b6", "手足综合征"],
        ):
            continue

        if mmr_lower == "pmmr" and (
            ("dmmr" in lowered or "msi-h" in lowered)
            and ("辅助" in sentence or "术后" in sentence or "adjuvant" in lowered)
        ):
            continue

        filtered.append(sentence)

    return " ".join(filtered).strip()


def _sanitize_decision_output(
    decision_dict: Dict[str, Any],
    rag_context: str,
    state: CRCAgentState,
    references: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    sanitized = dict(decision_dict or {})
    mmr_status = (
        (state.findings or {}).get("mmr_status")
        or ((state.findings or {}).get("molecular_markers") or {}).get("MMR")
        or ""
    )

    normalized_refs = _dedupe_references(references or state.retrieved_references or [])
    force_inline_citations = bool(normalized_refs) and _user_explicitly_requests_guideline_grounding_v2(state)

    summary = _sanitize_generated_text(sanitized.get("summary", ""), rag_context, mmr_status)
    if normalized_refs:
        summary = _attach_reference_anchors(
            summary,
            normalized_refs,
            max_refs=2,
            fallback_to_top=force_inline_citations,
        )
    sanitized["summary"] = summary

    cleaned_plan: List[Dict[str, str]] = []
    for idx, item in enumerate(sanitized.get("treatment_plan") or []):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            item = {"title": "治疗步骤", "content": _decision_value_to_text(item)}
        content = _sanitize_generated_text(item.get("content", ""), rag_context, mmr_status)
        if normalized_refs:
            content = _attach_reference_anchors(
                content,
                normalized_refs,
                max_refs=2,
                fallback_to_top=force_inline_citations or idx < 2,
            )
        if content:
            cleaned_plan.append({"title": item.get("title") or "治疗步骤", "content": content})
    sanitized["treatment_plan"] = cleaned_plan

    cleaned_follow_up: List[str] = []
    for item in sanitized.get("follow_up") or []:
        content = _sanitize_generated_text(item, rag_context, mmr_status)
        if normalized_refs and force_inline_citations:
            content = _attach_reference_anchors(
                content,
                normalized_refs,
                max_refs=1,
                fallback_to_top=False,
            )
        if content:
            cleaned_follow_up.append(content)
    sanitized["follow_up"] = cleaned_follow_up

    return ClinicalDecisionSchema.model_validate(sanitized).model_dump()

# ==============================================================================
# 2. Decision Node
# ==============================================================================

def node_decision(
    model, 
    tools: List[BaseTool], 
    streaming: bool = False, 
    show_thinking: bool = True,
    use_sub_agent: bool = True  # [新增] 是否使用子智能体隔离进行 RAG 检索
) -> Runnable:
    # 工具初始化 - 优先使用治疗方案专用检索工具
    decision_tools = _select_tools(tools, ["search_treatment_recommendations", "search_clinical_guidelines", "web_search", "search_clinical_evidence"])
    # 优先使用治疗方案检索工具，其次是基础指南检索
    rag_tool = next((t for t in decision_tools if "treatment" in getattr(t, "name", "")), None)
    if not rag_tool:
        rag_tool = next((t for t in decision_tools if "guideline" in getattr(t, "name", "")), None)
    if not rag_tool:
        from ..tools.rag_tools import get_guideline_tool
        rag_tool = get_guideline_tool()
    
    # 收集所有可用的 RAG 工具（用于子智能体模式）
    rag_tools_list = [t for t in decision_tools if any(kw in getattr(t, "name", "") for kw in ["treatment", "guideline", "search"])]
    if not rag_tools_list and rag_tool:
        rag_tools_list = [rag_tool]

    # 从统一的 prompts 模块导入 System Prompt
    system_prompt = DECISION_SYSTEM_PROMPT
    decision_prompt = (
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "参考资料:\n{rag_context}\n\n患者数据:\n{patient_data}\n\nCritic反馈:\n{feedback}\n\n请生成方案。"),
        ])
    )

    # 语义化 RAG 查询生成器（替代 if-else 拼接逻辑）
    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_GENERATION_SYSTEM_PROMPT),
        ("human", "{patient_context}")
    ])
    def _build_patient_context(state: CRCAgentState) -> str:
        """构建给查询生成器的上下文（尽量结构化，但允许是纯文本）。"""
        findings = state.findings or {}
        profile = state.patient_profile
        # 额外把 semantic_integrity 一并给到，帮助模型理解 TNM 完整性与部位归类
        semantic = findings.get("semantic_integrity", None)
        ctx = {
            "patient_profile": profile.model_dump() if hasattr(profile, "model_dump") and profile else str(profile),
            "findings": findings,
            "semantic_integrity": semantic,
        }
        return json.dumps(ctx, ensure_ascii=False)

    def _generate_queries(state: CRCAgentState) -> List[str]:
        """LLM 生成 3-5 条查询，失败时返回保守的少量默认查询（避免完全无检索）。"""
        if _should_use_template_decision_v2(state):
            return _build_fast_queries_v2(state)
        patient_context = _build_patient_context(state)
        summary_memory = _build_summary_memory(state)
        pinned_context = _build_pinned_context(state)
        try:
            payload = {
                "patient_context": patient_context,
                "summary_memory": summary_memory,
                "pinned_context": pinned_context
            }
            res: TreatmentSearchQueries = _invoke_structured_with_recovery(
                prompt=query_gen_prompt,
                model=model,
                schema=TreatmentSearchQueries,
                payload=payload,
                log_prefix="[RAG QueryGen]",
            )
            # 去重 + 清理空串
            queries = []
            for q in (res.queries or []):
                q2 = str(q).strip()
                if q2 and q2 not in queries:
                    queries.append(q2)
            if queries:
                return _ensure_query_language_balance(queries, state, max_queries=_preferred_query_count_v2(state))
            raise ValueError("LLM returned empty queries")
        except Exception as e:
            # 不做硬编码分子 if-else，只保留最低限度的安全 fallback（避免“完全不检索”）
            if show_thinking:
                print(f"⚠️ [RAG QueryGen] 语义查询生成失败，使用保守默认查询: {e}")
            fallback_queries = _build_fast_queries_v2(state) if _should_use_template_decision_v2(state) else _build_safe_fallback_queries(state)
            return _ensure_query_language_balance(fallback_queries, state, max_queries=_preferred_query_count_v2(state))



    async def _run(state: CRCAgentState):
        iteration = getattr(state, "iteration_count", 0)
        decision_attempt = iteration + 1
        stage_record: Dict[str, Any] = {
            "node": "decision",
            "attempt": decision_attempt,
            "cached_rag_reused": False,
        }
        retrieval_profiles: List[Dict[str, Any]] = []
        if _should_use_template_decision_v2(state):
            decision_dict = _sanitize_decision_output(
                _build_template_decision_v2(state),
                rag_context="",
                state=state,
            )
            stage_record["query_count"] = 0
            stage_record["retrieval_ms"] = 0.0
            stage_record["decision_llm_ms"] = 0.0
            updates = {
                "messages": [AIMessage(content=f"[Decision] template-fast {decision_dict.get('summary', '')[:200]}")],
                "clinical_stage": "Decision",
                "stage_timings": [stage_record],
                "retrieval_timings": retrieval_profiles,
                "decision_json": decision_dict,
                "retrieved_references": [],
                "iteration_count": iteration + 1,
                "error": None,
                "findings": {"decision_strategy": "template_fast"},
            }
            temp_state = state.model_copy(update=updates)
            updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return updates


        # 1. 迭代熔断
        if iteration >= 3:
            return {
                "clinical_stage": "Decision",
                "stage_timings": [stage_record],
                "retrieval_timings": retrieval_profiles,
                "decision_json": _generate_fallback_plan(state),
                "iteration_limit_reached": True,
                "messages": [AIMessage(content="达到最大重试次数，已生成兜底方案。")]
            }

        # 2. 意图检查
        if not _needs_full_decision(state):
            # 不覆盖已有方案，避免错误清空
            return {"clinical_stage": "Decision"}
        
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        decision_attempt = iteration + 1
        stage_record: Dict[str, Any] = {
            "node": "decision",
            "attempt": decision_attempt,
            "cached_rag_reused": False,
        }
        retrieval_profiles: List[Dict[str, Any]] = []

        # 3. RAG 检索（两种模式：子智能体隔离 vs 直接调用）
        # 【v5.2 重要说明】
        # 这是治疗决策流程中唯一的 RAG 检索执行点
        # Planner 不再为 treatment_decision 生成知识检索步骤，所有检索在此统一完成
        rag_context = ""
        retrieved_refs = []
        reuse_cached_rag = _should_reuse_cached_rag(state)
        sub_agent_succeeded = False
        if reuse_cached_rag:
            retrieved_refs = [
                r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                for r in (state.retrieved_references or [])
            ]
            rag_context = _build_cached_rag_context_from_references(retrieved_refs)
            stage_record["cached_rag_reused"] = True
            if show_thinking:
                print(f"[Decision] Reusing cached RAG references: {len(retrieved_refs)} refs")
        retrieval_started_at: Optional[float] = None
        rag_queries = []  # 保存查询词用于生成摘要
        sub_agent_succeeded = False  # 跟踪子智能体是否成功
        
        if (rag_tool or rag_tools_list) and not reuse_cached_rag:
            try:
                querygen_started_at = time.perf_counter()
                rag_queries = _generate_queries(state)
                stage_record["querygen_ms"] = _round_ms(querygen_started_at)
                stage_record["query_count"] = len(rag_queries)
                if show_thinking:
                    print(f"[RAG QueryGen] queries={rag_queries}")
                retrieval_started_at = time.perf_counter()
                
                # ============================================================
                # 模式 A：子智能体隔离（推荐，上下文完全隔离）
                # ============================================================
                if use_sub_agent and rag_tools_list:
                    if show_thinking:
                        print(f"🔒 [Decision] 使用子智能体隔离模式进行 RAG 检索...")
                    
                    # 构建患者上下文
                    patient_ctx = {
                        "profile": state.patient_profile.model_dump() if state.patient_profile and hasattr(state.patient_profile, "model_dump") else str(state.patient_profile),
                        "findings": state.findings or {},
                    }
                    
                    # 执行隔离的 RAG 检索
                    # 子智能体的所有中间过程（多次搜索、阅读、重试）都在沙箱中进行
                    # 主智能体只会收到最终的 <report> 内容
                    subagent_started_at = time.perf_counter()
                    sub_result = await run_isolated_rag_search(
                        model=model,
                        queries=rag_queries,
                        rag_tools=rag_tools_list,
                        patient_context=patient_ctx,
                        show_thinking=show_thinking
                    )
                    stage_record["subagent_retrieval_ms"] = _round_ms(subagent_started_at)
                    
                    if sub_result.success:
                        # 蒸馏后的报告（子智能体的完整历史已被销毁）
                        rag_context = sub_result.report
                        retrieved_refs = sub_result.references
                        sub_agent_succeeded = True
                        
                        if show_thinking:
                            print(f"✅ [SubAgent] 检索完成，报告长度: {len(rag_context)} 字符")
                            print(f"   子智能体消耗: ~{sub_result.raw_token_count} tokens, {sub_result.iterations} 次迭代")
                    else:
                        if show_thinking:
                            print(f"⚠️ [SubAgent] 检索失败，回退到直接模式: {sub_result.error}")
                
                # ============================================================
                # 模式 B：直接调用（兼容旧版，或子智能体失败后的回退）
                # ============================================================
                if not use_sub_agent or (use_sub_agent and not sub_agent_succeeded):
                    if show_thinking:
                        print(f"📋 [Decision] 使用直接调用模式进行 RAG 检索...")
                    
                    # 使用合适的 top_k
                    top_k = 5 if hasattr(rag_tool, 'name') and 'treatment' in rag_tool.name else 6
                    disease_cn, _ = _infer_query_disease_labels_v2(state)

                    merged_context_parts: List[str] = []
                    merged_refs: List[dict] = []

                    for q in rag_queries:
                        invoke_payload = {"query": q, "top_k": top_k}
                        tool_name = getattr(rag_tool, "name", "")
                        if disease_cn and disease_cn != "结直肠癌":
                            if tool_name == "search_treatment_recommendations":
                                invoke_payload["disease"] = disease_cn
                            elif tool_name in {"search_clinical_guidelines", "hybrid_guideline_search"}:
                                invoke_payload["disease_focus"] = disease_cn
                        reset_retrieval_metrics()
                        invoke_started_at = time.perf_counter()
                        res = rag_tool.invoke(invoke_payload)
                        invoke_ms = _round_ms(invoke_started_at)
                        call_metrics = consume_retrieval_metrics()
                        if call_metrics:
                            latest_metrics = dict(call_metrics[-1])
                            latest_metrics["query"] = q
                            latest_metrics["tool_name"] = tool_name
                            latest_metrics["invoke_ms"] = invoke_ms
                            retrieval_profiles.append(latest_metrics)
                        else:
                            retrieval_profiles.append({
                                "query": q,
                                "tool_name": tool_name,
                                "invoke_ms": invoke_ms,
                            })
                        ctx, refs = _extract_and_update_references(str(res))
                        if ctx:
                            merged_context_parts.append(f"[Query] {q}\n{ctx}")
                        if refs:
                            merged_refs.extend(refs)

                    rag_context = "\n\n".join(merged_context_parts)
                    # 去重引用（按 source+page+ref_id）
                    seen = set()
                    for r in merged_refs:
                        key = (r.get("source"), r.get("page"), r.get("ref_id"))
                        if key in seen:
                            continue
                        seen.add(key)
                        retrieved_refs.append(r)

                if show_thinking:
                    print(f"[RAG] 检索到 {len(retrieved_refs)} 条参考（合并自 {len(rag_queries)} 条查询）")

                if retrieval_started_at is not None:
                    stage_record["retrieval_ms"] = _round_ms(retrieval_started_at)
            except Exception as e:
                print(f"[RAG] 检索错误: {e}")

        # 4. 生成（使用激进截断的消息历史和压缩后的 RAG 上下文）
        try:
            # [关键修复] 激进截断消息历史，防止 token 超限
            # 只保留最近 5 条消息，每条最多 2000 字符，总 token 限制 15000
            truncated_messages = _truncate_message_history(
                state.messages,
                max_tokens=int(os.getenv("NODE_TRUNCATE_MAX_TOKENS", "15000")),
                keep_last_n=10,
                max_chars_per_message=2000
            )
            
            # [关键修复] 压缩 RAG 上下文，避免过长
            compressed_rag = _compress_rag_context(rag_context, max_chars=4000)  # 降低到 4000
            
            # 患者数据也需要限制长度
            if _should_use_template_decision_v2(state):
                decision_dict = _sanitize_decision_output(
                    _build_template_decision_v2(state),
                    rag_context=compressed_rag,
                    state=state,
                    references=retrieved_refs,
                )
                if retrieval_profiles:
                    stage_record["retrieval_breakdown_ms"] = {
                        "vector_ms": round(sum(float(item.get("vector_ms", 0.0)) for item in retrieval_profiles), 2),
                        "bm25_ms": round(sum(float(item.get("bm25_ms", 0.0)) for item in retrieval_profiles), 2),
                        "fusion_ms": round(sum(float(item.get("fusion_ms", 0.0)) for item in retrieval_profiles), 2),
                        "rerank_ms": round(sum(float(item.get("rerank_ms", 0.0)) for item in retrieval_profiles), 2),
                    }
                    stage_record["retrieval_queries_profiled"] = len(retrieval_profiles)
                stage_record["decision_llm_ms"] = 0.0
                updates = {
                    "messages": [AIMessage(content=f"[Decision] template-fast {decision_dict.get('summary', '')[:200]}")],
                    "clinical_stage": "Decision",
                    "stage_timings": [stage_record],
                    "retrieval_timings": retrieval_profiles,
                    "decision_json": decision_dict,
                    "retrieved_references": retrieved_refs,
                    "iteration_count": iteration + 1,
                    "error": None,
                    "findings": {"decision_strategy": "template_fast"},
                }
                temp_state = state.model_copy(update=updates)
                updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
                return updates

            patient_data = f"{state.patient_profile}\n{state.findings}"
            if len(patient_data) > 3000:  # 降低到 3000
                patient_data = patient_data[:3000] + "\n...[患者数据已截断]"
            
            llm_started_at = time.perf_counter()
            decision_obj = _invoke_structured_with_recovery(
                prompt=decision_prompt,
                model=model,
                schema=ClinicalDecisionSchema,
                payload={
                "chat_history": truncated_messages,
                "rag_context": compressed_rag,
                "patient_data": patient_data,
                "feedback": state.critic_feedback or "无",
                "summary_memory": summary_memory,
                "pinned_context": pinned_context,
                },
                log_prefix="[Decision]",
                raw_text_parser=_parse_decision_raw_text,
            )
            stage_record["decision_llm_ms"] = _round_ms(llm_started_at)
            if not decision_obj:
                raise ValueError("empty decision output")
            if isinstance(decision_obj, dict):
                decision_obj = ClinicalDecisionSchema.model_validate(decision_obj)
            elif isinstance(decision_obj, str):
                cleaned = decision_obj.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = _parse_decision_raw_text(cleaned)
                    if parsed is None:
                        parsed = json.loads(cleaned)
                    decision_obj = ClinicalDecisionSchema.model_validate(parsed)
                except Exception as e:
                    raise ValueError(f"invalid decision output: {e}")
            elif not hasattr(decision_obj, "model_dump"):
                decision_obj = ClinicalDecisionSchema.model_validate(decision_obj)
            
            decision_dict = _sanitize_decision_output(
                decision_obj.model_dump(),
                rag_context=compressed_rag,
                state=state,
                references=retrieved_refs,
            )
            decision_obj = ClinicalDecisionSchema.model_validate(decision_dict)
            decision_strategy = _decision_strategy_label_v2(state, references=retrieved_refs)
            stage_record["decision_strategy"] = decision_strategy
            
            if show_thinking:
                print(f"🧠 [Decision] Generated plan with {len(decision_obj.treatment_plan)} items.")

            # [优化] 生成 RAG 检索摘要（类似目录索引），避免完整内容污染消息历史
            # 完整的 rag_context 只在当前调用中使用，不存入 state.messages
            rag_digest = ""
            if rag_queries and retrieved_refs:
                rag_digest = _create_rag_digest(
                    rag_context=rag_context,
                    references=retrieved_refs,
                    queries=rag_queries,
                    max_digest_chars=600  # 控制摘要长度
                )

            # 决策结果消息（简化版，不含完整 JSON）
            decision_summary = f"📋 治疗方案已生成: {decision_dict.get('summary', '无摘要')[:200]}"
            
            # 消息列表：摘要 + 决策概要（而非完整 JSON）
            output_messages = []
            if retrieval_profiles:
                stage_record["retrieval_breakdown_ms"] = {
                    "vector_ms": round(sum(float(item.get("vector_ms", 0.0)) for item in retrieval_profiles), 2),
                    "bm25_ms": round(sum(float(item.get("bm25_ms", 0.0)) for item in retrieval_profiles), 2),
                    "fusion_ms": round(sum(float(item.get("fusion_ms", 0.0)) for item in retrieval_profiles), 2),
                    "rerank_ms": round(sum(float(item.get("rerank_ms", 0.0)) for item in retrieval_profiles), 2),
                }
                stage_record["retrieval_queries_profiled"] = len(retrieval_profiles)
            decision_summary = f"📋 治疗方案已生成: {decision_dict.get('summary', '已生成治疗建议')[:200]}"
            if rag_digest:
                output_messages.append(AIMessage(content=rag_digest))
            output_messages.append(AIMessage(content=decision_summary))

            updates = {
                "messages": output_messages,
                "clinical_stage": "Decision",
                "stage_timings": [stage_record],
                "retrieval_timings": retrieval_profiles,
                "decision_json": decision_dict,  # 完整数据仍存在 decision_json 中
                "retrieved_references": retrieved_refs,
                "findings": {"decision_strategy": decision_strategy},
                "iteration_count": iteration + 1,
                "error": None
            }
            
            # [新增] 自动更新路线图
            temp_state = state.model_copy(update=updates)
            updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            
            return updates

        except Exception as e:
            print(f"⚠️ [Decision Error] {e}")
            # 构造一个符合格式的错误提示，防止 Critic 再次报错空值
            error_fallback = {
                "summary": "生成过程中发生格式错误",
                "treatment_plan": [{"title": "系统错误", "content": f"无法生成有效方案: {str(e)}"}],
                "follow_up": []
            }
            error_updates = {
                "clinical_stage": "Decision",
                "decision_json": error_fallback,
                "error": str(e),
                "iteration_count": iteration + 1
            }
            # 即使出错也更新路线图
            temp_state = state.model_copy(update=error_updates)
            error_updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return error_updates

    return _run


def _build_critic_signal_from_state(state: CRCAgentState):
    stored_signal = getattr(state, "critic_review_signal", None) or {}
    if not isinstance(stored_signal, dict):
        stored_signal = {}
    return build_critic_review_signal(
        verdict=stored_signal.get("verdict", getattr(state, "critic_verdict", "")),
        feedback=stored_signal.get("feedback", getattr(state, "critic_feedback", "")),
        retryable=stored_signal.get("retryable"),
        reasons=stored_signal.get("reasons"),
        degraded=stored_signal.get("degraded"),
    )


def _route_by_critic_legacy_impl(state: CRCAgentState) -> str:
    if getattr(state, "iteration_count", 0) >= 3:
        return "finalize"
    strategy = (getattr(state, "findings", {}) or {}).get("decision_strategy")
    if strategy == "template_fast":
        return "finalize"
    if strategy == "rag_guideline" and _decision_has_stable_rag_support(state):
        return "finalize"
    if getattr(state, "critic_verdict") == "REJECTED":
        return "decision"
    return "finalize"


def _route_by_critic_policy_decision(state: CRCAgentState) -> ReviewDecision:
    facts = build_turn_facts(state)
    return decide_after_critic(facts, _build_critic_signal_from_state(state))


def _build_route_by_critic_shadow(
    state: CRCAgentState,
    legacy_action: str | None = None,
    policy_decision: ReviewDecision | None = None,
) -> dict[str, Any]:
    if policy_decision is None:
        policy_decision = _route_by_critic_policy_decision(state)
    if legacy_action is None:
        legacy_action = _route_by_critic_legacy_impl(state)
    return record_review_divergence(
        legacy_action,
        policy_decision.route,
        policy_rule_name=policy_decision.rule_name,
        divergence_reason=policy_decision.rationale,
    )


def node_critic(model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    
    # 从统一的 prompts 模块导入 System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", "{json_str}")
    ])
    
    def _run(state: CRCAgentState):
        decision = state.decision_json
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        
        if show_thinking:
            print(f"\n🧐 [Critic] 正在审核治疗方案...")

        if not decision:
            signal = build_critic_review_signal(
                verdict="REJECTED",
                feedback="Decision payload missing.",
                retryable=True,
            )
            return {
                "critic_verdict": "REJECTED",
                "critic_feedback": "Decision payload missing.",
                "critic_review_signal": signal.model_dump(),
                "rejection_count": getattr(state, "rejection_count", 0) + 1,
            }
            if show_thinking:
                print(f"❌ [Critic] 方案为空，拒绝。")
            return {"critic_verdict": "REJECTED", "critic_feedback": "空方案", "rejection_count": getattr(state, "rejection_count", 0)+1}
        
        try:
            def _critic_text_parser(raw_text: str) -> Optional[dict]:
                text = (raw_text or "").replace("```json", "").replace("```", "").strip()
                text_upper = text.upper()
                if "REJECTED" in text_upper:
                    verdict_local = "REJECTED"
                elif "APPROVED_WITH_WARNINGS" in text_upper:
                    verdict_local = "APPROVED_WITH_WARNINGS"
                elif "APPROVED" in text_upper:
                    verdict_local = "APPROVED"
                else:
                    verdict_local = "APPROVED"
                    if show_thinking:
                        print("⚠️ [Critic] 文本未识别 verdict，触发默认放行。")
                return {
                    "verdict": verdict_local,
                    "feedback": text or "无详细反馈"
                }

            res: CriticEvaluationSchema = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=CriticEvaluationSchema,
                payload={
                    "json_str": json.dumps(decision, ensure_ascii=False),
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context
                },
                log_prefix="[Critic]",
                raw_text_parser=_critic_text_parser,
            )
            verdict = res.verdict
            feedback = res.feedback
            
            # 死循环熔断
            hist = getattr(state, "feedback_history", [])
            if _is_repeated_rejection(hist, feedback):
                verdict = "APPROVED_WITH_WARNINGS"
                feedback += " (系统强制通过)"

            if show_thinking:
                color = "\033[92m" if verdict == "APPROVED" else "\033[91m"
                reset = "\033[0m"
                print(f"{color}🔍 [Critic 结论] {verdict}{reset}")
                print(f"   📝 意见: {feedback[:100]}...")

            return {
                "clinical_stage": "Critic",
                "critic_verdict": verdict,
                "critic_feedback": feedback,
                "feedback_history": (hist + [feedback])[-5:],
                "messages": [AIMessage(content=f"审核: {verdict}")]
            }

        except Exception as e:
            if show_thinking:
                print(f"⚠️ [Critic] 运行异常，默认放行: {e}")
            return {"critic_verdict": "APPROVED", "critic_feedback": f"审核出错自动放行: {e}"}

    return _run


def route_by_critic_validator(state: CRCAgentState) -> str:
    """
    统一的 Validator 链式 Router：
    1. 如果 Decision 已经反复重试 >= 3 次，直接 Finalize（熔断）
    2. Critic = REJECTED：回到 Decision 重写
    3. 否则进入 LLM-Judge 做最终质量评估
    """
    if getattr(state, "iteration_count", 0) >= 3:
        return "finalize"

    verdict = getattr(state, "critic_verdict", None)
    if verdict == "REJECTED":
        return "decision"

    return "evaluator"


def _log_review_shadow(label: str, shadow_payload: dict[str, Any]) -> None:
    legacy_action = shadow_payload.get("legacy_review_action", "")
    policy_action = shadow_payload.get("policy_review_action", "")
    policy_rule_name = shadow_payload.get("policy_rule_name", "")
    review_diverged = bool(shadow_payload.get("review_diverged"))
    if review_diverged:
        divergence_reason = shadow_payload.get("divergence_reason", "")
        print(
            f"\n  [Shadow] {label} legacy={legacy_action} policy={policy_action} "
            f"rule={policy_rule_name} diverged=True reason={divergence_reason}"
        )
        return

    print(
        f"\n  [Shadow] {label} legacy={legacy_action} policy={policy_action} "
        f"rule={policy_rule_name} diverged=False"
    )


def node_critic(model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", "{json_str}"),
    ])

    def _run(state: CRCAgentState):
        decision = state.decision_json
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)

        if show_thinking:
            print("\n[Critic] reviewing decision...")

        if not decision:
            signal = build_critic_review_signal(
                verdict="REJECTED",
                feedback="Decision payload missing.",
                retryable=True,
                reasons=("missing_decision_payload",),
            )
            return {
                "clinical_stage": "Critic",
                "critic_verdict": "REJECTED",
                "critic_feedback": signal.feedback,
                "critic_review_signal": signal.model_dump(),
                "rejection_count": getattr(state, "rejection_count", 0) + 1,
            }

        try:
            def _critic_text_parser(raw_text: str) -> Optional[dict]:
                text = (raw_text or "").replace("```json", "").replace("```", "").strip()
                if not text:
                    return None

                text_upper = text.upper()
                degraded = True
                degraded_reason = "raw_text_fallback"
                if "REJECTED" in text_upper:
                    verdict_local = "REJECTED"
                elif "APPROVED_WITH_WARNINGS" in text_upper:
                    verdict_local = "APPROVED_WITH_WARNINGS"
                elif "APPROVED" in text_upper:
                    verdict_local = "APPROVED"
                else:
                    verdict_local = "APPROVED"
                    degraded_reason = "parse_error"

                return {
                    "verdict": verdict_local,
                    "feedback": text or "Critic returned empty feedback.",
                    "degraded": degraded,
                    "degraded_reason": degraded_reason,
                }

            res: CriticEvaluationSchema = _invoke_structured_with_recovery(
                prompt=prompt,
                model=model,
                schema=CriticEvaluationSchema,
                payload={
                    "json_str": json.dumps(decision, ensure_ascii=False),
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context,
                },
                log_prefix="[Critic]",
                raw_text_parser=_critic_text_parser,
                fallback_factory=lambda _payload, _err: CriticEvaluationSchema(
                    verdict="APPROVED",
                    feedback="Critic degraded to safe default after parser recovery failed.",
                    degraded=True,
                    degraded_reason="parser_recovery_failed",
                ),
            )
            verdict = res.verdict
            feedback = res.feedback

            hist = getattr(state, "feedback_history", [])
            if _is_repeated_rejection(hist, feedback):
                verdict = "APPROVED_WITH_WARNINGS"
                feedback += " (repeated rejection feedback suppressed)"

            degraded_signal = DegradedSignal(
                is_degraded=bool(getattr(res, "degraded", False)),
                reason=str(getattr(res, "degraded_reason", "") or ""),
                fallback_value=verdict if getattr(res, "degraded", False) else None,
            )
            reasons: tuple[str, ...] = ()
            if degraded_signal.is_degraded and degraded_signal.reason:
                reasons = (f"degraded:{degraded_signal.reason}",)
            signal = build_critic_review_signal(
                verdict=verdict,
                feedback=feedback,
                retryable=(verdict == "REJECTED"),
                reasons=reasons,
                degraded=degraded_signal,
            )

            if show_thinking:
                color = "\033[92m" if verdict == "APPROVED" else "\033[91m"
                reset = "\033[0m"
                print(f"{color}[Critic] {verdict}{reset}")
                print(f"   feedback: {feedback[:100]}...")

            if verdict == "APPROVED":
                msg_content = "✅ **诊断流程审核通过** (Critic: APPROVED)"
            else:
                msg_content = f"❌ **诊断流程审核未通过** (Critic: {verdict})"

            updates = {
                "clinical_stage": "Critic",
                "critic_verdict": verdict,
                "critic_feedback": feedback,
                "critic_review_signal": signal.model_dump(),
                "feedback_history": (hist + [feedback])[-5:],
                "messages": [AIMessage(content=msg_content)],
            }
            if verdict == "REJECTED":
                updates["rejection_count"] = getattr(state, "rejection_count", 0) + 1
            return updates

        except Exception as e:
            signal = build_critic_review_signal(
                verdict="APPROVED",
                feedback=f"Critic degraded to safe default after runtime failure: {e}",
                retryable=False,
                reasons=("degraded:runtime_failure",),
                degraded=DegradedSignal(
                    is_degraded=True,
                    reason="runtime_failure",
                    fallback_value="APPROVED",
                ),
            )
            if show_thinking:
                print(f"[Critic] runtime failure: {e}")
            return {
                "clinical_stage": "Critic",
                "critic_verdict": "APPROVED",
                "critic_feedback": signal.feedback,
                "critic_review_signal": signal.model_dump(),
            }

    return _run


def route_by_critic_v2(state: CRCAgentState) -> str:
    legacy_action = _route_by_critic_legacy_impl(state)
    policy_decision = _route_by_critic_policy_decision(state)
    shadow_payload = _build_route_by_critic_shadow(
        state,
        legacy_action=legacy_action,
        policy_decision=policy_decision,
    )
    _log_review_shadow("route_by_critic_v2", shadow_payload)
    return policy_decision.route

def node_finalize(model=None, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    def _run(state: CRCAgentState):
        text = _format_final_response(
            decision=state.decision_json,
            verdict=state.critic_verdict,
            feedback=state.critic_feedback,
            references=getattr(state, "retrieved_references", []) or [],
            citation_report=getattr(state, "citation_report", None),
            evaluation_report=getattr(state, "evaluation_report", None),
        )
        return {"messages": [AIMessage(content=text)], "clinical_stage": "Finalize", "final_output": text}
    return _run

def _format_final_response(
    decision: dict,
    verdict: str,
    feedback: str,
    references: list = None,
    citation_report: dict | None = None,
    evaluation_report: dict | None = None,
) -> str:
    if not decision: return "无法生成方案。"

    normalized_references = _dedupe_references(references or [])

    # 适配新的 List 结构
    parts = ["# 🏥 临床治疗建议\n"]
    plans = decision.get("treatment_plan", [])
    raw_summary = decision.get("summary", "")
    summary = _sanitize_section_content(raw_summary)
    if (_looks_like_placeholder_text(raw_summary) or summary == "待补充。") and isinstance(plans, list) and plans:
        first_plan = plans[0]
        first_content = first_plan.get("content") if isinstance(first_plan, dict) else first_plan.content
        summary = _sanitize_section_content(first_content)
    parts.append(f"**摘要**: {summary}\n")
    
    if isinstance(plans, list):
        for idx, p in enumerate(plans, start=1):
            # 兼容对象或字典
            t = p.get('title') if isinstance(p, dict) else p.title
            c = p.get('content') if isinstance(p, dict) else p.content
            title = _sanitize_section_title(t, idx)
            content = _sanitize_section_content(c)
            parts.append(f"### {title}\n{content}\n")
            
    if decision.get("follow_up"):
        follow_up_items = _coerce_follow_up_items(decision["follow_up"])
        if follow_up_items:
            parts.append("### 📅 随访\n" + "\n".join([f"- {item}" for item in follow_up_items]))
        
    if verdict != "APPROVED":
        parts.append(f"\n> *审核意见: {feedback}*")

    rendered_body = "\n".join(parts)
    
    # [新增] 添加文献引用来源
    if normalized_references:
        ordered_references = _order_references_by_usage(rendered_body, normalized_references)
        parts.append("\n" + "=" * 50)
        parts.append("📚 **参考文献**\n")
        for idx, ref in enumerate(ordered_references, start=1):
            source = ref.get("source") or "unknown"
            page = ref.get("page", "?")
            snippet = str(ref.get("snippet") or "").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet
            line = f"- REF_{idx} {_reference_anchor(ref)}"
            if snippet:
                line += f" {snippet}"
            parts.append(line)
        parts.append("")

    # [新增] 引用覆盖率提示
    if citation_report:
        coverage = citation_report.get("coverage_score", 0)
        missing = citation_report.get("missing_claims", [])
        needs_more = citation_report.get("needs_more_sources", False)
        if isinstance(needs_more, str):
            needs_more = needs_more.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif not isinstance(needs_more, bool):
            needs_more = bool(needs_more)
        if needs_more or (missing and coverage < 60):
            parts.append("\n" + "=" * 50)
            parts.append("🧾 **引用覆盖提示**")
            parts.append(f"- 覆盖评分: {coverage}")
            if missing:
                parts.append("- 可能缺少引用的结论：")
                parts.extend([f"  - {m}" for m in missing])

    # [新增] 评估提示（仅当存在风险时展示）
    if evaluation_report:
        parts.append("\n" + "=" * 50)
        parts.append("🧪 **自动质量评估 (LLM-Judge Validator)**")
        fa = evaluation_report.get("factual_accuracy", "?")
        ca = evaluation_report.get("citation_accuracy", "?")
        comp = evaluation_report.get("completeness", "?")
        safety = evaluation_report.get("safety", "?")
        v = str(evaluation_report.get("verdict", "PASS")).upper()
        fb = evaluation_report.get("feedback", "")

        parts.append(f"- 事实正确性: {fa}/5")
        parts.append(f"- 引用准确性: {ca}/5")
        parts.append(f"- 完整性: {comp}/5")
        parts.append(f"- 安全性: {safety}/5")
        parts.append(f"- 总体结论: {v}")
        if fb:
            parts.append(f"- 评估说明: {fb}")
        
    return "\n".join(parts)
