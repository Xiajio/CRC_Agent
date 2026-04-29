from __future__ import annotations

import json
import re
from pathlib import PurePath
from typing import Any, Iterable
from uuid import uuid4


EVIDENCE_BLOCK_PATTERN = re.compile(
    r"<retrieved_evidence>(.*?)</retrieved_evidence>",
    re.DOTALL | re.IGNORECASE,
)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _basename(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return PurePath(text.replace("\\", "/")).name or text


def _snippet(text: str, limit: int = 500) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _score_from(scores: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _coerce_float(scores.get(name))
        if value is not None:
            return value
    return None


def build_evidence_from_document(
    doc: Any,
    *,
    index: int,
    query: str | None,
    tool_name: str | None,
    retrieval_profile: str = "general",
) -> dict[str, Any]:
    meta = dict(getattr(doc, "metadata", {}) or {})
    content = str(getattr(doc, "page_content", "") or "").strip()
    source = _basename(meta.get("source") or meta.get("file_name") or meta.get("doc_title"))
    page = _coerce_int(meta.get("page") or meta.get("page_number"))
    chunk_id = meta.get("chunk_id") or meta.get("id") or meta.get("doc_id")
    chunk_id = str(chunk_id).strip() if chunk_id is not None else None
    tool = str(tool_name or "rag").strip() or "rag"
    evidence_id = f"{tool}:{chunk_id}" if chunk_id else f"{tool}:{source or 'unknown'}:{page or index}:{index}"

    scores = {
        "vector": _coerce_float(meta.get("vector_score", meta.get("score"))),
        "bm25": _coerce_float(meta.get("bm25_score")),
        "fusion": _coerce_float(meta.get("fusion_score")),
        "rerank": _coerce_float(meta.get("rerank_score")),
    }

    title = source or "Unknown source"
    citation_label = f"{title} p.{page}" if page is not None else title
    preview = _snippet(content)

    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "source": source,
        "page": page,
        "section": meta.get("section") or meta.get("section_title") or meta.get("chapter"),
        "text": content,
        "snippet": preview,
        "query": query,
        "tool_name": tool_name,
        "retrieval_profile": retrieval_profile,
        "scores": scores,
        "provenance": {
            "parse_method": meta.get("parse_method"),
            "source_file_hash": meta.get("source_file_hash") or meta.get("file_hash"),
            "collection_version": meta.get("collection_version") or meta.get("index_version"),
        },
        "frontend": {
            "title": title,
            "citation_label": citation_label,
            "display_text": content,
        },
    }


def serialize_evidence_block(evidence: Iterable[dict[str, Any]]) -> str:
    return (
        "<retrieved_evidence>"
        + json.dumps(list(evidence or []), ensure_ascii=False)
        + "</retrieved_evidence>"
    )


def extract_evidence_block(text: str) -> list[dict[str, Any]]:
    match = EVIDENCE_BLOCK_PATTERN.search(text or "")
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def strip_evidence_block(text: str) -> str:
    return EVIDENCE_BLOCK_PATTERN.sub("", text or "").strip()


def dedupe_evidence(items: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    anonymous = 0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id:
            anonymous += 1
            evidence_id = f"anon:{anonymous}"
            item = {**item, "evidence_id": evidence_id}
        merged[evidence_id] = item
    return list(merged.values())


def evidence_to_references(evidence: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        frontend = item.get("frontend") if isinstance(item.get("frontend"), dict) else {}
        evidence_id = str(item.get("evidence_id") or "").strip()
        score = _score_from(scores, "rerank", "fusion", "vector", "bm25")
        references.append(
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "section": item.get("section"),
                "snippet": item.get("snippet") or "",
                "score": score,
                "ref_id": evidence_id,
                "source_id": evidence_id,
                "title": frontend.get("title") or item.get("source") or "",
                "evidence_id": evidence_id,
            }
        )
    return references


def make_rag_trace(
    *,
    tool_name: str,
    query: str,
    retrieval_profile: str,
    evidence: Iterable[dict[str, Any]],
    latency_ms: int | None = None,
    rerank_enabled: bool | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "trace_id": uuid4().hex,
        "tool_name": tool_name,
        "query": query,
        "retrieval_profile": retrieval_profile,
        "evidence_ids": [
            str(item.get("evidence_id"))
            for item in evidence or []
            if isinstance(item, dict) and item.get("evidence_id")
        ],
        "latency_ms": latency_ms,
        "rerank_enabled": rerank_enabled,
        "fallback_used": fallback_used,
    }
