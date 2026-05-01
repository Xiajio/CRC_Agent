from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePath
from typing import Any, Iterable, Mapping


TOOL_RETRIEVAL_PROFILES: dict[str, str] = {
    "search_clinical_guidelines": "general",
    "search_treatment_recommendations": "treatment",
    "search_staging_criteria": "staging",
    "search_drug_information": "drug",
    "search_by_guideline_source": "source_filter",
    "hybrid_guideline_search": "hybrid",
    "list_guideline_toc": "toc",
    "read_guideline_chapter": "chapter",
}

EVIDENCE_BLOCK_RE = re.compile(
    r"<retrieved_evidence>(.*?)</retrieved_evidence>",
    re.DOTALL | re.IGNORECASE,
)
METADATA_BLOCK_RE = re.compile(
    r"<retrieved_metadata>(.*?)</retrieved_metadata>",
    re.DOTALL | re.IGNORECASE,
)

_MISSING_TEXT = {"", "none", "null", "n/a", "na", "unknown"}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING_TEXT else text


def _source_basename(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    return PurePath(text.replace("\\", "/")).name or text


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT:
        return None
    match = re.search(r"[+-]?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _snippet(text: Any, limit: int = 220) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _hash_id(parts: Iterable[Any]) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _make_evidence_id(
    *,
    tool_name: str | None,
    chunk_id: Any,
    source: Any,
    page: Any,
    section: Any,
    text: Any,
    index: int | None = None,
) -> str:
    prefix = _clean_text(tool_name) or "rag"
    stable_part = chunk_id or _hash_id([source, page, section, str(text or "")[:500], index])
    digest = _hash_id([stable_part])
    return f"{prefix}:{digest}"


def normalize_evidence(item: Mapping[str, Any] | Any, *, index: int | None = None) -> dict[str, Any]:
    data = dict(item) if isinstance(item, Mapping) else {}
    scores = data.get("scores") if isinstance(data.get("scores"), Mapping) else {}
    provenance = data.get("provenance") if isinstance(data.get("provenance"), Mapping) else {}
    frontend = data.get("frontend") if isinstance(data.get("frontend"), Mapping) else {}

    source = _source_basename(data.get("source"))
    page = _coerce_int(data.get("page"))
    section = _clean_text(data.get("section"))
    text = str(data.get("text") or data.get("content") or data.get("snippet") or data.get("preview") or "")
    snippet = _snippet(data.get("snippet") or data.get("preview") or text)
    tool_name = _clean_text(data.get("tool_name"))
    retrieval_profile = _clean_text(data.get("retrieval_profile")) or "general"
    evidence_id = _clean_text(data.get("evidence_id")) or _make_evidence_id(
        tool_name=tool_name,
        chunk_id=data.get("chunk_id"),
        source=source,
        page=page,
        section=section,
        text=text,
        index=index,
    )

    title = _clean_text(frontend.get("title")) or source or "Unknown source"
    citation_label = _clean_text(frontend.get("citation_label"))
    if citation_label is None:
        citation_label = f"{title} p.{page}" if page is not None else title

    return {
        "evidence_id": evidence_id,
        "chunk_id": _clean_text(data.get("chunk_id")),
        "source": source,
        "page": page,
        "section": section,
        "text": text,
        "snippet": snippet,
        "query": _clean_text(data.get("query")),
        "tool_name": tool_name,
        "retrieval_profile": retrieval_profile,
        "scores": {
            "vector": _coerce_float(scores.get("vector")),
            "bm25": _coerce_float(scores.get("bm25")),
            "fusion": _coerce_float(scores.get("fusion")),
            "rerank": _coerce_float(scores.get("rerank")),
        },
        "provenance": {
            "parse_method": _clean_text(provenance.get("parse_method")),
            "source_file_hash": _clean_text(provenance.get("source_file_hash")),
            "collection_version": _clean_text(provenance.get("collection_version")),
        },
        "frontend": {
            "title": title,
            "citation_label": citation_label,
            "display_text": _clean_text(frontend.get("display_text")) or snippet,
        },
    }


def build_evidence_from_document(
    document: Any,
    *,
    index: int,
    query: str | None = None,
    tool_name: str | None = None,
    retrieval_profile: str | None = None,
) -> dict[str, Any]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    content = str(getattr(document, "page_content", "") or "").strip()
    source = _source_basename(metadata.get("source"))
    page = _coerce_int(metadata.get("page"))
    section = (
        _clean_text(metadata.get("section"))
        or _clean_text(metadata.get("section_h3"))
        or _clean_text(metadata.get("section_h2"))
    )
    profile = retrieval_profile or TOOL_RETRIEVAL_PROFILES.get(str(tool_name or ""), "general")

    scores = {
        "vector": _coerce_float(_first_present(metadata.get("vector_score"), metadata.get("vector"))),
        "bm25": _coerce_float(_first_present(metadata.get("bm25_score"), metadata.get("bm25"))),
        "fusion": _coerce_float(
            _first_present(
                metadata.get("fusion_score"),
                metadata.get("combined_score"),
                metadata.get("score"),
                metadata.get("relevance"),
            )
        ),
        "rerank": _coerce_float(_first_present(metadata.get("rerank_score"), metadata.get("rerank"))),
    }
    snippet = _snippet(content)
    title = _clean_text(metadata.get("doc_title")) or source or "Unknown source"

    return normalize_evidence(
        {
            "chunk_id": metadata.get("chunk_id") or metadata.get("id") or metadata.get("doc_id"),
            "source": source,
            "page": page,
            "section": section,
            "text": content,
            "snippet": snippet,
            "query": query,
            "tool_name": tool_name,
            "retrieval_profile": profile,
            "scores": scores,
            "provenance": {
                "parse_method": metadata.get("parse_method"),
                "source_file_hash": metadata.get("source_file_hash"),
                "collection_version": metadata.get("collection_version"),
            },
            "frontend": {
                "title": title,
                "citation_label": f"{title} p.{page}" if page is not None else title,
                "display_text": snippet,
            },
        },
        index=index,
    )


def serialize_retrieved_evidence(evidence: Iterable[Mapping[str, Any]]) -> str:
    normalized = [normalize_evidence(item, index=i) for i, item in enumerate(evidence, start=1)]
    return f"<retrieved_evidence>{json.dumps(normalized, ensure_ascii=False)}</retrieved_evidence>"


def _load_json_list(payload: str) -> list[Any]:
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_retrieved_evidence(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    blocks = EVIDENCE_BLOCK_RE.findall(str(text))
    if not blocks:
        stripped = str(text).strip()
        if stripped.startswith("["):
            blocks = [stripped]
    evidence: list[dict[str, Any]] = []
    for block in blocks:
        for index, item in enumerate(_load_json_list(block), start=1):
            if isinstance(item, Mapping):
                evidence.append(normalize_evidence(item, index=index))
    return evidence


def parse_retrieved_metadata(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    metadata: list[dict[str, Any]] = []
    for block in METADATA_BLOCK_RE.findall(str(text)):
        for item in _load_json_list(block):
            if isinstance(item, Mapping):
                metadata.append(dict(item))
    return metadata


def metadata_to_evidence(
    metadata: Iterable[Mapping[str, Any]],
    *,
    query: str | None = None,
    tool_name: str | None = None,
    retrieval_profile: str | None = None,
) -> list[dict[str, Any]]:
    profile = retrieval_profile or TOOL_RETRIEVAL_PROFILES.get(str(tool_name or ""), "general")
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(metadata, start=1):
        source = _source_basename(item.get("source") or item.get("title"))
        page = _coerce_int(item.get("page"))
        preview = item.get("preview") or item.get("snippet") or item.get("content") or ""
        score = _coerce_float(_first_present(item.get("score"), item.get("relevance")))
        evidence.append(
            normalize_evidence(
                {
                    "evidence_id": item.get("evidence_id"),
                    "chunk_id": item.get("chunk_id") or item.get("ref_id"),
                    "source": source,
                    "page": page,
                    "section": item.get("section"),
                    "text": preview,
                    "snippet": preview,
                    "query": item.get("query") or query,
                    "tool_name": item.get("tool_name") or tool_name,
                    "retrieval_profile": item.get("retrieval_profile") or profile,
                    "scores": {
                        "vector": item.get("vector_score"),
                        "bm25": item.get("bm25_score"),
                        "fusion": score,
                        "rerank": item.get("rerank_score"),
                    },
                    "provenance": {
                        "parse_method": item.get("parse_method"),
                        "source_file_hash": item.get("source_file_hash"),
                        "collection_version": item.get("collection_version"),
                    },
                    "frontend": {
                        "title": source or "Unknown source",
                        "citation_label": f"{source} p.{page}" if source and page is not None else source,
                        "display_text": _snippet(preview),
                    },
                },
                index=index,
            )
        )
    return evidence


def evidence_to_references(evidence: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        normalized = normalize_evidence(item, index=index)
        scores = normalized["scores"]
        score = _first_present(
            scores.get("rerank"),
            scores.get("fusion"),
            scores.get("vector"),
            scores.get("bm25"),
        )
        source = normalized.get("source")
        page = normalized.get("page")
        ref_id = f"{source}:{page}" if source and page is not None else normalized["evidence_id"]
        references.append(
            {
                "source": source,
                "page": page,
                "section": normalized.get("section"),
                "snippet": normalized.get("snippet") or "",
                "score": score,
                "ref_id": ref_id,
                "evidence_id": normalized["evidence_id"],
            }
        )
    return references


def dedupe_evidence_by_id(evidence: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence, start=1):
        normalized = normalize_evidence(item, index=index)
        merged[normalized["evidence_id"]] = normalized
    return list(merged.values())


def build_rag_trace(
    *,
    tool_name: str | None,
    query: str | None,
    retrieval_profile: str | None,
    evidence: Iterable[Mapping[str, Any]],
    latency_ms: int | None = None,
    rerank_enabled: bool | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    normalized = [normalize_evidence(item, index=i) for i, item in enumerate(evidence, start=1)]
    evidence_ids = [item["evidence_id"] for item in normalized]
    profile = retrieval_profile or (normalized[0]["retrieval_profile"] if normalized else "general")
    name = tool_name or (normalized[0].get("tool_name") if normalized else None) or "rag"
    q = query or (normalized[0].get("query") if normalized else None) or ""
    return {
        "trace_id": f"{name}:{_hash_id([q, profile, ','.join(evidence_ids)])}",
        "tool_name": name,
        "query": q,
        "retrieval_profile": profile,
        "evidence_ids": evidence_ids,
        "latency_ms": latency_ms,
        "rerank_enabled": rerank_enabled,
        "fallback_used": bool(fallback_used),
    }


def build_rag_traces_from_evidence(evidence: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}
    for index, item in enumerate(evidence, start=1):
        normalized = normalize_evidence(item, index=index)
        key = (
            normalized.get("tool_name"),
            normalized.get("query"),
            normalized.get("retrieval_profile"),
        )
        groups.setdefault(key, []).append(normalized)
    return [
        build_rag_trace(
            tool_name=tool_name,
            query=query,
            retrieval_profile=profile,
            evidence=items,
            latency_ms=None,
            rerank_enabled=None,
            fallback_used=False,
        )
        for (tool_name, query, profile), items in groups.items()
        if items
    ]


def strip_retrieval_payload_blocks(text: str | None) -> str:
    cleaned = EVIDENCE_BLOCK_RE.sub("", str(text or ""))
    cleaned = METADATA_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()
