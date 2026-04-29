# RAG Evidence Contract Design

## Goal

Build the P0 backend-only minimum loop for clinical RAG evidence handling:

```text
retriever -> rag tool -> direct node path -> graph state -> frontend-ready fields
```

The key change is to carry retrieved guideline evidence as structured data instead of relying on LLM-written reference labels or free-form text parsing.

## Scope

This spec covers the minimum backend loop:

- Define a stable `RetrievedEvidence` contract.
- Serialize structured evidence from RAG tools.
- Parse structured evidence in the direct `use_sub_agent=False` path.
- Merge evidence into graph state fields that future frontend code can consume.
- Add a small RAG trace object for debugging and later evaluation.
- Keep existing text output and legacy reference fields compatible.

This spec does not cover:

- Frontend rendering changes.
- Parent-child chunking.
- Table-aware chunking.
- Retrieval profile tuning beyond carrying a profile name.
- Golden-query retrieval evaluation.
- Chroma/BM25 atomic rebuild.
- Parallel vector/BM25 execution.

## Current State

The current RAG stack already has the right broad shape:

- `src/rag/retriever.py` performs hybrid retrieval over Chroma and BM25, with optional reranking.
- `src/tools/rag_tools.py` exposes guideline search tools and formats retrieved documents for LLM consumption.
- `src/nodes/knowledge_nodes.py` and `src/nodes/decision_nodes.py` call RAG tools directly when `use_sub_agent=False`.
- `src/nodes/sub_agent.py` can parse `<retrieved_metadata>` in sub-agent output.
- `src/nodes/node_utils.py` has a reference extraction path for direct node responses.

The main gap is that direct node paths do not have a single structured evidence contract from tool output into graph state. Evidence can be visible in the LLM prompt while still being weak or incomplete in `retrieved_references`.

## Design Principle

Evidence should be a first-class structured object across the backend:

```text
retriever result
  -> RetrievedEvidence
  -> tool output payload
  -> node extraction
  -> graph state
  -> frontend-ready API field
```

The LLM may cite evidence, summarize it, and use reference labels, but the backend must not depend on the LLM to recreate source, page, score, or snippet metadata.

## Data Contract

Create a lightweight module:

```text
src/rag/evidence.py
```

The module should expose typed helpers based on plain dictionaries or dataclasses. Pydantic is not required for this P0 loop.

### RetrievedEvidence

Required fields:

```python
{
    "evidence_id": str,
    "chunk_id": str | None,
    "source": str | None,
    "page": int | None,
    "section": str | None,
    "text": str,
    "snippet": str,
    "query": str | None,
    "tool_name": str | None,
    "retrieval_profile": str,
    "scores": {
        "vector": float | None,
        "bm25": float | None,
        "fusion": float | None,
        "rerank": float | None,
    },
    "provenance": {
        "parse_method": str | None,
        "source_file_hash": str | None,
        "collection_version": str | None,
    },
    "frontend": {
        "title": str,
        "citation_label": str,
        "display_text": str,
    },
}
```

Rules:

- `evidence_id` must be stable within a single tool call and deterministic when enough metadata exists.
- `text` is the full retrieved content available to the LLM.
- `snippet` is a compact preview for references and frontend display.
- `frontend.display_text` must not contain synthetic enhancement text when raw citation text is available.
- Missing metadata should be represented as `None`, not omitted.

### RagTrace

Each RAG tool call should be able to emit:

```python
{
    "trace_id": str,
    "tool_name": str,
    "query": str,
    "retrieval_profile": str,
    "evidence_ids": list[str],
    "latency_ms": int | None,
    "rerank_enabled": bool | None,
    "fallback_used": bool,
}
```

P0 trace is intentionally small. It should be enough to answer: what was searched, through which tool/profile, and which evidence objects were returned.

## Tool Output Contract

`src/tools/rag_tools.py` should continue returning human-readable text for the LLM, but also include a structured evidence block:

```xml
<retrieved_evidence>
[
  {
    "evidence_id": "search_clinical_guidelines:abc123",
    "chunk_id": "abc123",
    "source": "NCCN rectal cancer v5.2025.pdf",
    "page": 12,
    "section": "Treatment",
    "text": "...",
    "snippet": "...",
    "query": "III期结肠癌辅助化疗",
    "tool_name": "search_treatment_recommendations",
    "retrieval_profile": "treatment",
    "scores": {
      "vector": 0.72,
      "bm25": 3.14,
      "fusion": 0.81,
      "rerank": 0.67
    },
    "provenance": {
      "parse_method": null,
      "source_file_hash": null,
      "collection_version": null
    },
    "frontend": {
      "title": "NCCN rectal cancer v5.2025.pdf",
      "citation_label": "NCCN rectal cancer v5.2025.pdf p.12",
      "display_text": "..."
    }
  }
]
</retrieved_evidence>
```

The existing `<retrieved_metadata>` block should remain for compatibility during P0. New extraction code should prefer `<retrieved_evidence>` and fall back to `<retrieved_metadata>`.

## Node And State Contract

Direct node paths must extract structured evidence from tool outputs and merge it into graph state.

State fields:

```python
state["retrieved_evidence"] = list[RetrievedEvidence]
state["retrieved_references"] = list[dict]
state["rag_trace"] = list[RagTrace]
```

Rules:

- `retrieved_evidence` is the canonical structured evidence list.
- `retrieved_references` remains backward-compatible for existing prompts, UI, and logs.
- `rag_trace` is append-only per request.
- Duplicate evidence should be deduplicated by `evidence_id`.
- If `retrieved_evidence` exists, references should be derived from it rather than reparsed from LLM prose.

Compatibility reference shape:

```python
{
    "source": str | None,
    "page": int | None,
    "section": str | None,
    "snippet": str,
    "score": float | None,
    "ref_id": str,
    "evidence_id": str,
}
```

## Integration Points

### `src/rag/evidence.py`

Responsibilities:

- Build evidence objects from retrieved document objects.
- Normalize missing fields.
- Generate `evidence_id`.
- Serialize and parse `<retrieved_evidence>` blocks.
- Convert evidence to legacy reference dictionaries.
- Deduplicate evidence lists.

### `src/tools/rag_tools.py`

Responsibilities:

- Convert each retrieved document to `RetrievedEvidence`.
- Include `<retrieved_evidence>` in every RAG tool result.
- Preserve existing readable context format for LLM answer quality.
- Set `retrieval_profile` based on tool:
  - `search_clinical_guidelines`: `general`
  - `search_treatment_recommendations`: `treatment`
  - `search_staging_criteria`: `staging`
  - `search_drug_information`: `drug`
  - `list_guideline_toc`: `toc`
  - `read_guideline_chapter`: `chapter`

### `src/nodes/node_utils.py`

Responsibilities:

- Parse `<retrieved_evidence>` from direct tool outputs.
- Fall back to existing reference parsing when structured evidence is absent.
- Merge evidence into state without losing existing references.

### `src/nodes/knowledge_nodes.py`

Responsibilities:

- Preserve direct tool output evidence when knowledge retrieval runs without sub-agents.
- Ensure `retrieved_evidence`, `retrieved_references`, and `rag_trace` are present in the returned state.

### `src/nodes/decision_nodes.py`

Responsibilities:

- Preserve evidence returned during internal decision RAG.
- Ensure treatment decision output includes state-level evidence and trace.
- Keep existing decision prompt behavior unchanged for P0.

## Error Handling

Structured evidence extraction must be fail-soft:

- Malformed JSON in `<retrieved_evidence>` should not break the node.
- The parser should return an empty list on invalid payload and allow legacy parsing to continue.
- Missing metadata should produce valid evidence with `None` fields.
- Trace generation failure should not block answer generation.

## Testing Strategy

Use test-first implementation for the P0 loop.

Required tests:

1. Evidence serialization round trip
   - Build two evidence dictionaries.
   - Serialize into `<retrieved_evidence>`.
   - Parse them back.
   - Assert `evidence_id`, `source`, `page`, `snippet`, and `retrieval_profile` are preserved.

2. Legacy reference conversion
   - Convert evidence to references.
   - Assert reference dictionaries contain `source`, `page`, `snippet`, `ref_id`, and `evidence_id`.

3. Tool formatting includes structured evidence
   - Pass fake retrieved documents into the RAG tool formatter.
   - Assert output contains both readable context and `<retrieved_evidence>`.

4. Direct extraction prefers structured evidence
   - Pass a direct tool output containing `<retrieved_evidence>` into node extraction.
   - Assert `retrieved_evidence` and `retrieved_references` are populated without depending on LLM prose citations.

5. Malformed evidence payload is fail-soft
   - Pass invalid JSON inside `<retrieved_evidence>`.
   - Assert parser returns no evidence and no exception escapes.

## Migration Strategy

Implement this in a backward-compatible way:

1. Add `src/rag/evidence.py`.
2. Add tests for the evidence helpers before production changes.
3. Update RAG tool formatting to emit `<retrieved_evidence>` while keeping existing text and metadata blocks.
4. Update direct extraction to prefer structured evidence.
5. Update knowledge and decision state returns only where needed to preserve evidence fields.
6. Run focused tests first, then the existing relevant test suite.

No database migration or index rebuild is required for P0.

## Frontend Readiness

Frontend can later consume:

```python
state["retrieved_evidence"]
state["rag_trace"]
```

Expected frontend behavior in a later phase:

- Show evidence cards using `frontend.title`, `frontend.citation_label`, and `snippet`.
- Use `evidence_id` for stable expand/collapse interactions.
- Use `rag_trace` for debug panels or clinician audit views.

The P0 backend should not require frontend changes.

## Risks And Mitigations

Risk: tool output becomes too large.

Mitigation: keep `snippet` compact and allow `text` to be trimmed to the same content already passed to the LLM.

Risk: duplicate evidence appears across multiple tool calls.

Mitigation: deduplicate by `evidence_id` in `src/rag/evidence.py`.

Risk: old UI or prompt code expects `retrieved_references`.

Mitigation: keep `retrieved_references` and derive it from `retrieved_evidence`.

Risk: metadata fields are incomplete in existing chunks.

Mitigation: use `None` for unavailable provenance fields and avoid requiring index rebuild in P0.

## Acceptance Criteria

P0 is complete when:

- RAG tool output includes `<retrieved_evidence>` for guideline search results.
- Direct `use_sub_agent=False` paths can populate `retrieved_evidence`.
- `retrieved_references` remains populated for compatibility.
- `rag_trace` exists for RAG tool calls where data is available.
- Malformed evidence payloads do not break answer generation.
- Focused tests cover serialization, parsing, conversion, tool formatting, and fail-soft behavior.
