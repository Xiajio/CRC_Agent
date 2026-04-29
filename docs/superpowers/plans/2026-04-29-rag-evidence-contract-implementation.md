# RAG Evidence Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry clinical RAG evidence as structured data from RAG tool output into direct node state, while preserving legacy references and preparing frontend-readable fields.

**Architecture:** Add focused evidence helpers in `src/rag/evidence.py`, extend `CRCAgentState` and `RetrievedReference`, then wire the helpers into RAG tool formatting and direct/sub-agent extraction. Keep existing prompt-facing text unchanged and use `<retrieved_evidence>` as the canonical machine-readable payload.

**Tech Stack:** Python 3.10+, Pydantic v2, LangChain `Document`, pytest.

---

## File Map

- Create `src/rag/evidence.py`: evidence construction, XML serialization/parsing, reference conversion, trace creation, and deduplication helpers.
- Modify `src/state.py`: add `merge_evidence_by_id`, `retrieved_evidence`, `rag_trace`, and optional `evidence_id`/`section` on `RetrievedReference`.
- Modify `src/tools/rag_tools.py`: emit `<retrieved_evidence>` alongside existing `<retrieved_metadata>` and assign retrieval profiles by tool.
- Modify `src/nodes/node_utils.py`: expose structured extraction helpers and keep legacy reference extraction compatible.
- Modify `src/nodes/sub_agent.py`: prefer `<retrieved_evidence>` and fall back to `<retrieved_metadata>`.
- Modify `src/nodes/knowledge_nodes.py`: return structured evidence and trace from direct local RAG tool calls.
- Modify `src/nodes/decision_nodes.py`: return structured evidence and trace from direct decision RAG tool calls.
- Add `tests/backend/test_rag_evidence_contract.py`: focused helper, formatter, parser, state reducer, and sub-agent extraction tests.

---

### Task 1: Evidence Helper Red Tests

**Files:**
- Create: `tests/backend/test_rag_evidence_contract.py`
- Create later: `src/rag/evidence.py`

- [ ] **Step 1: Write failing evidence helper tests**

Add tests that import the wished-for helper API:

```python
from langchain_core.documents import Document

from src.rag.evidence import (
    build_evidence_from_document,
    dedupe_evidence,
    evidence_to_references,
    extract_evidence_block,
    make_rag_trace,
    serialize_evidence_block,
)


def test_evidence_serialization_round_trip_preserves_core_fields() -> None:
    evidence = [
        {
            "evidence_id": "search:e1",
            "chunk_id": "chunk-1",
            "source": "NCCN.pdf",
            "page": 12,
            "section": "Treatment",
            "text": "Full guideline text",
            "snippet": "Full guideline text",
            "query": "stage III treatment",
            "tool_name": "search_treatment_recommendations",
            "retrieval_profile": "treatment",
            "scores": {"vector": 0.7, "bm25": 2.0, "fusion": 0.8, "rerank": 0.9},
            "provenance": {"parse_method": None, "source_file_hash": None, "collection_version": None},
            "frontend": {"title": "NCCN.pdf", "citation_label": "NCCN.pdf p.12", "display_text": "Full guideline text"},
        }
    ]

    payload = serialize_evidence_block(evidence)
    parsed = extract_evidence_block(payload)

    assert parsed == evidence


def test_build_evidence_from_document_normalizes_metadata_and_frontend_fields() -> None:
    doc = Document(
        page_content="Guideline text about adjuvant chemotherapy.",
        metadata={
            "source": r"C:\\guidelines\\NCCN.pdf",
            "page": "12",
            "chunk_id": "chunk-12",
            "section": "Treatment",
            "score": 0.42,
            "bm25_score": 2.5,
            "fusion_score": 0.8,
            "rerank_score": 0.9,
        },
    )

    evidence = build_evidence_from_document(
        doc,
        index=1,
        query="stage III treatment",
        tool_name="search_treatment_recommendations",
        retrieval_profile="treatment",
    )

    assert evidence["evidence_id"] == "search_treatment_recommendations:chunk-12"
    assert evidence["source"] == "NCCN.pdf"
    assert evidence["page"] == 12
    assert evidence["retrieval_profile"] == "treatment"
    assert evidence["scores"]["bm25"] == 2.5
    assert evidence["frontend"]["citation_label"] == "NCCN.pdf p.12"


def test_evidence_to_references_preserves_evidence_id() -> None:
    references = evidence_to_references([
        {
            "evidence_id": "e1",
            "source": "NCCN.pdf",
            "page": 12,
            "section": "Treatment",
            "snippet": "Preview",
            "scores": {"rerank": 0.9, "fusion": 0.8},
            "frontend": {"title": "NCCN.pdf"},
        }
    ])

    assert references == [
        {
            "source": "NCCN.pdf",
            "page": 12,
            "section": "Treatment",
            "snippet": "Preview",
            "score": 0.9,
            "ref_id": "e1",
            "source_id": "e1",
            "title": "NCCN.pdf",
            "evidence_id": "e1",
        }
    ]


def test_malformed_evidence_payload_is_fail_soft() -> None:
    assert extract_evidence_block("<retrieved_evidence>{bad json</retrieved_evidence>") == []


def test_dedupe_evidence_later_value_wins() -> None:
    merged = dedupe_evidence([
        {"evidence_id": "e1", "snippet": "old"},
        {"evidence_id": "e1", "snippet": "new"},
        {"evidence_id": "e2", "snippet": "other"},
    ])

    assert merged == [
        {"evidence_id": "e1", "snippet": "new"},
        {"evidence_id": "e2", "snippet": "other"},
    ]


def test_make_rag_trace_uses_evidence_ids() -> None:
    trace = make_rag_trace(
        tool_name="search_clinical_guidelines",
        query="MSI-H",
        retrieval_profile="general",
        evidence=[{"evidence_id": "e1"}],
        latency_ms=15,
        rerank_enabled=True,
        fallback_used=False,
    )

    assert trace["tool_name"] == "search_clinical_guidelines"
    assert trace["evidence_ids"] == ["e1"]
    assert trace["latency_ms"] == 15
```

- [ ] **Step 2: Run helper tests to verify RED**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.evidence'`.

---

### Task 2: Evidence Helper Implementation

**Files:**
- Create: `src/rag/evidence.py`
- Test: `tests/backend/test_rag_evidence_contract.py`

- [ ] **Step 1: Implement minimal evidence helpers**

Implement:

```python
def build_evidence_from_document(doc, *, index, query, tool_name, retrieval_profile): ...
def serialize_evidence_block(evidence): ...
def extract_evidence_block(text): ...
def evidence_to_references(evidence): ...
def dedupe_evidence(items): ...
def make_rag_trace(...): ...
```

Use standard JSON, compact snippets, deterministic IDs, and fail-soft parsing.

- [ ] **Step 2: Run helper tests to verify GREEN**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: helper tests pass.

---

### Task 3: State Contract Red/Green

**Files:**
- Modify: `tests/backend/test_rag_evidence_contract.py`
- Modify: `src/state.py`

- [ ] **Step 1: Add failing state tests**

Add:

```python
from src.state import RetrievedReference, merge_evidence_by_id


def test_merge_evidence_by_id_deduplicates_and_keeps_later_value() -> None:
    merged = merge_evidence_by_id(
        [{"evidence_id": "e1", "snippet": "old"}],
        [{"evidence_id": "e1", "snippet": "new"}, {"evidence_id": "e2", "snippet": "other"}],
    )

    assert merged == [
        {"evidence_id": "e1", "snippet": "new"},
        {"evidence_id": "e2", "snippet": "other"},
    ]


def test_retrieved_reference_preserves_evidence_id_and_section() -> None:
    ref = RetrievedReference.model_validate(
        {
            "source": "NCCN.pdf",
            "page": 12,
            "section": "Treatment",
            "snippet": "Preview",
            "score": 0.8,
            "evidence_id": "e1",
        }
    )

    assert ref.source_id == "NCCN.pdf:12"
    assert ref.evidence_id == "e1"
    assert ref.section == "Treatment"
```

- [ ] **Step 2: Run state tests to verify RED**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: FAIL because `merge_evidence_by_id` or fields are missing.

- [ ] **Step 3: Implement state changes**

In `src/state.py`:

- Add `merge_evidence_by_id`.
- Add `evidence_id: Optional[str] = None` and `section: Optional[str] = None` to `RetrievedReference`.
- Pass `evidence_id` and `section` through `_normalize_legacy_reference`.
- Add `retrieved_evidence: Annotated[List[Dict[str, Any]], merge_evidence_by_id]`.
- Add `rag_trace: Annotated[List[Dict[str, Any]], append_list]`.

- [ ] **Step 4: Run state tests to verify GREEN**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: tests pass.

---

### Task 4: Tool Formatting Red/Green

**Files:**
- Modify: `tests/backend/test_rag_evidence_contract.py`
- Modify: `src/tools/rag_tools.py`

- [ ] **Step 1: Add failing `_format_docs` test**

Add:

```python
from src.rag.evidence import extract_evidence_block
from src.tools.rag_tools import _format_docs


def test_format_docs_emits_retrieved_evidence_and_legacy_metadata() -> None:
    doc = Document(
        page_content="Guideline text",
        metadata={"source": "NCCN.pdf", "page": 3, "chunk_id": "chunk-3", "score": 0.5},
    )

    output = _format_docs(
        [doc],
        tool_name="search_clinical_guidelines",
        query="guideline question",
        retrieval_profile="general",
    )

    evidence = extract_evidence_block(output)
    assert "[REF_1] [[Source:NCCN.pdf|Page:3]]" in output
    assert "<retrieved_metadata>" in output
    assert evidence[0]["evidence_id"] == "search_clinical_guidelines:chunk-3"
    assert evidence[0]["retrieval_profile"] == "general"
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py::test_format_docs_emits_retrieved_evidence_and_legacy_metadata -q`

Expected: FAIL because `_format_docs` does not accept tool/query/profile arguments and does not emit `<retrieved_evidence>`.

- [ ] **Step 3: Update tool formatter and callers**

Change `_format_docs` signature to:

```python
def _format_docs(
    docs: List[Document],
    include_metadata: bool = True,
    *,
    tool_name: str = "search_clinical_guidelines",
    query: str | None = None,
    retrieval_profile: str = "general",
) -> str:
```

Add profile constants for all RAG tools, including `source_filter` and `hybrid`. Each tool `_run` should pass its own name, query, and profile.

- [ ] **Step 4: Run formatter tests to verify GREEN**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: tests pass.

---

### Task 5: Direct And Sub-Agent Extraction Red/Green

**Files:**
- Modify: `tests/backend/test_rag_evidence_contract.py`
- Modify: `src/nodes/node_utils.py`
- Modify: `src/nodes/sub_agent.py`

- [ ] **Step 1: Add failing extraction tests**

Add:

```python
from src.nodes.node_utils import _extract_structured_evidence, _extract_and_update_references
from src.nodes.sub_agent import SubAgentContext


def test_direct_extraction_prefers_retrieved_evidence() -> None:
    payload = serialize_evidence_block([
        {
            "evidence_id": "e1",
            "source": "NCCN.pdf",
            "page": 4,
            "section": "Treatment",
            "snippet": "Evidence preview",
            "scores": {"rerank": 0.7},
            "frontend": {"title": "NCCN.pdf"},
        }
    ])

    cleaned, refs, evidence = _extract_structured_evidence("visible context\n" + payload)

    assert "retrieved_evidence" not in cleaned
    assert evidence[0]["evidence_id"] == "e1"
    assert refs[0]["evidence_id"] == "e1"


def test_direct_extraction_falls_back_to_legacy_metadata() -> None:
    text = '<retrieved_metadata>[{"ref_id":"REF_1","source":"NCCN.pdf","page":5,"preview":"Legacy"}]</retrieved_metadata>'

    cleaned, refs, evidence = _extract_structured_evidence(text)

    assert cleaned == ""
    assert evidence == []
    assert refs[0]["source"] == "NCCN.pdf"


def test_legacy_extract_and_update_references_keeps_backward_compatible_tuple() -> None:
    cleaned, refs = _extract_and_update_references("[1] source=NCCN.pdf\nSnippet")

    assert cleaned == ""
    assert refs == [{"index": 1, "source": "NCCN.pdf", "snippet": "Snippet"}]


def test_sub_agent_extract_references_prefers_retrieved_evidence() -> None:
    agent = SubAgentContext(model=None, system_prompt="", task_description="")
    payload = serialize_evidence_block([
        {
            "evidence_id": "e1",
            "source": "NCCN.pdf",
            "page": 4,
            "snippet": "Evidence preview",
            "scores": {"rerank": 0.7},
            "frontend": {"title": "NCCN.pdf"},
        }
    ])

    agent._extract_references(payload + '<retrieved_metadata>[{"source":"Legacy.pdf"}]</retrieved_metadata>')

    assert agent._collected_references[0]["evidence_id"] == "e1"
    assert agent._collected_references[0]["source"] == "NCCN.pdf"
```

- [ ] **Step 2: Run extraction tests to verify RED**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: FAIL because `_extract_structured_evidence` does not exist and sub-agent does not parse `<retrieved_evidence>`.

- [ ] **Step 3: Implement structured extraction**

In `src/nodes/node_utils.py`, add:

```python
def _extract_structured_evidence(content: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    ...
```

Preference order:

1. `<retrieved_evidence>`
2. `<retrieved_metadata>`
3. legacy `[1] source=...`

Keep `_extract_and_update_references` returning its existing two-tuple.

In `src/nodes/sub_agent.py`, update `_extract_references` to prefer structured evidence and convert it to references.

- [ ] **Step 4: Run extraction tests to verify GREEN**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py -q`

Expected: tests pass.

---

### Task 6: Knowledge And Decision State Wiring

**Files:**
- Modify: `src/nodes/knowledge_nodes.py`
- Modify: `src/nodes/decision_nodes.py`
- Optional Test: `tests/backend/test_rag_evidence_contract.py`

- [ ] **Step 1: Replace direct extraction calls**

Where direct RAG tool output currently uses:

```python
context, refs = _extract_and_update_references(str(raw_res))
```

change to:

```python
context, refs, evidence = _extract_structured_evidence(str(raw_res))
```

Accumulate `evidence` in local lists next to existing `refs`.

- [ ] **Step 2: Return state evidence fields**

When knowledge or decision nodes return local RAG references, include:

```python
"retrieved_evidence": accumulated_evidence,
"retrieved_references": accumulated_refs,
"rag_trace": accumulated_trace,
```

Use `make_rag_trace` where tool name/query/profile/evidence are available. If timing is not available in the direct path, set `latency_ms` to `None`.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py tests/backend/test_node_utils_streaming.py -q`

Expected: tests pass.

---

### Task 7: Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused backend tests**

Run: `python -m pytest tests/backend/test_rag_evidence_contract.py tests/backend/test_rag_retriever.py tests/backend/test_node_utils_streaming.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Inspect diff**

Run: `git diff --stat`

Expected: only planned files changed.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add src/rag/evidence.py src/state.py src/tools/rag_tools.py src/nodes/node_utils.py src/nodes/sub_agent.py src/nodes/knowledge_nodes.py src/nodes/decision_nodes.py tests/backend/test_rag_evidence_contract.py docs/superpowers/plans/2026-04-29-rag-evidence-contract-implementation.md
git commit -m "feat: carry structured RAG evidence through nodes"
```
