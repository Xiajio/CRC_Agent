from __future__ import annotations

from langchain_core.documents import Document

from src.nodes.node_utils import _extract_and_update_references, _extract_structured_evidence
from src.nodes.sub_agent import SubAgentContext
from src.rag.evidence import (
    build_evidence_from_document,
    dedupe_evidence,
    evidence_to_references,
    extract_evidence_block,
    make_rag_trace,
    serialize_evidence_block,
)
from src.state import CRCAgentState, RetrievedReference, merge_evidence_by_id
from src.tools.rag_tools import _format_docs


def _minimal_evidence(**overrides):
    evidence = {
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
        "provenance": {
            "parse_method": None,
            "source_file_hash": None,
            "collection_version": None,
        },
        "frontend": {
            "title": "NCCN.pdf",
            "citation_label": "NCCN.pdf p.12",
            "display_text": "Full guideline text",
        },
    }
    evidence.update(overrides)
    return evidence


def test_evidence_serialization_round_trip_preserves_core_fields() -> None:
    evidence = [_minimal_evidence()]

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
    references = evidence_to_references(
        [
            _minimal_evidence(
                evidence_id="e1",
                snippet="Preview",
                scores={"rerank": 0.9, "fusion": 0.8, "vector": None, "bm25": None},
                frontend={"title": "NCCN.pdf"},
            )
        ]
    )

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
    merged = dedupe_evidence(
        [
            {"evidence_id": "e1", "snippet": "old"},
            {"evidence_id": "e1", "snippet": "new"},
            {"evidence_id": "e2", "snippet": "other"},
        ]
    )

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


def test_crc_agent_state_exposes_frontend_ready_evidence_fields() -> None:
    state = CRCAgentState()

    assert state.retrieved_evidence == []
    assert state.rag_trace == []


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


def test_direct_extraction_prefers_retrieved_evidence() -> None:
    payload = serialize_evidence_block([_minimal_evidence(evidence_id="e1", snippet="Evidence preview")])

    cleaned, refs, evidence = _extract_structured_evidence("visible context\n" + payload)

    assert "retrieved_evidence" not in cleaned
    assert evidence[0]["evidence_id"] == "e1"
    assert refs[0]["evidence_id"] == "e1"


def test_direct_extraction_falls_back_to_legacy_metadata() -> None:
    text = (
        '<retrieved_metadata>[{"ref_id":"REF_1","source":"NCCN.pdf",'
        '"page":5,"preview":"Legacy"}]</retrieved_metadata>'
    )

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
    payload = serialize_evidence_block([_minimal_evidence(evidence_id="e1", snippet="Evidence preview")])

    agent._extract_references(payload + '<retrieved_metadata>[{"source":"Legacy.pdf"}]</retrieved_metadata>')

    assert agent._collected_references[0]["evidence_id"] == "e1"
    assert agent._collected_references[0]["source"] == "NCCN.pdf"
