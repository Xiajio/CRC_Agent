from __future__ import annotations

from langchain_core.documents import Document

from src.nodes.node_utils import _extract_rag_payload
from src.nodes.sub_agent import SubAgentContext
from src.rag import retriever as retriever_module
from src.rag.evidence import (
    build_rag_trace,
    evidence_to_references,
    parse_retrieved_evidence,
    serialize_retrieved_evidence,
)
from src.state import RetrievedReference, merge_evidence_by_id
from src.tools.rag_tools import _format_docs


def test_get_hybrid_retriever_forwards_caller_k_to_global_manager(monkeypatch) -> None:
    captured: dict[str, int] = {}

    class FakeManager:
        def get_retriever(self, candidate_k: int = 20):
            captured["candidate_k"] = candidate_k
            return "stub-retriever"

    monkeypatch.setattr(retriever_module, "_global_manager", FakeManager())

    result = retriever_module.get_hybrid_retriever(k=7)

    assert result == "stub-retriever"
    assert captured["candidate_k"] == 7


def _sample_evidence(evidence_id: str = "search_treatment_recommendations:abc123") -> dict:
    return {
        "evidence_id": evidence_id,
        "chunk_id": "abc123",
        "source": "NCCN_rectal.pdf",
        "page": 12,
        "section": "Treatment",
        "text": "Adjuvant chemotherapy is recommended for selected stage III patients.",
        "snippet": "Adjuvant chemotherapy is recommended.",
        "query": "stage III adjuvant chemotherapy",
        "tool_name": "search_treatment_recommendations",
        "retrieval_profile": "treatment",
        "scores": {
            "vector": 0.72,
            "bm25": 3.14,
            "fusion": 0.81,
            "rerank": 0.67,
        },
        "provenance": {
            "parse_method": None,
            "source_file_hash": None,
            "collection_version": None,
        },
        "frontend": {
            "title": "NCCN_rectal.pdf",
            "citation_label": "NCCN_rectal.pdf p.12",
            "display_text": "Adjuvant chemotherapy is recommended.",
        },
    }


def test_evidence_serialization_round_trip_preserves_core_fields() -> None:
    evidence = [
        _sample_evidence("search_treatment_recommendations:abc123"),
        {
            **_sample_evidence("search_treatment_recommendations:def456"),
            "chunk_id": "def456",
            "source": "CSCO_colon.pdf",
            "page": None,
            "retrieval_profile": "general",
        },
    ]

    block = serialize_retrieved_evidence(evidence)
    parsed = parse_retrieved_evidence(block)

    assert "<retrieved_evidence>" in block
    assert [item["evidence_id"] for item in parsed] == [
        "search_treatment_recommendations:abc123",
        "search_treatment_recommendations:def456",
    ]
    assert parsed[0]["source"] == "NCCN_rectal.pdf"
    assert parsed[0]["page"] == 12
    assert parsed[0]["snippet"] == "Adjuvant chemotherapy is recommended."
    assert parsed[1]["retrieval_profile"] == "general"


def test_evidence_to_references_preserves_evidence_id_and_section() -> None:
    refs = evidence_to_references([_sample_evidence()])

    assert refs == [
        {
            "source": "NCCN_rectal.pdf",
            "page": 12,
            "section": "Treatment",
            "snippet": "Adjuvant chemotherapy is recommended.",
            "score": 0.67,
            "ref_id": "NCCN_rectal.pdf:12",
            "evidence_id": "search_treatment_recommendations:abc123",
        }
    ]

    model_ref = RetrievedReference.model_validate(refs[0])
    assert model_ref.evidence_id == "search_treatment_recommendations:abc123"
    assert model_ref.section == "Treatment"


def test_format_docs_includes_readable_metadata_and_structured_evidence() -> None:
    doc = Document(
        page_content="Use FOLFOX or CAPOX as adjuvant chemotherapy in appropriate cases.",
        metadata={
            "chunk_id": "chunk-1",
            "source": "NCCN_colon.pdf",
            "page": "18",
            "section": "Adjuvant Therapy",
            "vector_score": 0.4,
            "bm25_score": 1.2,
            "score": 0.8,
            "rerank_score": 0.6,
            "parse_method": "pypdf",
        },
    )

    output = _format_docs(
        [doc],
        tool_name="search_treatment_recommendations",
        query="adjuvant chemotherapy",
        retrieval_profile="treatment",
    )
    parsed = parse_retrieved_evidence(output)

    assert "[REF_1]" in output
    assert "<retrieved_metadata>" in output
    assert "<retrieved_evidence>" in output
    assert parsed[0]["tool_name"] == "search_treatment_recommendations"
    assert parsed[0]["retrieval_profile"] == "treatment"
    assert parsed[0]["source"] == "NCCN_colon.pdf"
    assert parsed[0]["page"] == 18
    assert parsed[0]["scores"]["bm25"] == 1.2


def test_direct_extraction_prefers_structured_evidence_over_metadata() -> None:
    structured = _sample_evidence("structured:1")
    legacy_metadata = (
        '<retrieved_metadata>[{"ref_id":"REF_1","source":"legacy.pdf","page":99,'
        '"preview":"legacy","score":0.1}]</retrieved_metadata>'
    )
    tool_output = (
        "Readable context\n"
        f"{serialize_retrieved_evidence([structured])}\n"
        f"{legacy_metadata}"
    )

    payload = _extract_rag_payload(tool_output)

    assert payload["content"] == "Readable context"
    assert [item["evidence_id"] for item in payload["retrieved_evidence"]] == ["structured:1"]
    assert payload["retrieved_references"][0]["source"] == "NCCN_rectal.pdf"
    assert payload["retrieved_references"][0]["evidence_id"] == "structured:1"
    assert payload["rag_trace"][0]["evidence_ids"] == ["structured:1"]


def test_malformed_evidence_payload_is_fail_soft() -> None:
    payload = _extract_rag_payload("<retrieved_evidence>{bad json</retrieved_evidence>")

    assert payload["retrieved_evidence"] == []
    assert payload["retrieved_references"] == []
    assert payload["rag_trace"] == []


def test_state_reducer_deduplicates_evidence_by_id_and_later_wins() -> None:
    merged = merge_evidence_by_id(
        [{"evidence_id": "same", "snippet": "old"}, {"evidence_id": "left", "snippet": "left"}],
        [{"evidence_id": "same", "snippet": "new"}],
    )

    assert merged == [
        {"evidence_id": "same", "snippet": "new"},
        {"evidence_id": "left", "snippet": "left"},
    ]


def test_sub_agent_extraction_prefers_structured_evidence() -> None:
    agent = SubAgentContext(model=None, system_prompt="", task_description="", show_thinking=False)
    structured = _sample_evidence("subagent:structured")
    tool_output = (
        f"{serialize_retrieved_evidence([structured])}"
        '<retrieved_metadata>[{"ref_id":"REF_1","source":"legacy.pdf","page":2,'
        '"preview":"legacy"}]</retrieved_metadata>'
    )

    agent._extract_references(tool_output)

    assert agent._collected_references[0]["evidence_id"] == "subagent:structured"
    assert agent._collected_references[0]["source"] == "NCCN_rectal.pdf"


def test_rag_trace_uses_evidence_ids_and_tool_profile() -> None:
    trace = build_rag_trace(
        tool_name="search_treatment_recommendations",
        query="adjuvant chemotherapy",
        retrieval_profile="treatment",
        evidence=[_sample_evidence("trace:1")],
        latency_ms=None,
        rerank_enabled=True,
        fallback_used=False,
    )

    assert trace["tool_name"] == "search_treatment_recommendations"
    assert trace["query"] == "adjuvant chemotherapy"
    assert trace["retrieval_profile"] == "treatment"
    assert trace["evidence_ids"] == ["trace:1"]
    assert trace["rerank_enabled"] is True
