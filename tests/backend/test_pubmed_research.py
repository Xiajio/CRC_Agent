from __future__ import annotations

import httpx
import pytest

from src.services.pubmed_research import PubMedEvidenceRetriever, PubMedResearchError


PUBMED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123456</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
          <Title>Journal of Verified Evidence</Title>
        </Journal>
        <ArticleTitle>Circulating <i>DNA</i> and colorectal cancer recurrence</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Recurrence biomarkers require validation.</AbstractText>
          <AbstractText Label="RESULTS">The marker was associated with recurrence.</AbstractText>
        </Abstract>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>999999</PMID>
      <Article><ArticleTitle>No abstract article</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_retriever_uses_esearch_then_efetch_and_parses_verified_sources() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["123456", "999999", "bad"]}},
            )
        return httpx.Response(200, content=PUBMED_XML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retriever = PubMedEvidenceRetriever(
        client=client,
        now=lambda: "2026-07-19T08:00:00+00:00",
        api_key="test-key",
        email="research@example.org",
    )

    sources = retriever.retrieve("colorectal cancer recurrence biomarker", 5)

    assert len(sources) == 1
    source = sources[0]
    assert source.pmid == "123456"
    assert source.title == "Circulating DNA and colorectal cancer recurrence"
    assert source.publication_year == "2026"
    assert source.journal == "Journal of Verified Evidence"
    assert "BACKGROUND: Recurrence biomarkers" in source.abstract
    assert source.url == "https://pubmed.ncbi.nlm.nih.gov/123456/"
    assert len(requests) == 2
    assert requests[0].url.params["db"] == "pubmed"
    assert requests[0].url.params["term"] == "(colorectal cancer recurrence biomarker) AND hasabstract[text]"
    assert requests[0].url.params["retmax"] == "10"
    assert requests[0].url.params["api_key"] == "test-key"
    assert requests[1].url.params["id"] == "123456,999999"
    client.close()


def test_pubmed_retriever_returns_empty_without_calling_efetch() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retriever = PubMedEvidenceRetriever(client=client)

    assert retriever.retrieve("query with no result", 3) == []
    assert requests == 1
    client.close()


def test_pubmed_retriever_fails_closed_on_http_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, text="service unavailable")
        )
    )
    retriever = PubMedEvidenceRetriever(
        client=client,
        api_key="ncbi-secret-key",
        email="owner@example.org",
    )

    with pytest.raises(PubMedResearchError, match="PubMed ESearch failed") as exc_info:
        retriever.retrieve("colorectal cancer", 3)
    assert "ncbi-secret-key" not in str(exc_info.value)
    assert "owner@example.org" not in str(exc_info.value)
    assert "HTTP 503" in str(exc_info.value)
    client.close()


def test_pubmed_retriever_rejects_invalid_budget() -> None:
    retriever = PubMedEvidenceRetriever(client=httpx.Client())

    with pytest.raises(ValueError, match="between 1 and 20"):
        retriever.retrieve("colorectal cancer", 21)
    retriever._client.close()  # type: ignore[union-attr]
