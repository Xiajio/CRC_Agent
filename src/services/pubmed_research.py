from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import os
import re
from typing import Any
from xml.etree import ElementTree

import httpx

from src.contracts.auto_research import ResearchSource, make_research_source_id


class PubMedResearchError(RuntimeError):
    """Raised when PubMed cannot return a verifiable evidence batch."""


class PubMedEvidenceRetriever:
    """Retrieve source-grounded PubMed abstracts through NCBI E-utilities."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        now: Callable[[], str] | None = None,
        api_key: str | None = None,
        email: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._client = client
        self._now = now or (
            lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self._api_key = api_key if api_key is not None else os.getenv("NCBI_API_KEY", "")
        self._email = email if email is not None else os.getenv("NCBI_EMAIL", "")
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ncbi_pubmed_eutilities"

    def retrieve(self, question: str, max_sources: int) -> list[ResearchSource]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if isinstance(max_sources, bool) or not isinstance(max_sources, int):
            raise TypeError("max_sources must be an integer")
        if not 1 <= max_sources <= 20:
            raise ValueError("max_sources must be between 1 and 20")

        if self._client is not None:
            return self._retrieve_with_client(
                self._client,
                question=question.strip(),
                max_sources=max_sources,
            )

        headers = {"User-Agent": "LangG-AutoResearch/0.1"}
        with httpx.Client(timeout=self._timeout_seconds, headers=headers) as client:
            return self._retrieve_with_client(
                client,
                question=question.strip(),
                max_sources=max_sources,
            )

    def _retrieve_with_client(
        self,
        client: httpx.Client,
        *,
        question: str,
        max_sources: int,
    ) -> list[ResearchSource]:
        fetch_budget = min(max_sources * 2, 40)
        search_params = self._common_params()
        search_params.update(
            {
                "db": "pubmed",
                "term": f"({question}) AND hasabstract[text]",
                "retmode": "json",
                "retmax": str(fetch_budget),
                "sort": "relevance",
            }
        )
        try:
            search_response = client.get(self.ESEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_payload = search_response.json()
        except httpx.HTTPError as exc:
            raise PubMedResearchError(
                f"PubMed ESearch failed: {_http_error_summary(exc)}"
            ) from exc
        except ValueError as exc:
            raise PubMedResearchError(f"PubMed ESearch failed: {exc}") from exc

        raw_ids = (
            search_payload.get("esearchresult", {}).get("idlist", [])
            if isinstance(search_payload, dict)
            else []
        )
        pmids = _normalize_pmids(raw_ids, maximum=fetch_budget)
        if not pmids:
            return []

        fetch_params = self._common_params()
        fetch_params.update(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
        )
        try:
            fetch_response = client.get(self.EFETCH_URL, params=fetch_params)
            fetch_response.raise_for_status()
            root = ElementTree.fromstring(fetch_response.content)
        except httpx.HTTPError as exc:
            raise PubMedResearchError(
                f"PubMed EFetch failed: {_http_error_summary(exc)}"
            ) from exc
        except ElementTree.ParseError as exc:
            raise PubMedResearchError(f"PubMed EFetch failed: {exc}") from exc

        retrieved_at = self._now()
        sources: list[ResearchSource] = []
        seen_pmids: set[str] = set()
        for article in root.findall(".//PubmedArticle"):
            source = _source_from_pubmed_article(
                article,
                query=question,
                retrieved_at=retrieved_at,
            )
            if source is None or source.pmid in seen_pmids:
                continue
            seen_pmids.add(source.pmid or "")
            sources.append(source)
            if len(sources) >= max_sources:
                break
        return sources

    def _common_params(self) -> dict[str, str]:
        params = {"tool": "LangGAutoResearch"}
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email
        return params


def _normalize_pmids(value: Any, *, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        pmid = str(item).strip()
        if not pmid.isdigit() or pmid in normalized:
            continue
        normalized.append(pmid)
        if len(normalized) >= maximum:
            break
    return normalized


def _http_error_summary(exc: httpx.HTTPError) -> str:
    """Return a useful provider error without echoing credential-bearing URLs."""

    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _source_from_pubmed_article(
    article: ElementTree.Element,
    *,
    query: str,
    retrieved_at: str,
) -> ResearchSource | None:
    pmid = _element_text(article.find(".//MedlineCitation/PMID"))
    title = _element_text(article.find(".//Article/ArticleTitle"))
    abstract_parts: list[str] = []
    for item in article.findall(".//Article/Abstract/AbstractText"):
        text = _element_text(item)
        if not text:
            continue
        label = str(item.attrib.get("Label", "")).strip()
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts).strip()
    if not pmid or not pmid.isdigit() or not title or not abstract:
        return None

    journal = _element_text(article.find(".//Article/Journal/Title"))
    publication_year = _publication_year(article)
    publication_types = [
        _element_text(item)
        for item in article.findall(".//Article/PublicationTypeList/PublicationType")
    ]
    publication_types = [item for item in publication_types if item]
    source_type = publication_types[0] if publication_types else "Journal Article"

    return ResearchSource(
        source_id=make_research_source_id("pubmed", pmid),
        title=title,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        abstract=abstract,
        journal=journal,
        publication_year=publication_year,
        source_type=source_type,
        query=query,
        retrieved_at=retrieved_at,
        pmid=pmid,
    )


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _publication_year(article: ElementTree.Element) -> str:
    for path in (
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//DateCompleted/Year",
    ):
        year = _element_text(article.find(path))
        if re.fullmatch(r"(?:19|20)\d{2}", year):
            return year
    medline_date = _element_text(
        article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    )
    match = re.search(r"(?:19|20)\d{2}", medline_date)
    return match.group(0) if match else "unknown"


__all__ = ["PubMedEvidenceRetriever", "PubMedResearchError"]
