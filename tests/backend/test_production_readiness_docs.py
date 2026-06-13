from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_production_readiness_doc_captures_current_deployment_boundary() -> None:
    document = _read("docs/deployment/production-readiness.md")

    assert "single worker" in document
    assert "SESSION_STORE_BACKEND=sqlite" in document
    assert "does not provide a cross-process run lock" in document
    assert "POST SSE currently has no mid-run resume" in document
    assert "proxy_buffering off" in document
    assert "VITE_API_BEARER_TOKEN is a browser-visible token" in document


def test_readme_links_to_production_readiness_guide() -> None:
    readme = _read("README.md")

    assert "docs/deployment/production-readiness.md" in readme
    assert "single-worker production boundary" in readme
