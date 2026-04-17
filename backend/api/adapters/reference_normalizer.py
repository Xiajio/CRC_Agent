from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def _parse_page(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            try:
                return int(match.group())
            except ValueError:
                return None
    return None


def _read_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def normalize_reference(value: Any) -> dict[str, Any] | None:
    mapped = _coerce_mapping(value)
    if mapped is None:
        return None

    normalized = dict(mapped)
    page = _parse_page(mapped.get("page"))
    if page is not None:
        normalized["page"] = page

    source = _read_text(mapped.get("source"))
    title = _read_text(mapped.get("title"))
    snippet = _read_text(
        mapped.get("citation"),
        mapped.get("snippet"),
        mapped.get("preview"),
        mapped.get("content"),
    )

    if title is None and source is not None:
        normalized["title"] = source
        title = source

    if source is None and title is not None and (page is not None or snippet is not None):
        normalized["source"] = title
        source = title

    if snippet is not None and not _read_text(mapped.get("snippet")):
        normalized["snippet"] = snippet

    existing_source_id = _read_text(mapped.get("source_id"), mapped.get("ref_id"))
    existing_id = _read_text(mapped.get("id"))
    if not existing_source_id and source is not None and (page is not None or snippet is not None):
        normalized["source_id"] = existing_id or (f"{source}:{page}" if page is not None else source)

    return normalized


def normalize_reference_list(value: Sequence[Any] | Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    items: list[dict[str, Any]] = []
    for entry in value:
        normalized = normalize_reference(entry)
        if normalized is not None:
            items.append(normalized)
    return items
