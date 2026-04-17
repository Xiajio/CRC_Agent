from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CARD_PREVIEW_LIMIT = 8

_PREVIEW_IMAGE_FIELDS = {
    "image_name",
    "image_base64",
    "image_url",
    "image_path",
    "source_slide",
}

_CARD_PREVIEW_COLLECTIONS = {
    "imaging_card": "images",
    "tumor_detection_card": "sample_images_with_tumor",
    "pathology_slide_card": "images",
    "radiomics_report_card": "analyzed_images",
}


def strip_binary(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("base64", "blob", "binary", "bytes")):
                continue
            if key_text == "artifact":
                continue
            cleaned[key] = strip_binary(item)
        return cleaned

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [strip_binary(item) for item in value]

    return value


def _sanitize_preview_item(preview: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for field in _PREVIEW_IMAGE_FIELDS:
        value = preview.get(field)
        if value is None:
            continue
        cleaned[field] = value
    return cleaned


def sanitize_visual_card_payload(payload: Mapping[str, Any], preview_collection_key: str) -> dict[str, Any]:
    sanitized = strip_binary(dict(payload))
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return sanitized

    sanitized_data = sanitized.get("data")
    if not isinstance(sanitized_data, Mapping):
        sanitized_data = {}

    previews = data.get(preview_collection_key)
    if isinstance(previews, Sequence) and not isinstance(previews, (str, bytes)):
        sanitized_previews = [
            _sanitize_preview_item(image)
            for image in previews
            if isinstance(image, Mapping)
        ]
        sanitized_data = dict(sanitized_data)
        sanitized_data[preview_collection_key] = sanitized_previews[:CARD_PREVIEW_LIMIT]
        sanitized["data"] = sanitized_data

    return sanitized


def sanitize_card_payload(card_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    preview_collection_key = _CARD_PREVIEW_COLLECTIONS.get(card_type)
    if preview_collection_key:
        return sanitize_visual_card_payload(payload, preview_collection_key)
    return strip_binary(dict(payload))
