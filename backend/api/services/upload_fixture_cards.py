from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_UPLOAD_CARD_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "upload_cards"


def _sanitize_asset_filename(filename: str) -> str:
    candidate = Path(filename or "").name.strip()
    if not candidate:
        return "upload.bin"
    return candidate.replace("/", "_").replace("\\", "_")


def load_fixture_upload_card(filename: str) -> dict[str, Any]:
    sanitized_filename = _sanitize_asset_filename(filename)
    fixture_filename = Path(sanitized_filename).with_suffix(".json").name
    fixture_path = _UPLOAD_CARD_FIXTURE_ROOT / fixture_filename
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"Missing fixture upload card for {sanitized_filename!r}: {fixture_path}"
        )

    with fixture_path.open("r", encoding="utf-8") as fixture_file:
        card = json.load(fixture_file)

    if not isinstance(card, dict):
        raise ValueError(f"Fixture upload card must be a JSON object: {fixture_path}")
    return card
