from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import sanitize_asset_filename


class AssetNotFoundError(RuntimeError):
    pass


def load_asset_content(
    *,
    session_store: InMemorySessionStore,
    assets_root: Path,
    asset_id: str,
) -> dict[str, Any]:
    lookup = session_store.find_uploaded_asset(asset_id)
    if lookup is None:
        raise AssetNotFoundError(f"Asset not found: {asset_id}")

    meta, asset_record = lookup
    filename = str(asset_record.get("filename") or "upload.bin")
    original_path = (
        assets_root
        / meta.session_id
        / asset_id
        / "original"
        / sanitize_asset_filename(filename)
    )
    if not original_path.exists() or not original_path.is_file():
        raise AssetNotFoundError(f"Asset content not found: {asset_id}")

    return {
        "filename": filename,
        "content_type": str(asset_record.get("content_type") or "application/octet-stream"),
        "content": original_path.read_bytes(),
    }
