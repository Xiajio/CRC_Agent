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
    session_id: str,
    asset_id: str,
) -> dict[str, Any]:
    lookup = session_store.find_uploaded_asset(session_id, asset_id)
    if lookup is None:
        raise AssetNotFoundError(f"Asset not found: {asset_id}")

    meta, asset_record = lookup
    filename = str(asset_record.get("filename") or "upload.bin")
    storage_path = asset_record.get("storage_path")
    if isinstance(storage_path, str) and storage_path.strip():
        original_path = Path(storage_path)
    else:
        original_path = (
            assets_root
            / str(asset_record.get("patient_id") or meta.patient_id or meta.session_id)
            / str(asset_record.get("sha256") or asset_id)
            / "original"
            / sanitize_asset_filename(filename)
        )
    assets_root_resolved = assets_root.resolve(strict=False)
    original_path_resolved = original_path.resolve(strict=False)
    try:
        original_path_resolved.relative_to(assets_root_resolved)
    except ValueError as exc:
        raise AssetNotFoundError(f"Asset content not found: {asset_id}") from exc

    if not original_path_resolved.exists() or not original_path_resolved.is_file():
        raise AssetNotFoundError(f"Asset content not found: {asset_id}")

    return {
        "filename": filename,
        "content_type": str(asset_record.get("content_type") or "application/octet-stream"),
        "content": original_path_resolved.read_bytes(),
    }
