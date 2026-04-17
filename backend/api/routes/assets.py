from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response

from backend.api.services.asset_service import AssetNotFoundError, load_asset_content
from backend.api.services.session_store import InMemorySessionStore

router = APIRouter(prefix="/api", tags=["assets"])

SAFE_INLINE_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
}


def _build_content_disposition(filename: str, *, inline: bool) -> str:
    fallback = (
        filename.replace('"', "_")
        .replace("\r", "_")
        .replace("\n", "_")
    )
    disposition = "inline" if inline else "attachment"
    encoded_filename = quote(filename, safe="")
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{encoded_filename}"


def _get_runtime_dependency(request: Request) -> tuple[InMemorySessionStore, object]:
    runtime = getattr(request.app.state, "runtime", None)
    session_store = getattr(runtime, "session_store", None)
    assets_root = getattr(runtime, "assets_root", None)
    if session_store is None or assets_root is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized")
    return session_store, assets_root


@router.get("/assets/{asset_id}")
async def get_asset(asset_id: str, request: Request):
    session_store, assets_root = _get_runtime_dependency(request)
    try:
        payload = load_asset_content(
            session_store=session_store,
            assets_root=assets_root,
            asset_id=asset_id,
        )
    except AssetNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc

    content_type = str(payload["content_type"])
    is_safe_inline = content_type in SAFE_INLINE_CONTENT_TYPES

    return Response(
        content=payload["content"],
        media_type=content_type if is_safe_inline else "application/octet-stream",
        headers={
            "Content-Disposition": _build_content_disposition(
                str(payload["filename"]),
                inline=is_safe_inline,
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )
