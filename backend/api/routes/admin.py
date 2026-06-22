from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.config import load_settings
from src.tools.manifest import build_tool_manifest_response

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _web_search_enabled_from_request(request: Request) -> bool:
    runtime = getattr(request.app.state, "runtime", None)
    runtime_settings = getattr(runtime, "settings", None)
    runtime_web_search = getattr(runtime_settings, "web_search", None)
    runtime_enabled = getattr(runtime_web_search, "enabled", None)
    if runtime_enabled is not None:
        return bool(runtime_enabled)

    return bool(load_settings().web_search.enabled)


@router.get("/tools")
async def get_admin_tools(request: Request) -> dict[str, Any]:
    return build_tool_manifest_response(
        web_search_enabled=_web_search_enabled_from_request(request),
    )
