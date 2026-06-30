from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend.api.services.admin_release_dashboard import build_release_dashboard
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


@router.get("/release-dashboard")
async def get_admin_release_dashboard() -> dict[str, Any]:
    return build_release_dashboard()
