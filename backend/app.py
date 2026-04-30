from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.routes import assets as asset_routes
from backend.api.routes import chat as chat_routes
from backend.api.routes import database as database_routes
from backend.api.routes import patient_registry as patient_registry_routes
from backend.api.routes import sessions as session_routes
from backend.api.routes import uploads as upload_routes
from backend.api.services.context_maintenance import create_context_maintenance_service
from backend.api.services.graph_factory import (
    get_doctor_graph,
    get_patient_graph,
    get_runner_mode,
    load_backend_settings,
    should_warm_rag,
)
from backend.api.services.graph_service import DoctorGraphService, PatientGraphService, SceneGraphRouter
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.settings import RuntimeSettings, load_runtime_settings
from backend.api.services.session_store import InMemorySessionStore


@dataclass(slots=True)
class AppRuntime:
    settings: object
    runner_mode: str
    session_store: object
    patient_registry_service: object
    patient_command_service: object
    patient_graph: object
    doctor_graph: object
    patient_graph_service: object
    doctor_graph_service: object
    scene_router: object
    runtime_root: Path
    assets_root: Path


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: RuntimeSettings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):
        if self._settings.auth_mode != "bearer":
            return await call_next(request)

        if request.method.upper() == "OPTIONS" or not request.url.path.startswith("/api"):
            return await call_next(request)

        expected = self._settings.api_bearer_token
        authorization = request.headers.get("Authorization")
        if not expected:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        scheme, _, token = authorization.partition(" ") if authorization else ("", "", "")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, expected):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        return await call_next(request)


def _build_runtime_metadata(runner_mode: str) -> dict[str, str | None]:
    fixture_case = None
    if runner_mode == "fixture":
        fixture_case = os.getenv("GRAPH_FIXTURE_CASE", "database_case").strip() or "database_case"
    return {
        "runner_mode": runner_mode,
        "fixture_case": fixture_case,
    }


def _build_lifespan():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = load_backend_settings()
        runner_mode = get_runner_mode()
        rag_warmup = should_warm_rag()
        runtime_metadata = _build_runtime_metadata(runner_mode)

        runtime_root = Path("runtime")
        assets_root = runtime_root / "assets"
        assets_root.mkdir(parents=True, exist_ok=True)
        patient_registry_service = PatientRegistryService(runtime_root / "patient_registry.db")
        patient_command_service = PatientCommandService(patient_registry_service)

        session_store = InMemorySessionStore()
        session_routes.session_store = session_store
        session_routes.patient_registry_service = patient_registry_service
        session_routes.patient_command_service = patient_command_service
        patient_graph = get_patient_graph(
            settings,
            runner_mode=runner_mode,
            rag_warmup=rag_warmup,
        )
        doctor_graph = get_doctor_graph(
            settings,
            runner_mode=runner_mode,
            rag_warmup=rag_warmup,
        )
        context_maintenance_service = create_context_maintenance_service(
            settings,
            runner_mode=runner_mode,
        )
        patient_graph_service = PatientGraphService(
            patient_graph,
            session_store,
        )
        doctor_graph_service = DoctorGraphService(
            doctor_graph,
            session_store,
            patient_registry=patient_registry_service,
            context_finalizer=context_maintenance_service,
        )
        scene_router = SceneGraphRouter(
            patient_service=patient_graph_service,
            doctor_service=doctor_graph_service,
            session_store=session_store,
        )
        session_routes.load_agent_state = scene_router.load_agent_state
        session_routes.get_runtime_metadata = lambda: runtime_metadata

        app.state.runtime = AppRuntime(
            settings=settings,
            runner_mode=runner_mode,
            session_store=session_store,
            patient_registry_service=patient_registry_service,
            patient_command_service=patient_command_service,
            patient_graph=patient_graph,
            doctor_graph=doctor_graph,
            patient_graph_service=patient_graph_service,
            doctor_graph_service=doctor_graph_service,
            scene_router=scene_router,
            runtime_root=runtime_root,
            assets_root=assets_root,
        )
        try:
            yield
        finally:
            if hasattr(app.state, "runtime"):
                delattr(app.state, "runtime")

    return lifespan


def create_app() -> FastAPI:
    runtime_settings = load_runtime_settings()
    runner_mode = get_runner_mode()
    app = FastAPI(
        title="LangG BFF",
        lifespan=_build_lifespan(),
    )
    session_routes.get_runtime_metadata = lambda: _build_runtime_metadata(runner_mode)
    app.add_middleware(BearerAuthMiddleware, settings=runtime_settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.frontend_origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    app.include_router(session_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(database_routes.router)
    app.include_router(patient_registry_routes.router)
    app.include_router(upload_routes.router)
    app.include_router(asset_routes.router)
    return app


app = create_app()
