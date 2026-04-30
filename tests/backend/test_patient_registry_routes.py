from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import patient_registry as patient_registry_routes


class _StubPatientRegistry:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def list_recent_patients(self, *, limit: int) -> list[dict[str, object]]:
        self.limits.append(limit)
        return [
            {
                "patient_id": 7,
                "status": "active",
                "created_by_session_id": "sess_patient_1",
                "updated_at": "2026-04-30T00:00:00Z",
                "tumor_location": "rectum",
                "mmr_status": "dMMR",
                "clinical_stage": "cT3N1M0",
            }
        ]


def test_patient_registry_routes_use_app_runtime_service() -> None:
    registry = _StubPatientRegistry()
    app = FastAPI()
    app.state.runtime = SimpleNamespace(patient_registry_service=registry)
    app.include_router(patient_registry_routes.router)

    with TestClient(app) as client:
        response = client.get("/api/patient-registry/patients/recent?limit=1")

    assert response.status_code == 200
    assert response.json()["items"] == [
        {
            "patient_id": 7,
            "status": "active",
            "created_by_session_id": "sess_patient_1",
            "updated_at": "2026-04-30T00:00:00Z",
            "tumor_location": "rectum",
            "mmr_status": "dMMR",
            "clinical_stage": "cT3N1M0",
        }
    ]
    assert response.json()["total"] == 1
    assert registry.limits == [1]
