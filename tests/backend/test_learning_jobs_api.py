from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.learning_job_service import LearningJobValidationError


def _signal_payload(*, deidentified: bool = True) -> dict[str, Any]:
    return {
        "signal_type": "doctor_action_trace",
        "source_ref": {
            "kind": "doctor_action_trace",
            "id": "doctor_action_trace_crc_shadow_001",
            "projection": "aggregate_shadow_learning",
        },
        "reason_code": "unsafe_disposition",
        "target_area": "prompt",
        "severity": "high",
        "summary": "Aggregate deidentified shadow signal.",
        "deidentified": deidentified,
        "created_at": "2026-07-09T10:00:00+08:00",
    }


def _post_payload(*, deidentified: bool = True) -> dict[str, Any]:
    return {
        "signals": [_signal_payload(deidentified=deidentified)],
        "requested_by": "admin_user",
        "idempotency_key": "learning-job-001",
    }


def _client() -> TestClient:
    from backend.api.routes import learning_jobs

    app = FastAPI()
    app.include_router(learning_jobs.router)
    return TestClient(app)


def test_get_learning_jobs_returns_service_read_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.api.routes import learning_jobs

    class ServiceStub:
        def read_jobs(self) -> dict[str, Any]:
            return {
                "jobs": [],
                "candidates": [],
                "integrity": {"status": "verified", "warnings": []},
                "disabled_actions": [
                    {
                        "id": "apply",
                        "label": "Apply",
                        "disabled": True,
                        "reason": "shadow_learning_jobs_only",
                    }
                ],
                "runtime": {
                    "auth": "admin",
                    "source": "reports/learning_jobs",
                    "mode": "shadow_learning_jobs",
                },
            }

    monkeypatch.setattr(learning_jobs, "_learning_job_service", lambda: ServiceStub())
    client = _client()

    try:
        response = client.get("/api/admin/learning-jobs")

        assert response.status_code == 200
        assert response.json()["runtime"]["mode"] == "shadow_learning_jobs"
    finally:
        client.close()


def test_post_learning_jobs_derives_signal_ids_and_calls_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.api.routes import learning_jobs

    captured: dict[str, Any] = {}

    class ServiceStub:
        def create_job(
            self,
            signals: list[object],
            *,
            requested_by: str,
            idempotency_key: str,
        ) -> dict[str, Any]:
            captured["signals"] = signals
            captured["requested_by"] = requested_by
            captured["idempotency_key"] = idempotency_key
            return {
                "job": {"job_id": "learning_job_001", "status": "shadow_only"},
                "signals": [signals[0].to_dict()],
                "candidates": [],
                "disabled_actions": [],
                "runtime": {"auth": "admin", "mode": "shadow_learning_jobs"},
            }

    monkeypatch.setattr(learning_jobs, "_learning_job_service", lambda: ServiceStub())
    client = _client()

    try:
        response = client.post("/api/admin/learning-jobs", json=_post_payload())

        assert response.status_code == 200
        assert response.json()["job"]["status"] == "shadow_only"
        assert captured["requested_by"] == "admin_user"
        assert captured["idempotency_key"] == "learning-job-001"
        signal = captured["signals"][0]
        assert signal.signal_id.startswith("learning_signal_")
        assert signal.to_dict()["deidentified"] is True
    finally:
        client.close()


def test_post_learning_jobs_rejects_non_deidentified_signal() -> None:
    client = _client()

    try:
        response = client.post(
            "/api/admin/learning-jobs",
            json=_post_payload(deidentified=False),
        )

        assert response.status_code == 422
    finally:
        client.close()


@pytest.mark.parametrize(
    ("exc", "status_code"),
    [
        (LearningJobValidationError("invalid request"), 422),
        (TypeError("bad type"), 422),
        (ValueError("bad value"), 422),
        (FileExistsError("duplicate"), 409),
        (OSError("disk unavailable"), 500),
    ],
)
def test_post_learning_jobs_maps_service_errors(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    status_code: int,
) -> None:
    from backend.api.routes import learning_jobs

    class ServiceStub:
        def create_job(self, *args: object, **kwargs: object) -> dict[str, Any]:
            raise exc

    monkeypatch.setattr(learning_jobs, "_learning_job_service", lambda: ServiceStub())
    client = _client()

    try:
        response = client.post("/api/admin/learning-jobs", json=_post_payload())

        assert response.status_code == status_code
        assert str(exc) in response.json()["detail"]
    finally:
        client.close()


def test_post_learning_jobs_maps_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.api.routes import learning_jobs
    from backend.api.services.learning_job_store import LearningJobIntegrityError

    class ServiceStub:
        def create_job(self, *args: object, **kwargs: object) -> dict[str, Any]:
            raise LearningJobIntegrityError("integrity failed")

    monkeypatch.setattr(learning_jobs, "_learning_job_service", lambda: ServiceStub())
    client = _client()

    try:
        response = client.post("/api/admin/learning-jobs", json=_post_payload())

        assert response.status_code == 409
        assert response.json()["detail"] == "integrity failed"
    finally:
        client.close()


def test_learning_jobs_apply_endpoint_is_not_exposed() -> None:
    client = _client()

    try:
        response = client.post("/api/admin/learning-jobs/learning_job_1/apply")

        assert response.status_code == 404
    finally:
        client.close()
