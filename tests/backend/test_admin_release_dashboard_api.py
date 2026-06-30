from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin as admin_routes


def test_admin_release_dashboard_route_returns_service_payload(monkeypatch) -> None:
    payload = {
        "version_chain": {
            "agent_policy_version": "agent_policy_test",
            "clinical_safety_policy_version": "safety_test",
            "evidence_index_version": "evidence_test",
            "judge_rubric_version": "rubric_test",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_previous",
        "human_signoff": {"required": True, "status": "missing", "reason": "read-only"},
        "summary": {
            "hard_fail_count": 0,
            "p0_cases_total": 1,
            "p0_cases_passed": 1,
            "literature_claims": 2,
            "literature_isolation_violations": 0,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [],
        "blocking_gates": [],
        "disabled_actions": [],
        "runtime": {"auth": "admin", "source": "reports/static_release_artifacts", "mode": "read_only"},
    }
    calls = {"count": 0}

    def fake_build_release_dashboard():
        calls["count"] += 1
        return payload

    monkeypatch.setattr(admin_routes, "build_release_dashboard", fake_build_release_dashboard)
    app = FastAPI()
    app.include_router(admin_routes.router)
    client = TestClient(app)

    try:
        response = client.get("/api/admin/release-dashboard")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == payload
    assert calls["count"] == 1
