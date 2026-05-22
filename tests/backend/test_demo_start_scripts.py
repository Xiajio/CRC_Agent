from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_start_demo_sets_replay_environment():
    text = (ROOT / "scripts" / "start_demo.ps1").read_text(encoding="utf-8")

    assert 'LANGG_RUNTIME_ROOT = "runtime/demo"' in text
    assert 'VITE_DEMO_MODE = "replay"' in text
    assert "start_backend_fixture.ps1" in text
    assert "start_backend_real.ps1" not in text


def test_fixture_backend_default_remains_database_case():
    text = (ROOT / "scripts" / "start_backend_fixture.ps1").read_text(encoding="utf-8")

    assert '$FixtureCase = "database_case"' in text
    assert 'if (-not $env:AUTH_MODE)' in text
    assert '$env:AUTH_MODE = "bearer"' in text
    assert '$env:API_BEARER_TOKEN = "local-dev-token"' in text
    assert '$env:AUTH_MODE = "none"' not in text
    assert '$env:GRAPH_RUNNER_MODE = "fixture"' in text
    assert "$env:GRAPH_FIXTURE_CASE = $FixtureCase" in text


def test_demo_playwright_config_starts_demo_profile():
    config_text = (ROOT / "frontend" / "playwright.demo.config.ts").read_text(encoding="utf-8")
    runner_text = (ROOT / "scripts" / "run_demo_playwright.cjs").read_text(encoding="utf-8")

    assert "PLAYWRIGHT_DEMO_SKIP_WEBSERVER" in config_text
    assert "playwright_demo_server.cjs backend" in config_text
    assert "playwright_demo_server.cjs frontend" in config_text
    assert "playwright_demo_server.cjs" in runner_text
    assert "PLAYWRIGHT_DEMO_SKIP_WEBSERVER" in runner_text
    assert "killProcessTree" in runner_text
    assert "playwright.demo.config.ts" in runner_text
    assert r"D:\anaconda3\envs\LangG\npx.cmd" not in runner_text
    assert "node_modules" in runner_text
    assert "@playwright" in runner_text
    assert "cli.js" in runner_text
    assert "npx.cmd" not in runner_text


def test_demo_playwright_forwards_optional_bearer_token_to_frontend():
    runner_text = (ROOT / "scripts" / "run_demo_playwright.cjs").read_text(encoding="utf-8")
    server_text = (ROOT / "scripts" / "playwright_demo_server.cjs").read_text(encoding="utf-8")

    assert "VITE_API_BEARER_TOKEN" in runner_text
    assert "API_BEARER_TOKEN" in runner_text
    assert "VITE_API_BEARER_TOKEN" in server_text
    assert "API_BEARER_TOKEN" in server_text


def test_demo_playwright_server_wrapper_cleans_up_process_tree():
    text = (ROOT / "scripts" / "playwright_demo_server.cjs").read_text(encoding="utf-8")

    assert "start_backend_fixture.ps1" in text
    assert "-FixtureCase" in text
    assert "demo_doctor_decision" in text
    assert "-UploadConverterFixture" in text
    assert "LANGG_RUNTIME_ROOT" in text
    assert "VITE_DEMO_MODE" in text
    assert "taskkill" in text
    assert "/T" in text
