$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\anaconda3\envs\LangG\python.exe"
$seedPath = Join-Path $repoRoot "tests\fixtures\acceptance_case_db\seed.json"
$acceptanceDbRoot = Join-Path $repoRoot "runtime\acceptance_case_db"

if (-not (Test-Path $pythonExe)) {
  throw "LangG python not found at $pythonExe"
}

Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $pythonExe scripts/prepare_acceptance_case_db.py --seed $seedPath --output $acceptanceDbRoot
if ($LASTEXITCODE -ne 0) {
  throw "Acceptance case database preparation failed with exit code $LASTEXITCODE"
}

$env:AUTH_MODE = "none"
$env:GRAPH_RUNNER_MODE = "fixture"
$env:GRAPH_FIXTURE_CASE = "database_case"
$env:UPLOAD_CONVERTER_MODE = "fixture"
$env:CASE_DATABASE_PATH = $acceptanceDbRoot
$env:RAG_WARMUP = "false"
$env:FRONTEND_ORIGINS = "http://127.0.0.1:4173"

& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
