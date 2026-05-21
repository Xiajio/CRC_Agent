param(
  [string]$FixtureCase = "database_case",
  [switch]$UploadConverterFixture
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\anaconda3\envs\LangG\python.exe"

if (-not (Test-Path $pythonExe)) {
  throw "LangG python not found at $pythonExe"
}

Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:AUTH_MODE = "none"
$env:GRAPH_RUNNER_MODE = "fixture"
$env:GRAPH_FIXTURE_CASE = $FixtureCase
$env:RAG_WARMUP = "false"
$env:FRONTEND_ORIGINS = "http://127.0.0.1:4173"
if ($UploadConverterFixture) {
  $env:UPLOAD_CONVERTER_MODE = "fixture"
}

& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
