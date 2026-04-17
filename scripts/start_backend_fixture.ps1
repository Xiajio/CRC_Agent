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
$env:RAG_WARMUP = "false"
$env:FRONTEND_ORIGINS = "http://127.0.0.1:4173"

& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
