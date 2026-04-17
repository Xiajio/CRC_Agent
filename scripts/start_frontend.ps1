$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "frontend"
$nodeHome = "D:\anaconda3\envs\LangG"
$nodeExe = Join-Path $nodeHome "node.exe"
$npmCmd = Join-Path $nodeHome "npm.cmd"

if (-not (Test-Path $npmCmd)) {
  throw "LangG npm.cmd not found at $npmCmd"
}

if (-not (Test-Path $nodeExe)) {
  throw "LangG node.exe not found at $nodeExe"
}

Set-Location $frontendRoot

$env:PATH = "$nodeHome;$env:PATH"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"

& $npmCmd run dev:e2e
