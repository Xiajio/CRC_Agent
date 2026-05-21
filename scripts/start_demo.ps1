param(
  [string]$FixtureCase = "demo_doctor_decision"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendScript = Join-Path $PSScriptRoot "start_backend_fixture.ps1"
$frontendScript = Join-Path $PSScriptRoot "start_frontend.ps1"

Set-Location $repoRoot

$env:LANGG_RUNTIME_ROOT = "runtime/demo"
$env:VITE_DEMO_MODE = "replay"

$backendArgs = @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-File",
  $backendScript,
  "-FixtureCase",
  $FixtureCase,
  "-UploadConverterFixture"
)

Start-Process powershell -ArgumentList $backendArgs -WindowStyle Hidden

Write-Host "Demo backend fixture case: $FixtureCase"
Write-Host "Demo frontend mode: replay"
& $frontendScript
