param(
  [switch]$ListOnly,
  [switch]$CoreOnly,
  [switch]$SkipManualDocs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "frontend"
$pythonExe = "D:\anaconda3\envs\LangG\python.exe"
$npmCmd = "D:\anaconda3\envs\LangG\npm.cmd"
$nodeHome = Split-Path -Parent $npmCmd
$evidenceRoot = "output\acceptance"
$runbookPath = "docs\superpowers\acceptance\e2e-full-acceptance-runbook.md"
$checklistPath = "docs\superpowers\acceptance\e2e-manual-review-checklist.md"
$reportPath = "docs\superpowers\acceptance\e2e-release-report-template.md"
$backendTests = @(
  "tests/backend/test_payload_builder.py",
  "tests/backend/test_capture_graph_fixtures.py",
  "tests/backend/test_acceptance_case_db.py",
  "tests/backend/test_uploads_routes.py"
)
$backendTestArgs = @("-m", "pytest") + $backendTests + @("-v")
$playwrightFullArgs = @("--prefix", $frontendRoot, "run", "test:e2e:acceptance")
$playwrightCoreArgs = @("--prefix", $frontendRoot, "run", "test:e2e:acceptance", "--", "../tests/e2e/acceptance/workspace-core.spec.ts")

function Format-Command {
  param(
    [string]$Command,
    [string[]]$Arguments
  )

  $parts = @($Command) + $Arguments
  return ($parts | ForEach-Object {
      if ($_ -match "\s") {
        '"' + $_ + '"'
      } else {
        $_
      }
    }) -join " "
}

function Write-RunnerSummary {
  $backendTestCommand = Format-Command -Command $pythonExe -Arguments $backendTestArgs
  $playwrightFullCommand = Format-Command -Command $npmCmd -Arguments $playwrightFullArgs
  $playwrightCoreCommand = Format-Command -Command $npmCmd -Arguments $playwrightCoreArgs

  Write-Host "Full acceptance runner"
  Write-Host "Backend acceptance-support command:"
  Write-Host "  $backendTestCommand"
  Write-Host "Playwright full command:"
  Write-Host "  $playwrightFullCommand"
  Write-Host "Playwright core-only command:"
  Write-Host "  $playwrightCoreCommand"
  Write-Host "Playwright evidence:"
  Write-Host "  $evidenceRoot"
  Write-Host "Docs:"
  Write-Host "  $runbookPath"
  Write-Host "  $checklistPath"
  Write-Host "  $reportPath"
  if ($SkipManualDocs) {
    Write-Host "Manual docs reminder: suppressed by -SkipManualDocs"
  } else {
    Write-Host "Manual docs reminder: review the checklist and release report after automation passes"
  }
}

Write-RunnerSummary

if ($ListOnly) {
  Write-Host "Dry run only: no environment checks or tests were executed."
  return
}

foreach ($path in @($pythonExe, $npmCmd)) {
  if (-not (Test-Path $path)) {
    throw "Required tool not found at $path"
  }
}

Set-Location $repoRoot
$env:PATH = "$nodeHome;$env:PATH"

Write-Host "Running backend acceptance-support tests..."
& $pythonExe @backendTestArgs
if ($LASTEXITCODE -ne 0) {
  throw "Backend acceptance-support tests failed with exit code $LASTEXITCODE"
}

if ($CoreOnly) {
  Write-Host "Running core-only Playwright acceptance subset..."
  & $npmCmd @playwrightCoreArgs
} else {
  Write-Host "Running full Playwright acceptance pack..."
  & $npmCmd @playwrightFullArgs
}

if ($LASTEXITCODE -ne 0) {
  throw "Playwright acceptance run failed with exit code $LASTEXITCODE"
}

Write-Host "Playwright evidence lives at:"
Write-Host "  $evidenceRoot"
if (-not $SkipManualDocs) {
  Write-Host "Manual review handoff order:"
  Write-Host "  1. Inspect the evidence bundle under $evidenceRoot"
  Write-Host "  2. Complete $checklistPath"
  Write-Host "  3. Complete $reportPath"
  Write-Host "  4. Record any blockers before signing off"
}
