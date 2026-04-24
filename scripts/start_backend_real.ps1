param(
  [switch]$WarmupRag
)

$ErrorActionPreference = "Stop"

function Resolve-CommandPath {
  param(
    [string[]]$PreferredPaths = @(),
    [string[]]$CommandNames,
    [string]$DisplayName
  )

  foreach ($preferredPath in $PreferredPaths) {
    if ($preferredPath -and (Test-Path -LiteralPath $preferredPath)) {
      return $preferredPath
    }
  }

  foreach ($commandName in $CommandNames) {
    $command = Get-Command $commandName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
      if ($command.Source) { return $command.Source }
      if ($command.Path) { return $command.Path }
      if ($command.Definition) { return $command.Definition }
    }
  }

  throw "$DisplayName not found on PATH. Install it or add it to PATH."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultCondaEnv = "D:\anaconda3\envs\LangG"
if (Test-Path -LiteralPath $defaultCondaEnv) {
  $env:PATH = "$defaultCondaEnv;$env:PATH"
}
# New windows from Start-Process often have CONDA_PREFIX=base; prefer the repo's LangG
# env first so we do not run uvicorn with base Python (missing project deps like fastapi).
$preferredPythonPaths = @()
$langGPython = Join-Path $defaultCondaEnv "python.exe"
if (Test-Path -LiteralPath $langGPython) {
  $preferredPythonPaths += $langGPython
}
if ($env:CONDA_PREFIX) {
  $preferredPythonPaths += Join-Path $env:CONDA_PREFIX "python.exe"
}
$pythonExe = Resolve-CommandPath -PreferredPaths $preferredPythonPaths -CommandNames @("python", "py") -DisplayName "Python"

Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:AUTH_MODE = "none"
$env:GRAPH_RUNNER_MODE = "real"
$env:RAG_WARMUP = if ($WarmupRag) { "true" } else { "false" }
$env:FRONTEND_ORIGINS = "http://127.0.0.1:4173"

& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
