param(
  [switch]$WarmupRag
)

$ErrorActionPreference = "Stop"

function Resolve-CommandPath {
  param(
    [string[]]$CommandNames,
    [string]$DisplayName
  )

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
$pythonExe = Resolve-CommandPath -CommandNames @("python", "py") -DisplayName "Python"

Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:AUTH_MODE = "none"
$env:GRAPH_RUNNER_MODE = "real"
$env:RAG_WARMUP = if ($WarmupRag) { "true" } else { "false" }
$env:FRONTEND_ORIGINS = "http://127.0.0.1:4173"

& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
