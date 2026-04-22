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
$frontendRoot = Join-Path $repoRoot "frontend"
$npmCmd = Resolve-CommandPath -CommandNames @("npm") -DisplayName "npm"

Set-Location $frontendRoot

$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"

& $npmCmd run dev:e2e
