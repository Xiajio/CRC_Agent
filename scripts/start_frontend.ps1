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
$frontendRoot = Join-Path $repoRoot "frontend"
$defaultCondaEnv = "D:\anaconda3\envs\LangG"
if (Test-Path -LiteralPath $defaultCondaEnv) {
  $env:PATH = "$defaultCondaEnv;$env:PATH"
}
$preferredNpmPaths = @()
if ($env:CONDA_PREFIX) {
  $preferredNpmPaths += Join-Path $env:CONDA_PREFIX "npm.cmd"
}
$preferredNpmPaths += Join-Path $defaultCondaEnv "npm.cmd"
$npmCmd = Resolve-CommandPath -PreferredPaths $preferredNpmPaths -CommandNames @("npm") -DisplayName "npm"

Set-Location $frontendRoot

$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"

& $npmCmd run dev:e2e
