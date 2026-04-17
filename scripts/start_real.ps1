param(
  [switch]$WarmupRag
)

$ErrorActionPreference = "Stop"

# 强制 UTF-8 输出，避免中文乱码
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot   = Split-Path -Parent $PSScriptRoot
$pythonExe  = "D:\anaconda3\envs\LangG\python.exe"
$nodeHome   = "D:\anaconda3\envs\LangG"
$npmCmd     = Join-Path $nodeHome "npm.cmd"
$nodeExe    = Join-Path $nodeHome "node.exe"

if (-not (Test-Path $pythonExe)) { throw "LangG python not found at $pythonExe" }
if (-not (Test-Path $npmCmd))    { throw "LangG npm.cmd not found at $npmCmd" }
if (-not (Test-Path $nodeExe))   { throw "LangG node.exe not found at $nodeExe" }

# ── 后端：在独立窗口中运行，窗口关闭时进程也退出 ──────────────────────────
$backendArgs = @(
  "-NoExit", "-Command",
  "`$env:PYTHONUTF8='1'; `$env:AUTH_MODE='none'; `$env:GRAPH_RUNNER_MODE='real'; " +
  "`$env:RAG_WARMUP='$(if ($WarmupRag) { 'true' } else { 'false' })'; " +
  "`$env:FRONTEND_ORIGINS='http://127.0.0.1:4173'; " +
  "Set-Location '$repoRoot'; " +
  "& '$pythonExe' -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
)

Write-Host ">>> 启动后端 (新窗口)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList $backendArgs

# 稍等一秒让后端窗口先弹出
Start-Sleep -Seconds 1

# ── 前端：在当前窗口运行 ──────────────────────────────────────────────────
Write-Host ">>> 启动前端..." -ForegroundColor Cyan

Set-Location (Join-Path $repoRoot "frontend")

$env:PATH = "$nodeHome;$env:PATH"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"

& $npmCmd run dev:e2e
