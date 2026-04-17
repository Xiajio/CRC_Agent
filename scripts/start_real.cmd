@echo off
REM Run from repo root or anywhere: always cd to repo root (parent of scripts\)
cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_real.ps1" %*
