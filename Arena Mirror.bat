@echo off
REM ===========================================================================
REM  Arena -> XMage mirror launcher.
REM  Double-click this file to open the follower GUI: locate your MTGA logs,
REM  click Start, and play a match. XMage opens automatically on the first
REM  live game and mirrors it while recording a CABT replay bundle.
REM
REM  First run only: build XMage once with scripts\setup-arena-mirror.ps1
REM  (this launcher will tell you if that hasn't been done yet).
REM ===========================================================================
title Arena Mirror
cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\arena-mirror-gui.ps1" %*

if errorlevel 1 (
  echo.
  echo ---------------------------------------------------------------------------
  echo  The launcher hit an error.
  echo  If this is your first run, build XMage once ^(takes a while^), then retry:
  echo.
  echo      powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup-arena-mirror.ps1"
  echo ---------------------------------------------------------------------------
  echo.
  pause
)
