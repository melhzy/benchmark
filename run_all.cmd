@echo off
rem run_all.cmd -- Windows launcher for run_all.ps1 (same options as run_all.sh, e.g. run_all.cmd --verify).
rem Starts Windows PowerShell with the script execution policy bypassed for this one script only.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
exit /b %ERRORLEVEL%
