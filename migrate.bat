@echo off
title ProjectGabriel - Memory Migration
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\migrate_memories.py
) else (
    python scripts\migrate_memories.py
)
pause
