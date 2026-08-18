@echo off
title Fingerprint API Server (v4)
color 0B

echo ============================================
echo   Fingerprint Matching API Server  v4
echo ============================================
echo.

cd /d "%~dp0fingerprint_api_v4"

if not exist "..\fingerprint_model_v4.h5" (
    echo [!] fingerprint_model_v4.h5 not found in V4\
    echo [!] Train it first:  python V4\train_v4.py
    pause
    exit /b 1
)

echo [OK] v4 model found
echo [*]  starting API server on port 8001...
echo.
echo ============================================
echo   API URL : http://localhost:8001
echo   API Docs: http://localhost:8001/docs
echo.
echo   (v3 API on port 8000 can run in parallel)
echo ============================================
echo.
echo Press Ctrl+C to stop the server.
echo.

C:\Users\nakor\.conda\envs\fingerprint\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001

echo.
echo [!] Server stopped.
pause
