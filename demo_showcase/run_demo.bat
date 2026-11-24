@echo off
echo ==========================================================
echo   Launching 2D Skeleton Gait Biometrics Academic Demo
echo   Web Dashboard: http://127.0.0.1:8501
echo ==========================================================

call .\.venv\Scripts\activate.bat
python -m uvicorn demo_showcase.app:app --host 127.0.0.1 --port 8501 --reload
pause
