@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
echo ========================================================
echo  Starting Chemotherapy PROMIS Risk Streamlit Dashboard
echo ========================================================
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Streamlit exited with an error.
    pause
)
