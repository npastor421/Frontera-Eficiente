@echo off
title Calculadora de Frontera Eficiente
cd /d "%~dp0"
echo =========================================================
echo   Iniciando Calculadora de Frontera Eficiente...
echo =========================================================
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocurrio un error al iniciar la aplicacion.
    pause
)
