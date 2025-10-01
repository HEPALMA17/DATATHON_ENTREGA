@echo off
title Decision AI - Sistema de Recrutamento Inteligente
color 0A
cls

echo.
echo ============================================================
echo       DECISION AI - Sistema de Recrutamento Inteligente
echo ============================================================
echo.
echo [INFO] Iniciando o servidor Streamlit...
echo.
echo [URL]  http://localhost:8501
echo.
echo [DICA] O navegador abrira automaticamente
echo [AVISO] Para parar o servidor pressione Ctrl+C
echo.
echo ============================================================
echo.

cd /d "%~dp0"
streamlit run app\app.py --server.port 8501

pause

