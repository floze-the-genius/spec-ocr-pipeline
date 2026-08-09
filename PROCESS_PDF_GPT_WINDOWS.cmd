@echo off
setlocal
chcp 65001 >nul
title Spec OCR Process PDF GPT

cd /d "%~dp0"
echo Spec OCR: обработка PDF через GPT-5.5 Extra High
echo.

if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1" -Profile gpt
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1" -Profile gpt -PdfPath "%~1"
)
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Обработка через GPT завершена. Готовые Excel-файлы лежат в OUTPUT_RESULTS.
) else (
  echo Обработка через GPT завершилась с ошибкой. Код: %EXIT_CODE%
  echo Проверьте интернет и GPT-аккаунт, затем отправьте скриншот ошибки в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
