@echo off
setlocal
chcp 65001 >nul
title Spec OCR Process PDF

cd /d "%~dp0"
echo Spec OCR: обработка PDF с выбором модели
echo.

if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1" -PdfPath "%~1"
)
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Обработка завершена. Готовые Excel-файлы лежат в OUTPUT_RESULTS.
) else (
  echo Обработка завершилась с ошибкой. Код: %EXIT_CODE%
  echo Не закрывайте окно: сделайте скриншот или скопируйте последние строки ошибки и отправьте в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
