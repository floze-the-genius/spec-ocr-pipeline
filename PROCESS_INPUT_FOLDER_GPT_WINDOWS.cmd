@echo off
setlocal
chcp 65001 >nul
title Spec OCR Process INPUT_PDF GPT

cd /d "%~dp0"
echo Spec OCR: пакетная обработка PDF из INPUT_PDF через GPT-5.5 Extra High
echo.
echo Положите PDF-файлы в папку INPUT_PDF и запустите этот файл.
echo Результаты будут сохранены в OUTPUT_RESULTS.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1" -Profile gpt -Batch -InputDir "%~dp0INPUT_PDF" -OutputDir "%~dp0OUTPUT_RESULTS" %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Пакетная обработка через GPT завершена. Готовые Excel-файлы лежат в OUTPUT_RESULTS.
) else (
  echo Пакетная обработка через GPT завершилась с ошибкой. Код: %EXIT_CODE%
  echo Проверьте интернет и GPT-аккаунт, затем отправьте скриншот ошибки в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
