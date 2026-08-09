@echo off
setlocal
chcp 65001 >nul
title Spec OCR GPT Setup

cd /d "%~dp0"
echo Spec OCR: автоматическая установка GPT-режима
echo Для searchable PDF этот режим не требует WSL и прав администратора.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1" -GptFastPath %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo GPT-режим настроен. Теперь положите PDF-файлы в INPUT_PDF и запустите PROCESS_INPUT_FOLDER_GPT_WINDOWS.cmd.
) else (
  echo Настройка GPT-режима завершилась с ошибкой. Код: %EXIT_CODE%
  echo Проверьте интернет, затем отправьте скриншот ошибки в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
