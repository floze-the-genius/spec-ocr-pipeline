@echo off
setlocal
chcp 65001 >nul
title Spec OCR Local Qwen Setup

fltmc >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo Запрашиваю права администратора для настройки WSL...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

cd /d "%~dp0"
echo Spec OCR: автоматическая установка локального Qwen3.6 Safe
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1" --profile qwen36-safe --yes %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Локальный режим настроен. Теперь положите PDF-файлы в INPUT_PDF и запустите PROCESS_INPUT_FOLDER_WINDOWS.cmd.
) else if "%EXIT_CODE%"=="3010" (
  echo Windows включила WSL-компоненты и требует перезагрузку.
  echo Перезагрузите компьютер, затем снова запустите INSTALL_LOCAL_QWEN_SAFE_WINDOWS.cmd.
) else (
  echo Настройка локального режима завершилась с ошибкой. Код: %EXIT_CODE%
  echo Не закрывайте окно: сделайте скриншот или скопируйте последние строки ошибки и отправьте в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
