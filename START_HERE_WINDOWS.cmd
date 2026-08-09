@echo off
setlocal
chcp 65001 >nul
title Spec OCR Setup

fltmc >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo Запрашиваю права администратора для настройки WSL...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

cd /d "%~dp0"
echo Spec OCR: мастер настройки Windows/WSL2
echo.
echo Нажмите Enter в меню модели, чтобы выбрать рекомендуемый локальный Qwen3.6 Safe.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Готово. Теперь положите PDF-файлы в INPUT_PDF и запустите PROCESS_INPUT_FOLDER_WINDOWS.cmd.
) else if "%EXIT_CODE%"=="3010" (
  echo Windows включила WSL-компоненты и требует перезагрузку.
  echo Перезагрузите компьютер, затем снова запустите START_HERE_WINDOWS.cmd.
) else (
  echo Настройка завершилась с ошибкой. Код: %EXIT_CODE%
  echo Не закрывайте окно: сделайте скриншот или скопируйте последние строки ошибки и отправьте в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
