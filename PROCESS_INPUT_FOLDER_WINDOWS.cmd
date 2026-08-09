@echo off
setlocal
chcp 65001 >nul
title Spec OCR Process INPUT_PDF

cd /d "%~dp0"
echo Spec OCR: пакетная обработка PDF из INPUT_PDF с выбором модели
echo.
echo Положите PDF-файлы в папку INPUT_PDF и запустите этот файл.
echo Результаты будут сохранены в OUTPUT_RESULTS.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0process-pdf-windows.ps1" -Batch -InputDir "%~dp0INPUT_PDF" -OutputDir "%~dp0OUTPUT_RESULTS" %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
  echo Пакетная обработка завершена. Готовые Excel-файлы лежат в OUTPUT_RESULTS.
) else (
  echo Пакетная обработка завершилась с ошибкой. Код: %EXIT_CODE%
  echo Не закрывайте окно: сделайте скриншот или скопируйте последние строки ошибки и отправьте в поддержку.
)
echo.
pause
exit /b %EXIT_CODE%
