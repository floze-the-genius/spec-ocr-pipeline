param(
    [ValidateSet("local", "gpt", "kimi", "kimi-check", "gpt-check")]
    [string]$Mode = "",
    [string]$Profile = "",
    [string]$PdfPath = "",
    [string]$InputDir = "",
    [string]$OutputDir = "",
    [switch]$Batch,
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if ([string]::IsNullOrWhiteSpace($InputDir)) {
    $InputDir = Join-Path $ProjectDir "INPUT_PDF"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectDir "OUTPUT_RESULTS"
}
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$PaddleCudaPackage = "paddlepaddle-gpu==3.2.1"
$PaddleCudaIndexUrl = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
$PaddleOcrPackage = "paddleocr[doc-parser]>=3.0.0"
$PaddleSafetensorsWheel = "https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl"
$LlamaCppPackage = "llama-cpp-python[server]"
$LlamaCppCudaWheelPackage = "llama-cpp-python[server] @ https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.21-cu124/llama_cpp_python-0.3.21-py3-none-linux_x86_64.whl"
$LlamaCppCudaRuntimePackage = "nvidia-cuda-runtime-cu12==12.6.77"
$LlamaCppCudaBlasPackage = "nvidia-cublas-cu12==12.6.4.1"
$LlamaCppPaddleNumpyPin = "numpy<2.4,>=1.24"
$LlamaCppPaddlePyYamlPin = "PyYAML<=6.0.2"
$LlamaCppCudaLibraryPathScript = 'for d in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib /usr/local/cuda/lib64 /usr/lib/wsl/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done'
$SpecOcrHotfixVersion = "LocalModelSizeTolerance_v87"
$NativePdfTextTimeoutSeconds = 90
if (-not [string]::IsNullOrWhiteSpace($env:SPEC_OCR_NATIVE_PDF_TEXT_TIMEOUT_SEC)) {
    $parsedNativeTimeout = 0
    if ([int]::TryParse($env:SPEC_OCR_NATIVE_PDF_TEXT_TIMEOUT_SEC, [ref]$parsedNativeTimeout) -and $parsedNativeTimeout -gt 0) {
        $NativePdfTextTimeoutSeconds = $parsedNativeTimeout
    }
}

$ProcessingProfiles = @(
    @{
        Name = "gpt"
        Title = "GPT-5.5 Extra High"
        Description = "Основной онлайн-режим максимального качества: AI-разбор делает GPT через OpenCode, OCR-текстом и изображениями страниц. Требует входа в ваш собственный OpenCode аккаунт внутри WSL."
        Mode = "gpt"
        Model = "openai/gpt-5.5"
        ReasoningEffort = "xhigh"
        ServedModel = ""
        RuntimeInstall = ""
        ThirdLayerReviewer = "opencode-gpt"
        ThirdLayerModel = "openai/gpt-5.5"
        ThirdLayerReasoningEffort = "xhigh"
    },
    @{
        Name = "kimi-k2.6-turbo"
        Title = "Kimi K2.6 Turbo"
        Description = "Основной онлайн-режим: AI-разбор делает Kimi через Fireworks с OCR-текстом и изображениями страниц. Требует FIREWORKS_API_KEY в WSL; ключ не хранится в архиве."
        Mode = "fireworks"
        Model = "accounts/fireworks/routers/kimi-k2p6-turbo"
        ServedModel = "accounts/fireworks/routers/kimi-k2p6-turbo"
        BaseUrl = "https://api.fireworks.ai/inference/v1"
        ApiKeyEnv = "FIREWORKS_API_KEY"
        RuntimeInstall = ""
    },
    @{
        Name = "qwen36-safe"
        Title = "Qwen3.6 Safe"
        Description = "Основной локальный режим: стабильный текстовый разбор результата OCR."
        Mode = "local"
        Model = "qwen3.6-safe"
        ServedModel = "qwen3.6-safe-local"
        RuntimeInstall = "llama-cpp-python[server]"
    },
    @{
        Name = "qwen36-balanced"
        Title = "Qwen3.6 Balanced"
        Description = "Экспериментально: тяжелее safe, может не стартовать на 32GB RAM."
        Mode = "local"
        Model = "qwen3.6-balanced"
        ServedModel = "qwen3.6-balanced-local"
        RuntimeInstall = "llama-cpp-python[server]"
    },
    @{
        Name = "qwen36-quality"
        Title = "Qwen3.6 Quality"
        Description = "Экспериментально: самый тяжелый GGUF, для текущей машины не рекомендуется."
        Mode = "local"
        Model = "qwen3.6-quality"
        ServedModel = "qwen3.6-quality-local"
        RuntimeInstall = "llama-cpp-python[server]"
    },
    @{
        Name = "qwen36-max"
        Title = "Qwen3.6 Max"
        Description = "Экспериментальный GPTQ/vLLM вариант; для 16GB VRAM не гарантируется."
        Mode = "local"
        Model = "qwen36-max"
        ServedModel = "qwen36-max-local"
        RuntimeInstall = "vllm"
    },
    @{
        Name = "qwen36-safe-kimi-check"
        Title = "Qwen3.6 Safe + Kimi контроль"
        Description = "Основной разбор делает qwen36-safe; Kimi проверяет только подозрительные строки третьим слоем. Требует FIREWORKS_API_KEY."
        Mode = "local"
        Model = "qwen3.6-safe"
        ServedModel = "qwen3.6-safe-local"
        RuntimeInstall = "llama-cpp-python[server]"
        SetupProfile = "qwen36-safe"
        ThirdLayerReviewer = "kimi-fireworks"
        ThirdLayerModel = "accounts/fireworks/routers/kimi-k2p6-turbo"
        ThirdLayerBaseUrl = "https://api.fireworks.ai/inference/v1"
        ThirdLayerApiKeyEnv = "FIREWORKS_API_KEY"
        ThirdLayerReasoningEffort = "none"
    },
    @{
        Name = "qwen36-safe-gpt-check"
        Title = "Qwen3.6 Safe + GPT контроль"
        Description = "Основной разбор делает qwen36-safe; GPT/OpenCode проверяет только подозрительные строки третьим слоем."
        Mode = "local"
        Model = "qwen3.6-safe"
        ServedModel = "qwen3.6-safe-local"
        RuntimeInstall = "llama-cpp-python[server]"
        SetupProfile = "qwen36-safe"
        ThirdLayerReviewer = "opencode-gpt"
        ThirdLayerModel = "openai/gpt-5.5"
        ThirdLayerReasoningEffort = "xhigh"
    }
)

function Get-ProcessingProfile {
    param([string]$Name)
    foreach ($profileItem in $ProcessingProfiles) {
        if ($profileItem.Name -eq $Name) {
            return $profileItem
        }
    }
    return $null
}

function Select-ProcessingProfile {
    if (-not [string]::IsNullOrWhiteSpace($Profile)) {
        $selected = Get-ProcessingProfile -Name $Profile
        if ($null -eq $selected) {
            throw "Unknown processing profile: $Profile"
        }
        return $selected
    }
    if ($Mode -eq "gpt") {
        return Get-ProcessingProfile -Name "gpt"
    }
    if ($Mode -eq "kimi") {
        return Get-ProcessingProfile -Name "kimi-k2.6-turbo"
    }
    if ($Mode -eq "kimi-check") {
        return Get-ProcessingProfile -Name "qwen36-safe-kimi-check"
    }
    if ($Mode -eq "gpt-check") {
        return Get-ProcessingProfile -Name "qwen36-safe-gpt-check"
    }
    if ($Mode -eq "local") {
        return Get-ProcessingProfile -Name "qwen36-safe"
    }

    Write-Host "Выберите модель обработки:"
    for ($i = 0; $i -lt $ProcessingProfiles.Count; $i++) {
        $item = $ProcessingProfiles[$i]
        $number = $i + 1
        Write-Host "$number. $($item.Title)"
        Write-Host "   $($item.Name): $($item.Description)"
    }
    $answer = (Read-Host "Номер модели [1]").Trim()
    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $ProcessingProfiles[0]
    }
    $index = 0
    if ([int]::TryParse($answer, [ref]$index)) {
        if ($index -ge 1 -and $index -le $ProcessingProfiles.Count) {
            return $ProcessingProfiles[$index - 1]
        }
    }
    $selectedByName = Get-ProcessingProfile -Name $answer
    if ($null -ne $selectedByName) {
        return $selectedByName
    }
    throw "Unknown processing profile selection: $answer"
}

function Convert-ToWslPath {
    param([string]$Path)
    $normalized = $Path.Replace("\", "/")
    if ($normalized -match "^([A-Za-z]):/(.*)$") {
        return "/mnt/$($Matches[1].ToLower())/$($Matches[2])"
    }
    return $normalized
}

function Quote-BashArg {
    param([string]$Value)
    $quote = [string][char]39
    return $quote + $Value.Replace($quote, $quote + '\' + $quote + $quote) + $quote
}

function Quote-PowerShellLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Quote-WindowsProcessArg {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $doubleQuote = [string][char]34
    $backslash = [string][char]92
    $result = $doubleQuote
    $slashCount = 0
    foreach ($ch in $Value.ToCharArray()) {
        if ($ch -eq [char]92) {
            $slashCount += 1
            continue
        }
        if ($ch -eq [char]34) {
            if ($slashCount -gt 0) {
                $result += $backslash * ($slashCount * 2)
            }
            $result += $backslash + $doubleQuote
            $slashCount = 0
            continue
        }
        if ($slashCount -gt 0) {
            $result += $backslash * $slashCount
            $slashCount = 0
        }
        $result += [string]$ch
    }
    if ($slashCount -gt 0) {
        $result += $backslash * ($slashCount * 2)
    }
    $result += $doubleQuote
    return $result
}

function Write-Utf8NoBomFile {
    param(
        [string]$Path,
        [string]$Value
    )
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Value, $utf8NoBom)
}

function Get-PaddleCudaRepairScript {
    $pipBase = "python -m pip install --disable-pip-version-check --retries 10 --timeout 120"
    $check = "spec-ocr paddle-gpu-check --require-cuda --json"
    $repairSteps = @(
        "python -m pip uninstall -y paddlepaddle paddlepaddle-gpu",
        "$pipBase -U $(Quote-BashArg $PaddleOcrPackage)",
        "$pipBase -U $(Quote-BashArg $PaddleSafetensorsWheel)",
        "python -m pip uninstall -y paddlepaddle paddlepaddle-gpu",
        "$pipBase $(Quote-BashArg $PaddleCudaPackage) -i $(Quote-BashArg $PaddleCudaIndexUrl)",
        $check
    )
    return "$check || ($($repairSteps -join '; '))"
}

function Get-LlamaCudaRepairScript {
    $pipBase = "python -m pip install --disable-pip-version-check --retries 10 --timeout 120"
    $check = "$LlamaCppCudaLibraryPathScript; spec-ocr llama-gpu-check --require-cuda --json"
    $repairSteps = @(
        "python -m pip uninstall -y llama-cpp-python llama_cpp_python",
        "$pipBase --upgrade --force-reinstall --no-cache-dir --only-binary llama-cpp-python --prefer-binary $(Quote-BashArg $LlamaCppCudaWheelPackage) $(Quote-BashArg $LlamaCppCudaRuntimePackage) $(Quote-BashArg $LlamaCppCudaBlasPackage) $(Quote-BashArg $LlamaCppPaddleNumpyPin) $(Quote-BashArg $LlamaCppPaddlePyYamlPin)",
        "spec-ocr llama-link-cuda-libs --json",
        $check
    )
    return "$check || ($($repairSteps -join '; '))"
}

function Resolve-WslDistroName {
    param([string]$Requested)
    $rawNames = @(& wsl.exe --list --quiet 2>$null)
    $names = @()
    foreach ($raw in $rawNames) {
        $name = ($raw -replace "`0", "").Trim().TrimStart("*").Trim()
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $names += $name
        }
    }
    if ($names -contains $Requested) {
        return $Requested
    }
    foreach ($preferred in @("Ubuntu-24.04", "Ubuntu")) {
        if ($names -contains $preferred) {
            return $preferred
        }
    }
    $ubuntu = @($names | Where-Object { $_ -like "Ubuntu*" } | Select-Object -First 1)
    if ($ubuntu.Count -gt 0) {
        return $ubuntu[0]
    }
    throw "WSL Ubuntu distribution was not found. Run 00_INSTALL_WSL_AS_ADMIN.cmd, finish Ubuntu setup, then run this file again."
}

function Show-LogTail {
    param(
        [string]$Path,
        [int]$Lines = 80
    )
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return
    }
    Write-Host ""
    Write-Host "Настоящая ошибка из лога: $Path" -ForegroundColor Yellow
    Get-Content -LiteralPath $Path -Tail $Lines | ForEach-Object {
        Write-Host $_
    }
}

function Get-WslStepScriptPath {
    param(
        [string]$Title,
        [string]$LogPath
    )
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        return [System.IO.Path]::ChangeExtension($LogPath, ".sh")
    }
    $safeTitle = ($Title -replace "[^A-Za-z0-9А-Яа-я._-]+", "_").Trim("_")
    if ([string]::IsNullOrWhiteSpace($safeTitle)) {
        $safeTitle = "step"
    }
    return (Join-Path $OutputDir "_wsl-step-$safeTitle.sh")
}

function Invoke-WslStep {
    param(
        [string]$Title,
        [string]$Script,
        [string]$QualityGateRunDir = "",
        [switch]$ContinueOnQualityGateFailure,
        [string]$LogPath = ""
    )
    $script:LastWslStepExitCode = 0
    Write-Host ""
    Write-Host "== $Title =="
    $scriptPathWindows = Get-WslStepScriptPath -Title $Title -LogPath $LogPath
    $scriptText = (($Script -replace "`r`n", "`n") -replace "`r", "`n")
    Write-Utf8NoBomFile -Path $scriptPathWindows -Value $scriptText
    $scriptPathWsl = Convert-ToWslPath $scriptPathWindows
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
        Remove-Item -Force -ErrorAction SilentlyContinue $LogPath
        @(
            "== $Title ==",
            "time: $(Get-Date -Format o)",
            "distro: $Distro",
            "script: $scriptPathWindows",
            ""
        ) | Set-Content -Path $LogPath -Encoding UTF8
        & wsl.exe -d $Distro -- bash $scriptPathWsl 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
        Add-Content -Path $LogPath -Encoding UTF8 -Value @("", "[spec-ocr] exit_code=$exitCode")
    } else {
        & wsl.exe -d $Distro -- bash $scriptPathWsl
        $exitCode = $LASTEXITCODE
    }
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
        if (($exitCode -eq 2) -and -not [string]::IsNullOrWhiteSpace($QualityGateRunDir)) {
            $statusPath = Join-Path $QualityGateRunDir 'processing.status.txt'
            @(
                'quality_gate_failed',
                'Excel не создан, потому что распознавание содержит подозрительные места.',
                'Проверьте rows.ai.quality.json, rows.ai.jsonl и logs\02-ai-map.log в этой папке результата.'
            ) | Set-Content -Path $statusPath -Encoding UTF8
            Write-Host ""
            Write-Host "Остановлено контролем качества OCR." -ForegroundColor Yellow
            Write-Host "Excel не создан, потому что распознавание содержит подозрительные места."
            Write-Host "Это защита от выдачи плохого результата как готового."
            Write-Host ""
            Write-Host "Проверьте диагностические файлы в папке результата:"
            Write-Host "  $(Join-Path $QualityGateRunDir 'rows.ai.quality.json')"
            Write-Host "  $(Join-Path $QualityGateRunDir 'rows.ai.jsonl')"
            Write-Host "  $(Join-Path $QualityGateRunDir 'logs\02-ai-map.log')"
            Write-Host "  $statusPath"
            Write-Host ""
            Show-LogTail -Path $LogPath
            if ($ContinueOnQualityGateFailure) {
                Write-Host "Пакетная обработка продолжится со следующего PDF." -ForegroundColor Yellow
                $script:LastWslStepExitCode = 2
                return
            }
            exit 2
        }
        Show-LogTail -Path $LogPath
        if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
            throw "$Title failed with exit code $exitCode. Log: $LogPath"
        }
        throw "$Title failed with exit code $exitCode"
    }
}

function Test-WslFileNonEmpty {
    param([string]$Path)
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & wsl.exe -d $Distro -- bash -lc "test -s $(Quote-BashArg $Path)" *> $null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    return ($exitCode -eq 0)
}

function Test-GptAuthInWsl {
    param([string]$ProjectWsl)

    $script = @(
        "set -euo pipefail",
        "cd $(Quote-BashArg $ProjectWsl)",
        'auth="$HOME/.local/share/opencode/auth.json"',
        'if [ ! -s "$auth" ]; then',
        '  echo "OpenCode is not authenticated in this WSL account." >&2',
        '  echo "Open a WSL terminal, sign in with your own OpenCode account, and rerun this command." >&2',
        '  echo "Never copy or share another person''s auth.json session." >&2',
        '  exit 21',
        'fi',
        'echo "OpenCode auth is present for the current WSL user."'
    ) -join "`n"
    Invoke-WslStep -Title "Проверка личной OpenCode-сессии в WSL" -Script $script -LogPath (Join-Path $OutputDir "_gpt-auth-check.log")
}

function Ensure-OpenCodeCliInWsl {
    param([string]$ProjectWsl)

    $script = @(
        "set -euo pipefail",
        "cd $(Quote-BashArg $ProjectWsl)",
        'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"',
        'if ! command -v opencode >/dev/null 2>&1; then',
        '  echo "OpenCode CLI is not installed in this WSL account." >&2',
        '  echo "Install it from the official OpenCode documentation, review the installer, then rerun this command." >&2',
        '  exit 127',
        'fi',
        'echo "OpenCode CLI found: $(command -v opencode)"',
        'opencode --version || true'
    ) -join "`n"
    Invoke-WslStep -Title "Проверка OpenCode CLI в WSL" -Script $script -LogPath (Join-Path $OutputDir "_gpt-opencode-check.log")
}

function Test-GptOpenCodeInWsl {
    param([string]$ProjectWsl)

    $prompt = Quote-BashArg "Ответь ровно OK и больше ничего."
    $script = @"
set -euo pipefail
cd $(Quote-BashArg $ProjectWsl)
. .venv-wsl/bin/activate
export PATH="`$HOME/.opencode/bin:`$HOME/.local/bin:`$PATH"
set +e
output=`$(opencode run --model openai/gpt-5.5 --agent build --variant xhigh --format default $prompt 2>&1)
code=`$?
set -e
printf '%s\n' "`$output"
# opencode иногда печатает "Error: Forbidden / Token refresh failed", но возвращает код 0,
# поэтому классифицируем ошибки провайдера/авторизации по ВЫВОДУ, не только по коду выхода.
  if printf '%s\n' "`$output" | grep -Eiq 'forbidden|gateway|proxy|permission|provider settings|access this resource'; then
    cat >&2 <<'SPEC_OCR_GPT_FORBIDDEN'
SPEC_OCR_GPT_ACCOUNT_FORBIDDEN
GPT/OpenCode дошел до провайдера, но запрос отклонен: Forbidden/gateway/proxy/provider settings.
Это не ошибка PDF, Python, WSL или Excel.
Что проверить:
1. На этом компьютере/в WSL должен быть доступ к OpenAI/OpenCode без региональной или корпоративной блокировки. Если сеть блокирует OpenAI, включите VPN/proxy именно для WSL/Ubuntu.
2. Войдите в собственный OpenCode-аккаунт внутри этого WSL-пользователя.
3. Проверьте, что ваш аккаунт имеет доступ к выбранной модели. Никогда не копируйте чужой auth.json.
SPEC_OCR_GPT_FORBIDDEN
    exit 23
  fi
  if printf '%s\n' "`$output" | grep -Eiq 'token refresh failed|invalid_grant|unauthorized|failed: 401'; then
    cat >&2 <<'SPEC_OCR_GPT_AUTH_STALE_EOF'
SPEC_OCR_GPT_AUTH_STALE
GPT/OpenCode не смог обновить токен доступа (401).
Обычно это значит, что личная OpenCode-сессия этого WSL-пользователя устарела.
Это не ошибка PDF, Python, WSL или Excel.
Что делать:
1. Откройте WSL-терминал и повторно войдите в свой OpenCode-аккаунт.
2. Не передавайте и не коммитьте auth.json, токены или файлы сессии.
3. Запустите обработку снова.
SPEC_OCR_GPT_AUTH_STALE_EOF
    exit 24
  fi
if [ "`$code" -ne 0 ]; then
  echo "GPT/OpenCode smoke failed with exit code `$code" >&2
  exit "`$code"
fi
clean=`$(printf '%s\n' "`$output" | python3 -c 'import re,sys; text=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",sys.stdin.read()).replace("\r",""); lines=[line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("> ")]; print(lines[-1] if lines else "")')
if [ "`$clean" != "OK" ]; then
  echo "GPT/OpenCode smoke expected OK, got: `$clean" >&2
  exit 22
fi
echo "GPT/OpenCode smoke: OK"
"@
    Invoke-WslStep -Title "Проверка GPT/OpenCode аккаунта в WSL" -Script $script -LogPath (Join-Path $OutputDir "_gpt-opencode-smoke.log")
}

function Get-WindowsPowerShellExe {
    if ($env:SystemRoot) {
        $candidate = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return "powershell.exe"
}

function Test-Admin {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

function Test-WslVenvReady {
    param(
        [string]$RequestedDistro,
        [string]$ProjectWsl
    )
    $check = "set -euo pipefail; cd $(Quote-BashArg $ProjectWsl); test -f .venv-wsl/bin/activate"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & wsl.exe -d $RequestedDistro -- bash -lc $check *> $null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    return ($exitCode -eq 0)
}

function Ensure-GptWslFallbackEnvironment {
    param(
        [string]$RequestedDistro,
        [string]$ProjectWsl
    )
    if (Test-WslVenvReady -RequestedDistro $RequestedDistro -ProjectWsl $ProjectWsl) {
        return
    }

    Write-Host ""
    Write-Host "PDF не содержит пригодный PDF_TABLE_LAYER; включаю полный GPT/WSL fallback." -ForegroundColor Yellow
    Write-Host "Сейчас будет настроен WSL. OpenCode должен быть установлен и авторизован вашим собственным аккаунтом внутри WSL."
    Write-Host "Если Windows попросит перезагрузку, перезагрузите компьютер и запустите обработку еще раз."

    $setupScript = Join-Path $ProjectDir "setup-windows.ps1"
    $powershellExe = Get-WindowsPowerShellExe
    $setupExitCode = 0
    if (Test-Admin) {
        & $powershellExe -NoProfile -ExecutionPolicy Bypass -File $setupScript --profile gpt --yes
        $setupExitCode = $LASTEXITCODE
    } else {
        Write-Host "Запрашиваю права администратора для первичной настройки WSL..."
        $setupArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$setupScript`" --profile gpt --yes"
        $setupProcess = Start-Process -FilePath $powershellExe -ArgumentList $setupArguments -Verb RunAs -Wait -PassThru
        $setupExitCode = $setupProcess.ExitCode
    }
    if (($setupExitCode -eq 3010) -or ($setupExitCode -eq 1641)) {
        throw "Автонастройка GPT/WSL включила компоненты Windows и требует перезагрузку. Перезагрузите компьютер, затем снова запустите этот файл."
    }
    if ($setupExitCode -ne 0) {
        throw "Автонастройка GPT/WSL fallback failed with exit code $setupExitCode. Запустите INSTALL_GPT_WINDOWS.cmd от администратора, затем повторите обработку."
    }
    if (-not (Test-WslVenvReady -RequestedDistro $RequestedDistro -ProjectWsl $ProjectWsl)) {
        throw "Автонастройка GPT/WSL fallback завершилась, но .venv-wsl не найден. Если Windows просила перезагрузку, перезагрузите компьютер и повторите обработку."
    }
}

function Get-LocalServerModelIds {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" -TimeoutSec 2
        if ($null -eq $response.data) {
            return @()
        }
        $ids = @()
        foreach ($model in $response.data) {
            $ids += $model.id
        }
        return $ids
    } catch {
        return @()
    }
}

function Ensure-LocalServer {
    param(
        [string]$ProjectWsl,
        [string]$Model,
        [string]$ServedModel
    )
    $runningModels = @(Get-LocalServerModelIds)
    if ($runningModels -contains $ServedModel) {
        throw "Порт 8000 уже занят Qwen-сервером $ServedModel. Закройте старое окно Spec OCR Local Qwen Server и запустите обработку снова, чтобы новый CUDA runtime точно применился."
    }
    if ($runningModels.Count -gt 0) {
        throw "Порт 8000 уже занят другой моделью: $($runningModels -join ', '). Закройте окно локального сервера и запустите обработку снова."
    }

    Write-Host "Запускаю локальный сервер $ServedModel в отдельном окне..."
    $serverScript = "set -euo pipefail; cd $(Quote-BashArg $ProjectWsl); . .venv-wsl/bin/activate; $LlamaCppCudaLibraryPathScript; spec-ocr serve-local $Model --cache-dir artifacts/models --host 127.0.0.1 --port 8000 --served-model-name $ServedModel"
    $serverDistro = ($Distro -replace "[^A-Za-z0-9._-]", "").Trim()
    $serverPsPath = Join-Path $ProjectDir "run-local-qwen-server.ps1"
    $serverFailPath = Join-Path $ProjectDir "run-local-qwen-server.failed.txt"
    Remove-Item -Force -ErrorAction SilentlyContinue $serverFailPath
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $ProjectDir "run-local-qwen-server.cmd")
    $serverPs = @"
`$ErrorActionPreference = "Continue"
`$Host.UI.RawUI.WindowTitle = "Spec OCR Local Qwen Server"
Set-Location `$PSScriptRoot
`$Distro = $(Quote-PowerShellLiteral $serverDistro)
`$ServerScript = $(Quote-PowerShellLiteral $serverScript)
`$FailPath = $(Quote-PowerShellLiteral $serverFailPath)

function Invoke-QwenServerWsl {
    param(
        [string[]]`$WslArgs,
        [string]`$Label
    )
    Write-Host "Starting Qwen server via `$Label"
    & wsl.exe @WslArgs -- bash -lc `$ServerScript
    return `$LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace(`$Distro)) {
    `$code = Invoke-QwenServerWsl -WslArgs @() -Label "default WSL distro"
} else {
    `$code = Invoke-QwenServerWsl -WslArgs @("-d", `$Distro) -Label "WSL distro `$Distro"
    if (`$code -ne 0) {
        Write-Host ""
        Write-Host "WSL launch failed for distro `$Distro. Retrying default WSL distro..."
        `$code = Invoke-QwenServerWsl -WslArgs @() -Label "default WSL distro"
    }
}

if (`$code -ne 0) {
    "Local Qwen server failed with exit code `$code" | Set-Content -Path `$FailPath -Encoding UTF8
    Write-Host ""
    Write-Host "Local Qwen server failed with exit code `$code."
    Write-Host "Failure marker: `$FailPath"
    Read-Host "Press Enter to close this server window"
    exit `$code
}
"@
    Set-Content -Path $serverPsPath -Value $serverPs -Encoding UTF8
    $powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $serverProcess = Start-Process -FilePath $powershellExe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $serverPsPath) -PassThru

    Write-Host "Жду готовности сервера http://127.0.0.1:8000/v1/models ..."
    $deadline = (Get-Date).AddMinutes(10)
    while ((Get-Date) -lt $deadline) {
        $runningModels = @(Get-LocalServerModelIds)
        if ($runningModels -contains $ServedModel) {
            Write-Host "Локальный сервер $ServedModel готов."
            return
        }
        if (Test-Path $serverFailPath) {
            $serverError = Get-Content -Path $serverFailPath -Raw
            throw "Окно локального сервера завершилось с ошибкой: $serverError"
        }
        if ($serverProcess.HasExited) {
            throw "Окно локального сервера завершилось до готовности. Проверьте run-local-qwen-server.ps1 или окно Spec OCR Local Qwen Server."
        }
        Start-Sleep -Seconds 5
        Write-Host "." -NoNewline
    }
    Write-Host ""
    throw "Локальный сервер $ServedModel не запустился за 10 минут. Проверьте окно сервера."
}

function Get-PdfFilesToProcess {
    if (-not [string]::IsNullOrWhiteSpace($PdfPath)) {
        return @((Resolve-Path -LiteralPath $PdfPath.Trim().Trim('"')).Path)
    }
    if ($Batch) {
        $pdfs = @(Get-ChildItem -LiteralPath $InputDir -Filter "*.pdf" -File | Sort-Object Name)
        if ($pdfs.Count -eq 0) {
            Write-Host "В папке INPUT_PDF нет PDF-файлов: $InputDir"
            Write-Host "Положите PDF-файлы в INPUT_PDF и запустите этот файл еще раз."
            explorer.exe $InputDir | Out-Null
            exit 2
        }
        return @($pdfs | ForEach-Object { $_.FullName })
    }

    Write-Host "Перетащите PDF в это окно или вставьте полный путь к PDF и нажмите Enter:"
    $typedPath = (Read-Host "PDF").Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($typedPath)) {
        Write-Host "Путь пустой. Открываю INPUT_PDF: $InputDir"
        explorer.exe $InputDir | Out-Null
        exit 2
    }
    return @((Resolve-Path -LiteralPath $typedPath).Path)
}

function New-PdfRunContext {
    param(
        [string]$ResolvedPdfPath,
        [string]$ProjectWsl,
        [hashtable]$SelectedProfile
    )
    $pdfWsl = Convert-ToWslPath $ResolvedPdfPath
    $baseName = [IO.Path]::GetFileNameWithoutExtension($ResolvedPdfPath)
    $safeName = $baseName -replace "[^A-Za-z0-9А-Яа-я._-]+", "_"
    if ([string]::IsNullOrWhiteSpace($safeName)) {
        $safeName = "document"
    }

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $runId = "$safeName-$($SelectedProfile.Name)-$timestamp"
    $runDirWindows = Join-Path $OutputDir $runId
    $runZipWindows = Join-Path $OutputDir "$runId.zip"
    New-Item -ItemType Directory -Force -Path $runDirWindows | Out-Null
    $runDirWsl = Convert-ToWslPath $runDirWindows

    $ocrFileName = "ocr.paddleocr-vl.json"
    if ($SelectedProfile.Mode -eq "gpt") {
        $ocrFileName = "ocr.pdf-text.json"
    }
    $ocrJson = "$runDirWsl/$ocrFileName"
    $rowsJsonl = "$runDirWsl/rows.ai.jsonl"
    $qualityJson = "$runDirWsl/rows.ai.quality.json"
    $performanceJson = "$runDirWsl/rows.ai.performance.json"
    $plainXlsx = "$runDirWsl/result.plain.xlsx"
    $kpXlsx = "$runDirWsl/result.kp.xlsx"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "PDF: $ResolvedPdfPath"
    Write-Host "Результаты: $runDirWindows"
    Write-Host "============================================================"

    return @{
        ResolvedPdfPath = $ResolvedPdfPath
        ProjectWsl = $ProjectWsl
        SelectedProfile = $SelectedProfile
        PdfWsl = $pdfWsl
        RunId = $runId
        RunDirWindows = $runDirWindows
        RunZipWindows = $runZipWindows
        OcrJson = $ocrJson
        RowsJsonl = $rowsJsonl
        QualityJson = $qualityJson
        PerformanceJson = $performanceJson
        PlainXlsx = $plainXlsx
        KpXlsx = $kpXlsx
    }
}

function Get-RunLogPath {
    param(
        [hashtable]$Run,
        [string]$Name
    )
    $logsDir = Join-Path $Run["RunDirWindows"] "logs"
    New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
    return (Join-Path $logsDir "$Name.log")
}

function Invoke-PdfOcr {
    param([hashtable]$Run)

    $projectWsl = $Run["ProjectWsl"]
    $pdfWsl = $Run["PdfWsl"]
    $ocrJson = $Run["OcrJson"]
    $extract = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr extract-ocr $(Quote-BashArg $pdfWsl) --provider auto --output $(Quote-BashArg $ocrJson) --pdf-text-overlay true --model-cache-dir artifacts/models --paddle-device gpu:0 --paddle-require-cuda --paddle-use-queues true --paddle-split-wide-pages true --paddle-band-height 260 --paddle-band-overlap 40 --paddle-max-pixels 501760 --paddle-max-new-tokens 1024 --paddle-predict-timeout 900 --paddle-predict-heartbeat 60"
    Invoke-WslStep -Title "OCR PDF auto (PDF table layer или PaddleOCR-VL CUDA watchdog)" -Script $extract -LogPath (Get-RunLogPath -Run $Run -Name "01-ocr")
}

function Invoke-PdfGptVisualInput {
    param([hashtable]$Run)

    $projectWsl = $Run["ProjectWsl"]
    $pdfWsl = $Run["PdfWsl"]
    $ocrJson = $Run["OcrJson"]
    $extract = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr extract-ocr $(Quote-BashArg $pdfWsl) --provider pdf-text --output $(Quote-BashArg $ocrJson)"
    Invoke-WslStep -Title "Рендер страниц PDF для GPT vision fallback" -Script $extract -LogPath (Get-RunLogPath -Run $Run -Name "01-render-pdf-text")
}

function Get-ThirdLayerMapArgs {
    param([hashtable]$SelectedProfile)

    if (-not $SelectedProfile.ContainsKey("ThirdLayerReviewer")) {
        return ""
    }
    $args = @(
        "--third-layer-reviewer $($SelectedProfile.ThirdLayerReviewer)"
    )
    if ($SelectedProfile.ContainsKey("ThirdLayerModel") -and -not [string]::IsNullOrWhiteSpace($SelectedProfile.ThirdLayerModel)) {
        $args += "--third-layer-model $(Quote-BashArg $SelectedProfile.ThirdLayerModel)"
    }
    if ($SelectedProfile.ContainsKey("ThirdLayerBaseUrl") -and -not [string]::IsNullOrWhiteSpace($SelectedProfile.ThirdLayerBaseUrl)) {
        $args += "--third-layer-base-url $(Quote-BashArg $SelectedProfile.ThirdLayerBaseUrl)"
    }
    if ($SelectedProfile.ContainsKey("ThirdLayerServedModel") -and -not [string]::IsNullOrWhiteSpace($SelectedProfile.ThirdLayerServedModel)) {
        $args += "--third-layer-served-model $(Quote-BashArg $SelectedProfile.ThirdLayerServedModel)"
    }
    if ($SelectedProfile.ContainsKey("ThirdLayerApiKeyEnv") -and -not [string]::IsNullOrWhiteSpace($SelectedProfile.ThirdLayerApiKeyEnv)) {
        $args += "--third-layer-api-key-env $($SelectedProfile.ThirdLayerApiKeyEnv)"
    }
    if ($SelectedProfile.ContainsKey("ThirdLayerReasoningEffort") -and -not [string]::IsNullOrWhiteSpace($SelectedProfile.ThirdLayerReasoningEffort)) {
        $args += "--third-layer-reasoning-effort $($SelectedProfile.ThirdLayerReasoningEffort)"
    }
    return " " + ($args -join " ")
}

function Invoke-PdfAiAndExport {
    param([hashtable]$Run)
    $script:LastPdfRunSucceeded = $false

    $projectWsl = $Run["ProjectWsl"]
    $selectedProfile = $Run["SelectedProfile"]
    $pdfWsl = $Run["PdfWsl"]
    $runId = $Run["RunId"]
    $rowsJsonl = $Run["RowsJsonl"]
    $ocrJson = $Run["OcrJson"]
    $qualityJson = $Run["QualityJson"]
    $performanceJson = $Run["PerformanceJson"]
    $plainXlsx = $Run["PlainXlsx"]
    $kpXlsx = $Run["KpXlsx"]
    $runDirWindows = $Run["RunDirWindows"]
    $thirdLayerArgs = Get-ThirdLayerMapArgs -SelectedProfile $selectedProfile
    $forcePdfTableAiArg = ""
    if ($selectedProfile.ContainsKey("ForceAiOnPdfTableLayer") -and $selectedProfile.ForceAiOnPdfTableLayer) {
        $forcePdfTableAiArg = " --ai-force-on-pdf-table-layer"
    }
    $aiReasoningArg = ""
    if ($selectedProfile.ContainsKey("ReasoningEffort") -and -not [string]::IsNullOrWhiteSpace($selectedProfile.ReasoningEffort)) {
        $aiReasoningArg = " --ai-reasoning-effort $($selectedProfile.ReasoningEffort)"
    }
    $aiPageBatchSize = "1"
    if ($selectedProfile.Mode -eq "fireworks") {
        $aiPageBatchSize = "3"
    }
    $thirdLayerPath = ""
    if ($selectedProfile.ContainsKey("ThirdLayerReviewer") -and $selectedProfile.ThirdLayerReviewer -eq "opencode-gpt") {
        $thirdLayerPath = 'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"; '
    }

    if ($selectedProfile.Mode -eq "local") {
        $map = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; ${thirdLayerPath}spec-ocr map-ocr $(Quote-BashArg $runId) --input-ocr $(Quote-BashArg $ocrJson) --source-pdf $(Quote-BashArg $pdfWsl) --ai-client openai-compatible --ai-model $($selectedProfile.Model) --ai-served-model $($selectedProfile.ServedModel) --ai-base-url http://127.0.0.1:8000/v1 --ai-timeout 1800 --ai-max-tokens 8192 --ai-page-batch-size $aiPageBatchSize --ai-context-tokens 12288 --ai-context-margin 512 --model-cache-dir artifacts/models$thirdLayerArgs$forcePdfTableAiArg --output $(Quote-BashArg $rowsJsonl) --quality-json $(Quote-BashArg $qualityJson) --quality-fail-on uncorrected-warning --performance-json $(Quote-BashArg $performanceJson)"
    } elseif ($selectedProfile.Mode -eq "fireworks") {
        $map = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr map-ocr $(Quote-BashArg $runId) --input-ocr $(Quote-BashArg $ocrJson) --source-pdf $(Quote-BashArg $pdfWsl) --ai-client openai-compatible --ai-model $(Quote-BashArg $selectedProfile.Model) --ai-served-model $(Quote-BashArg $selectedProfile.ServedModel) --ai-base-url $(Quote-BashArg $selectedProfile.BaseUrl) --ai-external-openai-compatible --ai-external-supports-images --ai-api-key-env $($selectedProfile.ApiKeyEnv) --ai-reasoning-effort none --ai-timeout 1800 --ai-max-tokens 8192 --ai-page-batch-size $aiPageBatchSize --ai-context-tokens 12288 --ai-context-margin 512$forcePdfTableAiArg --output $(Quote-BashArg $rowsJsonl) --quality-json $(Quote-BashArg $qualityJson) --quality-fail-on uncorrected-warning --performance-json $(Quote-BashArg $performanceJson)"
    } else {
        $opencodePath = 'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"'
        $map = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; $opencodePath; spec-ocr map-ocr $(Quote-BashArg $runId) --input-ocr $(Quote-BashArg $ocrJson) --source-pdf $(Quote-BashArg $pdfWsl) --ai-client opencode --ai-model $($selectedProfile.Model) --ai-agent build$aiReasoningArg --ai-timeout 1800 --ai-page-batch-size $aiPageBatchSize --ai-context-tokens 12288 --ai-context-margin 512$forcePdfTableAiArg --output $(Quote-BashArg $rowsJsonl) --quality-json $(Quote-BashArg $qualityJson) --quality-fail-on uncorrected-warning --performance-json $(Quote-BashArg $performanceJson)"
    }
    Invoke-WslStep -Title "AI-разбор спецификации" -Script $map -QualityGateRunDir $runDirWindows -ContinueOnQualityGateFailure:$Batch -LogPath (Get-RunLogPath -Run $Run -Name "02-ai-map")
    if ($script:LastWslStepExitCode -eq 2) {
        return
    }
    if (-not (Test-WslFileNonEmpty -Path $rowsJsonl)) {
        $statusPath = Join-Path $runDirWindows 'processing.status.txt'
        $aiLogPath = Join-Path $runDirWindows 'logs\02-ai-map.log'
        $smokeLogPath = Join-Path $OutputDir '_gpt-opencode-smoke.log'
        @(
            'ai_output_missing',
            'Excel не создан, потому что AI-разбор завершился без rows.ai.jsonl.',
            'Экспорт не запускался: это защита от пустого или битого результата.',
            "AI log: $aiLogPath",
            "GPT smoke log: $smokeLogPath",
            'Проверьте указанные логи и OCR JSON в этой папке результата.'
        ) | Set-Content -Path $statusPath -Encoding UTF8
        Write-Host ""
        Write-Host "AI-разбор не создал rows.ai.jsonl; Excel не экспортирую." -ForegroundColor Yellow
        Write-Host "Проверьте диагностические файлы:"
        Write-Host "  $aiLogPath"
        Write-Host "  $smokeLogPath"
        Write-Host "  $statusPath"
        Show-LogTail -Path $aiLogPath
        Show-LogTail -Path $smokeLogPath
        return
    }

    $exports = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr export-jsonl-xlsx $(Quote-BashArg $rowsJsonl) --output $(Quote-BashArg $plainXlsx); spec-ocr export-kp-xlsx $(Quote-BashArg $rowsJsonl) --output $(Quote-BashArg $kpXlsx)"
    Invoke-WslStep -Title "Экспорт Excel и КП" -Script $exports -LogPath (Get-RunLogPath -Run $Run -Name "03-export")

    Write-Host ""
    Write-Host "Готово: $runDirWindows"
    Write-Host "  result.plain.xlsx"
    Write-Host "  result.kp.xlsx"
    Write-Host "  rows.ai.performance.json"
    Write-Host "  rows.ai.quality.json"
    $script:LastPdfRunSucceeded = $true
}

function New-PerPdfResultArchive {
    param([hashtable]$Run)

    $runDirWindows = $Run["RunDirWindows"]
    $runZipWindows = $Run["RunZipWindows"]
    if (Test-Path -LiteralPath $runZipWindows) {
        Remove-Item -LiteralPath $runZipWindows -Force
    }
    Compress-Archive -LiteralPath $runDirWindows -DestinationPath $runZipWindows -CompressionLevel Optimal -Force
    Write-Host "  Архив этого PDF: $runZipWindows"
}

function Invoke-NativeSpecOcr {
    param(
        [string[]]$CliArgs,
        [int]$TimeoutSeconds = 0
    )

    $bootstrapPython = Get-WindowsBootstrapPython
    $env:PYTHONPATH = Join-Path $ProjectDir "src"
    if ($TimeoutSeconds -gt 0) {
        $allArgs = @("-m", "spec_pipeline.cli") + $CliArgs
        $argumentLine = ($allArgs | ForEach-Object { Quote-WindowsProcessArg ([string]$_) }) -join " "
        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = $bootstrapPython
        $processInfo.Arguments = $argumentLine
        $processInfo.WorkingDirectory = $ProjectDir
        $processInfo.UseShellExecute = $false
        $processInfo.RedirectStandardOutput = $true
        $processInfo.RedirectStandardError = $true
        $processInfo.CreateNoWindow = $true
        $processInfo.EnvironmentVariables["PYTHONPATH"] = Join-Path $ProjectDir "src"
        $processInfo.EnvironmentVariables["PYTHONUTF8"] = "1"
        $processInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $processInfo
        $null = $process.Start()
        $completed = $process.WaitForExit($TimeoutSeconds * 1000)
        if (-not $completed) {
            try {
                $process.Kill()
            } catch {
            }
            throw "spec-ocr превысил таймаут $TimeoutSeconds сек: $($CliArgs -join ' ')"
        }
        $stdoutText = $process.StandardOutput.ReadToEnd()
        $stderrText = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if (-not [string]::IsNullOrWhiteSpace($stdoutText)) {
            $stdoutText -split "`r?`n" | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
        }
        if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
            $stderrText -split "`r?`n" | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
        }
        if ($process.ExitCode -ne 0) {
            throw "spec-ocr завершился с кодом $($process.ExitCode)"
        }
        return
    }

    & $bootstrapPython -m spec_pipeline.cli @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "spec-ocr завершился с кодом $LASTEXITCODE"
    }
}

function Test-WindowsBootstrapPythonReady {
    param([string]$PythonExe)
    # setup-windows.ps1 writes .spec-ocr-bootstrap-ready next to the bootstrap
    # Python only after pip + spec-ocr + pypdfium2/Pillow are installed.
    # Require the marker before trusting a Python.
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $false
    }
    $marker = Join-Path (Split-Path -Parent $PythonExe) ".spec-ocr-bootstrap-ready"
    return (Test-Path -LiteralPath $marker)
}

function Get-WindowsBootstrapPython {
    $venvPython = Join-Path $ProjectDir ".venv-windows-bootstrap\Scripts\python.exe"
    if (Test-WindowsBootstrapPythonReady -PythonExe $venvPython) {
        return $venvPython
    }

    Write-Host "Windows bootstrap Python не готов (нет .spec-ocr-bootstrap-ready); запускаю автоматическую подготовку GPT fast path..."
    $setupScript = Join-Path $ProjectDir "setup-windows.ps1"
    if (-not (Test-Path -LiteralPath $setupScript)) {
        throw "setup-windows.ps1 не найден рядом со скриптом обработки: $setupScript"
    }
    $powershellExe = Get-WindowsPowerShellExe
    $setupOutput = & $powershellExe -NoProfile -ExecutionPolicy Bypass -File $setupScript -GptFastPath 2>&1
    $setupExitCode = $LASTEXITCODE
    foreach ($line in $setupOutput) {
        Write-Host $line
    }
    if ($setupExitCode -ne 0) {
        throw "Автоматическая подготовка GPT fast path завершилась с кодом $setupExitCode"
    }
    if (Test-WindowsBootstrapPythonReady -PythonExe $venvPython) {
        return $venvPython
    }
    throw "Windows bootstrap Python не готов даже после автоматической подготовки GPT fast path."
}

function Ensure-NativePdfTextDependencies {
    $bootstrapPython = Get-WindowsBootstrapPython
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $bootstrapPython -c "import pypdfium2, PIL.Image" *> $null
    $importExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($importExitCode -eq 0) {
        return
    }
    Write-Host "Устанавливаю pypdfium2/Pillow для Windows-native PDF_TABLE_LAYER..."
    & $bootstrapPython -m pip install --disable-pip-version-check -q pypdfium2 Pillow
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось установить pypdfium2/Pillow для Windows-native PDF обработки"
    }
}

function Invoke-NativeGptPdfTableLayerRun {
    param([hashtable]$Run)

    $script:LastPdfRunSucceeded = $false
    $script:NativeGptPdfTableLayerSucceeded = $false
    $resolvedPdfPath = $Run["ResolvedPdfPath"]
    $runId = $Run["RunId"]
    $runDirWindows = $Run["RunDirWindows"]
    $ocrJson = Join-Path $runDirWindows "ocr.pdf-text.json"
    $rowsJsonl = Join-Path $runDirWindows "rows.ai.jsonl"
    $qualityJson = Join-Path $runDirWindows "rows.ai.quality.json"
    $performanceJson = Join-Path $runDirWindows "rows.ai.performance.json"
    $plainXlsx = Join-Path $runDirWindows "result.plain.xlsx"
    $kpXlsx = Join-Path $runDirWindows "result.kp.xlsx"
    $selectedProfile = $Run["SelectedProfile"]

    Write-Host "Windows-native fast path: проверяю PDF_TABLE_LAYER без WSL (таймаут $NativePdfTextTimeoutSeconds сек)..."
    Ensure-NativePdfTextDependencies
    try {
        Invoke-NativeSpecOcr -TimeoutSeconds $NativePdfTextTimeoutSeconds -CliArgs @(
            "extract-ocr",
            $resolvedPdfPath,
            "--provider",
            "pdf-text",
            "--output",
            $ocrJson
        )
    } catch {
        Write-Host "Windows-native PDF_TABLE_LAYER не завершился: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Продолжаю через WSL/OCR fallback для этого PDF."
        return
    }
    if (-not (Test-Path -LiteralPath $ocrJson)) {
        Write-Host "Windows-native PDF_TABLE_LAYER не создал OCR JSON; продолжаю через WSL/OCR fallback." -ForegroundColor Yellow
        return
    }
    $ocrText = Get-Content -LiteralPath $ocrJson -Raw -Encoding UTF8
    $pdfTableRowCount = ([regex]::Matches($ocrText, "PDF_TABLE_ROW\b")).Count
    Write-Host "PDF_TABLE_ROW count: $pdfTableRowCount"
    if ($pdfTableRowCount -lt 5) {
        Write-Host "PDF_TABLE_LAYER недостаточен; нужен WSL/OCR fallback."
        return
    }

    $aiPageBatchSize = "1"
    if ($selectedProfile.Mode -eq "fireworks") {
        $aiPageBatchSize = "3"
    }
    $mapArgs = @(
        "map-ocr",
        $runId,
        "--input-ocr",
        $ocrJson,
        "--source-pdf",
        $resolvedPdfPath
    )
    if ($selectedProfile.Mode -eq "local") {
        $mapArgs += @(
            "--ai-client",
            "openai-compatible",
            "--ai-model",
            $selectedProfile.Model,
            "--ai-served-model",
            $selectedProfile.ServedModel,
            "--ai-base-url",
            "http://127.0.0.1:8000/v1"
        )
    } elseif ($selectedProfile.Mode -eq "fireworks") {
        $mapArgs += @(
            "--ai-client",
            "openai-compatible",
            "--ai-model",
            $selectedProfile.Model,
            "--ai-served-model",
            $selectedProfile.ServedModel,
            "--ai-base-url",
            $selectedProfile.BaseUrl,
            "--ai-external-openai-compatible",
            "--ai-external-supports-images",
            "--ai-api-key-env",
            $selectedProfile.ApiKeyEnv,
            "--ai-reasoning-effort",
            "none"
        )
    } else {
        $mapArgs += @(
            "--ai-client",
            "opencode",
            "--ai-model",
            $selectedProfile.Model,
            "--ai-agent",
            "build"
        )
        if ($selectedProfile.ContainsKey("ReasoningEffort") -and -not [string]::IsNullOrWhiteSpace($selectedProfile.ReasoningEffort)) {
            $mapArgs += @("--ai-reasoning-effort", $selectedProfile.ReasoningEffort)
        }
    }
    $mapArgs += @(
        "--ai-timeout",
        "1800",
        "--ai-page-batch-size",
        $aiPageBatchSize,
        "--ai-context-tokens",
        "12288",
        "--ai-context-margin",
        "512",
        "--output",
        $rowsJsonl,
        "--quality-json",
        $qualityJson,
        "--quality-fail-on",
        "uncorrected-warning",
        "--performance-json",
        $performanceJson
    )
    Invoke-NativeSpecOcr -CliArgs $mapArgs
    Invoke-NativeSpecOcr -CliArgs @(
        "export-jsonl-xlsx",
        $rowsJsonl,
        "--output",
        $plainXlsx
    )
    Invoke-NativeSpecOcr -CliArgs @(
        "export-kp-xlsx",
        $rowsJsonl,
        "--output",
        $kpXlsx
    )

    Write-Host ""
    Write-Host "Готово без WSL: $runDirWindows"
    Write-Host "  result.plain.xlsx"
    Write-Host "  result.kp.xlsx"
    Write-Host "  rows.ai.performance.json"
    Write-Host "  rows.ai.quality.json"
    $script:LastPdfRunSucceeded = $true
    $script:NativeGptPdfTableLayerSucceeded = $true
    return
}

Write-Host "SpecOCR hotfix: $SpecOcrHotfixVersion"

$script:GptVisualFallbackNeeded = $false
$projectWsl = Convert-ToWslPath $ProjectDir
$pdfFiles = @(Get-PdfFilesToProcess)
$selectedProfile = Select-ProcessingProfile
Write-Host ""
Write-Host "Выбрана модель обработки: $($selectedProfile.Name) - $($selectedProfile.Title)"
Write-Host "$($selectedProfile.Description)"

Write-Host ""
Write-Host "Пробую Windows-native fast path для searchable PDF: PDF_TABLE_LAYER -> Excel без WSL."
$nativeProcessed = 0
$nativeFailed = 0
$fallbackPdfFiles = @()
foreach ($pdf in $pdfFiles) {
    $run = New-PdfRunContext -ResolvedPdfPath $pdf -ProjectWsl $projectWsl -SelectedProfile $selectedProfile
    try {
        $script:NativeGptPdfTableLayerSucceeded = $false
        Invoke-NativeGptPdfTableLayerRun -Run $run
        $nativeOk = $script:NativeGptPdfTableLayerSucceeded
        if (-not $nativeOk) {
            $fallbackPdfFiles += $pdf
            continue
        }
        New-PerPdfResultArchive -Run $run
        $nativeProcessed += 1
    } catch {
        if (-not $Batch) {
            throw
        }
        $nativeFailed += 1
        $message = $_.Exception.Message
        $statusPath = Join-Path $run["RunDirWindows"] "processing.status.txt"
        @(
            'native_processing_failed',
            $message
        ) | Set-Content -Path $statusPath -Encoding UTF8
        Write-Host ""
        Write-Host "Ошибка Windows-native обработки PDF: $($run['ResolvedPdfPath'])" -ForegroundColor Red
        Write-Host $message
        New-PerPdfResultArchive -Run $run
    }
}
if ($fallbackPdfFiles.Count -eq 0) {
    Write-Host ""
    Write-Host "Обработка завершена."
    Write-Host "Успешно обработано PDF: $nativeProcessed"
    if ($nativeFailed -gt 0) {
        Write-Host "PDF с диагностикой без Excel: $nativeFailed" -ForegroundColor Yellow
    }
    Write-Host "Папка результатов: $OutputDir"
    if (Test-Path $OutputDir) {
        explorer.exe $OutputDir | Out-Null
    }
    if (($nativeProcessed -eq 0) -and ($nativeFailed -gt 0)) {
        exit 2
    }
    exit 0
}
Write-Host "PDF без пригодного PDF_TABLE_LAYER: $($fallbackPdfFiles.Count). Продолжаю для них через WSL/OCR fallback."
$pdfFiles = @($fallbackPdfFiles)
if ($selectedProfile.Mode -eq "gpt") {
    $script:GptVisualFallbackNeeded = $true
    Ensure-GptWslFallbackEnvironment -RequestedDistro $Distro -ProjectWsl $projectWsl
}

$Distro = Resolve-WslDistroName -Requested $Distro
Write-Host "WSL distro: $Distro"

$preflight = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); test -f .venv-wsl/bin/activate"
Invoke-WslStep -Title "Проверка установленного окружения" -Script $preflight -LogPath (Join-Path $OutputDir "_wsl-env-check.log")

if ($selectedProfile.Mode -ne "gpt") {
    $paddleCuda = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; $(Get-PaddleCudaRepairScript)"
    Invoke-WslStep -Title "Проверка CUDA для PaddleOCR-VL" -Script $paddleCuda -LogPath (Join-Path $OutputDir "_paddle-cuda-check.log")

    $paddleDownload = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr models --cache-dir artifacts/models download paddleocr-vl --json"
    Invoke-WslStep -Title "Проверка/скачивание PaddleOCR-VL" -Script $paddleDownload -LogPath (Join-Path $OutputDir "_paddle-download.log")
} elseif ($script:GptVisualFallbackNeeded) {
    $renderDeps = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; python -m pip install --disable-pip-version-check --retries 10 --timeout 120 pypdfium2 Pillow"
    Invoke-WslStep -Title "Проверка рендера PDF для GPT vision fallback" -Script $renderDeps -LogPath (Join-Path $OutputDir "_gpt-render-deps.log")
    Ensure-OpenCodeCliInWsl -ProjectWsl $projectWsl
    Test-GptAuthInWsl -ProjectWsl $projectWsl
    Test-GptOpenCodeInWsl -ProjectWsl $projectWsl
}

if ($selectedProfile.Mode -eq "local") {
    if ($selectedProfile.RuntimeInstall -eq $LlamaCppPackage) {
        $runtimeInstallScript = Get-LlamaCudaRepairScript
    } else {
        $runtimeInstallArg = Quote-BashArg $selectedProfile.RuntimeInstall
        $runtimeInstallScript = "python -m pip install --disable-pip-version-check --retries 10 --timeout 120 $runtimeInstallArg"
    }
    $installRuntime = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; $runtimeInstallScript"
    Invoke-WslStep -Title "Проверка runtime для $($selectedProfile.Name)" -Script $installRuntime -LogPath (Join-Path $OutputDir "_local-runtime-check.log")
    $setupProfile = $selectedProfile.Name
    if ($selectedProfile.ContainsKey("SetupProfile") -and -not [string]::IsNullOrWhiteSpace($selectedProfile.SetupProfile)) {
        $setupProfile = $selectedProfile.SetupProfile
    }
    $downloadModel = "set -euo pipefail; cd $(Quote-BashArg $projectWsl); . .venv-wsl/bin/activate; spec-ocr setup-local --profile $setupProfile --download --json --cache-dir artifacts/models"
    Invoke-WslStep -Title "Проверка/скачивание модели $setupProfile" -Script $downloadModel -LogPath (Join-Path $OutputDir "_local-model-download.log")
} elseif ($selectedProfile.Mode -eq "gpt") {
    if ($script:GptVisualFallbackNeeded) {
        Write-Host "GPT/OpenCode будет читать изображения страниц. CUDA/PaddleOCR-VL для этого fallback не требуется."
    } else {
        Write-Host "GPT/OpenCode будет вызван на этапе AI-разбора, если PDF не содержит готовый табличный текстовый слой."
        Write-Host "Для searchable PDF с PDF_TABLE_LAYER CUDA/PaddleOCR-VL не требуется."
    }
} elseif ($selectedProfile.Mode -eq "fireworks") {
    $apiKeyEnv = $selectedProfile.ApiKeyEnv
    $cloudPreflight = "set -euo pipefail; printenv $apiKeyEnv >/dev/null || (echo 'Не задан $apiKeyEnv в WSL. Задайте ключ Fireworks перед запуском Kimi.' >&2; exit 1)"
    Invoke-WslStep -Title "Проверка Fireworks/Kimi API key" -Script $cloudPreflight -LogPath (Join-Path $OutputDir "_fireworks-api-key-check.log")
}
if ($selectedProfile.ContainsKey("ThirdLayerApiKeyEnv") -and -not [string]::IsNullOrWhiteSpace($selectedProfile.ThirdLayerApiKeyEnv)) {
    $thirdLayerApiKeyEnv = $selectedProfile.ThirdLayerApiKeyEnv
    $thirdLayerCloudPreflight = "set -euo pipefail; printenv $thirdLayerApiKeyEnv >/dev/null || (echo 'Не задан $thirdLayerApiKeyEnv в WSL. Задайте ключ для третьего слоя перед запуском.' >&2; exit 1)"
    Invoke-WslStep -Title "Проверка API key третьего слоя" -Script $thirdLayerCloudPreflight -LogPath (Join-Path $OutputDir "_third-layer-api-key-check.log")
}
if ($selectedProfile.ContainsKey("ThirdLayerReviewer") -and $selectedProfile.ThirdLayerReviewer -eq "opencode-gpt") {
    Write-Host "GPT/OpenCode третьего слоя будет вызван только для подозрительных строк."
}

$runs = @()
foreach ($pdf in $pdfFiles) {
    $run = New-PdfRunContext -ResolvedPdfPath $pdf -ProjectWsl $projectWsl -SelectedProfile $selectedProfile
    $runs += $run
}

Write-Host ""
if ($selectedProfile.Mode -eq "gpt" -and $script:GptVisualFallbackNeeded) {
    Write-Host "Сначала рендерю страницы PDF для GPT vision fallback."
    foreach ($run in $runs) {
        Invoke-PdfGptVisualInput -Run $run
    }
} else {
    Write-Host "Сначала выполняю OCR через CUDA без загруженного Qwen-сервера, чтобы снизить расход памяти."
    foreach ($run in $runs) {
        Invoke-PdfOcr -Run $run
    }
}

if ($selectedProfile.Mode -eq "local") {
    Ensure-LocalServer -ProjectWsl $projectWsl -Model $selectedProfile.Model -ServedModel $selectedProfile.ServedModel
}

$processed = 0
$failed = 0
foreach ($run in $runs) {
    $script:LastPdfRunSucceeded = $false
    try {
        Invoke-PdfAiAndExport -Run $run
        if ($script:LastPdfRunSucceeded) {
            $processed += 1
        } else {
            $failed += 1
            Write-Host ""
            Write-Host "PDF оставлен с диагностикой без Excel: $($run['ResolvedPdfPath'])" -ForegroundColor Yellow
        }
    } catch {
        if (-not $Batch) {
            throw
        }
        $failed += 1
        $message = $_.Exception.Message
        $statusPath = Join-Path $run["RunDirWindows"] "processing.status.txt"
        @(
            'processing_failed',
            $message
        ) | Set-Content -Path $statusPath -Encoding UTF8
        Write-Host ""
        Write-Host "Ошибка обработки PDF: $($run['ResolvedPdfPath'])" -ForegroundColor Red
        Write-Host $message
        Write-Host "Пакетная обработка продолжится со следующего PDF." -ForegroundColor Yellow
    }
    New-PerPdfResultArchive -Run $run
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "Обработка завершена."
Write-Host "Успешно обработано PDF: $processed"
if ($failed -gt 0) {
    Write-Host "PDF с диагностикой без Excel: $failed" -ForegroundColor Yellow
}
Write-Host "Папка результатов: $OutputDir"
if (Test-Path $OutputDir) {
    explorer.exe $OutputDir | Out-Null
}
if (($processed -eq 0) -and ($failed -gt 0)) {
    exit 2
}
