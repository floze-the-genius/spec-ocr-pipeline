#requires -Version 5.1

param(
    [switch]$GptFastPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SpecOcrArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $ProjectDir "src"

$PythonVersion = "3.11.9"
$PythonInstallerName = "python-$PythonVersion-amd64.exe"
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonInstallerName"
$PythonInstallerSha256 = "5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde"
$DownloadDir = Join-Path $env:TEMP "spec-ocr-downloads"
$DownloadedPythonInstaller = Join-Path $DownloadDir $PythonInstallerName
$LocalPythonDir = Join-Path $env:LocalAppData "SpecOCR\Python311"
$LocalPython = Join-Path $LocalPythonDir "python.exe"
$BootstrapVenv = Join-Path $ProjectDir ".venv-windows-bootstrap"
$BootstrapPython = Join-Path $BootstrapVenv "Scripts\python.exe"
$BootstrapReady = Join-Path $BootstrapVenv ".spec-ocr-bootstrap-ready"

function Test-SupportedPython {
    param(
        [string]$Exe,
        [string[]]$PythonArgs = @(),
        [bool]$RequireVenv = $true
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $versionText = (& $Exe @PythonArgs --version 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $versionText -notmatch "Python\s+(\d+)\.(\d+)") {
            return $false
        }
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ne 3 -or $minor -lt 11 -or $minor -gt 14) {
            return $false
        }

        $imports = "encodings, sys, sysconfig"
        if ($RequireVenv) {
            $imports += ", venv"
        }
        $healthScript = "import $imports; assert (3, 11) <= sys.version_info[:2] <= (3, 14); assert sysconfig.get_paths().get('stdlib')"
        & $Exe @PythonArgs -c $healthScript *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Find-SupportedPython {
    if ((Test-Path $LocalPython) -and (Test-SupportedPython -Exe $LocalPython)) {
        return @{ Exe = $LocalPython; Args = @() }
    }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        foreach ($selector in @("-3.14", "-3.13", "-3.12", "-3.11", "-3")) {
            if (Test-SupportedPython -Exe $launcher.Source -PythonArgs @($selector)) {
                return @{ Exe = $launcher.Source; Args = @($selector) }
            }
        }
    }

    foreach ($commandName in @("python", "python3")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command -and (Test-SupportedPython -Exe $command.Source)) {
            return @{ Exe = $command.Source; Args = @() }
        }
    }

    if ($env:LocalAppData) {
        $pythonRoot = Join-Path $env:LocalAppData "Programs\Python"
        if (Test-Path $pythonRoot) {
            foreach ($directory in Get-ChildItem -Path $pythonRoot -Directory -Filter "Python3*" -ErrorAction SilentlyContinue) {
                $candidate = Join-Path $directory.FullName "python.exe"
                if ((Test-Path $candidate) -and (Test-SupportedPython -Exe $candidate)) {
                    return @{ Exe = $candidate; Args = @() }
                }
            }
        }
    }
    return $null
}

function Assert-InstallerHash {
    param([string]$InstallerPath)

    $actual = (Get-FileHash -Algorithm SHA256 -Path $InstallerPath).Hash.ToLowerInvariant()
    if ($actual -ne $PythonInstallerSha256) {
        throw "Python installer checksum mismatch. Expected $PythonInstallerSha256, got $actual"
    }
}

function Install-VerifiedPython {
    if ($env:PROCESSOR_ARCHITECTURE -match "ARM") {
        throw "Windows ARM64 is not currently supported by this installer."
    }

    New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
    if (Test-Path $DownloadedPythonInstaller) {
        try {
            Assert-InstallerHash -InstallerPath $DownloadedPythonInstaller
        } catch {
            Remove-Item -Force $DownloadedPythonInstaller
        }
    }
    if (-not (Test-Path $DownloadedPythonInstaller)) {
        Write-Host "Downloading Python $PythonVersion from python.org..."
        Invoke-WebRequest -UseBasicParsing -Uri $PythonInstallerUrl -OutFile $DownloadedPythonInstaller
    }
    Assert-InstallerHash -InstallerPath $DownloadedPythonInstaller

    New-Item -ItemType Directory -Force -Path $LocalPythonDir | Out-Null
    $installArguments = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=`"$LocalPythonDir`"",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "Shortcuts=0",
        "PrependPath=0"
    )
    Write-Host "Installing a private Python runtime in $LocalPythonDir..."
    $process = Start-Process -FilePath $DownloadedPythonInstaller -ArgumentList $installArguments -Wait -PassThru
    if ($process.ExitCode -notin @(0, 3010)) {
        throw "Python installer returned exit code $($process.ExitCode)."
    }
    if (-not ((Test-Path $LocalPython) -and (Test-SupportedPython -Exe $LocalPython))) {
        if ($process.ExitCode -eq 3010) {
            Write-Host "Python installation requires a Windows restart."
            exit 3010
        }
        throw "Python was installed but failed its runtime health check."
    }
}

function Ensure-BootstrapVenv {
    param(
        [string]$PythonExe,
        [string[]]$PythonArgs = @()
    )

    if ((Test-Path $BootstrapPython) -and (-not (Test-SupportedPython -Exe $BootstrapPython))) {
        Write-Host "Rebuilding a damaged project environment..."
        Remove-Item -Recurse -Force $BootstrapVenv
    }
    if (-not (Test-Path $BootstrapPython)) {
        Write-Host "Creating the project environment..."
        & $PythonExe @PythonArgs -m venv $BootstrapVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Python could not create the project virtual environment."
        }
    }

    Write-Host "Installing Spec OCR Pipeline and the lightweight PDF renderer..."
    & $BootstrapPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip in the project environment."
    }
    & $BootstrapPython -m pip install --disable-pip-version-check -e ".[ocr]"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install Spec OCR Pipeline."
    }
    Set-Content -Path $BootstrapReady -Value (Get-Date -Format o) -Encoding ASCII
}

$python = Find-SupportedPython
if ($null -eq $python) {
    Install-VerifiedPython
    $python = Find-SupportedPython
}
if ($null -eq $python) {
    throw "A supported Python 3.11-3.14 runtime could not be prepared."
}

Ensure-BootstrapVenv -PythonExe $python.Exe -PythonArgs $python.Args
& $BootstrapPython -m spec_pipeline.cli doctor
if ($LASTEXITCODE -ne 0) {
    throw "Spec OCR Pipeline failed its post-install health check."
}

if ($GptFastPath) {
    Write-Host ""
    Write-Host "Windows searchable-PDF mode is ready."
    Write-Host "For GPT/OpenCode, sign in with your own account; never copy a shared auth session into the repository."
    exit 0
}

& $BootstrapPython -m spec_pipeline.cli setup-windows @SpecOcrArgs
exit $LASTEXITCODE
