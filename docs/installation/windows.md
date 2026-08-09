# Install on Windows

## Supported scope

The core searchable-PDF and Excel workflow runs natively on Windows. Tesseract OCR is supported
through a system-installed `tesseract.exe` available on `PATH`.

The supported Windows local-AI path is Windows x64 with WSL2 and a compatible NVIDIA GPU. This
project does not claim native-Windows CUDA support or Windows ARM support.

## Install the native core CLI

Use a 64-bit CPython 3.11-3.14. In PowerShell, from a source
checkout:

```powershell
py -3.11 -m venv ".venv"
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade "pip"
python -m pip install -e ".[ocr]"
```

If script execution policy prevents activation, either use an approved organization policy or
call the environment's Python directly:

```powershell
& ".\.venv\Scripts\python.exe" -m spec_pipeline.cli doctor
```

Verify the native core workflow with the synthetic demo:

```powershell
spec-ocr --version
spec-ocr doctor
spec-ocr process "examples\demo\specification.pdf" --output-dir "demo-output" --quote
```

## Add Tesseract

Install a maintained Windows Tesseract distribution from a source you trust. Select the required
language data during installation and add its installation directory to `PATH`. Start a new
PowerShell window, then verify:

```powershell
tesseract --version
tesseract --list-langs
spec-ocr extract-ocr "C:\Documents\scanned-specification.pdf" --provider "tesseract" --lang "eng" --output "demo-output\scanned-ocr.json"
```

The repository must not contain a copied `tesseract.exe`, DLLs, or language-data bundle.

## Local AI through WSL2 and NVIDIA

Use WSL2 for GPU-backed local AI. Install the NVIDIA Windows driver with WSL support, install an
approved Linux distribution, and verify the GPU inside WSL before downloading models:

```powershell
wsl --update
wsl nvidia-smi
```

The CLI can print a setup plan without changing the machine:

```powershell
spec-ocr setup-wsl --profile "qwen36-safe" --distro "Ubuntu" --json
```

Review the plan, dependency sources, model licenses, disk requirements, and bind addresses. Only
then may an operator intentionally add both `--execute` and `--confirm`. Do not use native Windows
CUDA as a fallback; it is outside the support contract.

Keep private PDFs in a WSL path with appropriate permissions, and do not expose a local model
server beyond the loopback interface unless you have separately secured it.

## Next steps

- [Usage](../usage.md)
- [Providers](../providers.md)
- [Privacy](../privacy.md)
- [Troubleshooting](../troubleshooting.md)
