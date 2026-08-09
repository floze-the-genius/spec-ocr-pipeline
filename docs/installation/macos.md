# Install on macOS

## Supported scope

The core searchable-PDF and Excel workflow runs natively on macOS. Tesseract OCR is supported
through a system-installed `tesseract` executable.

The supported local-AI configuration is an Apple Silicon Mac using MLX. Intel Macs can use the
core workflow, but this project does not claim Intel Mac local-AI support.

## Requirements

- CPython 3.11-3.14
- A virtual environment
- Git, when installing from a source checkout
- Optional: Tesseract and its language packs
- Optional local AI: Apple Silicon and enough unified memory for the chosen model

## Install the core CLI

Run these commands from a source checkout:

```bash
python3 -m venv ".venv"
source ".venv/bin/activate"
python -m pip install --upgrade "pip"
python -m pip install -e ".[ocr]"
```

Verify the installation with the synthetic demo:

```bash
spec-ocr --version
spec-ocr doctor
spec-ocr process "examples/demo/specification.pdf" --output-dir "demo-output" --quote
```

## Add Tesseract

One common Homebrew installation is:

```bash
brew install "tesseract"
tesseract --version
tesseract --list-langs
```

Install the language data required by your documents using a trusted package source. Verify that
the requested language appears in `tesseract --list-langs` before running OCR.

```bash
spec-ocr extract-ocr "/path/to/scanned-specification.pdf" --provider "tesseract" --lang "eng" --output "demo-output/scanned-ocr.json"
```

The project uses the executable on `PATH`; it does not bundle Tesseract.

## Add local AI on Apple Silicon

Install the MLX runtime only after the core demo works:

```bash
python -m pip install -e ".[ocr,local-models,macos-local]"
spec-ocr models status --json
```

Models are not bundled. Review the model's license, disk size, and memory requirements before
downloading it. See [Providers](../providers.md) for provider selection and [Privacy](../privacy.md)
for the local/external boundary.

## Next steps

- [Usage](../usage.md)
- [Providers](../providers.md)
- [Privacy](../privacy.md)
- [Troubleshooting](../troubleshooting.md)
