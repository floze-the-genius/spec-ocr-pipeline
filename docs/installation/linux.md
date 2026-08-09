# Install on Linux

## Supported scope

The core workflow runs natively on Linux: read a searchable PDF, normalize rows, and create
Excel workbooks. Tesseract OCR is supported through a system-installed `tesseract` executable.

Local AI is a separate, hardware-specific path. The supported Linux local-AI configuration is
an x86_64 machine with a compatible NVIDIA GPU and driver. CPU-only AI, non-NVIDIA accelerators,
and other Linux architectures may work, but they are not part of the support contract.

## Requirements

- CPython 3.11-3.14
- A virtual environment
- Git, when installing from a source checkout
- Optional: Tesseract and the language packs required by your documents
- Optional local AI: Linux x86_64, NVIDIA GPU, a working driver, and enough VRAM for the chosen
  model

## Install the core CLI

Run these commands from a source checkout:

```bash
python3 -m venv ".venv"
source ".venv/bin/activate"
python -m pip install --upgrade "pip"
python -m pip install -e ".[ocr]"
```

The `ocr` extra adds PDF rendering support. It is useful for Tesseract and image-based
providers; the searchable-PDF text-layer provider itself remains the lightest path.

Verify the installation:

```bash
spec-ocr --version
spec-ocr doctor
spec-ocr process "examples/demo/specification.pdf" --output-dir "demo-output" --quote
```

The demo is synthetic and safe to use in logs and screenshots.

## Add Tesseract

Install Tesseract with the operating system's package manager. For example, on Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install "tesseract-ocr" "tesseract-ocr-eng"
tesseract --version
tesseract --list-langs
```

Install additional language packs from the same distribution repository. Then test a scanned
PDF explicitly:

```bash
spec-ocr extract-ocr "/path/to/scanned-specification.pdf" --provider "tesseract" --lang "eng" --output "demo-output/scanned-ocr.json"
```

The CLI does not download, embed, or replace the Tesseract system binary.

## Local AI on NVIDIA

Keep the core installation working before adding model runtimes. Install the project extras
needed by the selected provider, then follow [Providers](../providers.md). Model downloads can
be large and may be governed by separate licenses.

```bash
python -m pip install -e ".[ocr,local-models,dots-mocr-transformers]"
spec-ocr models status --json
nvidia-smi
```

The `dots-mocr` runtime does not download a model automatically. Use only an audited local snapshot
at an immutable revision; see [Providers](../providers.md) before enabling model-supplied code.

Do not treat a successful package installation as proof that CUDA is active. Confirm GPU use
with the provider-specific diagnostic and a synthetic document before processing private data.

## Next steps

- [Usage](../usage.md)
- [Providers](../providers.md)
- [Privacy](../privacy.md)
- [Troubleshooting](../troubleshooting.md)
