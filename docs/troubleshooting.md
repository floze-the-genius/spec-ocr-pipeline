# Troubleshooting

Start with version and machine-readable diagnostics:

```text
spec-ocr --version
spec-ocr doctor --json
```

Use `python -m spec_pipeline.cli doctor --json` if the console script is unavailable. Do not post
the complete diagnostic output publicly until file paths, account names, endpoints, and other
sensitive values have been reviewed.

## `spec-ocr` is not found

Activate the same virtual environment in which the package was installed, or call its Python
directly. Reinstall the editable project from the repository root:

```text
python -m pip install -e ".[ocr]"
python -m spec_pipeline.cli --version
```

## PDF rendering reports a missing module

Install the `ocr` extra in the active environment:

```text
python -m pip install -e ".[ocr]"
```

Confirm that `python`, `pip`, and `spec-ocr` all resolve to the same environment.

## Tesseract is not found

The Python package does not include Tesseract. Install the system executable, open a new shell, and
run:

```text
tesseract --version
tesseract --list-langs
```

If a requested language is missing, install its language pack from the same trusted source as the
binary. Do not copy an executable or DLL bundle into the repository.

## `pdf-text` returns empty or disordered text

The PDF may be an image scan, use custom glyph encoding, or have a text layer whose reading order
does not match the table. Try a small page range with system Tesseract. Use a VLM provider only in
one of the supported hardware configurations and only after reviewing [Privacy](privacy.md).

## The output directory does not exist

Create it before an exporter writes the file:

```text
python -c "from pathlib import Path; Path('work').mkdir(parents=True, exist_ok=True)"
```

Quote paths that contain spaces. On Windows, avoid reserved file names and verify that ransomware
protection or corporate policy is not blocking the destination.

## MLX fails to install or import

MLX local AI is supported only on Apple Silicon macOS. Confirm `uname -m` reports `arm64`, use a
supported Python version, and install the Apple Silicon extra:

```text
python -m pip install -e ".[ocr,local-models,macos-local]"
```

Intel Mac local AI is outside the support contract; use the native core workflow instead.

## NVIDIA is not visible

On native Linux or inside WSL2, first run:

```text
nvidia-smi
```

If that fails, fix the operating-system or WSL driver path before changing Python packages. On
Windows, local NVIDIA AI is supported through WSL2 only; native-Windows CUDA is not a supported
fallback. Also confirm the selected model fits available VRAM.

## A local OpenAI-compatible server cannot be reached

Check that the server is listening on the same loopback URL passed to `--ai-base-url`, that the
served model name matches `--ai-served-model`, and that no proxy rewrites loopback traffic. Prefer
`127.0.0.1` to a public bind address. A connection failure must not trigger an unreviewed external
fallback.

## External API returns 401, 403, or a timeout

Confirm that `--ai-api-key-env` names an environment variable present in the current process; do
not paste the token into the command. Verify the provider URL, model entitlement, account limits,
and retention approval. Never request another person's session file or credential as a workaround.

## The quality gate fails

Open the JSON file passed to `--quality-json`, inspect the referenced rows, and compare them with
the source pages. Do not weaken `--quality-fail-on` merely to obtain a successful exit. Correct the
input/provider problem or document an authorized human review.

## Excel opens with a repair warning

Keep the JSONL and source PDF, record the CLI version, and reproduce with the synthetic demo. Test
the workbook in a supported spreadsheet application without overwriting the failing artifact.
Never attach a customer workbook to a public issue; create a synthetic minimal reproduction.
