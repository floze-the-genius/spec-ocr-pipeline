# Providers

Provider choice determines accuracy, hardware requirements, and whether document data leaves the
machine. Select providers explicitly in automated or privacy-sensitive workflows.

## OCR extraction providers

| Provider | Data path | Best fit | Requirements |
| --- | --- | --- | --- |
| `pdf-text` | Local | Searchable or CAD-generated PDF with a usable text layer | Core Python dependencies |
| `tesseract` | Local | Scanned pages and conventional printed text | `ocr` extra, system Tesseract binary, language data |
| `paddleocr-vl` | Local when configured with a local runtime | Layout-heavy or visually complex pages | Optional packages, model files, supported local-AI hardware |
| `dots-mocr` | Local when configured with a local runtime | Experimental document/VLM extraction | Optional packages, model files, supported local-AI hardware |
| `auto` | Depends on the selected result | Interactive diagnosis | Inspect output metadata; avoid it when a fixed data path is required |

The Tesseract executable is never bundled. Install it with the operating system and confirm it is
on `PATH`.

`dots-mocr` model repositories can contain executable Python. Automatic model downloads and model
code execution are disabled. Obtain an immutable, approved snapshot separately, review its code
and license, pass its local directory with `--dots-model`, and add `--dots-trust-model-code` only
when you explicitly accept that code-execution boundary. A floating repository name is rejected.

## Mapping providers

| `--ai-client` | Data path | Supported public configuration |
| --- | --- | --- |
| `none` (default) | Local | Deterministic table-layer mapping only; unsupported inputs fail closed |
| `qwen-mlx` | Local | Apple Silicon macOS with MLX |
| `openai-compatible` | Local by default | Loopback vLLM, SGLang, or llama.cpp server on Linux x86_64 NVIDIA, or Windows x64 through WSL2 NVIDIA |
| `openai-compatible` with `--ai-external-openai-compatible` | External | User-selected API; OCR text is transmitted and images only when explicitly enabled |
| `opencode` | Configuration-dependent and potentially external | User's own OpenCode installation and authenticated account |

The local-AI support contract is deliberately narrow:

- macOS: Apple Silicon with MLX
- Linux: x86_64 with a compatible NVIDIA GPU
- Windows: x64 through WSL2 with a compatible NVIDIA GPU

There is no support claim for Intel Mac local AI, Windows ARM, or native-Windows CUDA. The core
searchable-PDF and Excel path remains native on Linux, macOS, and Windows.

## Selection guide

1. Try `pdf-text` when the PDF text is selectable and reading order is usable.
2. Use system Tesseract when pages are scans and a traditional OCR path is sufficient.
3. Use a supported local VLM only when layout or recognition quality requires it and the hardware
   envelope is met.
4. Use an external service only after the document owner approves the provider, region, retention,
   account, and transmitted content.

For deterministic and reproducible runs, record the provider, model identifier, model revision,
CLI version, page range, and quality report with the output.

## Local server example

The following endpoint is loopback-only and therefore represents a local server:

```text
spec-ocr map-ocr "example" --input-ocr "work/ocr.json" --source-pdf "/path/to/specification.pdf" --output "work/rows.jsonl" --ai-client "openai-compatible" --ai-base-url "http://127.0.0.1:8000/v1" --ai-served-model "local-model" --no-image-attachments
```

Do not bind a model server to `0.0.0.0` without authentication, transport security, network policy,
and an explicit operational reason.

Cleartext `http://` OpenAI-compatible endpoints are accepted only on loopback addresses such as
`127.0.0.1`, `::1`, and `localhost`. Remote endpoints must use HTTPS so bearer tokens and document
content are encrypted in transit.

## External API example

This synthetic endpoint uses the reserved `.invalid` domain and cannot make a real request:

```text
spec-ocr map-ocr "example" --input-ocr "work/ocr.json" --source-pdf "/path/to/specification.pdf" --output "work/rows.jsonl" --ai-client "openai-compatible" --ai-external-openai-compatible --ai-base-url "https://api.example.invalid/v1" --ai-served-model "example-model" --ai-api-key-env "SPEC_OCR_API_TOKEN" --no-image-attachments
```

Set `SPEC_OCR_API_TOKEN` through your shell, CI secret store, or operating-system credential tool.
Never put a token in a command, repository file, example, screenshot, or issue. Add
`--ai-external-supports-images` only when page-image transmission is approved.

Read [Privacy](privacy.md) before processing real documents.
