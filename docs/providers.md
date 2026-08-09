# Providers and supported models

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

## Managed local-model catalog

`spec-ocr models status --json` reads the following catalog. Aliases are the stable values accepted
by the model-management and setup commands. Disk and memory figures are planning minimums, not
performance guarantees.

| Primary alias | Upstream model ID | Backend | Supported platform | Minimum resources | Approx. disk |
| --- | --- | --- | --- | --- | --- |
| `paddleocr-vl` | `PaddlePaddle/PaddleOCR-VL` | Paddle | macOS, Windows WSL2 | Runtime-dependent | 3 GB |
| `qwen3-8b` | `mlx-community/Qwen3-8B-4bit` | MLX | Apple Silicon macOS | 16 GB unified memory | 6 GB |
| `qwen3-14b` | `mlx-community/Qwen3-14B-4bit` | MLX | Apple Silicon macOS | 32 GB unified memory | 10 GB |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-4bit` | MLX-VLM | Apple Silicon macOS | 64 GB unified memory | 21 GB |
| `qwen36-safe` | `bartowski/Qwen_Qwen3.6-27B-GGUF` IQ3_XXS | llama.cpp | Linux, Windows WSL2 | 32 GB RAM | 12.5 GB |
| `qwen36-balanced` | `bartowski/Qwen_Qwen3.6-27B-GGUF` Q4_K_S | llama.cpp | Linux, Windows WSL2 | 48 GB RAM | 16.5 GB |
| `qwen36-quality` | `bartowski/Qwen_Qwen3.6-27B-GGUF` Q4_K_M | llama.cpp | Linux, Windows WSL2 | 48 GB RAM | 17.6 GB |
| `qwen3-8b-cuda` | `Qwen/Qwen3-8B` | vLLM | Linux, Windows WSL2 + NVIDIA | 12 GB VRAM | 8 GB |
| `qwen3-14b-cuda` | `Qwen/Qwen3-14B` | vLLM | Linux, Windows WSL2 + NVIDIA | 16 GB VRAM | 12 GB |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | SGLang | Linux, Windows WSL2 + NVIDIA | 16 GB VRAM | 10 GB |
| `qwen36-max` (experimental) | `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit` | vLLM | Linux, Windows WSL2 + NVIDIA | 48 GB RAM, 24 GB VRAM | 19 GB |

Additional aliases are shown by `spec-ocr models status --json` and resolve to the same entries.
The catalog is an integration manifest: models are downloaded separately, have their own licenses,
and may change upstream. Review the exact revision, files, license, and checksums before operational
use. Hardware-specific inference is supported only after a real synthetic smoke test on that
machine.

dots.mocr is intentionally not managed as an automatic remote download. Supply an audited local
snapshot with `--dots-model`; use `--dots-trust-model-code` only after reviewing model-supplied
Python. The commonly used Apple Silicon snapshot name is `mlx-community/dots.mocr-4bit`, but the
CLI requires the resolved local directory rather than silently fetching it.

`openai-compatible` is a protocol integration rather than a fixed model allowlist. A local vLLM,
SGLang, or llama.cpp server may expose one of the managed models above or another model that can
follow the mapping schema. `opencode` likewise uses the model configured in the user's own OpenCode
account. Compatibility with an arbitrary API model is not implied until the synthetic mapping
fixture succeeds and the returned rows pass quality review.

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
