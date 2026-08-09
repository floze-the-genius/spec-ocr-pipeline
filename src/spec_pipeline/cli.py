from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from spec_pipeline import __version__
from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult
from spec_pipeline.ai.openai_compatible_client import (
    OpenAiCompatibleClientConfig,
    OpenAiCompatibleExtractionClient,
)
from spec_pipeline.ai.opencode_client import OpenCodeClientConfig, OpenCodeExtractionClient
from spec_pipeline.ai.prompt import build_extraction_prompt
from spec_pipeline.ai.quality import assess_extracted_rows
from spec_pipeline.ai.qwen_mlx_client import QwenMlxClientConfig, QwenMlxExtractionClient
from spec_pipeline.ai.schema import ai_row_to_specification_row, validate_ai_rows
from spec_pipeline.benchmark.runner import LocalVsGptBenchmarkConfig, run_local_vs_gpt_benchmark
from spec_pipeline.models import looks_like_shifted_item_name

try:
    from spec_pipeline.ai import vision_repair as _vision_repair
except ImportError:  # Stage-1 packages can omit the optional vision layer.
    _vision_repair = None
from spec_pipeline.exporters.kp_xlsx import (
    ASCII_KP_TEMPLATE_NAME,
    default_kp_template_path,
    export_jsonl_to_kp_xlsx,
)
from spec_pipeline.exporters.xlsx import (
    default_xlsx_output_path,
    export_jsonl_to_xlsx,
    export_sample_to_xlsx,
)
from spec_pipeline.local.deps import install_local_dependencies
from spec_pipeline.local.hardware import collect_hardware_report
from spec_pipeline.local.llama_runtime import (
    inspect_llama_cpp_runtime,
    link_llama_cpp_cuda_libraries,
)
from spec_pipeline.local.model_manager import (
    ModelCacheLockError,
    ModelCacheManager,
    ModelNotReadyError,
    default_model_cache_dir,
)
from spec_pipeline.local.server_launcher import (
    LocalServerLaunchConfig,
    build_local_server_command,
    launch_local_server,
)
from spec_pipeline.local.wsl_installer import (
    WslInstallConfig,
    build_wsl_install_plan,
    execute_install_plan,
)
from spec_pipeline.ocr.base import OcrPage
from spec_pipeline.ocr.providers.tesseract import (
    TesseractConfig,
    TesseractOcrProvider,
    parse_page_selection,
)


def _prefer_utf8_console_on_windows() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_prefer_utf8_console_on_windows()


class _NoVisionRepairPlan:
    needed = False
    page_refs: list[str] = []
    issues: list[Any] = []


def detect_vision_repair_plan(rows: list[dict[str, object]]) -> Any:
    if _vision_repair is None:
        return _NoVisionRepairPlan()
    return _vision_repair.detect_vision_repair_plan(rows)


def existing_image_pages(pages: list[OcrPage], page_refs: set[str]) -> list[OcrPage]:
    if _vision_repair is None:
        return []
    return _vision_repair.existing_image_pages(pages, page_refs)


def merge_repaired_page_rows(
    original_rows: list[dict[str, object]],
    repaired_rows: list[dict[str, object]],
    repaired_page_refs: set[str],
) -> list[dict[str, object]]:
    if _vision_repair is None:
        return original_rows
    return _vision_repair.merge_repaired_page_rows(original_rows, repaired_rows, repaired_page_refs)


def apply_row_patches(
    original_rows: list[dict[str, object]],
    patches: list[dict[str, object]],
    affected_page_refs: set[str],
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]], list[dict[str, object]]]:
    if _vision_repair is None:
        return original_rows, ["third-layer module is not included in this package"], [], patches
    return _vision_repair.apply_row_patches(original_rows, patches, affected_page_refs)


from spec_pipeline.ocr.paddle_runtime import inspect_paddle_runtime
from spec_pipeline.ocr.providers.dots_mocr import DotsMocrConfig, DotsMocrProvider
from spec_pipeline.ocr.providers.paddleocr_vl import PaddleOcrVlConfig, PaddleOcrVlProvider
from spec_pipeline.ocr.providers.pdf_text import (
    PdfTextOcrProvider,
    compact_pdf_text_overlay,
    parse_pdf_table_layer_rows,
)
from spec_pipeline.parsers import PdfMinerSpec8Parser
from spec_pipeline.samples import (
    DEFAULT_MANIFEST_PATH,
    load_ground_truth_rows,
    load_sample_manifest,
    summarize_manifest,
)
from spec_pipeline.workflows.compare import compare_sample_against_jsonl, write_compare_summary
from spec_pipeline.workflows.normalize import default_output_path, normalize_sample_to_jsonl
from spec_pipeline.workflows.ocr import (
    default_ocr_output_path,
    default_ocr_page_dir,
    load_ocr_pages_json,
    write_ocr_pages_json,
)
from spec_pipeline.workflows.parse import (
    default_diagnostics_output_path,
    default_parsed_output_path,
    parse_sample_to_jsonl,
    write_diagnostics,
    write_parsed_rows_jsonl,
)
from spec_pipeline.workflows.pdf_probe import probe_pdf_text_layer


def _default_workspace_root() -> Path:
    configured = os.environ.get("SPEC_OCR_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()

    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / "pyproject.toml").is_file():
        return source_checkout
    return Path.cwd().resolve()


PROJECT_ROOT = _default_workspace_root()
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
PARSER_BACKENDS = {
    "pdfminer-spec8-text": PdfMinerSpec8Parser,
}

WINDOWS_SETUP_PROFILES = (
    (
        "qwen36-safe",
        "Qwen3.6 Безопасный",
        "Рекомендуется для RTX 4080 Super 16GB: самый надежный запуск, качество ниже.",
    ),
    (
        "gpt",
        "GPT-5.5 онлайн",
        "Лучшее качество и скорость при наличии интернета.",
    ),
    (
        "qwen36-balanced",
        "Qwen3.6 Сбалансированный",
        "Качество выше, но больше нагрузка на память и диск. Пробовать после безопасного режима.",
    ),
    (
        "qwen36-quality",
        "Qwen3.6 Качественный",
        "Лучший GGUF-вариант по качеству, самый медленный; вероятно понадобится CPU offload.",
    ),
    (
        "qwen36-max",
        "Qwen3.6 Max (экспериментальный)",
        "GPTQ/vLLM вариант. Пробовать только если безопасные GGUF-профили не подходят.",
    ),
    (
        "standard",
        "Qwen3 8B быстрый запасной вариант",
        "Это не Qwen3.6, но самый простой быстрый fallback, если Qwen3.6 слишком медленный.",
    ),
)


def parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec-ocr",
        description="Turn engineering specification PDFs into structured data and Excel",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Print project scope and current status")
    doctor_parser = subparsers.add_parser("doctor", help="Check local project and runtime status")
    doctor_parser.add_argument("--json", action="store_true", help="Print status as JSON")

    paddle_gpu_parser = subparsers.add_parser(
        "paddle-gpu-check",
        help="Check whether PaddlePaddle can use CUDA for PaddleOCR-VL",
    )
    paddle_gpu_parser.add_argument("--json", action="store_true", help="Print status as JSON")
    paddle_gpu_parser.add_argument(
        "--device",
        default="gpu:0",
        help="Paddle device to smoke test, for example gpu:0",
    )
    paddle_gpu_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return failure unless PaddlePaddle CUDA is available",
    )
    paddle_gpu_parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip tiny tensor allocation on the selected CUDA device",
    )

    llama_gpu_parser = subparsers.add_parser(
        "llama-gpu-check",
        help="Check whether llama-cpp-python supports CUDA/GPU offload",
    )
    llama_gpu_parser.add_argument("--json", action="store_true", help="Print status as JSON")
    llama_gpu_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return failure unless llama.cpp GPU offload is available",
    )

    llama_link_parser = subparsers.add_parser(
        "llama-link-cuda-libs",
        help="Link PyPI NVIDIA CUDA libraries into llama_cpp/lib for WSL CUDA loading",
    )
    llama_link_parser.add_argument("--json", action="store_true", help="Print status as JSON")

    models_parser = subparsers.add_parser("models", help="Manage local model cache")
    models_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Model cache directory. Defaults to artifacts/models",
    )
    models_parser.add_argument(
        "--json",
        action="store_true",
        help="Print status as JSON. For subcommands this can also be passed after the action.",
    )
    models_subparsers = models_parser.add_subparsers(dest="models_action")

    models_status_parser = models_subparsers.add_parser("status", help="Show local model status")
    models_status_parser.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="Print status as JSON"
    )

    for action_name in ("download", "verify", "repair", "update", "remove"):
        action_parser = models_subparsers.add_parser(
            action_name, help=f"{action_name} a local model"
        )
        action_parser.add_argument("model", help="Model alias, for example qwen3-8b")
        action_parser.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS, help="Print result as JSON"
        )
        if action_name in {"update", "remove"}:
            action_parser.add_argument(
                "--confirm",
                action="store_true",
                help="Required for destructive or revision-changing operations",
            )

    setup_local_parser = subparsers.add_parser(
        "setup-local",
        help="Настройка локального OCR/ИИ профиля",
    )
    setup_local_parser.add_argument(
        "--profile",
        choices=[
            "fast",
            "standard",
            "hq",
            "qwen36-safe",
            "qwen36-balanced",
            "qwen36-quality",
            "qwen36-max",
        ],
        default="qwen36-safe",
        help="Профиль качества: qwen36-safe по умолчанию, либо fast/standard/hq/другой Qwen3.6 профиль",
    )
    setup_local_parser.add_argument(
        "--download",
        action="store_true",
        help="Скачать отсутствующие модели. Без этого флага команда только проверяет статус.",
    )
    setup_local_parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install local runtime dependencies before checking/downloading models",
    )
    setup_local_parser.add_argument(
        "--skip-paddle-install",
        action="store_true",
        help="When installing deps, skip PaddleOCR-VL optional dependencies",
    )
    setup_local_parser.add_argument("--json", action="store_true", help="Print result as JSON")
    setup_local_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Model cache directory. Defaults to artifacts/models",
    )

    setup_wsl_parser = subparsers.add_parser(
        "setup-wsl",
        help="Создать или выполнить план установки Windows/WSL2 CUDA",
    )
    setup_wsl_parser.add_argument(
        "--profile",
        choices=[
            "fast",
            "standard",
            "hq",
            "qwen36-safe",
            "qwen36-balanced",
            "qwen36-quality",
            "qwen36-max",
            "gpt",
        ],
        default="qwen36-safe",
        help="Local quality profile to install. Defaults to qwen36-safe for RTX 4080 Super 16GB.",
    )
    setup_wsl_parser.add_argument("--distro", default="Ubuntu-24.04", help="WSL distro name")
    setup_wsl_parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory as seen from Windows or WSL. Defaults to the current directory.",
    )
    setup_wsl_parser.add_argument("--venv", default=".venv-wsl", help="WSL virtualenv path")
    setup_wsl_parser.add_argument(
        "--cache-dir",
        default="artifacts/models",
        help="Model cache directory inside the project/WSL environment",
    )
    setup_wsl_parser.add_argument("--host", default="127.0.0.1", help="Server bind host")
    setup_wsl_parser.add_argument("--port", type=int, default=8000, help="Server bind port")
    setup_wsl_parser.add_argument(
        "--served-model-name",
        default=None,
        help="Stable OpenAI served model name. Defaults from profile.",
    )
    setup_wsl_parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "sglang", "llama-cpp"],
        default="auto",
        help="serve-local backend validation mode",
    )
    setup_wsl_parser.add_argument(
        "--runtime",
        choices=["auto", "vllm", "sglang", "llama-cpp"],
        default="auto",
        help="Runtime package to install inside WSL",
    )
    setup_wsl_parser.add_argument(
        "--skip-runtime-install",
        action="store_true",
        help="Do not install vLLM/SGLang/llama.cpp runtime packages",
    )
    setup_wsl_parser.add_argument(
        "--skip-paddle-install",
        action="store_true",
        help="Install project local-models extra without PaddleOCR-VL extra",
    )
    setup_wsl_parser.add_argument(
        "--gpt-auth-file",
        default=None,
        help=(
            "Advanced opt-in path to your own OpenCode auth.json. By default no credential "
            "is imported and the current WSL user signs in with `opencode auth login`. "
            "Never commit, publish, or share this file."
        ),
    )
    setup_wsl_parser.add_argument(
        "--skip-gpt-smoke",
        action="store_true",
        help="Do not run the tiny GPT smoke test during GPT setup",
    )
    setup_wsl_parser.add_argument("--execute", action="store_true", help="Run generated commands")
    setup_wsl_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required together with --execute for mutable setup commands",
    )
    setup_wsl_parser.add_argument("--json", action="store_true", help="Print plan/result as JSON")

    setup_windows_parser = subparsers.add_parser(
        "setup-windows",
        aliases=["setup-customer", "setup-wizard"],
        help="Удобный мастер установки Windows с выбором модели",
    )
    setup_windows_parser.add_argument(
        "--profile",
        choices=[choice[0] for choice in WINDOWS_SETUP_PROFILES],
        default=None,
        help="Preselect model profile. If omitted, an interactive menu is shown.",
    )
    setup_windows_parser.add_argument("--distro", default="Ubuntu-24.04", help="WSL distro name")
    setup_windows_parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory as seen from Windows or WSL. Defaults to the current directory.",
    )
    setup_windows_parser.add_argument("--venv", default=".venv-wsl", help="WSL virtualenv path")
    setup_windows_parser.add_argument(
        "--cache-dir",
        default="artifacts/models",
        help="Model cache directory inside the project/WSL environment",
    )
    setup_windows_parser.add_argument("--host", default="127.0.0.1", help="Server bind host")
    setup_windows_parser.add_argument("--port", type=int, default=8000, help="Server bind port")
    setup_windows_parser.add_argument(
        "--yes",
        action="store_true",
        help="Run installation after selecting the profile without asking again",
    )
    setup_windows_parser.add_argument(
        "--skip-runtime-install",
        action="store_true",
        help="Do not install vLLM/SGLang/llama.cpp runtime packages",
    )
    setup_windows_parser.add_argument(
        "--skip-paddle-install",
        action="store_true",
        help="Install project local-models extra without PaddleOCR-VL extra",
    )
    setup_windows_parser.add_argument(
        "--gpt-auth-file",
        default=None,
        help=(
            "Расширенный opt-in: путь к вашему собственному OpenCode auth.json. "
            "По умолчанию credential не импортируется — текущий пользователь входит "
            "через `opencode auth login` внутри WSL. Не коммитьте, не публикуйте "
            "и не передавайте этот файл."
        ),
    )
    setup_windows_parser.add_argument(
        "--skip-gpt-smoke",
        action="store_true",
        help="Не выполнять короткий GPT smoke-test во время установки",
    )
    setup_windows_parser.add_argument(
        "--json", action="store_true", help="Print selected plan as JSON"
    )

    export_gpt_auth_parser = subparsers.add_parser(
        "export-gpt-auth",
        help=(
            "Расширенный opt-in: экспортировать ваш собственный OpenCode credential "
            "для приватного переноса"
        ),
    )
    export_gpt_auth_parser.add_argument(
        "--source",
        default=str(Path.home() / ".local" / "share" / "opencode" / "auth.json"),
        help="Ваш собственный исходный OpenCode auth.json",
    )
    export_gpt_auth_parser.add_argument(
        "--output",
        required=True,
        help="Явный приватный путь назначения; не используйте путь внутри Git-репозитория",
    )
    export_gpt_auth_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Подтвердить приватный перенос credential; файл нельзя коммитить или передавать",
    )
    export_gpt_auth_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    serve_local_parser = subparsers.add_parser(
        "serve-local",
        help="Launch a local vLLM/SGLang/llama.cpp OpenAI-compatible server from cache",
    )
    serve_local_parser.add_argument(
        "model",
        help="Local model alias, for example qwen3-8b-cuda or qwen3-vl-8b",
    )
    serve_local_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Model cache directory. Defaults to artifacts/models",
    )
    serve_local_parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "sglang", "llama-cpp"],
        default="auto",
        help="Server backend. auto uses the model manifest backend.",
    )
    serve_local_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_local_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_local_parser.add_argument(
        "--served-model-name",
        default=None,
        help="OpenAI served model name. Defaults to the resolved local model path.",
    )
    serve_local_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without launching the server",
    )
    serve_local_parser.add_argument("--json", action="store_true", help="Print result as JSON")
    serve_local_parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra backend argument appended to the generated command. Repeat as needed; use --extra-arg=--flag for flag values.",
    )

    samples_parser = subparsers.add_parser("samples", help="Inspect fixture directories")
    samples_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    samples_parser.add_argument(
        "--json",
        action="store_true",
        help="Print fixture directory status as JSON",
    )
    samples_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate referenced external artifact paths and row counts",
    )

    rows_parser = subparsers.add_parser("rows", help="Print normalized rows for a sample")
    rows_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    rows_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    rows_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of rows to print",
    )
    rows_parser.add_argument(
        "--json",
        action="store_true",
        help="Print rows as JSON",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Write normalized JSONL for a sample from its ground-truth TSV",
    )
    normalize_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    normalize_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    normalize_parser.add_argument(
        "--output",
        default=None,
        help="Output path for JSONL, or '-' for stdout",
    )

    export_parser = subparsers.add_parser(
        "export-xlsx",
        help="Write a single-sheet Excel workbook for a sample",
    )
    export_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    export_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    export_parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional normalized JSONL input path",
    )
    export_parser.add_argument(
        "--output",
        default=None,
        help="Output XLSX path",
    )

    export_jsonl_parser = subparsers.add_parser(
        "export-jsonl-xlsx",
        help="Write a single-sheet Excel workbook directly from normalized JSONL",
    )
    export_jsonl_parser.add_argument("input_jsonl", help="Input normalized JSONL path")
    export_jsonl_parser.add_argument("--output", required=True, help="Output XLSX path")

    export_kp_parser = subparsers.add_parser(
        "export-kp-xlsx",
        help="Write a КП-style Excel workbook from normalized JSONL using the customer template",
    )
    export_kp_parser.add_argument("input_jsonl", help="Input normalized JSONL path")
    export_kp_parser.add_argument("--output", required=True, help="Output XLSX path")
    export_kp_parser.add_argument(
        "--template",
        default=None,
        help="Path to КП template XLSX. Defaults to examples/kp-template.xlsx.",
    )
    export_kp_parser.add_argument(
        "--sheet",
        default=None,
        help="Template sheet name. Defaults to Quote.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare predicted JSONL rows against the ground-truth sample",
    )
    compare_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    compare_parser.add_argument("predicted_jsonl", help="Predicted normalized JSONL path")
    compare_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    compare_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the comparison summary as JSON",
    )

    parse_parser = subparsers.add_parser(
        "parse-pdf",
        help="Parse a supported PDF sample into normalized JSONL",
    )
    parse_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    parse_parser.add_argument(
        "--backend",
        default="pdfminer-spec8-text",
        choices=sorted(PARSER_BACKENDS),
        help="Parser backend",
    )
    parse_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    parse_parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path",
    )

    diagnose_parser = subparsers.add_parser(
        "diagnose-pdf",
        help="Run parser diagnostics for a supported PDF sample",
    )
    diagnose_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    diagnose_parser.add_argument(
        "--backend",
        default="pdfminer-spec8-text",
        choices=sorted(PARSER_BACKENDS),
        help="Parser backend",
    )
    diagnose_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )
    diagnose_parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of sample clusters to keep per page in diagnostics",
    )
    diagnose_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional diagnostics JSON path",
    )

    probe_parser = subparsers.add_parser(
        "probe-pdf",
        help="Probe whether a sample PDF has a usable text layer or needs OCR",
    )
    probe_parser.add_argument("sample_id", help="Sample identifier from the manifest")
    probe_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample manifest JSON",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Convert a searchable specification PDF to JSONL and Excel without external AI",
    )
    process_parser.add_argument("pdf_path", help="Path to a searchable PDF file")
    process_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to ./<pdf-name>-spec-ocr",
    )
    process_parser.add_argument(
        "--pages",
        default=None,
        help="Optional page selection, for example '1', '1-3', or '1,3-5'",
    )
    process_parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    process_parser.add_argument(
        "--quote",
        action="store_true",
        help="Also create a commercial-proposal workbook from the packaged template",
    )
    process_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing output files in the selected directory",
    )

    ocr_parser = subparsers.add_parser(
        "extract-ocr",
        help="Render PDF pages and extract OCR text/tokens for AI mapping",
    )
    ocr_parser.add_argument("pdf_path", help="Path to a PDF file")
    ocr_parser.add_argument(
        "--provider",
        choices=["auto", "tesseract", "pdf-text", "paddleocr-vl", "dots-mocr"],
        default="tesseract",
        help=(
            "Extraction provider: auto, Tesseract OCR, PDF text layer, PaddleOCR-VL, or dots.mocr"
        ),
    )
    ocr_parser.add_argument(
        "--pages",
        default=None,
        help="Page selection, for example '1', '1-3', or '1,3-5'",
    )
    ocr_parser.add_argument(
        "--output",
        default=None,
        help="Output OCR JSON path",
    )
    ocr_parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    ocr_parser.add_argument(
        "--lang",
        default="rus+eng",
        help="Tesseract language expression",
    )
    ocr_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode",
    )
    ocr_parser.add_argument(
        "--pdf-text-overlay",
        type=parse_cli_bool,
        default=None,
        help="Append any usable PDF text layer to OCR output for searchable/CAD PDFs",
    )
    ocr_parser.add_argument(
        "--paddle-model",
        default="PaddlePaddle/PaddleOCR-VL",
        help="PaddleOCR-VL model name or local path",
    )
    ocr_parser.add_argument(
        "--paddle-device",
        default=None,
        help="PaddleOCR-VL device. On Apple Silicon direct PaddlePaddle mode uses cpu; use MLX-VLM for acceleration.",
    )
    ocr_parser.add_argument(
        "--paddle-pipeline-version",
        choices=["v1", "v1.5"],
        default="v1",
        help="PaddleOCR-VL pipeline version. Managed paddleocr-vl cache currently uses v1.",
    )
    ocr_parser.add_argument(
        "--paddle-vl-backend",
        choices=[
            "native",
            "mlx-vlm-server",
            "vllm-server",
            "sglang-server",
            "fastdeploy-server",
            "llama-cpp-server",
        ],
        default=None,
        help="PaddleOCR-VL recognition backend. Use mlx-vlm-server on Apple Silicon for faster VLM inference.",
    )
    ocr_parser.add_argument(
        "--paddle-vl-server-url",
        default=None,
        help="VLM server URL for server backends, for example http://localhost:8111/",
    )
    ocr_parser.add_argument(
        "--paddle-vl-api-model",
        default=None,
        help="Model name/path exposed by the VLM server",
    )
    ocr_parser.add_argument(
        "--paddle-max-new-tokens",
        type=int,
        default=None,
        help="Maximum VLM tokens for PaddleOCR-VL diagnostics/performance tuning",
    )
    ocr_parser.add_argument("--paddle-min-pixels", type=int, default=None)
    ocr_parser.add_argument("--paddle-max-pixels", type=int, default=None)
    ocr_parser.add_argument(
        "--paddle-repetition-penalty",
        type=float,
        default=None,
        help="Override PaddleOCR-VL generation repetition penalty",
    )
    ocr_parser.add_argument(
        "--paddle-temperature",
        type=float,
        default=None,
        help="Override PaddleOCR-VL generation temperature",
    )
    ocr_parser.add_argument(
        "--paddle-top-p",
        type=float,
        default=None,
        help="Override PaddleOCR-VL generation top-p",
    )
    ocr_parser.add_argument(
        "--paddle-split-wide-pages",
        type=parse_cli_bool,
        default=None,
        help="Split wide table pages into OCR bands before PaddleOCR-VL recognition",
    )
    ocr_parser.add_argument(
        "--paddle-band-height",
        type=int,
        default=520,
        help="Rendered-pixel height for --paddle-split-wide-pages OCR bands",
    )
    ocr_parser.add_argument(
        "--paddle-band-overlap",
        type=int,
        default=80,
        help="Rendered-pixel overlap for --paddle-split-wide-pages OCR bands",
    )
    ocr_parser.add_argument(
        "--paddle-predict-timeout",
        type=int,
        default=None,
        help="Maximum seconds to wait for one PaddleOCR-VL image/band prediction",
    )
    ocr_parser.add_argument(
        "--paddle-predict-heartbeat",
        type=int,
        default=60,
        help="Seconds between PaddleOCR-VL progress heartbeats during long predictions",
    )
    ocr_parser.add_argument(
        "--paddle-use-layout-detection",
        type=parse_cli_bool,
        default=None,
        help="Override PaddleOCR-VL layout detection: true or false",
    )
    ocr_parser.add_argument(
        "--paddle-use-queues",
        type=parse_cli_bool,
        default=None,
        help="Override PaddleOCR-VL internal queues: true or false",
    )
    ocr_parser.add_argument("--paddle-engine", default=None, help="PaddleOCR engine override")
    ocr_parser.add_argument("--paddle-cpu-threads", type=int, default=None)
    ocr_parser.add_argument(
        "--paddle-require-cuda",
        action="store_true",
        help="Fail instead of silently falling back to CPU when CUDA PaddlePaddle is unavailable",
    )
    ocr_parser.add_argument(
        "--paddle-enable-hpi",
        type=parse_cli_bool,
        default=None,
        help="Override PaddleOCR high-performance inference: true or false",
    )
    ocr_parser.add_argument(
        "--paddle-use-tensorrt",
        type=parse_cli_bool,
        default=None,
        help="Override PaddleOCR TensorRT usage: true or false",
    )
    ocr_parser.add_argument(
        "--paddle-precision", default=None, help="PaddleOCR precision, e.g. fp32 or fp16"
    )
    ocr_parser.add_argument(
        "--paddle-vl-max-concurrency",
        type=int,
        default=None,
        help="Max concurrency for VLM server backend",
    )
    ocr_parser.add_argument(
        "--dots-model",
        default="mlx-community/dots.mocr-4bit",
        help="dots.mocr model name or local path",
    )
    ocr_parser.add_argument(
        "--dots-backend",
        choices=["mlx-vlm", "transformers"],
        default="mlx-vlm",
        help="dots.mocr runtime backend: mlx-vlm for Apple Silicon, transformers for Windows/Linux checks",
    )
    ocr_parser.add_argument(
        "--dots-device",
        default=None,
        help="dots.mocr transformers device: auto, cuda, cpu, or mps",
    )
    ocr_parser.add_argument(
        "--dots-device-map",
        default=None,
        help="Optional transformers device_map, for example auto",
    )
    ocr_parser.add_argument(
        "--dots-torch-dtype",
        default="auto",
        help="dots.mocr transformers dtype: auto, float16, bfloat16, or float32",
    )
    ocr_parser.add_argument(
        "--dots-attn-implementation",
        default=None,
        help="Optional transformers attention implementation, for example sdpa or eager",
    )
    ocr_parser.add_argument(
        "--dots-trust-model-code",
        action="store_true",
        help=(
            "Allow an audited local dots.mocr snapshot to execute model-supplied code. "
            "Remote model identifiers are never downloaded automatically."
        ),
    )
    ocr_parser.add_argument(
        "--dots-upscale",
        type=float,
        default=2.0,
        help=(
            "Scale rendered/cropped table image before dots.mocr. "
            "0 enables auto mode: 2x for red/changed quantities, otherwise 1x."
        ),
    )
    ocr_parser.add_argument(
        "--dots-max-image-edge",
        type=int,
        default=2600,
        help="Downscale preprocessed dots.mocr image if its longest edge exceeds this value; 0 disables",
    )
    ocr_parser.add_argument(
        "--dots-max-tokens",
        type=int,
        default=4096,
        help="Maximum dots.mocr output tokens per page",
    )
    ocr_parser.add_argument(
        "--dots-no-crop",
        action="store_true",
        help="Disable automatic non-white content crop before dots.mocr",
    )
    ocr_parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Local model cache directory for managed local models",
    )

    map_parser = subparsers.add_parser(
        "map-ocr",
        help="Map OCR JSON into normalized rows using a deterministic table layer or an explicit AI provider",
    )
    map_parser.add_argument("sample_id", help="Identifier to write into extracted rows")
    map_parser.add_argument("--input-ocr", required=True, help="OCR JSON from extract-ocr")
    map_parser.add_argument("--source-pdf", required=True, help="Source PDF path")
    map_parser.add_argument("--output", default=None, help="Output JSONL path")
    map_parser.add_argument(
        "--quality-json", default=None, help="Optional quality report JSON path"
    )
    map_parser.add_argument(
        "--quality-fail-on",
        choices=["none", "error", "uncorrected-warning", "warning"],
        default="error",
        help=(
            "Quality gate for map-ocr: none never fails, error fails on quality errors, "
            "uncorrected-warning allows export-normalized OCR warnings, "
            "warning fails on any quality warning or error"
        ),
    )
    map_parser.add_argument(
        "--performance-json",
        default=None,
        help="Optional performance/benchmark summary JSON path",
    )
    map_parser.add_argument(
        "--ai-client",
        choices=["none", "opencode", "qwen-mlx", "openai-compatible"],
        default="none",
        help=(
            "AI extraction backend. Defaults to none (fail closed); selecting opencode may "
            "transmit OCR text and page images to an external provider."
        ),
    )
    map_parser.add_argument(
        "--ai-model",
        default="openai/gpt-5.5",
        help="OpenCode model identifier or local model alias for local AI backends",
    )
    map_parser.add_argument("--ai-agent", default="build", help="OpenCode primary agent name")
    map_parser.add_argument(
        "--ai-timeout",
        type=int,
        default=600,
        help="OpenCode command timeout in seconds",
    )
    map_parser.add_argument(
        "--ai-max-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for local OpenAI-compatible or MLX backends",
    )
    map_parser.add_argument(
        "--ai-page-batch-size",
        type=int,
        default=0,
        help="Use a fixed number of OCR pages per AI request; 0 enables context-aware batching if --ai-context-tokens is set",
    )
    map_parser.add_argument(
        "--ai-context-tokens",
        type=int,
        default=0,
        help="Total context window for context-aware AI page batching; 0 disables dynamic batching",
    )
    map_parser.add_argument(
        "--ai-context-margin",
        type=int,
        default=512,
        help="Token safety margin reserved when planning context-aware AI batches",
    )
    map_parser.add_argument(
        "--no-image-attachments",
        action="store_true",
        help="Send OCR text only without attaching rendered page images",
    )
    map_parser.add_argument(
        "--ai-force-on-pdf-table-layer",
        action="store_true",
        help=(
            "Run AI even when deterministic PDF_TABLE_LAYER rows exist; "
            "fallback to the deterministic rows if AI returns too few rows"
        ),
    )
    map_parser.add_argument(
        "--ai-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible server base URL for vLLM/SGLang/llama.cpp or cloud providers",
    )
    map_parser.add_argument(
        "--ai-served-model",
        default=None,
        help="Optional served model name for the OpenAI-compatible local server",
    )
    map_parser.add_argument(
        "--ai-external-openai-compatible",
        action="store_true",
        help="Use --ai-base-url as an external OpenAI-compatible API and skip local model cache checks",
    )
    map_parser.add_argument(
        "--ai-external-supports-images",
        action="store_true",
        help="Allow an external OpenAI-compatible API to receive page image attachments",
    )
    map_parser.add_argument(
        "--ai-api-key-env",
        default=None,
        help="Environment variable containing the Bearer token for an external OpenAI-compatible API",
    )
    map_parser.add_argument(
        "--ai-reasoning-effort",
        default=None,
        help=(
            "Optional reasoning effort. For OpenAI-compatible providers this is sent as "
            "reasoning_effort; for OpenCode this is sent as --variant, for example xhigh."
        ),
    )
    map_parser.add_argument(
        "--third-layer-reviewer",
        choices=["none", "kimi-fireworks", "opencode-gpt", "openai-compatible"],
        default="none",
        help="Optional third-layer row patch reviewer; primary extraction remains unchanged",
    )
    map_parser.add_argument(
        "--third-layer-model",
        default=None,
        help="Model for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--third-layer-agent",
        default="build",
        help="OpenCode agent for --third-layer-reviewer opencode-gpt",
    )
    map_parser.add_argument(
        "--third-layer-base-url",
        default=None,
        help="OpenAI-compatible base URL for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--third-layer-served-model",
        default=None,
        help="Served model name for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--third-layer-api-key-env",
        default=None,
        help="Environment variable containing the optional third-layer API key",
    )
    map_parser.add_argument(
        "--third-layer-reasoning-effort",
        default=None,
        help="Optional reasoning_effort value for the third-layer reviewer",
    )
    map_parser.add_argument(
        "--third-layer-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--third-layer-max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--no-third-layer-image-attachments",
        action="store_true",
        help="Disable page image attachments for the optional third-layer reviewer",
    )
    map_parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Local model cache directory for local AI backends",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Time qwen-mlx against OpenCode GPT-5.5 on the same OCR JSON",
    )
    benchmark_parser.add_argument("sample_id", help="Identifier to write into extracted rows")
    benchmark_parser.add_argument("--input-ocr", required=True, help="OCR JSON from extract-ocr")
    benchmark_parser.add_argument("--source-pdf", required=True, help="Source PDF path")
    benchmark_parser.add_argument("--output-dir", default=None, help="Benchmark output directory")
    benchmark_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Optional sample manifest for ground-truth comparison",
    )
    benchmark_parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Local model cache directory",
    )
    benchmark_parser.add_argument(
        "--local-ai-client",
        choices=["qwen-mlx", "openai-compatible"],
        default="qwen-mlx",
        help="Local backend to compare against GPT. Use openai-compatible for Windows/WSL2 CUDA.",
    )
    benchmark_parser.add_argument(
        "--qwen-model",
        "--local-model",
        dest="qwen_model",
        default=None,
        help="Local model alias. Defaults to qwen3-8b for MLX or qwen3-8b-cuda for openai-compatible.",
    )
    benchmark_parser.add_argument(
        "--local-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible local server URL for Windows/WSL2 CUDA benchmark",
    )
    benchmark_parser.add_argument(
        "--local-served-model",
        default=None,
        help="Served model name exposed by the local OpenAI-compatible server",
    )
    benchmark_parser.add_argument(
        "--gpt-model", default="openai/gpt-5.5", help="OpenCode model identifier"
    )
    benchmark_parser.add_argument("--gpt-agent", default="build", help="OpenCode primary agent")
    benchmark_parser.add_argument("--ai-timeout", type=int, default=600, help="AI timeout seconds")
    benchmark_parser.add_argument(
        "--no-image-attachments",
        action="store_true",
        help="Do not attach page images to GPT/OpenCode request",
    )
    benchmark_parser.add_argument("--json", action="store_true", help="Print report as JSON")

    return parser


def command_info() -> int:
    print(f"Spec OCR Pipeline {__version__}")
    print("goal: PDF specifications -> normalized rows -> Excel")
    print("status: public alpha")
    print(f"workspace: {PROJECT_ROOT}")
    return 0


def command_doctor(as_json: bool = False) -> int:
    core_modules = {
        "openpyxl": importlib.util.find_spec("openpyxl") is not None,
        "pdfminer": importlib.util.find_spec("pdfminer") is not None,
    }
    optional_modules = {
        "pillow": importlib.util.find_spec("PIL") is not None,
        "pypdfium2": importlib.util.find_spec("pypdfium2") is not None,
        "paddleocr": importlib.util.find_spec("paddleocr") is not None,
        "mlx": importlib.util.find_spec("mlx") is not None,
    }
    executables = {
        "tesseract": shutil.which("tesseract"),
        "opencode": shutil.which("opencode"),
    }
    python_supported = sys.version_info >= (3, 11)
    missing_core = sorted(name for name, available in core_modules.items() if not available)
    hardware = collect_hardware_report()
    payload = {
        "ok": python_supported and not missing_core,
        "version": __version__,
        "workspace": str(PROJECT_ROOT),
        "python_supported": python_supported,
        "core_modules": core_modules,
        "missing_core_modules": missing_core,
        "optional_modules": optional_modules,
        "executables": executables,
        # Retained for compatibility with earlier machine-readable doctor output.
        "missing_paths": [],
        "missing_required_paths": [],
        "missing_optional_paths": [],
        "hardware": hardware.to_dict(),
        "model_cache_dir": str(default_model_cache_dir(PROJECT_ROOT)),
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["ok"] else 1

    if not python_supported:
        print("doctor: Python 3.11 or newer is required")
        return 1
    if missing_core:
        print("doctor: missing core Python modules")
        for module in missing_core:
            print(f"- {module}")
        return 1

    print("doctor: ok")
    print(f"version: {__version__}")
    print(f"workspace: {PROJECT_ROOT}")
    print(f"hardware: {hardware.os}, {hardware.machine}, python {hardware.python}")
    if hardware.nvidia_gpu:
        print(f"gpu: {hardware.nvidia_gpu}")
    elif hardware.nvidia_smi:
        print("gpu: nvidia-smi found, GPU name unavailable")
    else:
        print("gpu: NVIDIA not found")
    print("mlx: available" if hardware.mlx_available else "mlx: unavailable")
    print("tesseract: available" if executables["tesseract"] else "tesseract: unavailable")
    print("pdfium: available" if optional_modules["pypdfium2"] else "pdfium: unavailable")
    return 0


def command_paddle_gpu_check(
    *,
    as_json: bool = False,
    device: str = "gpu:0",
    require_cuda: bool = False,
    smoke_test: bool = True,
) -> int:
    info = inspect_paddle_runtime(device=device, smoke_test=smoke_test)
    ok = info.cuda_ready if require_cuda else info.installed
    payload = info.to_dict() | {"ok": ok, "required_cuda": require_cuda}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if info.installed:
            print(f"paddle: {info.version}")
            print("paddle cuda: yes" if info.compiled_with_cuda else "paddle cuda: no")
            print(f"cuda devices: {info.cuda_device_count}")
            if info.selected_device:
                print(f"selected device: {info.selected_device}")
            if info.smoke_test_ok is not None:
                print("cuda smoke test: ok" if info.smoke_test_ok else "cuda smoke test: failed")
        else:
            print("paddle: not installed")
        if info.error:
            print(f"error: {info.error}")
    return 0 if ok else 1


def command_llama_gpu_check(*, as_json: bool = False, require_cuda: bool = False) -> int:
    info = inspect_llama_cpp_runtime()
    ok = info.cuda_ready if require_cuda else info.installed
    payload = info.to_dict() | {"ok": ok, "required_cuda": require_cuda}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if info.installed:
            print(f"llama-cpp-python: {info.version}")
            print(
                "llama.cpp GPU offload: yes"
                if info.supports_gpu_offload
                else "llama.cpp GPU offload: no"
            )
        else:
            print("llama-cpp-python: not installed")
        if info.error:
            print(f"error: {info.error}")
    return 0 if ok else 1


def command_llama_link_cuda_libs(*, as_json: bool = False) -> int:
    info = link_llama_cpp_cuda_libraries()
    payload = info.to_dict() | {"ok": info.ok}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if info.llama_lib_dir:
            print(f"llama lib dir: {info.llama_lib_dir}")
        print(f"linked cuda libs: {len(info.linked)}")
        print(f"copied cuda libs: {len(info.copied)}")
        print(f"existing cuda libs: {len(info.skipped)}")
        if info.error:
            print(f"error: {info.error}")
    return 0 if info.ok else 1


def command_models(
    action: str | None,
    model: str | None,
    cache_dir: str | None,
    as_json: bool,
    confirm: bool = False,
) -> int:
    manager = ModelCacheManager(
        cache_dir=Path(cache_dir).expanduser()
        if cache_dir
        else default_model_cache_dir(PROJECT_ROOT)
    )
    try:
        if action is None or action == "status":
            statuses = manager.list_statuses()
            payload = {
                "cache_dir": str(manager.cache_dir),
                "models": [status.to_dict() for status in statuses],
            }
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"Кеш моделей: {manager.cache_dir}")
                for status in statuses:
                    marker = "OK" if status.status == "ready" else "--"
                    print(f"[{marker}] {status.alias}: {status.message}")
            return 0

        if model is None:
            print("models: model alias is required", file=sys.stderr)
            return 1

        if action == "download":
            result = manager.download(model)
        elif action == "verify":
            result = manager.verify(model)
        elif action == "repair":
            result = manager.repair(model)
        elif action == "update":
            result = manager.update(model, confirm=confirm)
        elif action == "remove":
            result = manager.remove(model, confirm=confirm)
        else:
            print(f"models: unknown action {action}", file=sys.stderr)
            return 2
    except (KeyError, ModelCacheLockError, ValueError) as exc:
        print(f"models: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        marker = "OK" if result.ok else "ERR"
        print(f"[{marker}] {result.model.alias}: {result.message}")
        if result.model.issues:
            for issue in result.model.issues:
                print(f"- {issue}")
    return 0 if result.ok else 1


def command_setup_local(
    *,
    profile: str,
    download: bool,
    install_deps: bool,
    skip_paddle_install: bool,
    as_json: bool,
    cache_dir: str | None,
) -> int:
    dependency_install = None
    if install_deps:
        dependency_install = install_local_dependencies(
            project_root=PROJECT_ROOT, include_paddle=not skip_paddle_install
        )
        if not dependency_install.ok:
            payload = {
                "ok": False,
                "install_deps": install_deps,
                "dependency_install": dependency_install.to_dict(),
            }
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                reason = dependency_install.skipped_reason or dependency_install.stderr
                print(f"setup-local: dependency install failed: {reason}", file=sys.stderr)
            return 1

    hardware = collect_hardware_report()
    manager = ModelCacheManager(
        cache_dir=Path(cache_dir).expanduser()
        if cache_dir
        else default_model_cache_dir(PROJECT_ROOT)
    )
    backend_name, model_suffix, warning = select_local_backend(hardware)
    ai_client = local_ai_client_for_suffix(model_suffix)
    qwen_model = setup_local_profile_model(profile, model_suffix)
    model_keys = ["paddleocr_vl", qwen_model]

    try:
        statuses = [manager.status(key) for key in model_keys]
        missing = [status for status in statuses if status.status != "ready"]
        needed_disk_gb = sum(manager.manifest.resolve(status.key).disk_gb for status in missing)
    except (KeyError, ModelCacheLockError, ValueError) as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"setup-local: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, object] = {
        "ok": not missing,
        "profile": profile,
        "backend": backend_name,
        "ai_client": ai_client,
        "warning": warning,
        "download": download,
        "install_deps": install_deps,
        "dependency_install": dependency_install.to_dict() if dependency_install else None,
        "cache_dir": str(manager.cache_dir),
        "needed_disk_gb": needed_disk_gb,
        "models": [status.to_dict() for status in statuses],
    }

    if not download:
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if dependency_install is not None:
                print(
                    "[OK] Установлены локальные зависимости: "
                    f".[{','.join(dependency_install.extras)}]"
                )
            print_setup_local_status(
                profile=profile,
                backend_name=backend_name,
                ai_client=ai_client,
                warning=warning,
                statuses=statuses,
                needed_disk_gb=needed_disk_gb,
                cache_dir=manager.cache_dir,
            )
        return 0 if not missing else 1

    results = []
    all_ok = True
    if not as_json:
        if dependency_install is not None:
            print(
                f"[OK] Установлены локальные зависимости: .[{','.join(dependency_install.extras)}]"
            )
        print(f"Оборудование: {hardware.os}, {hardware.machine}. Выбран бекенд: {backend_name}")
        if warning:
            print(f"Предупреждение: {warning}")
        print(f"Профиль: {profile}")
        print(f"Локальный AI-клиент: {ai_client}")
        print(f"Кеш моделей: {manager.cache_dir}")
        print(f"Загрузка недостающих моделей (примерно {needed_disk_gb:.1f} ГБ)...")
    for status in statuses:
        result = manager.download(status.key)
        results.append(result.to_dict())
        all_ok = all_ok and result.ok
        if not as_json:
            marker = "OK" if result.ok else "ERR"
            print(f"[{marker}] {result.model.title}: {result.message}")
            for issue in result.model.issues:
                print(f"  - {issue}")

    payload["download_results"] = results
    payload["ok"] = all_ok
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif all_ok:
        print(f"Настройка профиля {profile} завершена успешно.")
    else:
        print("Возникли ошибки при скачивании моделей.", file=sys.stderr)
    return 0 if all_ok else 1


def select_local_backend(hardware) -> tuple[str, str, str | None]:
    if "Darwin" in hardware.os and hardware.mlx_available:
        return "MLX", "_mlx", None
    if hardware.nvidia_smi:
        return "CUDA/WSL2", "_cuda", None
    if hardware.mlx_available:
        return "MLX", "_mlx", None
    return (
        "CUDA/WSL2",
        "_cuda",
        "NVIDIA GPU не найдена. Модели можно скачать, но для запуска потребуется CUDA/WSL2 или другой backend.",
    )


def setup_local_profile_model(profile: str, model_suffix: str) -> str:
    if profile == "hq":
        return f"qwen3_14b{model_suffix}"
    if profile in {"qwen36-safe", "qwen36-balanced", "qwen36-quality"}:
        if model_suffix == "_mlx":
            return "qwen3_6_35b_a3b_mlx_vlm"
        if profile == "qwen36-safe":
            return "qwen3_6_27b_gguf_iq3_xxs"
        if profile == "qwen36-balanced":
            return "qwen3_6_27b_gguf_q4_k_s"
        return "qwen3_6_27b_gguf_q4_k_m"
    if profile == "qwen36-max":
        if model_suffix == "_mlx":
            return "qwen3_6_35b_a3b_mlx_vlm"
        return "qwen3_6_27b_gptq_cuda"
    return f"qwen3_8b{model_suffix}"


def local_ai_client_for_suffix(model_suffix: str) -> str:
    return "qwen-mlx" if model_suffix == "_mlx" else "openai-compatible"


def print_setup_local_status(
    *,
    profile: str,
    backend_name: str,
    ai_client: str,
    warning: str | None,
    statuses,
    needed_disk_gb: float,
    cache_dir: Path,
) -> None:
    print(f"Выбран локальный бекенд: {backend_name}")
    if warning:
        print(f"Предупреждение: {warning}")
    print(f"Профиль: {profile}")
    print(f"Локальный AI-клиент: {ai_client}")
    print(f"Кеш моделей: {cache_dir}")
    print("Статус локальных моделей:")
    missing_count = 0
    for status in statuses:
        marker = "OK" if status.status == "ready" else "--"
        if status.status != "ready":
            missing_count += 1
        print(f"[{marker}] {status.title}: {status.message}")
    if missing_count == 0:
        print("Все модели готовы к работе.")
        return
    print(f"Итог: не хватает {missing_count} модели(ей).")
    print(f"Для загрузки потребуется примерно {needed_disk_gb:.1f} ГБ.")
    print("Чтобы скачать недостающие модели, запустите:")
    print(f"  spec-ocr setup-local --profile {profile} --download")


def command_setup_wsl(
    *,
    profile: str,
    distro: str,
    project_dir: str | None,
    venv: str,
    cache_dir: str,
    host: str,
    port: int,
    served_model_name: str | None,
    backend: str,
    runtime: str,
    skip_runtime_install: bool,
    skip_paddle_install: bool,
    gpt_auth_file: str | None,
    skip_gpt_smoke: bool,
    execute: bool,
    confirm: bool,
    as_json: bool,
) -> int:
    config = WslInstallConfig(
        distro=distro,
        profile=profile,
        project_path=Path(project_dir).expanduser() if project_dir else Path.cwd(),
        venv_path=Path(venv),
        cache_dir=Path(cache_dir),
        host=host,
        port=port,
        served_model_name=served_model_name,
        backend=backend,
        runtime=runtime,
        install_runtime=not skip_runtime_install,
        install_paddle=not skip_paddle_install,
        gpt_auth_file=Path(gpt_auth_file).expanduser() if gpt_auth_file else None,
        gpt_smoke_test=not skip_gpt_smoke,
        execute=execute,
        confirm=confirm,
    )
    try:
        plan = build_wsl_install_plan(config)
    except ValueError as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"setup-wsl: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_wsl_install_plan(plan)
    if not plan.ok:
        for error in plan.errors:
            print(f"setup-wsl: {error}", file=sys.stderr)
        return 2 if execute and not confirm else 1
    if not execute:
        return 0
    return execute_install_plan(plan)


def command_setup_windows(
    *,
    profile: str | None,
    distro: str,
    project_dir: str | None,
    venv: str,
    cache_dir: str,
    host: str,
    port: int,
    yes: bool,
    skip_runtime_install: bool,
    skip_paddle_install: bool,
    gpt_auth_file: str | None,
    skip_gpt_smoke: bool,
    as_json: bool,
) -> int:
    selected_profile = profile or select_windows_setup_profile(
        interactive=(not as_json and sys.stdin.isatty())
    )
    if selected_profile is None:
        print("setup-windows: установка отменена", file=sys.stderr)
        return 1

    execute = yes
    if not as_json:
        print_windows_setup_summary(selected_profile)
        if not yes:
            if sys.stdin.isatty():
                answer = input("Запустить установку сейчас? [y/N]: ").strip().lower()
                execute = answer in {"y", "yes", "д", "да"}
            else:
                print(
                    "Интерактивная консоль недоступна. Показываю dry-run; для установки добавьте --yes."
                )
            if not execute:
                print("Установка не запущена. Ниже dry-run план и команды для проверки.")

    return command_setup_wsl(
        profile=selected_profile,
        distro=distro,
        project_dir=project_dir,
        venv=venv,
        cache_dir=cache_dir,
        host=host,
        port=port,
        served_model_name=None,
        backend="auto",
        runtime="auto",
        skip_runtime_install=skip_runtime_install,
        skip_paddle_install=skip_paddle_install,
        gpt_auth_file=gpt_auth_file,
        skip_gpt_smoke=skip_gpt_smoke,
        execute=execute,
        confirm=execute,
        as_json=as_json,
    )


def select_windows_setup_profile(*, interactive: bool) -> str | None:
    if not interactive:
        return WINDOWS_SETUP_PROFILES[0][0]
    print("Настройка Windows/WSL2 для локального ИИ")
    print("Выберите модель. Если не уверены, нажмите Enter.")
    for index, (profile, title, description) in enumerate(WINDOWS_SETUP_PROFILES, start=1):
        default_marker = " (рекомендуется)" if index == 1 else ""
        print(f"{index}. {title}{default_marker}")
        print(f"   {profile}: {description}")
    for _ in range(3):
        answer = input("Номер модели [1]: ").strip()
        if not answer:
            return WINDOWS_SETUP_PROFILES[0][0]
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(WINDOWS_SETUP_PROFILES):
                return WINDOWS_SETUP_PROFILES[index - 1][0]
        for profile, _title, _description in WINDOWS_SETUP_PROFILES:
            if answer == profile:
                return profile
        print("Не понял выбор. Введите номер из списка или нажмите Enter для рекомендуемой модели.")
    return None


def print_windows_setup_summary(profile: str) -> None:
    profile_info = next(
        (item for item in WINDOWS_SETUP_PROFILES if item[0] == profile),
        None,
    )
    if profile_info is None:
        print(f"Выбран профиль: {profile}")
        return
    profile_name, title, description = profile_info
    print(f"Выбран профиль: {profile_name} - {title}")
    print(description)
    if profile_name == "qwen36-max":
        print("Важно: это экспериментальный vLLM/GPTQ вариант. Для 16GB VRAM сначала лучше Safe.")


def print_wsl_install_plan(plan) -> None:
    mode = "установка" if plan.execute else "dry-run"
    print(f"План настройки WSL ({mode})")
    print(f"Профиль: {plan.profile}")
    print(f"Дистрибутив: {plan.distro}")
    print(f"Проект: {plan.project_path}")
    print(f"Python-окружение: {plan.venv_path}")
    print(f"Модель: {plan.model}")
    print(f"Имя модели в API: {plan.served_model_name}")
    print(f"Адрес API: {plan.base_url}")
    if plan.warnings:
        print("Предупреждения:")
        for warning in plan.warnings:
            print(f"- {warning}")
    if plan.errors:
        print("Ошибки:")
        for error in plan.errors:
            print(f"- {error}")
    print("Шаги:")
    for index, step in enumerate(plan.steps, start=1):
        mutation = "изменяет" if step.mutates else "проверка"
        print(f"{index}. [{step.shell}/{mutation}] {step.title}")
        print(f"   {step.command}")
    print("Команды после настройки:")
    for command in plan.next_commands:
        print(f"  {command}")


def command_export_gpt_auth(*, source: str, output: str, confirm: bool, as_json: bool) -> int:
    source_path = Path(source).expanduser()
    output_path = Path(output).expanduser()
    if not confirm:
        payload = {
            "ok": False,
            "error": (
                "Нужно явно передать --confirm для приватного переноса credential. "
                "Не коммитьте, не публикуйте и не передавайте экспортированный файл."
            ),
            "source": str(source_path),
            "output": str(output_path),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["error"], file=sys.stderr)
        return 2
    if not source_path.is_file():
        payload = {
            "ok": False,
            "error": f"OpenCode auth.json не найден: {source_path}",
            "source": str(source_path),
            "output": str(output_path),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["error"], file=sys.stderr)
        return 1
    try:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        payload = {
            "ok": False,
            "error": f"OpenCode auth.json не является JSON: {exc}",
            "source": str(source_path),
            "output": str(output_path),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["error"], file=sys.stderr)
        return 1
    if not isinstance(source_payload, dict):
        payload = {
            "ok": False,
            "error": "OpenCode auth.json должен быть JSON object.",
            "source": str(source_path),
            "output": str(output_path),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["error"], file=sys.stderr)
        return 1
    openai_key = (
        "openai" if "openai" in source_payload else "OpenAI" if "OpenAI" in source_payload else None
    )
    if openai_key is None:
        payload = {
            "ok": False,
            "error": "В OpenCode auth.json не найден OpenAI credential.",
            "source": str(source_path),
            "output": str(output_path),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["error"], file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_payload = {openai_key: source_payload[openai_key]}
    output_path.write_text(
        json.dumps(exported_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path.chmod(0o600)
    payload = {
        "ok": True,
        "source": str(source_path),
        "output": str(output_path),
        "message": (
            "Ваш OpenAI credential экспортирован для приватного переноса. "
            "Не коммитьте, не публикуйте и не передавайте этот файл."
        ),
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["message"])
        print(f"Файл: {output_path}")
    return 0


def command_serve_local(
    *,
    model: str,
    cache_dir: str | None,
    backend: str,
    host: str,
    port: int,
    served_model_name: str | None,
    dry_run: bool,
    as_json: bool,
    extra_args: list[str],
) -> int:
    resolved_cache_dir = (
        Path(cache_dir).expanduser() if cache_dir else default_model_cache_dir(PROJECT_ROOT)
    )
    manager = ModelCacheManager(cache_dir=resolved_cache_dir)
    try:
        command = build_local_server_command(
            LocalServerLaunchConfig(
                model=model,
                cache_dir=resolved_cache_dir,
                backend=backend,
                host=host,
                port=port,
                served_model_name=served_model_name,
                extra_args=tuple(extra_args),
            ),
            manager=manager,
        )
    except (KeyError, ModelCacheLockError, ModelNotReadyError, ValueError) as exc:
        hint = f"spec-pipeline models download {model}"
        if as_json:
            print(
                json.dumps(
                    {"ok": False, "error": str(exc), "hint": hint},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"serve-local: локальная модель недоступна: {exc}", file=sys.stderr)
            if isinstance(exc, ModelNotReadyError):
                print(f"Подсказка: выполните {hint}", file=sys.stderr)
        return 1

    payload = {"ok": True, "dry_run": dry_run, **command.to_dict()}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Backend: {command.backend}")
        print(f"Model path: {command.model_path}")
        print(f"Served model: {command.served_model_name}")
        print(f"Base URL: {command.base_url}")
        print("Offline env:")
        for name, value in command.env.items():
            print(f"  {name}={value}")
        print("Run:")
        print(f"  {shlex.join(command.args)}")
    if dry_run:
        return 0
    return launch_local_server(command)


def command_samples(manifest_path: str, as_json: bool, validate: bool) -> int:
    fixture_root = PROJECT_ROOT / "tests/fixtures"
    manifest = load_sample_manifest(manifest_path)
    payload = {
        "input_exists": (fixture_root / "input").exists(),
        "expected_exists": (fixture_root / "expected").exists(),
        "input_files": sorted(
            path.name for path in (fixture_root / "input").glob("**/*") if path.is_file()
        ),
        "expected_files": sorted(
            path.name for path in (fixture_root / "expected").glob("**/*") if path.is_file()
        ),
        "manifest": summarize_manifest(manifest, validate_paths=validate),
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"input_exists: {payload['input_exists']}")
        print(f"expected_exists: {payload['expected_exists']}")
        print(f"input_files: {len(payload['input_files'])}")
        print(f"expected_files: {len(payload['expected_files'])}")
        print(f"manifest_samples: {payload['manifest']['sample_count']}")

    return 0


def command_rows(manifest_path: str, sample_id: str, limit: int, as_json: bool) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    rows = load_ground_truth_rows(sample)[:limit]

    if as_json:
        print(json.dumps([row.to_dict() for row in rows], ensure_ascii=False, indent=2))
    else:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}] {row.to_dict()}")

    return 0


def command_normalize(manifest_path: str, sample_id: str, output: str | None) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    rows = load_ground_truth_rows(sample)

    if output == "-":
        for row in rows:
            sys.stdout.write(json.dumps(row.to_dict(), ensure_ascii=False))
            sys.stdout.write("\n")
        return 0

    written_path, row_count = normalize_sample_to_jsonl(
        sample=sample,
        project_root=PROJECT_ROOT,
        output_path=output,
    )
    if output is None:
        print(
            f"normalize: wrote {row_count} rows to {default_output_path(PROJECT_ROOT, sample_id)}"
        )
    else:
        print(f"normalize: wrote {row_count} rows to {written_path}")
    return 0


def command_export_xlsx(
    manifest_path: str,
    sample_id: str,
    input_jsonl: str | None,
    output: str | None,
) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    written_path, row_count = export_sample_to_xlsx(
        sample=sample,
        project_root=PROJECT_ROOT,
        output_path=output,
        normalized_input_path=input_jsonl,
    )
    if output is None:
        print(
            f"export-xlsx: wrote {row_count} rows to {default_xlsx_output_path(PROJECT_ROOT, sample_id)}"
        )
    else:
        print(f"export-xlsx: wrote {row_count} rows to {written_path}")
    return 0


def command_export_jsonl_xlsx(input_jsonl: str, output: str) -> int:
    written_path, row_count = export_jsonl_to_xlsx(input_jsonl_path=input_jsonl, output_path=output)
    print(f"export-jsonl-xlsx: wrote {row_count} rows to {written_path}")
    return 0


def command_export_kp_xlsx(
    input_jsonl: str, output: str, template: str | None, sheet: str | None
) -> int:
    template_path = resolve_kp_template_path(template, output)
    written_path, row_count = export_jsonl_to_kp_xlsx(
        input_jsonl_path=input_jsonl,
        output_path=output,
        template_path=template_path,
        sheet_name=sheet,
    )
    print(f"export-kp-xlsx: wrote {row_count} item rows to {written_path}")
    return 0


def resolve_kp_template_path(template: str | None, output: str) -> Path:
    if template:
        return Path(template).expanduser()

    default_path = default_kp_template_path(PROJECT_ROOT)
    candidates = [PACKAGE_DATA_DIR / ASCII_KP_TEMPLATE_NAME, default_path]

    output_path = Path(output).expanduser()
    template_names = (ASCII_KP_TEMPLATE_NAME, default_path.name)
    for template_name in template_names:
        for parent in (output_path.parent, *output_path.parents):
            candidates.append(parent / "examples" / template_name)

    cwd = Path.cwd()
    for template_name in template_names:
        candidates.extend(
            [
                cwd / "examples" / template_name,
                cwd.parent / "examples" / template_name,
                PROJECT_ROOT.parent / "examples" / template_name,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return default_path


def command_process(
    pdf_path: str,
    output_dir: str | None,
    pages: str | None,
    dpi: int,
    quote: bool,
    force: bool,
) -> int:
    source_pdf = Path(pdf_path).expanduser()
    if not source_pdf.is_file():
        print(f"process: PDF not found: {source_pdf}", file=sys.stderr)
        return 2
    if source_pdf.suffix.lower() != ".pdf":
        print(f"process: expected a .pdf file: {source_pdf}", file=sys.stderr)
        return 2

    target_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path.cwd() / f"{source_pdf.stem}-spec-ocr"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    ocr_path = target_dir / "ocr.json"
    rows_path = target_dir / "rows.jsonl"
    xlsx_path = target_dir / "result.xlsx"
    quote_path = target_dir / "quote.xlsx"
    metadata_path = target_dir / "run.json"
    outputs = [ocr_path, rows_path, xlsx_path, metadata_path]
    if quote:
        outputs.append(quote_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        print(
            f"process: output already exists; use --force to replace it: {joined}", file=sys.stderr
        )
        return 2

    extract_args = [
        "extract-ocr",
        str(source_pdf),
        "--provider",
        "pdf-text",
        "--output",
        str(ocr_path),
        "--dpi",
        str(dpi),
    ]
    if pages is not None:
        extract_args.extend(["--pages", pages])
    extract_exit = main(extract_args)
    if extract_exit != 0:
        return extract_exit

    ocr_pages = load_ocr_pages_json(ocr_path)
    table_row_count = sum(len(parse_pdf_table_layer_rows(page.text)) for page in ocr_pages)
    if table_row_count == 0:
        print(
            "process: no supported table layer was found. This safe one-command path does "
            "not invoke external AI. See docs/providers.md for scanned-PDF workflows.",
            file=sys.stderr,
        )
        return 2

    sample_id = re.sub(r"[^A-Za-z0-9._-]+", "-", source_pdf.stem).strip("-") or "document"
    map_exit = main(
        [
            "map-ocr",
            sample_id,
            "--input-ocr",
            str(ocr_path),
            "--source-pdf",
            str(source_pdf),
            "--output",
            str(rows_path),
            "--no-image-attachments",
        ]
    )
    if map_exit != 0:
        return map_exit
    export_exit = main(["export-jsonl-xlsx", str(rows_path), "--output", str(xlsx_path)])
    if export_exit != 0:
        return export_exit
    if quote:
        quote_exit = main(["export-kp-xlsx", str(rows_path), "--output", str(quote_path)])
        if quote_exit != 0:
            return quote_exit

    metadata = {
        "version": __version__,
        "source_pdf": str(source_pdf),
        "output_dir": str(target_dir),
        "pages": pages,
        "table_row_count": table_row_count,
        "external_ai_used": False,
        "files": {
            "ocr": ocr_path.name,
            "rows": rows_path.name,
            "xlsx": xlsx_path.name,
            "quote": quote_path.name if quote else None,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"process: complete; outputs are in {target_dir}")
    return 0


def command_compare(
    manifest_path: str,
    sample_id: str,
    predicted_jsonl: str,
    output_json: str | None,
) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    result = compare_sample_against_jsonl(sample=sample, predicted_jsonl_path=predicted_jsonl)

    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if output_json is not None:
        write_compare_summary(result, output_json)

    is_clean = result.row_delta == 0 and result.exact_match_rate == 1.0
    return 0 if is_clean else 1


def command_parse_pdf(
    manifest_path: str,
    sample_id: str,
    backend: str,
    output: str | None,
) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    parser = PARSER_BACKENDS[backend]()
    written_path, row_count = parse_sample_to_jsonl(
        parser=parser,
        sample=sample,
        project_root=PROJECT_ROOT,
        output_path=output,
    )
    if output is None:
        print(
            f"parse-pdf: wrote {row_count} rows to {default_parsed_output_path(PROJECT_ROOT, sample_id)}"
        )
    else:
        print(f"parse-pdf: wrote {row_count} rows to {written_path}")
    return 0


def command_diagnose_pdf(
    manifest_path: str,
    sample_id: str,
    backend: str,
    limit: int,
    output_json: str | None,
) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    parser = PARSER_BACKENDS[backend]()
    diagnostics = parser.diagnose(sample, limit=limit)
    payload = diagnostics.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target_path = output_json or str(
        default_diagnostics_output_path(PROJECT_ROOT, sample_id, backend)
    )
    write_diagnostics(diagnostics, target_path)
    return 0


def command_probe_pdf(manifest_path: str, sample_id: str) -> int:
    manifest = load_sample_manifest(manifest_path)
    sample = manifest.get_sample(sample_id)
    probe = probe_pdf_text_layer(sample)
    print(json.dumps(probe.to_dict(), ensure_ascii=False, indent=2))
    return 0


def command_extract_ocr(
    pdf_path: str,
    provider_name: str,
    pages: str | None,
    output: str | None,
    dpi: int,
    lang: str,
    psm: int,
    pdf_text_overlay: bool | None,
    paddle_model: str,
    paddle_device: str | None,
    paddle_pipeline_version: str,
    paddle_vl_backend: str | None,
    paddle_vl_server_url: str | None,
    paddle_vl_api_model: str | None,
    paddle_max_new_tokens: int | None,
    paddle_min_pixels: int | None,
    paddle_max_pixels: int | None,
    paddle_repetition_penalty: float | None,
    paddle_temperature: float | None,
    paddle_top_p: float | None,
    paddle_split_wide_pages: bool | None,
    paddle_band_height: int,
    paddle_band_overlap: int,
    paddle_predict_timeout: int | None,
    paddle_predict_heartbeat: int,
    paddle_use_layout_detection: bool | None,
    paddle_use_queues: bool | None,
    paddle_engine: str | None,
    paddle_cpu_threads: int | None,
    paddle_require_cuda: bool,
    paddle_enable_hpi: bool | None,
    paddle_use_tensorrt: bool | None,
    paddle_precision: str | None,
    paddle_vl_max_concurrency: int | None,
    dots_model: str,
    dots_backend: str,
    dots_device: str | None,
    dots_device_map: str | None,
    dots_torch_dtype: str,
    dots_attn_implementation: str | None,
    dots_trust_model_code: bool,
    dots_upscale: float,
    dots_max_image_edge: int,
    dots_max_tokens: int,
    dots_no_crop: bool,
    model_cache_dir: str | None,
) -> int:
    path = Path(pdf_path).expanduser()
    page_selection = parse_page_selection(pages)
    if provider_name == "auto":
        text_provider = PdfTextOcrProvider(render_dpi=dpi)
        text_pages = text_provider.extract_pages(path, pages=page_selection, output_dir=None)
        pdf_table_row_count = _pdf_table_layer_row_count(text_pages)
        print(f"extract-ocr: auto route PDF_TABLE_ROW count={pdf_table_row_count}")
        if pdf_table_row_count >= 5:
            output_path = (
                Path(output).expanduser()
                if output
                else default_ocr_output_path(PROJECT_ROOT, path, text_provider.provider_name)
            )
            written_path = write_ocr_pages_json(text_pages, output_path)
            print(
                "extract-ocr: PDF_TABLE_LAYER found; skipping PaddleOCR-VL for this searchable PDF"
            )
            print(f"extract-ocr: wrote {len(text_pages)} pages to {written_path}")
            return 0
        print("extract-ocr: no usable PDF_TABLE_LAYER; falling back to PaddleOCR-VL")
        provider_name = "paddleocr-vl"
    if provider_name == "pdf-text":
        provider = PdfTextOcrProvider(render_dpi=dpi)
    elif provider_name == "dots-mocr":
        provider = DotsMocrProvider(
            config=DotsMocrConfig(
                model_name=dots_model,
                backend=dots_backend,
                model_cache_dir=Path(model_cache_dir).expanduser()
                if model_cache_dir is not None
                else None,
                device=dots_device,
                device_map=dots_device_map,
                torch_dtype=dots_torch_dtype,
                attn_implementation=dots_attn_implementation,
                trust_model_code=dots_trust_model_code,
                dpi=dpi,
                upscale=dots_upscale,
                max_image_edge=dots_max_image_edge,
                max_tokens=dots_max_tokens,
                crop_to_content=not dots_no_crop,
                progress=True,
            )
        )
    elif provider_name == "paddleocr-vl":
        try:
            resolved_paddle_model = resolve_paddleocr_vl_model(paddle_model, model_cache_dir)
        except (ModelNotReadyError, KeyError, ModelCacheLockError, ValueError) as exc:
            print(f"extract-ocr: локальная PaddleOCR-VL модель недоступна: {exc}", file=sys.stderr)
            print(
                "Подсказка: выполните spec-pipeline models download paddleocr-vl", file=sys.stderr
            )
            return 1
        provider = PaddleOcrVlProvider(
            config=PaddleOcrVlConfig(
                model_name=resolved_paddle_model,
                dpi=dpi,
                device=paddle_device,
                pipeline_version=paddle_pipeline_version,
                vl_rec_backend=paddle_vl_backend,
                vl_rec_server_url=paddle_vl_server_url,
                vl_rec_api_model_name=paddle_vl_api_model,
                vl_rec_max_concurrency=paddle_vl_max_concurrency,
                engine=paddle_engine,
                cpu_threads=paddle_cpu_threads,
                require_cuda=paddle_require_cuda,
                enable_hpi=paddle_enable_hpi,
                use_tensorrt=paddle_use_tensorrt,
                precision=paddle_precision,
                use_layout_detection=paddle_use_layout_detection,
                use_queues=paddle_use_queues,
                min_pixels=paddle_min_pixels,
                max_pixels=paddle_max_pixels,
                max_new_tokens=paddle_max_new_tokens,
                repetition_penalty=paddle_repetition_penalty,
                temperature=paddle_temperature,
                top_p=paddle_top_p,
                split_wide_pages=bool(paddle_split_wide_pages),
                split_band_height=paddle_band_height,
                split_band_overlap=paddle_band_overlap,
                predict_timeout_seconds=paddle_predict_timeout,
                predict_heartbeat_seconds=paddle_predict_heartbeat,
                progress=True,
            )
        )
    else:
        provider = TesseractOcrProvider(
            config=TesseractConfig(language=lang, page_segmentation_mode=psm, dpi=dpi)
        )
    page_dir = default_ocr_page_dir(PROJECT_ROOT, path, provider.provider_name)
    provider_output_dir = None if provider_name == "pdf-text" else page_dir
    extracted = provider.extract_pages(path, pages=page_selection, output_dir=provider_output_dir)
    if pdf_text_overlay and provider_name != "pdf-text":
        extracted = append_pdf_text_overlay(extracted, path, page_selection)
    output_path = (
        Path(output).expanduser()
        if output
        else default_ocr_output_path(PROJECT_ROOT, path, provider.provider_name)
    )
    written_path = write_ocr_pages_json(extracted, output_path)
    print(f"extract-ocr: wrote {len(extracted)} pages to {written_path}")
    return 0


def _has_usable_pdf_table_layer(pages: list[OcrPage]) -> bool:
    return _pdf_table_layer_row_count(pages) >= 5


def _pdf_table_layer_row_count(pages: list[OcrPage]) -> int:
    return sum(len(parse_pdf_table_layer_rows(page.text)) for page in pages)


def append_pdf_text_overlay(
    pages: list[OcrPage], pdf_path: Path, page_selection: list[int] | None
) -> list[OcrPage]:
    overlay_pages = PdfTextOcrProvider().extract_pages(pdf_path, pages=page_selection)
    overlays_by_ref = {page.page_ref: page for page in overlay_pages if page.text.strip()}
    if not overlays_by_ref:
        return pages
    merged: list[OcrPage] = []
    for page in pages:
        overlay = overlays_by_ref.get(page.page_ref)
        if overlay is None:
            merged.append(page)
            continue
        merged.append(
            OcrPage(
                page_index=page.page_index,
                page_ref=page.page_ref,
                width=page.width,
                height=page.height,
                text=f"{page.text}\n\nPDF_TEXT_LAYER:\n{compact_pdf_text_overlay(overlay.text)}"
                if page.text
                else f"PDF_TEXT_LAYER:\n{compact_pdf_text_overlay(overlay.text)}",
                tokens=[*page.tokens, *overlay.tokens],
                provider=page.provider,
                image_path=page.image_path,
            )
        )
    return merged


def resolve_paddleocr_vl_model(paddle_model: str, model_cache_dir: str | None) -> str:
    candidate = Path(paddle_model).expanduser()
    if candidate.exists():
        return str(candidate)
    manager = ModelCacheManager(
        cache_dir=Path(model_cache_dir).expanduser()
        if model_cache_dir
        else default_model_cache_dir(PROJECT_ROOT)
    )
    if paddle_model == "PaddlePaddle/PaddleOCR-VL":
        return str(manager.resolve_for_run("paddleocr-vl"))
    return str(manager.resolve_for_run(paddle_model))


def command_map_ocr(
    sample_id: str,
    input_ocr: str,
    source_pdf: str,
    output: str | None,
    quality_json: str | None,
    quality_fail_on: str,
    performance_json: str | None,
    ai_client: str,
    ai_model: str,
    ai_agent: str,
    ai_timeout: int,
    ai_max_tokens: int | None,
    ai_page_batch_size: int,
    ai_context_tokens: int,
    ai_context_margin: int,
    ai_base_url: str,
    ai_served_model: str | None,
    ai_external_openai_compatible: bool,
    ai_external_supports_images: bool,
    ai_api_key_env: str | None,
    ai_reasoning_effort: str | None,
    third_layer_reviewer: str,
    third_layer_model: str | None,
    third_layer_agent: str,
    third_layer_base_url: str | None,
    third_layer_served_model: str | None,
    third_layer_api_key_env: str | None,
    third_layer_reasoning_effort: str | None,
    third_layer_timeout: int,
    third_layer_max_tokens: int,
    no_third_layer_image_attachments: bool,
    no_image_attachments: bool,
    ai_force_on_pdf_table_layer: bool,
    model_cache_dir: str | None,
) -> int:
    started = time.perf_counter()
    pages = load_ocr_pages_json(input_ocr)
    deterministic_direct_available = (
        not ai_force_on_pdf_table_layer
        and _should_use_pdf_table_direct(pdf_table_row_count=_pdf_table_layer_row_count(pages))
    )
    if ai_client == "none" and not deterministic_direct_available:
        print(
            "map-ocr: no supported deterministic table layer was found and AI is disabled. "
            "Choose an explicit local or external --ai-client after reviewing docs/privacy.md.",
            file=sys.stderr,
        )
        return 2
    requested_local_model: str | None = None
    effective_model = ai_model
    effective_base_url = (
        "none" if ai_client == "none" else ("opencode" if ai_client == "opencode" else ai_base_url)
    )
    effective_served_model = ai_served_model
    if ai_client == "none":
        effective_max_tokens = 0
        client = None
    elif ai_client == "opencode":
        effective_max_tokens = 0
        client = OpenCodeExtractionClient(
            config=OpenCodeClientConfig(
                model=ai_model,
                agent=ai_agent,
                variant=ai_reasoning_effort,
                timeout_seconds=ai_timeout,
                attach_images=not no_image_attachments,
                cache_dir=PROJECT_ROOT / "artifacts" / "ai" / "opencode_requests",
                cwd=PROJECT_ROOT,
            )
        )
    elif ai_client == "qwen-mlx":
        effective_max_tokens = ai_max_tokens if ai_max_tokens is not None else 8192
        local_model = "qwen3-8b" if ai_model == "openai/gpt-5.5" else ai_model
        requested_local_model = local_model
        effective_model = local_model
        effective_base_url = "local-mlx"
        client = QwenMlxExtractionClient(
            config=QwenMlxClientConfig(
                model=local_model,
                attach_images=not no_image_attachments,
                max_tokens=effective_max_tokens,
                project_root=PROJECT_ROOT,
                model_cache_dir=Path(model_cache_dir).expanduser()
                if model_cache_dir is not None
                else None,
            )
        )
    elif ai_client == "openai-compatible":
        effective_max_tokens = ai_max_tokens if ai_max_tokens is not None else 8192
        local_model = (
            ai_model
            if ai_external_openai_compatible
            else ("qwen3.6-safe" if ai_model == "openai/gpt-5.5" else ai_model)
        )
        requested_local_model = None if ai_external_openai_compatible else local_model
        effective_model = local_model
        effective_base_url = ai_base_url
        effective_served_model = ai_served_model or (
            local_model
            if ai_external_openai_compatible
            else _default_local_served_model(local_model, "openai-compatible")
        )
        client = OpenAiCompatibleExtractionClient(
            config=OpenAiCompatibleClientConfig(
                model=local_model,
                base_url=ai_base_url,
                served_model_name=effective_served_model,
                require_local_model=not ai_external_openai_compatible,
                api_key_env=ai_api_key_env,
                reasoning_effort=ai_reasoning_effort,
                external_supports_images=ai_external_supports_images,
                max_tokens=effective_max_tokens,
                timeout_seconds=ai_timeout,
                attach_images=not no_image_attachments,
                project_root=PROJECT_ROOT,
                model_cache_dir=Path(model_cache_dir).expanduser()
                if model_cache_dir is not None
                else None,
            )
        )
    else:
        raise ValueError(f"Unknown AI client: {ai_client}")
    third_layer_client = _build_third_layer_reviewer(
        reviewer=third_layer_reviewer,
        model=third_layer_model,
        agent=third_layer_agent,
        base_url=third_layer_base_url,
        served_model=third_layer_served_model,
        api_key_env=third_layer_api_key_env,
        reasoning_effort=third_layer_reasoning_effort,
        timeout_seconds=third_layer_timeout,
        max_tokens=third_layer_max_tokens,
        attach_images=not no_third_layer_image_attachments,
    )
    try:
        result = _extract_rows_with_page_batches(
            client=client,
            third_layer_client=third_layer_client,
            third_layer_name=third_layer_reviewer,
            sample_id=sample_id,
            source_pdf=Path(source_pdf).expanduser(),
            pages=pages,
            page_batch_size=ai_page_batch_size,
            context_tokens=ai_context_tokens,
            max_tokens=effective_max_tokens,
            context_margin=ai_context_margin,
            force_ai_on_pdf_table_layer=ai_force_on_pdf_table_layer,
        )
        validate_ai_rows(result.rows)
    except (ModelNotReadyError, RuntimeError, ValueError, KeyError) as exc:
        if ai_client == "opencode":
            message = str(exc)
            print(f"map-ocr: OpenCode/GPT extraction failed: {message}", file=sys.stderr)
            lower_message = message.lower()
            if any(
                marker in lower_message
                for marker in (
                    "forbidden",
                    "gateway",
                    "proxy",
                    "permission",
                    "provider settings",
                    "access this resource",
                )
            ):
                print(
                    "Подсказка: GPT/OpenCode дошел до провайдера, но запрос отклонен. "
                    "Это обычно означает блокировку OpenAI/OpenCode сетью/VPN/proxy "
                    "или аккаунт без доступа к выбранной модели. Проверьте доступ из WSL "
                    "и выполните `opencode auth login` от имени текущего пользователя WSL.",
                    file=sys.stderr,
                )
            elif any(
                marker in lower_message
                for marker in (
                    "multi-auth",
                    "no available accounts",
                    "invalidated",
                    "refresh failed",
                    "401",
                    "auth",
                )
            ):
                print(
                    "Подсказка: выполните `opencode auth login` внутри WSL и войдите "
                    "в собственный OpenCode/OpenAI аккаунт.",
                    file=sys.stderr,
                )
            elif "no json object" in lower_message or "requested json" in lower_message:
                print(
                    "Подсказка: OpenCode не вернул JSON со строками. "
                    "Смотрите logs/02-ai-map.log в архиве результата.",
                    file=sys.stderr,
                )
            elif "timed out" in lower_message or "timeout" in lower_message:
                print(
                    "Подсказка: страница не успела обработаться даже после повторных "
                    "попыток со сниженным reasoning. Проверьте стабильность интернета "
                    "и запустите обработку этого PDF еще раз.",
                    file=sys.stderr,
                )
            return 1
        if ai_client in {"qwen-mlx", "openai-compatible"}:
            if ai_client == "openai-compatible" and ai_external_openai_compatible:
                print(
                    f"map-ocr: облачная OpenAI-compatible модель недоступна: {exc}", file=sys.stderr
                )
                if ai_api_key_env:
                    print(
                        f"Подсказка: проверьте, что переменная окружения {ai_api_key_env} задана в WSL.",
                        file=sys.stderr,
                    )
                return 1
            print(f"map-ocr: локальная модель недоступна: {exc}", file=sys.stderr)
            if isinstance(exc, ModelNotReadyError):
                print(
                    "Подсказка: выполните "
                    f"spec-pipeline models download {requested_local_model or 'qwen3-8b'}",
                    file=sys.stderr,
                )
            elif isinstance(exc, ValueError) and "not an MLX backend" in str(exc):
                print(
                    "Подсказка: для --ai-client qwen-mlx используйте MLX-модель, "
                    "например qwen3-8b, qwen3-14b или qwen3.6-35b-a3b",
                    file=sys.stderr,
                )
            elif isinstance(exc, ValueError) and "OpenAI-compatible" in str(exc):
                print(
                    "Подсказка: для --ai-client openai-compatible используйте CUDA-модель, "
                    "например qwen3-8b-cuda, qwen3-14b-cuda или qwen3.6-safe",
                    file=sys.stderr,
                )
            elif (
                ai_client == "openai-compatible"
                and isinstance(exc, RuntimeError)
                and "timed out" in str(exc).lower()
            ):
                print(
                    "Подсказка: локальный Qwen работает, но ответ занял слишком много времени. "
                    "Увеличьте --ai-timeout или уменьшите размер пакета страниц.",
                    file=sys.stderr,
                )
            elif ai_client == "openai-compatible" and isinstance(exc, RuntimeError):
                print(
                    "Подсказка: запустите локальный vLLM/SGLang/llama.cpp сервер "
                    "и передайте его адрес через --ai-base-url.",
                    file=sys.stderr,
                )
            return 1
        raise
    rows = [ai_row_to_specification_row(sample_id, row) for row in result.rows]
    output_path = (
        Path(output).expanduser()
        if output is not None
        else PROJECT_ROOT / "artifacts" / "parsed" / f"{sample_id}.ai.jsonl"
    )
    write_parsed_rows_jsonl([cast(dict[str, object], row.to_dict()) for row in rows], output_path)
    quality = assess_extracted_rows(rows)
    runtime_sec = time.perf_counter() - started
    quality_path: Path | None = None
    if quality_json is not None:
        quality_path = Path(quality_json).expanduser()
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(
            json.dumps(quality.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    performance_path: Path | None = None
    extraction_mode = _map_ocr_extraction_mode(result.raw_response)
    deterministic_direct = extraction_mode == "pdf_table_layer_direct"
    performance = _map_ocr_performance_summary(
        sample_id=sample_id,
        ai_client="none" if deterministic_direct else ai_client,
        ai_model=None if deterministic_direct else effective_model,
        ai_agent=ai_agent if ai_client == "opencode" and not deterministic_direct else None,
        ai_base_url=None if deterministic_direct else effective_base_url,
        ai_served_model=None if deterministic_direct else effective_served_model,
        extraction_mode=extraction_mode,
        page_count=len(pages),
        row_count=len(rows),
        runtime_sec=runtime_sec,
        quality_errors=quality.error_count,
        quality_warnings=quality.warning_count,
        input_ocr=Path(input_ocr).expanduser(),
        source_pdf=Path(source_pdf).expanduser(),
        output_path=output_path,
        quality_path=quality_path,
    )
    if performance_json is not None:
        performance_path = Path(performance_json).expanduser()
        performance_path.parent.mkdir(parents=True, exist_ok=True)
        performance["performance_json"] = str(performance_path)
        performance_path.write_text(
            json.dumps(performance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"map-ocr: wrote {len(rows)} rows to {output_path}")
    print(f"map-ocr quality: {quality.error_count} errors, {quality.warning_count} warnings")
    _print_map_ocr_performance(performance, performance_path=performance_path)
    if _quality_gate_failed(quality, fail_on=quality_fail_on):
        print(
            "map-ocr quality gate failed: "
            f"{quality.error_count} errors, {quality.warning_count} warnings "
            f"(--quality-fail-on {quality_fail_on})",
            file=sys.stderr,
        )
        return 2
    return 0


def _quality_gate_failed(quality, *, fail_on: str) -> bool:
    if fail_on == "none":
        return False
    if fail_on == "error":
        return quality.error_count > 0
    if fail_on == "uncorrected-warning":
        if quality.error_count > 0:
            return True
        return any(
            issue.severity == "warning" and not _is_export_normalized_quality_warning(issue)
            for issue in quality.issues
        )
    if fail_on == "warning":
        return quality.issue_count > 0
    raise ValueError(f"Unknown quality gate mode: {fail_on}")


def _is_export_normalized_quality_warning(issue) -> bool:
    return (issue.field, issue.message) in {
        ("unit", "unit looks OCR-damaged"),
        ("name", "dimension text looks OCR-damaged"),
        ("type_mark", "dimension text looks OCR-damaged"),
    }


def _build_third_layer_reviewer(
    *,
    reviewer: str,
    model: str | None,
    agent: str,
    base_url: str | None,
    served_model: str | None,
    api_key_env: str | None,
    reasoning_effort: str | None,
    timeout_seconds: int,
    max_tokens: int,
    attach_images: bool,
):
    if reviewer == "none":
        return None
    if reviewer == "opencode-gpt":
        return OpenCodeExtractionClient(
            config=OpenCodeClientConfig(
                model=model or "openai/gpt-5.5",
                agent=agent,
                variant=reasoning_effort,
                timeout_seconds=timeout_seconds,
                attach_images=attach_images,
                cache_dir=PROJECT_ROOT / "artifacts" / "ai" / "opencode_requests",
                cwd=PROJECT_ROOT,
            )
        )
    if reviewer in {"kimi-fireworks", "openai-compatible"}:
        if reviewer == "kimi-fireworks":
            resolved_model = model or "accounts/fireworks/routers/kimi-k2p6-turbo"
            resolved_base_url = base_url or "https://api.fireworks.ai/inference/v1"
            resolved_api_key_env = api_key_env or "FIREWORKS_API_KEY"
            resolved_reasoning_effort = reasoning_effort or "none"
        else:
            if model is None:
                raise ValueError("--third-layer-model is required for openai-compatible reviewer")
            resolved_model = model
            resolved_base_url = base_url or "http://127.0.0.1:8000/v1"
            resolved_api_key_env = api_key_env
            resolved_reasoning_effort = reasoning_effort
        return OpenAiCompatibleExtractionClient(
            config=OpenAiCompatibleClientConfig(
                model=resolved_model,
                base_url=resolved_base_url,
                served_model_name=served_model or resolved_model,
                require_local_model=False,
                api_key_env=resolved_api_key_env,
                reasoning_effort=resolved_reasoning_effort,
                external_supports_images=True,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                attach_images=attach_images,
                project_root=PROJECT_ROOT,
            )
        )
    raise ValueError(f"Unknown third-layer reviewer: {reviewer}")


def _extract_rows_with_page_batches(
    *,
    client,
    third_layer_client,
    third_layer_name: str,
    sample_id: str,
    source_pdf: Path,
    pages: list[OcrPage],
    page_batch_size: int,
    context_tokens: int,
    max_tokens: int,
    context_margin: int,
    force_ai_on_pdf_table_layer: bool,
) -> AiExtractionResult:
    pdf_table_rows = _ai_rows_from_pdf_table_layer(pages)
    if not force_ai_on_pdf_table_layer and _should_use_pdf_table_direct(
        pdf_table_row_count=len(pdf_table_rows)
    ):
        print("map-ocr: PDF_TABLE_LAYER found; using deterministic rows without AI")
        validate_ai_rows(pdf_table_rows)
        warnings = ["Used deterministic PDF_TABLE_LAYER rows for searchable PDF"]
        raw_response = {
            "fallback": "pdf_table_layer_direct",
            "pdf_table_row_count": len(pdf_table_rows),
            "page_batch_size": page_batch_size,
            "context_tokens": context_tokens,
            "max_tokens": max_tokens,
            "context_margin": context_margin,
        }
        rows, repair_raw = _apply_third_layer_review_if_needed(
            reviewer=third_layer_client,
            reviewer_name=third_layer_name,
            sample_id=sample_id,
            source_pdf=source_pdf,
            pages=pages,
            rows=pdf_table_rows,
            warnings=warnings,
        )
        if repair_raw is not None:
            raw_response["third_layer_review"] = repair_raw
        return AiExtractionResult(
            rows=rows,
            warnings=warnings,
            raw_response=raw_response,
        )
    batches = _plan_ai_page_batches(
        sample_id=sample_id,
        pages=pages,
        page_batch_size=page_batch_size,
        context_tokens=context_tokens,
        max_tokens=max_tokens,
        context_margin=context_margin,
    )
    print(f"map-ocr: AI batching: {len(batches)} request(s)")
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    raw_batches: list[dict[str, object]] = []
    for batch in batches:
        page_refs = [page.page_ref for page in batch]
        result = _extract_ai_batch_with_context_retry(
            client=client,
            sample_id=sample_id,
            source_pdf=source_pdf,
            pages=batch,
        )
        validate_ai_rows(result.rows)
        rows.extend(result.rows)
        warnings.extend(f"pages {','.join(page_refs)}: {warning}" for warning in result.warnings)
        raw_batches.append(
            {
                "page_refs": page_refs,
                "row_count": len(result.rows),
                "warnings": result.warnings,
                "raw_response": result.raw_response,
            }
        )
    if _should_use_pdf_table_fallback(
        ai_row_count=len(rows), pdf_table_row_count=len(pdf_table_rows)
    ):
        warnings.append(
            "AI returned too few rows for a searchable PDF; used deterministic PDF_TABLE_LAYER rows"
        )
        validate_ai_rows(pdf_table_rows)
        return AiExtractionResult(
            rows=pdf_table_rows,
            warnings=warnings,
            raw_response={
                "batches": raw_batches,
                "fallback": "pdf_table_layer",
                "ai_row_count": len(rows),
                "pdf_table_row_count": len(pdf_table_rows),
                "page_batch_size": page_batch_size,
                "context_tokens": context_tokens,
                "max_tokens": max_tokens,
                "context_margin": context_margin,
            },
        )
    rows, repair_raw = _apply_third_layer_review_if_needed(
        reviewer=third_layer_client,
        reviewer_name=third_layer_name,
        sample_id=sample_id,
        source_pdf=source_pdf,
        pages=pages,
        rows=rows,
        warnings=warnings,
    )
    if repair_raw is not None:
        raw_batches.append(repair_raw)
    return AiExtractionResult(
        rows=rows,
        warnings=warnings,
        raw_response={
            "batches": raw_batches,
            "page_batch_size": page_batch_size,
            "context_tokens": context_tokens,
            "max_tokens": max_tokens,
            "context_margin": context_margin,
        },
    )


def _should_use_pdf_table_fallback(*, ai_row_count: int, pdf_table_row_count: int) -> bool:
    return pdf_table_row_count >= 5 and ai_row_count < max(3, int(pdf_table_row_count * 0.5))


def _should_use_pdf_table_direct(*, pdf_table_row_count: int) -> bool:
    return pdf_table_row_count >= 5


def _apply_third_layer_review_if_needed(
    *,
    reviewer,
    reviewer_name: str,
    sample_id: str,
    source_pdf: Path,
    pages: list[OcrPage],
    rows: list[dict[str, object]],
    warnings: list[str],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if reviewer is None:
        return rows, None

    plan = detect_vision_repair_plan(rows)
    if not plan.needed:
        return rows, None

    repair_pages = existing_image_pages(pages, set(plan.page_refs))
    if not repair_pages:
        warnings.append("third-layer review needed but no rendered scan page images are available")
        return rows, {"purpose": "third_layer_review", "skipped": "no_images"}

    request = AiExtractionRequest(sample_id=sample_id, source_pdf=source_pdf, pages=repair_pages)
    review = getattr(reviewer, "review_row_patches", None)
    if not callable(review):
        warnings.append("third-layer reviewer cannot return row patches")
        return rows, {
            "purpose": "third_layer_review",
            "skipped": "patch_reviewer_unavailable",
            "reviewer": reviewer_name,
        }

    page_refs = {page.page_ref for page in repair_pages}
    try:
        result = review(
            request=request,
            rows=rows,
            issues=[issue for issue in plan.issues if issue.page_ref in page_refs],
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        warnings.append(f"third-layer review failed: {exc}")
        return rows, {
            "purpose": "third_layer_review",
            "skipped": "review_failed",
            "reviewer": reviewer_name,
            "error": str(exc),
            "page_refs": sorted(page_refs),
        }

    result_patches = getattr(result, "patches", [])
    result_warnings = getattr(result, "warnings", [])
    result_raw_response = getattr(result, "raw_response", {})
    patched_rows, patch_warnings, accepted, rejected = apply_row_patches(
        rows, result_patches, page_refs
    )
    warnings.extend(patch_warnings)
    validate_ai_rows(patched_rows)
    return patched_rows, {
        "purpose": "third_layer_review",
        "reviewer": reviewer_name,
        "page_refs": sorted(page_refs),
        "patch_count": len(result_patches),
        "accepted_patch_count": len(accepted),
        "rejected_patch_count": len(rejected),
        "warnings": result_warnings,
        "patch_warnings": patch_warnings,
        "accepted_patches": accepted,
        "rejected_patches": rejected,
        "raw_response": result_raw_response,
        "issues": [issue.to_dict() for issue in plan.issues if issue.page_ref in page_refs],
    }


def _ai_rows_from_pdf_table_layer(pages: list[OcrPage]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_section: str | None = None
    current_subsection: str | None = None
    pending_context_name: str | None = None
    raw_rows: list[dict[str, object]] = []
    for page in pages:
        raw_rows.extend(parse_pdf_table_layer_rows(page.text))
    raw_rows = _repair_pdf_table_raw_rows(raw_rows)

    for index, raw_row in enumerate(raw_rows):
        if _is_pdf_table_group_row(raw_row):
            name = _string_or_none(raw_row.get("name"))
            if name is None:
                continue
            if _is_pdf_table_lowercase_continuation_header(name):
                previous = _last_pdf_table_data_row(rows)
                if previous is not None:
                    previous["name"] = _join_pdf_table_text(
                        _string_or_none(previous.get("name")), name
                    )
                pending_context_name = None
                continue
            if _is_pdf_table_context_header(raw_rows, index):
                if _can_use_pdf_table_context_header(name):
                    pending_context_name = name
                    continue
            if current_section is None or _is_top_level_pdf_section(name):
                current_section = name
                current_subsection = None
                rows.append(_pdf_table_grouping_ai_row(raw_row, current_section, None))
            else:
                current_subsection = name
                rows.append(
                    _pdf_table_grouping_ai_row(raw_row, current_section, current_subsection)
                )
            pending_context_name = None
            continue

        normalized = _normalize_pdf_table_ai_row(
            raw_row,
            current_section=current_section,
            current_subsection=current_subsection,
            pending_context_name=pending_context_name,
        )
        if normalized is not None:
            rows.append(normalized)
            if pending_context_name is not None:
                pending_context_name = None
    return rows


def _repair_pdf_table_raw_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    repaired: list[dict[str, object]] = []
    for raw_row in rows:
        row = dict(raw_row)
        _repair_unit_embedded_quantity(row)
        _repair_pdf_table_mass_as_quantity(row)
        _repair_pdf_table_pos_only_group_row(row)
        if _is_pdf_table_footer_noise_row(row):
            continue
        if repaired and _is_pdf_table_continuation_without_quantity(repaired[-1], row):
            _merge_pdf_table_continuation_into_row(row, repaired[-1])
            continue
        if repaired and _is_pdf_table_type_mark_tail_row(row):
            previous = repaired[-1]
            previous_type_mark = _string_or_none(previous.get("type_mark"))
            tail_type_mark = _string_or_none(row.get("type_mark"))
            if previous_type_mark and tail_type_mark:
                separator = "" if previous_type_mark.endswith(("-", "_", "/")) else " "
                previous["type_mark"] = f"{previous_type_mark}{separator}{tail_type_mark}"
                continue
        if repaired and _is_pdf_table_tail_row(row):
            previous = repaired[-1]
            tail_quantity = _string_or_none(row.get("quantity"))
            tail_note = _string_or_none(row.get("note"))
            if tail_quantity and _can_attach_pdf_quantity_tail(previous, tail_quantity):
                previous_quantity = _string_or_none(previous.get("quantity"))
                previous["quantity"] = f"{previous_quantity or ''}{tail_quantity}"
                tail_mass = _string_or_none(row.get("mass_kg"))
                if tail_mass:
                    previous["mass_kg"] = (
                        f"{_string_or_none(previous.get('mass_kg')) or ''}{tail_mass}"
                    )
                if tail_note:
                    previous["note"] = " ".join(
                        part for part in (_string_or_none(previous.get("note")), tail_note) if part
                    )
                continue
            if tail_note and _can_attach_pdf_note_tail(previous, row):
                previous["note"] = _join_pdf_table_text(
                    _string_or_none(previous.get("note")), tail_note
                )
                continue
            if (
                tail_quantity
                and _string_or_none(previous.get("quantity")) is None
                and _has_pdf_table_item_text(previous)
            ):
                previous["quantity"] = tail_quantity
                continue
        if repaired:
            _repair_pdf_table_ditto_note(row, repaired)
        repaired.append(row)
    return _merge_pdf_table_suffix_rows(_merge_pdf_table_prefix_rows(repaired))


def _repair_pdf_table_mass_as_quantity(row: dict[str, object]) -> None:
    if _string_or_none(row.get("quantity")) is not None:
        return
    mass_kg = _string_or_none(row.get("mass_kg"))
    if mass_kg is None or not _looks_like_pdf_quantity_text(mass_kg):
        return
    if not any(_string_or_none(row.get(field)) for field in ("unit", "pos", "name")):
        return
    row["quantity"] = mass_kg
    row["mass_kg"] = None


def _repair_pdf_table_pos_only_group_row(row: dict[str, object]) -> None:
    pos = _string_or_none(row.get("pos"))
    if pos is None or row.get("name") is not None:
        return
    if any(
        _string_or_none(row.get(field))
        for field in ("type_mark", "code", "manufacturer", "unit", "quantity", "mass_kg", "note")
    ):
        return
    if looks_like_shifted_item_name(pos):
        row["name"] = pos
        row["pos"] = None


PDF_TABLE_DITTO_NOTES = {"-//-", "//", "- // -"}


def _repair_pdf_table_ditto_note(
    row: dict[str, object], previous_rows: list[dict[str, object]]
) -> None:
    note = _string_or_none(row.get("note"))
    if note not in PDF_TABLE_DITTO_NOTES:
        return
    page_ref = _string_or_none(row.get("page_ref"))
    for previous in reversed(previous_rows):
        if _string_or_none(previous.get("page_ref")) != page_ref:
            break
        previous_note = _string_or_none(previous.get("note"))
        if previous_note and previous_note not in PDF_TABLE_DITTO_NOTES:
            row["note"] = previous_note
            return


def _merge_pdf_table_prefix_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        if index + 1 < len(rows) and _is_pdf_table_numbered_section_row(row, rows[index + 1]):
            row["pos"] = None
            merged.append(row)
            index += 1
            continue
        if index + 1 < len(rows) and _is_pdf_table_prefix_row(row, rows[index + 1]):
            next_row = dict(rows[index + 1])
            _merge_pdf_table_prefix_into_row(row, next_row)
            merged.append(next_row)
            index += 2
            continue
        merged.append(row)
        index += 1
    return merged


def _is_pdf_table_numbered_section_row(row: dict[str, object], next_row: dict[str, object]) -> bool:
    if _string_or_none(row.get("page_ref")) != _string_or_none(next_row.get("page_ref")):
        return False
    pos = _string_or_none(row.get("pos"))
    if pos is None or re.fullmatch(r"\d+\*?", pos) is None:
        return False
    if _string_or_none(row.get("name")) is None:
        return False
    if any(
        _string_or_none(row.get(field))
        for field in ("type_mark", "code", "manufacturer", "unit", "quantity", "mass_kg", "note")
    ):
        return False
    next_text = _pdf_table_full_item_text(next_row)
    if next_text is None:
        return False
    next_pos, _next_name = _split_leading_pdf_position(next_text)
    return bool(next_pos and next_pos.startswith(f"{pos}."))


def _is_pdf_table_prefix_row(row: dict[str, object], next_row: dict[str, object]) -> bool:
    if _string_or_none(row.get("page_ref")) != _string_or_none(next_row.get("page_ref")):
        return False
    if any(_string_or_none(row.get(field)) for field in ("unit", "quantity", "mass_kg")):
        return False
    prefix_text = _pdf_table_full_item_text(row)
    next_text = _pdf_table_full_item_text(next_row)
    if prefix_text is None or next_text is None:
        return False
    if "в составе" in prefix_text.lower():
        return False
    if not any(_string_or_none(next_row.get(field)) for field in ("unit", "quantity")):
        return False
    if any(
        _string_or_none(row.get(field)) for field in ("type_mark", "code", "manufacturer", "note")
    ):
        return True
    if _has_pdf_leading_position(prefix_text):
        return True
    normalized = prefix_text.lower()
    return any(marker in normalized for marker in ("радиатор", "плита", "коллектор"))


def _merge_pdf_table_prefix_into_row(
    prefix_row: dict[str, object], target_row: dict[str, object]
) -> None:
    prefix_text = _pdf_table_full_item_text(prefix_row)
    target_text = _pdf_table_full_item_text(target_row)
    if prefix_text is None:
        return

    prefix_pos, prefix_name = _split_leading_pdf_position(prefix_text)
    if prefix_pos is not None:
        target_row["pos"] = prefix_pos
        target_row["name"] = _join_pdf_table_text(prefix_name, target_text)
    else:
        split_pos, split_name = _split_pdf_leading_row_number(target_text or "")
        if split_pos is None:
            split_pos, split_name = _split_shifted_pdf_position_name(target_text or "")
        if split_pos is not None:
            target_row["pos"] = split_pos
            target_row["name"] = _join_pdf_table_text(prefix_text, split_name)
        else:
            target_row["pos"] = None
            target_row["name"] = _join_pdf_table_text(prefix_text, target_text)

    target_row["type_mark"] = _join_pdf_table_code_text(
        _string_or_none(prefix_row.get("type_mark")), _string_or_none(target_row.get("type_mark"))
    )
    for field in ("code", "manufacturer", "note"):
        target_row[field] = _join_pdf_table_text(
            _string_or_none(prefix_row.get(field)), _string_or_none(target_row.get(field))
        )


def _is_pdf_table_continuation_without_quantity(
    previous: dict[str, object], row: dict[str, object]
) -> bool:
    if _string_or_none(previous.get("page_ref")) != _string_or_none(row.get("page_ref")):
        return False
    if any(_string_or_none(previous.get(field)) for field in ("unit", "quantity", "mass_kg")):
        return False
    if any(_string_or_none(row.get(field)) for field in ("unit", "quantity", "mass_kg")):
        return False
    previous_text = _pdf_table_row_text(previous)
    text = _pdf_table_row_text(row)
    if previous_text is None or text is None:
        return False
    stripped = text.strip()
    previous_lower = previous_text.lower()
    text_lower = stripped.lower()
    has_open_parenthesis = previous_text.count("(") > previous_text.count(")")
    if LEADING_PDF_POSITION_RE.match(stripped):
        return False
    if PDF_LEADING_ROW_NUMBER_RE.match(stripped) and not (
        has_open_parenthesis or previous_lower.endswith((":", ";"))
    ):
        return False
    return (
        has_open_parenthesis
        or previous_lower.endswith((":", ";"))
        or previous_lower.endswith("в1:")
        or ("установка" in previous_lower and "в составе" in text_lower)
    )


def _merge_pdf_table_continuation_into_row(
    continuation_row: dict[str, object], target_row: dict[str, object]
) -> None:
    continuation_text = _pdf_table_row_text(continuation_row)
    if continuation_text is not None:
        if _string_or_none(target_row.get("name")) is not None:
            target_row["name"] = _join_pdf_table_text(
                _string_or_none(target_row.get("name")), continuation_text
            )
        else:
            target_row["pos"] = _join_pdf_table_text(
                _string_or_none(target_row.get("pos")), continuation_text
            )

    target_row["type_mark"] = _join_pdf_table_code_text(
        _string_or_none(target_row.get("type_mark")),
        _string_or_none(continuation_row.get("type_mark")),
    )
    for field in ("code", "manufacturer", "note"):
        target_row[field] = _join_pdf_table_text(
            _string_or_none(target_row.get(field)),
            _string_or_none(continuation_row.get(field)),
        )


def _is_pdf_table_footer_noise_row(row: dict[str, object]) -> bool:
    text = " ".join(
        value
        for value in (
            _string_or_none(row.get("pos")),
            _string_or_none(row.get("name")),
            _string_or_none(row.get("type_mark")),
            _string_or_none(row.get("code")),
            _string_or_none(row.get("manufacturer")),
            _string_or_none(row.get("unit")),
            _string_or_none(row.get("quantity")),
            _string_or_none(row.get("mass_kg")),
            _string_or_none(row.get("note")),
        )
        if value
    ).lower()
    return any(
        marker in text
        for marker in (
            "проектом допускается",
            "замена указанного в спецификации",
            "по своим характеристикам оборудование",
            "оборудование и материалы других производителей",
            "других производителей",
            "строительство жилого дома по адресу",
            "строительство жилых домов",
            "при монтаже количество труб",
            "отопления может отличатся",
            "проектного значения",
            "многоквартирные жилые дома",
            "ii этап строительства",
        )
    )
    stamp_markers = ("изм.", "кол.уч", "№док", "подп", "стадия лист", "листов", "гип")
    return sum(1 for marker in stamp_markers if marker in text) >= 2


def _pdf_table_row_text(row: dict[str, object]) -> str | None:
    return _string_or_none(row.get("name")) or _string_or_none(row.get("pos"))


def _pdf_table_full_item_text(row: dict[str, object]) -> str | None:
    pos = _string_or_none(row.get("pos"))
    name = _string_or_none(row.get("name"))
    if pos is not None and name is not None:
        return _join_pdf_table_text(pos, name)
    return name or pos


def _merge_pdf_table_suffix_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for row in rows:
        if merged and _is_pdf_table_suffix_row(merged[-1], row):
            _merge_pdf_table_suffix_into_row(row, merged[-1])
            continue
        merged.append(dict(row))
    return merged


def _is_pdf_table_suffix_row(previous: dict[str, object], row: dict[str, object]) -> bool:
    if _string_or_none(previous.get("page_ref")) != _string_or_none(row.get("page_ref")):
        return False
    if not any(_string_or_none(previous.get(field)) for field in ("unit", "quantity")):
        return False
    if any(_string_or_none(row.get(field)) for field in ("unit", "quantity", "mass_kg")):
        return False
    text = _pdf_table_row_text(row)
    if text is None or len(text) > 90:
        return False
    first = text[:1]
    return bool(first and first.islower())


def _merge_pdf_table_suffix_into_row(
    suffix_row: dict[str, object], target_row: dict[str, object]
) -> None:
    suffix_text = _pdf_table_row_text(suffix_row)
    target_name = _string_or_none(target_row.get("name"))
    target_pos = _string_or_none(target_row.get("pos"))
    if target_name is None and target_pos is not None:
        split_pos, split_name = _split_shifted_pdf_position_name(target_pos)
        if split_pos is not None:
            target_row["pos"] = split_pos
            target_name = split_name
    target_row["name"] = _join_pdf_table_text(target_name, suffix_text)
    target_row["type_mark"] = _join_pdf_table_code_text(
        _string_or_none(target_row.get("type_mark")), _string_or_none(suffix_row.get("type_mark"))
    )
    for field in ("code", "manufacturer", "note"):
        target_row[field] = _join_pdf_table_text(
            _string_or_none(target_row.get(field)), _string_or_none(suffix_row.get(field))
        )


UNIT_EMBEDDED_QUANTITY_RE = re.compile(
    r"^(?P<unit>.*?[^\d\s])\s+(?P<quantity>\d+(?:[,.]\d+)?)\s*/\s*(?P<mass>\d+(?:[,.]\d*)?)$"
)


def _repair_unit_embedded_quantity(row: dict[str, object]) -> None:
    unit = _string_or_none(row.get("unit"))
    if unit is None:
        return
    match = UNIT_EMBEDDED_QUANTITY_RE.match(unit)
    if not match:
        return
    quantity = match.group("quantity")
    mass = match.group("mass")
    tail = _string_or_none(row.get("quantity"))
    if tail and _looks_like_pdf_numeric_tail(tail):
        mass = f"{mass}{tail}"
        row["quantity"] = None
    elif row.get("quantity") is not None:
        return
    row["unit"] = match.group("unit")
    row["quantity"] = quantity
    row["mass_kg"] = mass


def _is_pdf_table_tail_row(row: dict[str, object]) -> bool:
    quantity = _string_or_none(row.get("quantity"))
    mass_kg = _string_or_none(row.get("mass_kg"))
    note = _string_or_none(row.get("note"))
    if not any([quantity, mass_kg, note]):
        return False
    return not any(
        _string_or_none(row.get(field))
        for field in ("pos", "name", "type_mark", "code", "manufacturer", "unit")
    )


def _can_attach_pdf_note_tail(previous: dict[str, object], row: dict[str, object]) -> bool:
    if _string_or_none(previous.get("page_ref")) != _string_or_none(row.get("page_ref")):
        return False
    if _string_or_none(previous.get("note")) is None:
        return False
    return _has_pdf_table_item_text(previous)


def _is_pdf_table_type_mark_tail_row(row: dict[str, object]) -> bool:
    type_mark = _string_or_none(row.get("type_mark"))
    if type_mark is None:
        return False
    return not any(
        _string_or_none(row.get(field))
        for field in ("pos", "name", "code", "manufacturer", "unit", "quantity", "mass_kg", "note")
    )


def _can_attach_pdf_quantity_tail(previous: dict[str, object], tail: str) -> bool:
    if not _looks_like_pdf_numeric_tail(tail):
        return False
    previous_quantity = _string_or_none(previous.get("quantity"))
    if previous_quantity is None:
        return _has_pdf_table_item_text(previous)
    return "/" in previous_quantity or previous_quantity.endswith((",", "/"))


def _looks_like_pdf_numeric_tail(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:[,.]\d{1,3})?|[,.]\d{1,3}", value.strip()))


def _has_pdf_table_item_text(row: dict[str, object]) -> bool:
    return any(
        _string_or_none(row.get(field))
        for field in ("name", "type_mark", "code", "manufacturer", "unit")
    )


def _normalize_pdf_table_ai_row(
    row: dict[str, object],
    *,
    current_section: str | None,
    current_subsection: str | None,
    pending_context_name: str | None = None,
) -> dict[str, object] | None:
    name = _string_or_none(row.get("name"))
    pos = _string_or_none(row.get("pos"))
    unit = _string_or_none(row.get("unit"))
    quantity = _string_or_none(row.get("quantity"))
    type_mark = _string_or_none(row.get("type_mark"))
    code = _string_or_none(row.get("code"))
    manufacturer = _string_or_none(row.get("manufacturer"))
    note = _string_or_none(row.get("note"))
    mass_kg = _string_or_none(row.get("mass_kg"))
    if name is None and pos is not None:
        split_pos, split_name = _split_shifted_pdf_position_name(pos)
        if split_pos is not None:
            pos = split_pos
            name = split_name
        elif looks_like_shifted_item_name(pos):
            name = pos
            pos = None
        elif _should_use_pdf_pos_as_name(
            pos=pos,
            unit=unit,
            quantity=quantity,
            type_mark=type_mark,
            code=code,
            manufacturer=manufacturer,
            mass_kg=mass_kg,
        ):
            name = pos
            pos = None
    if name is not None and pos is not None:
        split_pos, split_name = _split_leading_pdf_position(pos)
        if split_pos is None:
            split_pos, split_name = _split_shifted_pdf_position_name(pos)
        if split_pos is not None and split_name is not None:
            pos = split_pos
            name = _join_pdf_table_text(split_name, name)
    if pos is None and name is not None:
        extracted_pos, extracted_name = _split_leading_pdf_position(name)
        if extracted_pos is not None:
            pos = extracted_pos
            name = extracted_name
        elif any([unit, quantity, type_mark, code, manufacturer, mass_kg]):
            extracted_pos, extracted_name = _split_pdf_leading_item_number(name)
            if extracted_pos is not None:
                pos = extracted_pos
                name = extracted_name
    if name is not None:
        name = _repair_pdf_table_reordered_name(name)
    if name is None and _is_pdf_table_noise_without_name(
        pos=pos,
        unit=unit,
        quantity=quantity,
        type_mark=type_mark,
        code=code,
        manufacturer=manufacturer,
        mass_kg=mass_kg,
        note=note,
    ):
        return None
    if name and pos and not any([unit, quantity, type_mark, code, manufacturer, mass_kg, note]):
        if _is_pdf_table_positioned_header_row(pos=pos, name=name):
            return {
                "page_ref": _string_or_none(row.get("page_ref")),
                "section": current_section,
                "subsection": current_subsection,
                "pos": pos,
                "name": name,
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": None,
                "quantity": None,
                "mass_kg": None,
                "note": None,
                "status": _string_or_none(row.get("status")),
            }
        return None
    if name and not any([pos, unit, quantity, type_mark, code, manufacturer, mass_kg, note]):
        return None
    if not any([pos, name, unit, quantity, type_mark, code, manufacturer, mass_kg]):
        return None
    if pending_context_name and name and pos is None and not LEADING_PDF_POSITION_RE.match(name):
        name = f"{pending_context_name} {name}"
    return {
        "page_ref": _string_or_none(row.get("page_ref")),
        "section": current_section,
        "subsection": current_subsection,
        "pos": pos,
        "name": name,
        "type_mark": type_mark,
        "code": code,
        "manufacturer": manufacturer,
        "unit": unit,
        "quantity": quantity,
        "mass_kg": mass_kg,
        "note": note,
        "status": _string_or_none(row.get("status")),
    }


LEADING_PDF_POSITION_RE = re.compile(r"^(?P<pos>\d+(?:\.\d+)+)\s+(?P<name>.+)$")
SHIFTED_PDF_POSITION_NAME_RE = re.compile(
    r"^(?P<pos>(?:[A-Za-zА-Яа-я]{1,4}\d+(?:\.\d+)?|\d+\*?))\s+(?P<name>[A-ZА-ЯЁ].+)$"
)
PDF_LEADING_ROW_NUMBER_RE = re.compile(r"^(?P<pos>\d+\*?)\s+(?P<name>.+)$")
PDF_LEADING_ITEM_NUMBER_RE = re.compile(r"^(?P<pos>\d{1,2}\*?)\s+(?P<name>.+)$")
PDF_TABLE_REORDER_MARKERS = (
    "Вентилятор ",
    "Клапан ",
    "Кожух ",
    "Забивка ",
    "Наружная решетка ",
    "Переход ",
)
PDF_TABLE_SYSTEM_POSITION_RE = re.compile(
    r"^(?=.*[A-Za-zА-Яа-яЁё])(?=.*\d)[A-Za-zА-Яа-яЁё\d]+(?:\.\d+)*\*?$"
)


def _is_pdf_table_positioned_header_row(*, pos: str, name: str) -> bool:
    normalized_name = re.sub(r"\s+", " ", name).strip().lower()
    if "в составе" in normalized_name:
        return True
    return (
        normalized_name.endswith(":")
        and PDF_TABLE_SYSTEM_POSITION_RE.fullmatch(pos.strip()) is not None
    )


def _split_leading_pdf_position(value: str) -> tuple[str | None, str | None]:
    match = LEADING_PDF_POSITION_RE.match(value.strip())
    if not match:
        return None, value
    return match.group("pos"), match.group("name").strip() or None


def _split_shifted_pdf_position_name(value: str) -> tuple[str | None, str | None]:
    match = SHIFTED_PDF_POSITION_NAME_RE.match(value.strip())
    if not match:
        return None, value
    return match.group("pos"), match.group("name").strip() or None


def _split_pdf_leading_row_number(value: str) -> tuple[str | None, str | None]:
    match = PDF_LEADING_ROW_NUMBER_RE.match(value.strip())
    if not match:
        return None, value
    return match.group("pos"), match.group("name").strip() or None


def _split_pdf_leading_item_number(value: str) -> tuple[str | None, str | None]:
    match = PDF_LEADING_ITEM_NUMBER_RE.match(value.strip())
    if not match:
        return None, value
    return match.group("pos"), match.group("name").strip() or None


def _has_pdf_leading_position(value: str) -> bool:
    return (
        LEADING_PDF_POSITION_RE.match(value.strip()) is not None
        or SHIFTED_PDF_POSITION_NAME_RE.match(value.strip()) is not None
    )


def _should_use_pdf_pos_as_name(
    *,
    pos: str,
    unit: str | None,
    quantity: str | None,
    type_mark: str | None,
    code: str | None,
    manufacturer: str | None,
    mass_kg: str | None,
) -> bool:
    if not any([unit, quantity, type_mark, code, manufacturer, mass_kg]):
        return False
    if re.fullmatch(r"\d+\*?", pos):
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]", pos))


def _repair_pdf_table_reordered_name(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    for marker in PDF_TABLE_REORDER_MARKERS:
        marker_index = text.find(marker)
        if marker_index <= 0:
            continue
        prefix = text[:marker_index].strip()
        body = text[marker_index:].strip()
        if not prefix or not body:
            continue
        normalized_prefix = prefix.lower()
        prefix_starts_as_tail = (
            prefix[:1].islower()
            or bool(re.match(r"\d", prefix))
            or normalized_prefix.startswith(("pc=", "pс=", "рс="))
            or "электродвигател" in normalized_prefix
        )
        if not prefix_starts_as_tail:
            continue
        return f"{body} {prefix}".strip()
    return text


def _is_pdf_table_noise_without_name(
    *,
    pos: str | None,
    unit: str | None,
    quantity: str | None,
    type_mark: str | None,
    code: str | None,
    manufacturer: str | None,
    mass_kg: str | None,
    note: str | None,
) -> bool:
    if quantity is not None and not _looks_like_pdf_quantity_text(quantity):
        return True
    if quantity is not None and not any([pos, unit, type_mark, code, manufacturer, mass_kg, note]):
        return True
    if unit is not None and not any([pos, quantity, type_mark, code, manufacturer, mass_kg]):
        return True
    if manufacturer is not None and not any([pos, unit, quantity, type_mark, code, mass_kg]):
        return True
    if pos is not None and not any([unit, quantity, type_mark, code, manufacturer, mass_kg]):
        return True
    text = " ".join(
        part for part in (pos, unit, quantity, type_mark, code, manufacturer, mass_kg, note) if part
    )
    lowered = text.lower()
    if any(
        marker in lowered for marker in ("изме-", "рения", "испопн", "жилой дом №", "подп", "гип")
    ):
        return True
    return False


def _looks_like_pdf_quantity_text(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[,.]\d+)?(?:/\d*(?:[,.]\d*)?)?", value.strip()))


def _can_use_pdf_table_context_header(name: str) -> bool:
    normalized = name.strip().lower()
    if LEADING_PDF_POSITION_RE.match(normalized):
        return False
    return any(
        marker in normalized
        for marker in (
            "воздуховод",
            "коллектор",
            "радиатор",
            "плита",
            "теплоизоляц",
            "теплый пол",
        )
    )


def _is_pdf_table_lowercase_continuation_header(name: str) -> bool:
    stripped = name.strip()
    if len(stripped) > 80:
        return False
    first = stripped[:1]
    return bool(first and first.islower())


def _last_pdf_table_data_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in reversed(rows):
        if any(_string_or_none(row.get(field)) for field in ("name", "unit", "quantity")):
            return row
    return None


def _join_pdf_table_text(*parts: str | None) -> str | None:
    values = [part.strip() for part in parts if part and part.strip()]
    return " ".join(values) if values else None


def _join_pdf_table_code_text(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    separator = "" if left.endswith(("-", "_", "/")) else " "
    return f"{left}{separator}{right}"


def _pdf_table_grouping_ai_row(
    row: dict[str, object], section: str | None, subsection: str | None
) -> dict[str, object]:
    return {
        "page_ref": _string_or_none(row.get("page_ref")),
        "section": section,
        "subsection": subsection,
        "pos": None,
        "name": None,
        "type_mark": None,
        "code": None,
        "manufacturer": None,
        "unit": None,
        "quantity": None,
        "mass_kg": None,
        "note": None,
        "status": _string_or_none(row.get("status")),
    }


def _is_pdf_table_group_row(row: dict[str, object]) -> bool:
    name = _string_or_none(row.get("name"))
    return name is not None and not any(
        _string_or_none(row.get(field))
        for field in (
            "pos",
            "type_mark",
            "code",
            "manufacturer",
            "unit",
            "quantity",
            "mass_kg",
            "note",
        )
    )


def _is_pdf_table_context_header(rows: list[dict[str, object]], index: int) -> bool:
    for next_row in rows[index + 1 :]:
        if _is_pdf_table_group_row(next_row):
            return False
        if not any(_string_or_none(next_row.get(field)) for field in AI_ROW_DATA_FIELDS):
            continue
        return _string_or_none(next_row.get("pos")) is None
    return False


def _is_top_level_pdf_section(name: str) -> bool:
    return "часть" not in name.lower()


AI_ROW_DATA_FIELDS = (
    "pos",
    "name",
    "type_mark",
    "code",
    "manufacturer",
    "unit",
    "quantity",
    "mass_kg",
    "note",
)


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _plan_ai_page_batches(
    *,
    sample_id: str,
    pages: list[OcrPage],
    page_batch_size: int,
    context_tokens: int,
    max_tokens: int,
    context_margin: int,
) -> list[list[OcrPage]]:
    if not pages:
        return []
    if page_batch_size > 0:
        return [
            pages[index : index + page_batch_size]
            for index in range(0, len(pages), page_batch_size)
        ]
    if context_tokens <= 0 or max_tokens <= 0 or len(pages) == 1:
        return [pages]

    input_budget = context_tokens - max_tokens - max(context_margin, 0)
    if input_budget <= 0:
        return [[page] for page in pages]

    batches: list[list[OcrPage]] = []
    current: list[OcrPage] = []
    for page in pages:
        candidate = [*current, page]
        prompt = build_extraction_prompt(sample_id=sample_id, pages=candidate)
        if current and _estimate_prompt_tokens(prompt) > input_budget:
            batches.append(current)
            current = [page]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _extract_ai_batch_with_context_retry(
    *,
    client,
    sample_id: str,
    source_pdf: Path,
    pages: list[OcrPage],
) -> AiExtractionResult:
    page_refs = [page.page_ref for page in pages]
    batch_id = f"{sample_id}-pages-{'-'.join(page_refs)}"
    try:
        return client.extract_rows(
            AiExtractionRequest(sample_id=batch_id, source_pdf=source_pdf, pages=pages)
        )
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        retry_reason = _ai_batch_split_retry_reason(exc)
        if retry_reason is None:
            raise
        if len(pages) <= 1:
            if retry_reason != "timeout":
                raise
            return _retry_single_page_batch_on_timeout(
                client=client,
                batch_id=batch_id,
                source_pdf=source_pdf,
                pages=pages,
                first_error=exc,
            )
        print(
            "map-ocr: AI batch pages "
            f"{','.join(page_refs)} failed with {retry_reason}; retrying as smaller batches"
        )
        split_at = max(1, len(pages) // 2)
        left = _extract_ai_batch_with_context_retry(
            client=client,
            sample_id=sample_id,
            source_pdf=source_pdf,
            pages=pages[:split_at],
        )
        right = _extract_ai_batch_with_context_retry(
            client=client,
            sample_id=sample_id,
            source_pdf=source_pdf,
            pages=pages[split_at:],
        )
        return AiExtractionResult(
            rows=[*left.rows, *right.rows],
            warnings=[
                f"pages {','.join(page_refs)} split due to {retry_reason}",
                *left.warnings,
                *right.warnings,
            ],
            raw_response={
                "split_due_to": retry_reason,
                "page_refs": page_refs,
                "left": left.raw_response,
                "right": right.raw_response,
            },
        )


_AI_SINGLE_PAGE_TIMEOUT_ATTEMPTS = 3


def _retry_single_page_batch_on_timeout(
    *,
    client,
    batch_id: str,
    source_pdf: Path,
    pages: list[OcrPage],
    first_error: BaseException,
) -> AiExtractionResult:
    page_ref = pages[0].page_ref
    attempts = _AI_SINGLE_PAGE_TIMEOUT_ATTEMPTS
    warnings = [f"page {page_ref} attempt 1/{attempts} timed out"]
    attempt_client = client
    last_error: BaseException = first_error
    for attempt in range(2, attempts + 1):
        reduce_reasoning = getattr(attempt_client, "with_reduced_reasoning", None)
        reduced_client = reduce_reasoning() if callable(reduce_reasoning) else None
        variant_note = ""
        if reduced_client is not None:
            attempt_client = reduced_client
            variant = getattr(getattr(attempt_client, "config", None), "variant", None)
            if variant:
                variant_note = f" with reduced reasoning variant {variant}"
        print(
            f"map-ocr: AI page {page_ref} timed out; "
            f"retrying attempt {attempt}/{attempts}{variant_note}"
        )
        try:
            result = attempt_client.extract_rows(
                AiExtractionRequest(sample_id=batch_id, source_pdf=source_pdf, pages=pages)
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            if _ai_batch_split_retry_reason(exc) != "timeout":
                raise
            warnings.append(f"page {page_ref} attempt {attempt}/{attempts} timed out{variant_note}")
            last_error = exc
            continue
        return AiExtractionResult(
            rows=result.rows,
            warnings=[*warnings, *result.warnings],
            raw_response={
                "timeout_retry": {
                    "page_ref": page_ref,
                    "succeeded_attempt": attempt,
                    "attempts": attempts,
                },
                "result": result.raw_response,
            },
        )
    raise RuntimeError(
        f"AI extraction for page {page_ref} timed out after {attempts} attempts "
        "(reasoning effort was reduced on retries when supported)"
    ) from last_error


def _ai_batch_split_retry_reason(exc: BaseException) -> str | None:
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text or "timeoutexpired" in text:
        return "timeout"
    if "context_length" in text or "maximum context length" in text:
        return "context_length"
    return None


def _estimate_prompt_tokens(prompt: str) -> int:
    # Conservative character heuristic; exact tokenization depends on the served model.
    return max(1, int(len(prompt) / 2.7) + 128)


def _map_ocr_performance_summary(
    *,
    sample_id: str,
    ai_client: str,
    ai_model: str | None,
    ai_agent: str | None,
    ai_base_url: str | None,
    ai_served_model: str | None,
    extraction_mode: str,
    page_count: int,
    row_count: int,
    runtime_sec: float,
    quality_errors: int,
    quality_warnings: int,
    input_ocr: Path,
    source_pdf: Path,
    output_path: Path,
    quality_path: Path | None,
) -> dict[str, object]:
    sec_per_page = runtime_sec / page_count if page_count else None
    sec_per_row = runtime_sec / row_count if row_count else None
    rows_per_min = row_count / runtime_sec * 60 if runtime_sec > 0 else None
    return {
        "sample_id": sample_id,
        "ai_client": ai_client,
        "ai_model": ai_model,
        "ai_agent": ai_agent,
        "ai_base_url": ai_base_url,
        "ai_served_model": ai_served_model,
        "extraction_mode": extraction_mode,
        "page_count": page_count,
        "row_count": row_count,
        "runtime_sec": round(runtime_sec, 3),
        "runtime_human": _format_benchmark_duration(runtime_sec),
        "sec_per_page": _round_optional(sec_per_page),
        "sec_per_row": _round_optional(sec_per_row),
        "rows_per_min": _round_optional(rows_per_min),
        "quality_errors": quality_errors,
        "quality_warnings": quality_warnings,
        "input_ocr": str(input_ocr),
        "source_pdf": str(source_pdf),
        "output": str(output_path),
        "quality_json": str(quality_path) if quality_path is not None else None,
    }


def _print_map_ocr_performance(
    performance: dict[str, object], *, performance_path: Path | None
) -> None:
    print("map-ocr performance:")
    for key in (
        "sample_id",
        "ai_client",
        "ai_model",
        "ai_agent",
        "ai_base_url",
        "ai_served_model",
        "extraction_mode",
        "page_count",
        "row_count",
        "runtime_sec",
        "runtime_human",
        "sec_per_page",
        "sec_per_row",
        "rows_per_min",
        "quality_errors",
        "quality_warnings",
        "output",
        "quality_json",
    ):
        value = performance.get(key)
        if value is not None:
            print(f"  {key}: {value}")
    if performance_path is not None:
        print(f"  performance_json: {performance_path}")


def _map_ocr_extraction_mode(raw_response: dict[str, object]) -> str:
    fallback = raw_response.get("fallback")
    if fallback == "pdf_table_layer_direct":
        return "pdf_table_layer_direct"
    if fallback == "pdf_table_layer":
        return "pdf_table_layer_fallback"
    return "ai"


def _round_optional(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _format_benchmark_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining:02d}s"
    return f"{minutes}m{remaining:02d}s"


def command_benchmark(
    *,
    sample_id: str,
    input_ocr: str,
    source_pdf: str,
    output_dir: str | None,
    manifest_path: str,
    model_cache_dir: str | None,
    local_ai_client: str,
    qwen_model: str | None,
    local_base_url: str,
    local_served_model: str | None,
    gpt_model: str,
    gpt_agent: str,
    ai_timeout: int,
    no_image_attachments: bool,
    as_json: bool,
) -> int:
    sample = None
    try:
        sample = load_sample_manifest(manifest_path).get_sample(sample_id)
    except (FileNotFoundError, KeyError, ValueError):
        sample = None
    resolved_output_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else PROJECT_ROOT / "artifacts" / "benchmarks" / sample_id
    )
    resolved_local_model = qwen_model or (
        "qwen3.6-safe" if local_ai_client == "openai-compatible" else "qwen3-8b"
    )
    resolved_served_model = local_served_model or _default_local_served_model(
        resolved_local_model, local_ai_client
    )
    report = run_local_vs_gpt_benchmark(
        LocalVsGptBenchmarkConfig(
            sample_id=sample_id,
            input_ocr=Path(input_ocr).expanduser(),
            source_pdf=Path(source_pdf).expanduser(),
            output_dir=resolved_output_dir,
            project_root=PROJECT_ROOT,
            model_cache_dir=Path(model_cache_dir).expanduser() if model_cache_dir else None,
            local_ai_client=local_ai_client,
            qwen_model=resolved_local_model,
            local_base_url=local_base_url,
            local_served_model=resolved_served_model,
            gpt_model=gpt_model,
            gpt_agent=gpt_agent,
            ai_timeout=ai_timeout,
            attach_images=not no_image_attachments,
        ),
        sample=sample,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"benchmark: wrote {report['report_json']}")
        results_obj = report.get("results")
        for result in results_obj if isinstance(results_obj, list) else []:
            if not isinstance(result, dict):
                continue
            marker = "OK" if result.get("ok") else "ERR"
            runtime = result.get("runtime_sec")
            runtime_text = f"{runtime:.2f}s" if isinstance(runtime, (int, float)) else "n/a"
            print(f"[{marker}] {result.get('backend')}: {runtime_text}")
    results = report.get("results")
    return (
        0
        if isinstance(results, list)
        and all(item.get("ok") for item in results if isinstance(item, dict))
        else 1
    )


def _default_local_served_model(model: str, local_ai_client: str) -> str | None:
    if local_ai_client != "openai-compatible":
        return None
    if model in {
        "qwen3_6_27b_gguf_iq3_xxs",
        "qwen36-safe",
        "qwen3.6-safe",
        "qwen3.6-27b-safe",
        "qwen3.6-27b-iq3-gguf",
        "qwen3.6-27b-iq3-xxs-gguf",
    }:
        return "qwen3.6-safe-local"
    if model in {
        "qwen3_6_27b_gguf_q4_k_s",
        "qwen36-balanced",
        "qwen3.6-balanced",
        "qwen3.6-27b-balanced",
        "qwen3.6-27b-q4ks-gguf",
        "qwen3.6-27b-q4-k-s-gguf",
    }:
        return "qwen3.6-balanced-local"
    if model in {
        "qwen3_6_27b_gguf_q4_k_m",
        "qwen36-quality",
        "qwen3.6-quality",
        "qwen3.6-27b-quality",
        "qwen3.6-27b-q4km-gguf",
        "qwen3.6-27b-q4-k-m-gguf",
    }:
        return "qwen3.6-quality-local"
    if model in {
        "qwen3_vl_8b_cuda",
        "qwen3-vl-8b",
        "qwen3-vl-8b-cuda",
    }:
        return "qwen3-vl-local"
    if model in {
        "qwen3_6_27b_gptq_cuda",
        "qwen36-max",
        "qwen36-max-vllm",
        "qwen3.6-vllm-experimental",
    }:
        return "qwen36-max-local"
    if "14b" in model:
        return "qwen3-14b-local"
    return "qwen3-8b-local"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return command_info()
    if args.command == "doctor":
        return command_doctor(args.json)
    if args.command == "paddle-gpu-check":
        return command_paddle_gpu_check(
            as_json=args.json,
            device=args.device,
            require_cuda=args.require_cuda,
            smoke_test=not args.no_smoke_test,
        )
    if args.command == "llama-gpu-check":
        return command_llama_gpu_check(as_json=args.json, require_cuda=args.require_cuda)
    if args.command == "llama-link-cuda-libs":
        return command_llama_link_cuda_libs(as_json=args.json)
    if args.command == "models":
        return command_models(
            args.models_action,
            getattr(args, "model", None),
            args.cache_dir,
            args.json,
            getattr(args, "confirm", False),
        )
    if args.command == "setup-local":
        return command_setup_local(
            profile=args.profile,
            download=args.download,
            install_deps=args.install_deps,
            skip_paddle_install=args.skip_paddle_install,
            as_json=args.json,
            cache_dir=args.cache_dir,
        )
    if args.command == "setup-wsl":
        return command_setup_wsl(
            profile=args.profile,
            distro=args.distro,
            project_dir=args.project_dir,
            venv=args.venv,
            cache_dir=args.cache_dir,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name,
            backend=args.backend,
            runtime=args.runtime,
            skip_runtime_install=args.skip_runtime_install,
            skip_paddle_install=args.skip_paddle_install,
            gpt_auth_file=args.gpt_auth_file,
            skip_gpt_smoke=args.skip_gpt_smoke,
            execute=args.execute,
            confirm=args.confirm,
            as_json=args.json,
        )
    if args.command in {"setup-windows", "setup-customer", "setup-wizard"}:
        return command_setup_windows(
            profile=args.profile,
            distro=args.distro,
            project_dir=args.project_dir,
            venv=args.venv,
            cache_dir=args.cache_dir,
            host=args.host,
            port=args.port,
            yes=args.yes,
            skip_runtime_install=args.skip_runtime_install,
            skip_paddle_install=args.skip_paddle_install,
            gpt_auth_file=args.gpt_auth_file,
            skip_gpt_smoke=args.skip_gpt_smoke,
            as_json=args.json,
        )
    if args.command == "export-gpt-auth":
        return command_export_gpt_auth(
            source=args.source,
            output=args.output,
            confirm=args.confirm,
            as_json=args.json,
        )
    if args.command == "serve-local":
        return command_serve_local(
            model=args.model,
            cache_dir=args.cache_dir,
            backend=args.backend,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name,
            dry_run=args.dry_run,
            as_json=args.json,
            extra_args=args.extra_arg,
        )
    if args.command == "samples":
        return command_samples(args.manifest, args.json, args.validate)
    if args.command == "rows":
        return command_rows(args.manifest, args.sample_id, args.limit, args.json)
    if args.command == "normalize":
        return command_normalize(args.manifest, args.sample_id, args.output)
    if args.command == "export-xlsx":
        return command_export_xlsx(args.manifest, args.sample_id, args.input_jsonl, args.output)
    if args.command == "export-jsonl-xlsx":
        return command_export_jsonl_xlsx(args.input_jsonl, args.output)
    if args.command == "export-kp-xlsx":
        return command_export_kp_xlsx(args.input_jsonl, args.output, args.template, args.sheet)
    if args.command == "compare":
        return command_compare(
            args.manifest, args.sample_id, args.predicted_jsonl, args.output_json
        )
    if args.command == "parse-pdf":
        return command_parse_pdf(args.manifest, args.sample_id, args.backend, args.output)
    if args.command == "diagnose-pdf":
        return command_diagnose_pdf(
            args.manifest,
            args.sample_id,
            args.backend,
            args.limit,
            args.output_json,
        )
    if args.command == "probe-pdf":
        return command_probe_pdf(args.manifest, args.sample_id)
    if args.command == "process":
        return command_process(
            args.pdf_path,
            args.output_dir,
            args.pages,
            args.dpi,
            args.quote,
            args.force,
        )
    if args.command == "extract-ocr":
        return command_extract_ocr(
            args.pdf_path,
            args.provider,
            args.pages,
            args.output,
            args.dpi,
            args.lang,
            args.psm,
            args.pdf_text_overlay,
            args.paddle_model,
            args.paddle_device,
            args.paddle_pipeline_version,
            args.paddle_vl_backend,
            args.paddle_vl_server_url,
            args.paddle_vl_api_model,
            args.paddle_max_new_tokens,
            args.paddle_min_pixels,
            args.paddle_max_pixels,
            args.paddle_repetition_penalty,
            args.paddle_temperature,
            args.paddle_top_p,
            args.paddle_split_wide_pages,
            args.paddle_band_height,
            args.paddle_band_overlap,
            args.paddle_predict_timeout,
            args.paddle_predict_heartbeat,
            args.paddle_use_layout_detection,
            args.paddle_use_queues,
            args.paddle_engine,
            args.paddle_cpu_threads,
            args.paddle_require_cuda,
            args.paddle_enable_hpi,
            args.paddle_use_tensorrt,
            args.paddle_precision,
            args.paddle_vl_max_concurrency,
            args.dots_model,
            args.dots_backend,
            args.dots_device,
            args.dots_device_map,
            args.dots_torch_dtype,
            args.dots_attn_implementation,
            args.dots_trust_model_code,
            args.dots_upscale,
            args.dots_max_image_edge,
            args.dots_max_tokens,
            args.dots_no_crop,
            args.model_cache_dir,
        )
    if args.command == "map-ocr":
        return command_map_ocr(
            args.sample_id,
            args.input_ocr,
            args.source_pdf,
            args.output,
            args.quality_json,
            args.quality_fail_on,
            args.performance_json,
            args.ai_client,
            args.ai_model,
            args.ai_agent,
            args.ai_timeout,
            args.ai_max_tokens,
            args.ai_page_batch_size,
            args.ai_context_tokens,
            args.ai_context_margin,
            args.ai_base_url,
            args.ai_served_model,
            args.ai_external_openai_compatible,
            args.ai_external_supports_images,
            args.ai_api_key_env,
            args.ai_reasoning_effort,
            args.third_layer_reviewer,
            args.third_layer_model,
            args.third_layer_agent,
            args.third_layer_base_url,
            args.third_layer_served_model,
            args.third_layer_api_key_env,
            args.third_layer_reasoning_effort,
            args.third_layer_timeout,
            args.third_layer_max_tokens,
            args.no_third_layer_image_attachments,
            args.no_image_attachments,
            args.ai_force_on_pdf_table_layer,
            args.model_cache_dir,
        )
    if args.command == "benchmark":
        return command_benchmark(
            sample_id=args.sample_id,
            input_ocr=args.input_ocr,
            source_pdf=args.source_pdf,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            model_cache_dir=args.model_cache_dir,
            local_ai_client=args.local_ai_client,
            qwen_model=args.qwen_model,
            local_base_url=args.local_base_url,
            local_served_model=args.local_served_model,
            gpt_model=args.gpt_model,
            gpt_agent=args.gpt_agent,
            ai_timeout=args.ai_timeout,
            no_image_attachments=args.no_image_attachments,
            as_json=args.json,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
