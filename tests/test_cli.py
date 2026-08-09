from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

import spec_pipeline.cli as cli_module
from spec_pipeline.cli import _ai_rows_from_pdf_table_layer, _default_local_served_model, main
from spec_pipeline.local.llama_runtime import LlamaCppRuntimeInfo
from spec_pipeline.local.wsl_installer import windows_or_posix_project_path_to_wsl
from spec_pipeline.ocr.base import OcrPage, OcrToken
from spec_pipeline.ocr.paddle_runtime import PaddleRuntimeInfo
from spec_pipeline.workflows.pdf_probe import PdfTextLayerProbe


def _write_cli_manifest(tmp_path: Path) -> Path:
    tsv_path = tmp_path / "sample.tsv"
    tsv_path.write_text(
        "page\tsection\tsubsection\tpos\tname\ttype_mark\tcode\tmanufacturer\tunit\tquantity\tmass_kg\tnote\n"
        "001\tПВ7\t\t1\tУстановка\t\t\tСВОК\tшт\t1\t\t\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "1",
                "name": "cli-test",
                "samples": [
                    {
                        "sample_id": "sample_1",
                        "title": "Sample 1",
                        "source_pdf": str(tmp_path / "input.pdf"),
                        "ground_truth_tsv": str(tsv_path),
                        "expected_xlsx": str(tmp_path / "expected.xlsx"),
                        "visual_xlsx": str(tmp_path / "visual.xlsx"),
                        "page_field": "page",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_info_command_returns_success() -> None:
    assert main(["info"]) == 0


def test_doctor_command_returns_success() -> None:
    assert main(["doctor"]) == 0


def test_doctor_json_returns_success(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "hardware" in payload
    assert "model_cache_dir" in payload


def test_doctor_json_allows_packaged_runtime_without_test_fixtures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    for relative_path in (
        "src/spec_pipeline/parsers",
        "src/spec_pipeline/exporters",
        "src/spec_pipeline/workflows",
    ):
        (tmp_path / relative_path).mkdir(parents=True)

    monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)

    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["missing_paths"] == []
    assert payload["missing_required_paths"] == []
    assert payload["missing_optional_paths"] == []
    assert payload["workspace"] == str(tmp_path)
    assert payload["core_modules"]["openpyxl"] is True
    assert payload["core_modules"]["pdfminer"] is True


def test_models_status_json_returns_success(tmp_path: Path, capsys) -> None:
    assert main(["models", "--cache-dir", str(tmp_path), "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_dir"] == str(tmp_path)
    assert payload["models"]
    assert {model["status"] for model in payload["models"]} == {"missing"}


def test_models_parent_json_flag_applies_to_status(tmp_path: Path, capsys) -> None:
    assert main(["models", "--cache-dir", str(tmp_path), "--json", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_dir"] == str(tmp_path)


def test_models_download_unknown_alias_returns_error(tmp_path: Path, capsys) -> None:
    assert main(["models", "--cache-dir", str(tmp_path), "download", "unknown-model"]) == 1
    captured = capsys.readouterr()
    assert "Unknown local model" in captured.err


def test_models_status_corrupt_lock_returns_error(tmp_path: Path, capsys) -> None:
    (tmp_path / "manifest.lock.json").write_text("{bad", encoding="utf-8")

    assert main(["models", "--cache-dir", str(tmp_path), "status"]) == 1
    assert "Corrupt model lock file" in capsys.readouterr().err


class _FakeHardware:
    os = "Darwin 25.3.0"
    machine = "arm64"
    python = "3.13"
    nvidia_smi = False
    nvidia_gpu = None
    mlx_available = True

    def to_dict(self):
        return {
            "os": self.os,
            "machine": self.machine,
            "python": self.python,
            "nvidia_smi": self.nvidia_smi,
            "nvidia_gpu": self.nvidia_gpu,
            "mlx_available": self.mlx_available,
        }


class _FakeCudaHardware(_FakeHardware):
    os = "Linux 6.8.0 WSL2"
    machine = "x86_64"
    nvidia_smi = True
    nvidia_gpu = "NVIDIA GeForce RTX 4080 SUPER"
    mlx_available = False


class _FakeStatus:
    def __init__(self, key: str, title: str, status: str = "missing") -> None:
        self.key = key
        self.title = title
        self.alias = key
        self.hf_id = key
        self.revision = "rev"
        self.backend = "test"
        self.path = f"/models/{key}"
        self.status = status
        self.installed = status == "ready"
        self.message = "модель установлена и готова" if self.installed else "модель не установлена"
        self.issues: list[str] = []

    def to_dict(self):
        return {
            "key": self.key,
            "title": self.title,
            "alias": self.alias,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "backend": self.backend,
            "path": self.path,
            "status": self.status,
            "installed": self.installed,
            "message": self.message,
            "issues": self.issues,
        }


class _FakeOperationResult:
    def __init__(self, model: _FakeStatus) -> None:
        self.ok = True
        self.status = "ready"
        self.message = "модель уже установлена, скачивание не требуется"
        self.model = model

    def to_dict(self):
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "model": self.model.to_dict(),
        }


class _FakeManifest:
    def resolve(self, key: str):
        return type("FakeSpec", (), {"disk_gb": 10.0 if "14b" in key else 3.0})()


class _FakeModelCacheManager:
    instances: list["_FakeModelCacheManager"] = []

    def __init__(self, *, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.manifest = _FakeManifest()
        self.downloaded: list[str] = []
        self.__class__.instances.append(self)

    def status(self, key: str):
        title = "PaddleOCR-VL" if key == "paddleocr_vl" else key
        return _FakeStatus(key, title)

    def download(self, key: str):
        self.downloaded.append(key)
        return _FakeOperationResult(_FakeStatus(key, key, status="ready"))


def test_setup_local_dry_run_outputs_russian_download_hint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    assert main(["setup-local", "--cache-dir", str(tmp_path), "--profile", "standard"]) == 1
    output = capsys.readouterr().out
    assert "Выбран локальный бекенд: MLX" in output
    assert "qwen3_8b_mlx" in output
    assert "--download" in output


def test_setup_local_json_hq_selects_qwen14b_mlx(tmp_path: Path, monkeypatch, capsys) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    assert main(["setup-local", "--cache-dir", str(tmp_path), "--profile", "hq", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "MLX"
    assert payload["ai_client"] == "qwen-mlx"
    assert [model["key"] for model in payload["models"]] == ["paddleocr_vl", "qwen3_14b_mlx"]


def test_setup_local_json_cuda_selects_openai_compatible_client(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeCudaHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    assert main(["setup-local", "--cache-dir", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "CUDA/WSL2"
    assert payload["ai_client"] == "openai-compatible"
    assert [model["key"] for model in payload["models"]] == [
        "paddleocr_vl",
        "qwen3_6_27b_gguf_iq3_xxs",
    ]


def test_setup_local_qwen36_cuda_profile_selects_gguf_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeCudaHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    assert (
        main(
            [
                "setup-local",
                "--cache-dir",
                str(tmp_path),
                "--profile",
                "qwen36-safe",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert [model["key"] for model in payload["models"]] == [
        "paddleocr_vl",
        "qwen3_6_27b_gguf_iq3_xxs",
    ]


def test_setup_local_download_invokes_model_downloads(tmp_path: Path, monkeypatch) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    assert main(["setup-local", "--cache-dir", str(tmp_path), "--download"]) == 0
    assert _FakeModelCacheManager.instances[0].downloaded == [
        "paddleocr_vl",
        "qwen3_6_35b_a3b_mlx_vlm",
    ]


def test_setup_local_install_deps_invokes_dependency_installer(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakeModelCacheManager.instances = []
    calls = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    class FakeDepsResult:
        ok = True
        extras = ("local-models", "macos-local")

        def to_dict(self):
            return {"ok": True, "extras": list(self.extras)}

    def fake_install(**kwargs):
        calls.append(kwargs)
        return FakeDepsResult()

    monkeypatch.setattr("spec_pipeline.cli.install_local_dependencies", fake_install)

    assert (
        main(
            [
                "setup-local",
                "--cache-dir",
                str(tmp_path),
                "--install-deps",
                "--skip-paddle-install",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert calls
    assert calls[0]["include_paddle"] is False
    assert payload["install_deps"] is True
    assert payload["dependency_install"]["extras"] == ["local-models", "macos-local"]


def test_setup_local_does_not_install_deps_without_flag(tmp_path: Path, monkeypatch) -> None:
    _FakeModelCacheManager.instances = []
    monkeypatch.setattr("spec_pipeline.cli.collect_hardware_report", lambda: _FakeHardware())
    monkeypatch.setattr("spec_pipeline.cli.ModelCacheManager", _FakeModelCacheManager)

    def fail_install(**kwargs):
        raise AssertionError("must not install deps without --install-deps")

    monkeypatch.setattr("spec_pipeline.cli.install_local_dependencies", fail_install)

    assert main(["setup-local", "--cache-dir", str(tmp_path)]) == 1


def test_setup_wsl_dry_run_prints_commands(capsys) -> None:
    assert main(["setup-wsl"]) == 0
    output = capsys.readouterr().out
    assert "План настройки WSL (dry-run)" in output
    assert "wsl.exe --status" in output
    assert "nvidia-smi" in output
    assert "python3 -m venv .venv-wsl" in output
    assert "spec-ocr setup-local --profile qwen36-safe --download" in output
    assert "spec-ocr serve-local qwen3.6-safe" in output


def test_setup_wsl_json_outputs_plan(capsys) -> None:
    assert main(["setup-wsl", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["profile"] == "qwen36-safe"
    assert payload["model"] == "qwen3.6-safe"
    assert any(step["id"] == "wsl.models.download" for step in payload["steps"])


def test_paddle_gpu_check_json_requires_cuda(monkeypatch, capsys) -> None:
    def fake_inspect(*, device: str, smoke_test: bool) -> PaddleRuntimeInfo:
        assert device == "gpu:0"
        assert smoke_test is True
        return PaddleRuntimeInfo(
            installed=True,
            version="3.2.1",
            compiled_with_cuda=True,
            cuda_device_count=1,
            selected_device="gpu:0",
            smoke_test_ok=True,
        )

    monkeypatch.setattr("spec_pipeline.cli.inspect_paddle_runtime", fake_inspect)

    assert main(["paddle-gpu-check", "--require-cuda", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["cuda_ready"] is True


def test_paddle_gpu_check_fails_for_cpu_paddle(monkeypatch, capsys) -> None:
    def fake_inspect(*, device: str, smoke_test: bool) -> PaddleRuntimeInfo:
        return PaddleRuntimeInfo(installed=True, version="3.2.1", compiled_with_cuda=False)

    monkeypatch.setattr("spec_pipeline.cli.inspect_paddle_runtime", fake_inspect)

    assert main(["paddle-gpu-check", "--require-cuda", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["compiled_with_cuda"] is False


def test_llama_gpu_check_json_requires_cuda(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "spec_pipeline.cli.inspect_llama_cpp_runtime",
        lambda: LlamaCppRuntimeInfo(
            installed=True,
            version="0.3.21",
            supports_gpu_offload=True,
        ),
    )

    assert main(["llama-gpu-check", "--require-cuda", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["cuda_ready"] is True


def test_llama_gpu_check_fails_for_cpu_wheel(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "spec_pipeline.cli.inspect_llama_cpp_runtime",
        lambda: LlamaCppRuntimeInfo(
            installed=True,
            version="0.3.21",
            supports_gpu_offload=False,
        ),
    )

    assert main(["llama-gpu-check", "--require-cuda", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["supports_gpu_offload"] is False


def test_setup_wsl_default_project_dir_uses_current_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["setup-wsl", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["project_path"] == windows_or_posix_project_path_to_wsl(tmp_path)


def test_windows_bootstrap_scripts_are_packaged() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "setup-windows.cmd").is_file()
    assert (project_root / "START_HERE_WINDOWS.cmd").is_file()
    assert (project_root / "INSTALL_GPT_WINDOWS.cmd").is_file()
    assert (project_root / "INSTALL_LOCAL_QWEN_SAFE_WINDOWS.cmd").is_file()
    assert (project_root / "PROCESS_PDF_WINDOWS.cmd").is_file()
    assert (project_root / "PROCESS_PDF_GPT_WINDOWS.cmd").is_file()
    assert (project_root / "PROCESS_INPUT_FOLDER_WINDOWS.cmd").is_file()
    assert (project_root / "PROCESS_INPUT_FOLDER_GPT_WINDOWS.cmd").is_file()
    assert (project_root / "process-pdf-windows.ps1").is_file()
    assert (project_root / "INPUT_PDF" / "README.txt").is_file()
    assert (project_root / "OUTPUT_RESULTS" / "README.txt").is_file()
    assert not (project_root / "tools" / "windows" / "python-3.11.9-amd64.exe").exists()
    assert not (project_root / "tools" / "windows" / "python-3.11.9-embed-amd64.zip").exists()
    assert not (project_root / "tools" / "windows" / "get-pip.py").exists()
    assert not (project_root / "tools" / "windows" / "wheelhouse").exists()
    script = (project_root / "setup-windows.ps1").read_text(encoding="utf-8")
    start_script = (project_root / "START_HERE_WINDOWS.cmd").read_text(encoding="utf-8")
    gpt_script = (project_root / "INSTALL_GPT_WINDOWS.cmd").read_text(encoding="utf-8")
    qwen_script = (project_root / "INSTALL_LOCAL_QWEN_SAFE_WINDOWS.cmd").read_text(encoding="utf-8")
    process_script = (project_root / "process-pdf-windows.ps1").read_text(encoding="utf-8")
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    process_batch_cmd = (project_root / "PROCESS_INPUT_FOLDER_WINDOWS.cmd").read_text(
        encoding="utf-8"
    )
    process_batch_gpt_cmd = (project_root / "PROCESS_INPUT_FOLDER_GPT_WINDOWS.cmd").read_text(
        encoding="utf-8"
    )
    process_cmd = (project_root / "PROCESS_PDF_WINDOWS.cmd").read_text(encoding="utf-8")
    process_gpt_cmd = (project_root / "PROCESS_PDF_GPT_WINDOWS.cmd").read_text(encoding="utf-8")
    assert "PYTHONPATH" in script
    assert "spec_pipeline.cli setup-windows" in script
    assert '$PythonInstallerName = "python-$PythonVersion-amd64.exe"' in script
    assert "$PythonInstallerSha256" in script
    assert "$PythonInstallerUrl" in script
    assert "Invoke-WebRequest -UseBasicParsing" in script
    assert "Install-VerifiedPython" in script
    assert "Windows ARM64 is not currently supported" in script
    assert "Test-SupportedPython" in script
    assert "RequireVenv" in script
    assert "encodings, sys, sysconfig" in script
    assert '$imports += ", venv"' in script
    assert 'pip install --disable-pip-version-check -e ".[ocr]"' in script
    assert "Assert-InstallerHash" in script
    assert ".venv-windows-bootstrap" in script
    assert ".spec-ocr-bootstrap-ready" in script
    assert "get-pip.py" not in script
    assert "python-embed" not in script
    assert "wheelhouse" not in script
    assert "pause" in start_script.lower()
    assert "PROCESS_INPUT_FOLDER_WINDOWS.cmd" in start_script
    assert "-GptFastPath" in gpt_script
    assert "--profile qwen36-safe --yes" in qwen_script
    assert '-PdfPath "%~1"' in process_cmd
    assert '-Profile gpt -PdfPath "%~1"' in process_gpt_cmd
    assert "PaddleOCR-VL" in process_script
    assert "Ensure-LocalServer" in process_script
    assert "run-local-qwen-server.ps1" in process_script
    assert "run-local-qwen-server.cmd" in process_script
    assert (
        'Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $ProjectDir "run-local-qwen-server.cmd")'
        in process_script
    )
    assert "Start-Process -FilePath $powershellExe" in process_script
    assert '$env:PYTHONUTF8 = "1"' in process_script
    assert '$env:PYTHONIOENCODING = "utf-8"' in process_script
    assert "WSL launch failed for distro" in process_script
    assert "Retrying default WSL distro" in process_script
    assert "run-local-qwen-server.failed.txt" in process_script
    assert "$serverProcess.HasExited" in process_script
    assert "paddle-gpu-check --require-cuda --json" in process_script
    assert "paddlepaddle-gpu==3.2.1" in process_script
    assert "llama-gpu-check --require-cuda --json" in process_script
    assert "v0.3.21-cu124/llama_cpp_python-0.3.21-py3-none-linux_x86_64.whl" in process_script
    assert (
        "--force-reinstall --no-cache-dir --only-binary llama-cpp-python --prefer-binary"
        in process_script
    )
    assert (
        "llama-cpp-python[server] @ https://github.com/abetlen/llama-cpp-python" in process_script
    )
    assert "nvidia-cuda-runtime-cu12==12.6.77" in process_script
    assert "nvidia-cublas-cu12==12.6.4.1" in process_script
    assert "llama-link-cuda-libs --json" in process_script
    assert "LD_LIBRARY_PATH" in process_script
    assert '"$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib' in process_script
    assert "numpy<2.4,>=1.24" in process_script
    assert "PyYAML<=6.0.2" in process_script
    assert "https://abetlen.github.io/llama-cpp-python/whl/cu125" not in process_script
    assert "SpecOCR hotfix: $SpecOcrHotfixVersion" in process_script
    assert '$SpecOcrHotfixVersion = "LocalModelSizeTolerance_v87"' in process_script
    assert "$NativePdfTextTimeoutSeconds = 90" in process_script
    assert "SPEC_OCR_NATIVE_PDF_TEXT_TIMEOUT_SEC" in process_script
    assert "Windows-native PDF_TABLE_LAYER не завершился" in process_script
    assert "Продолжаю через WSL/OCR fallback для этого PDF" in process_script
    assert "tools\\windows\\wheelhouse" not in process_script
    assert (
        "Windows bootstrap Python не готов (нет .spec-ocr-bootstrap-ready); "
        "запускаю автоматическую подготовку GPT fast path"
    ) in process_script
    assert "$setupOutput = & $powershellExe" in process_script
    assert "$setupExitCode = $LASTEXITCODE" in process_script
    assert "$setupExitCode -eq 3010" in process_script
    assert "требует перезагрузку" in process_script
    assert "-GptFastPath" in process_script
    assert (
        "Windows bootstrap Python не готов даже после автоматической подготовки GPT fast path"
        in process_script
    )
    assert "qwen-vl-safe" not in process_script
    assert "qwen3-vl-8b" not in process_script
    assert "--provider auto" in process_script
    assert "--paddle-device gpu:0 --paddle-require-cuda" in process_script
    assert "--pdf-text-overlay true" in process_script
    assert "--paddle-split-wide-pages true" in process_script
    assert "--paddle-band-height 260" in process_script
    assert "--paddle-band-overlap 40" in process_script
    assert "--paddle-max-pixels 501760" in process_script
    assert "--paddle-max-new-tokens 1024" in process_script
    assert "--paddle-predict-timeout 900 --paddle-predict-heartbeat 60" in process_script
    assert "OCR PDF auto (PDF table layer или PaddleOCR-VL CUDA watchdog)" in process_script
    assert "Ensure-GptWslFallbackEnvironment" in process_script
    assert "Рендер страниц PDF для GPT vision fallback" in process_script
    assert "Проверка рендера PDF для GPT vision fallback" in process_script
    assert "spec-ocr extract-ocr $(Quote-BashArg $pdfWsl) --provider pdf-text" in process_script
    assert "GPT/OpenCode будет читать изображения страниц" in process_script
    assert "export-jsonl-xlsx" in process_script
    assert "export-kp-xlsx" in process_script
    assert "rows.ai.performance.json" in process_script
    assert "--quality-fail-on uncorrected-warning" in process_script
    assert "--quality-fail-on none" not in process_script
    assert "Остановлено контролем качества OCR" in process_script
    assert "Excel не создан" in process_script
    assert "ContinueOnQualityGateFailure" in process_script
    assert "Пакетная обработка продолжится со следующего PDF" in process_script
    assert "Test-WslFileNonEmpty" in process_script
    assert "AI-разбор не создал rows.ai.jsonl; Excel не экспортирую." in process_script
    assert "ai_output_missing" in process_script
    assert "logs\\02-ai-map.log" in process_script
    assert "Tee-Object -FilePath $LogPath" in process_script
    assert "Add-Content -Path $LogPath" in process_script
    assert "[spec-ocr] exit_code=$exitCode" in process_script
    assert "Write-Utf8NoBomFile" in process_script
    assert "Get-WslStepScriptPath" in process_script
    assert "-- bash $scriptPathWsl" in process_script
    assert "script: $scriptPathWindows" in process_script
    assert '$ErrorActionPreference = "Continue"' in process_script
    assert "Show-LogTail" in process_script
    assert "Настоящая ошибка из лога" in process_script
    assert "Log: $LogPath" in process_script
    assert "01-render-pdf-text" in process_script
    assert "03-export" in process_script
    assert "Test-GptAuthInWsl" in process_script
    assert "Ensure-OpenCodeCliInWsl" in process_script
    assert "Test-GptOpenCodeInWsl" in process_script
    assert "_gpt-auth-check.log" in process_script
    assert "_gpt-opencode-check.log" in process_script
    assert "_gpt-opencode-smoke.log" in process_script
    assert "_wsl-env-check.log" in process_script
    assert "_gpt-render-deps.log" in process_script
    assert "Never copy or share another person''s auth.json session." in process_script
    assert "customer-secrets" not in process_script
    assert "curl -fsSL" not in process_script
    assert "SPEC_OCR_GPT_AUTH_STALE" in process_script
    assert "не смог обновить токен доступа (401)" in process_script
    assert "GPT/OpenCode smoke: OK" in process_script
    assert "SPEC_OCR_GPT_ACCOUNT_FORBIDDEN" in process_script
    assert "Forbidden/gateway/proxy/provider settings" in process_script
    assert "VPN/proxy именно для WSL/Ubuntu" in process_script
    assert "PDF с диагностикой без Excel" in process_script
    assert "processing.status.txt" in process_script
    assert "RunZipWindows" in process_script
    assert "Compress-Archive -LiteralPath $runDirWindows" in process_script
    assert "Архив этого PDF" in process_script
    assert "INPUT_PDF" in process_script
    assert "OUTPUT_RESULTS" in process_script
    assert "Выберите модель обработки" in process_script
    assert process_script.index('Name = "gpt"') < process_script.index('Name = "kimi-k2.6-turbo"')
    assert process_script.index('Name = "kimi-k2.6-turbo"') < process_script.index(
        'Name = "qwen36-safe"'
    )
    assert "qwen36-balanced" in process_script
    assert "qwen36-quality" in process_script
    assert "qwen36-max" in process_script
    assert "GPT-5.5 Extra High" in process_script
    assert 'ReasoningEffort = "xhigh"' in process_script
    assert 'ThirdLayerReasoningEffort = "xhigh"' in process_script
    assert "ForceAiOnPdfTableLayer = $true" not in process_script
    assert "Kimi K2.6 Turbo" in process_script
    assert "Qwen3.6 Safe + Kimi контроль" in process_script
    assert "Qwen3.6 Safe + GPT контроль" in process_script
    assert "accounts/fireworks/routers/kimi-k2p6-turbo" in process_script
    assert "https://api.fireworks.ai/inference/v1" in process_script
    assert "FIREWORKS_API_KEY" in process_script
    assert "--ai-external-openai-compatible" in process_script
    assert "--ai-external-supports-images" in process_script
    assert "--ai-api-key-env $($selectedProfile.ApiKeyEnv)" in process_script
    assert "--ai-reasoning-effort none" in process_script
    assert "--no-image-attachments" not in process_script
    assert "--third-layer-reviewer $($SelectedProfile.ThirdLayerReviewer)" in process_script
    assert "--third-layer-api-key-env $($SelectedProfile.ThirdLayerApiKeyEnv)" in process_script
    assert "Проверка API key третьего слоя" in process_script
    assert (
        "GPT/OpenCode третьего слоя будет вызван только для подозрительных строк." in process_script
    )
    assert "Проверка Fireworks/Kimi API key" in process_script
    assert "spec-ocr setup-local --profile $setupProfile --download" in process_script
    assert "--ai-model $($selectedProfile.Model)" in process_script
    assert (
        "--ai-timeout 1800 --ai-max-tokens 8192 --ai-page-batch-size $aiPageBatchSize --ai-context-tokens 12288 --ai-context-margin 512"
        in process_script
    )
    assert '$selectedProfile.Mode -eq "fireworks"' in process_script
    assert (
        '$selectedProfile.Mode -eq "gpt" -or $selectedProfile.Mode -eq "fireworks"'
        not in process_script
    )
    assert "GPT/OpenCode будет вызван на этапе AI-разбора" in process_script
    assert "Для searchable PDF с PDF_TABLE_LAYER CUDA/PaddleOCR-VL не требуется." in process_script
    assert "PDF без пригодного PDF_TABLE_LAYER" in process_script
    assert "--ai-client opencode" in process_script
    assert "--ai-reasoning-effort $($selectedProfile.ReasoningEffort)" in process_script
    assert "--ai-force-on-pdf-table-layer" in process_script
    assert process_script.index("$pdfFiles = @(Get-PdfFilesToProcess)") < process_script.index(
        "Проверка CUDA для PaddleOCR-VL"
    )
    assert process_script.index("Сначала выполняю OCR через CUDA") < process_script.index(
        "Ensure-LocalServer -ProjectWsl"
    )
    assert "Get-PdfFilesToProcess" in process_script
    assert "*.pdf" in process_script
    assert "INPUT_PDF/*" in gitignore
    assert "!INPUT_PDF/README.txt" in gitignore
    assert "OUTPUT_RESULTS/*" in gitignore
    assert "!OUTPUT_RESULTS/README.txt" in gitignore
    assert ".venv-wsl/" in gitignore
    assert "Готовые Excel-файлы лежат в OUTPUT_RESULTS" in process_batch_cmd
    assert "Готовые Excel-файлы лежат в OUTPUT_RESULTS" in process_batch_gpt_cmd
    assert "Готовые Excel-файлы лежат в OUTPUT_RESULTS" in process_cmd
    assert "Готовые Excel-файлы лежат в OUTPUT_RESULTS" in process_gpt_cmd
    assert "скриншот" in start_script
    assert "скриншот" in gpt_script
    assert "скриншот" in qwen_script


def test_setup_wsl_hq_json_selects_qwen14b(capsys) -> None:
    assert main(["setup-wsl", "--profile", "hq", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "qwen3-14b-cuda"
    assert payload["served_model_name"] == "qwen3-14b-local"


def test_setup_wsl_qwen36_safe_json_selects_llama_cpp(capsys) -> None:
    assert main(["setup-wsl", "--profile", "qwen36-safe", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "qwen3.6-safe"
    assert payload["served_model_name"] == "qwen3.6-safe-local"
    assert any("llama-cpp-python[server]" in step["command"] for step in payload["steps"])


def test_setup_wsl_qwen36_max_json_selects_vllm(capsys) -> None:
    assert main(["setup-wsl", "--profile", "qwen36-max", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "qwen36-max"
    assert payload["served_model_name"] == "qwen36-max-local"
    assert any("python -m pip install vllm" in step["command"] for step in payload["steps"])


def test_setup_windows_json_defaults_to_qwen36_safe_profile(capsys) -> None:
    assert main(["setup-windows", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "qwen36-safe"
    assert payload["model"] == "qwen3.6-safe"
    assert isinstance(payload["warnings"], list)


def test_setup_windows_gpt_yes_uses_current_user_auth_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured_plan = {}

    def fake_execute(plan):
        captured_plan["plan"] = plan
        return 0

    monkeypatch.setattr("spec_pipeline.cli.execute_install_plan", fake_execute)

    assert (
        main(
            [
                "setup-windows",
                "--profile",
                "gpt",
                "--project-dir",
                str(tmp_path),
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert not payload["errors"]
    assert any("opencode auth login" in warning for warning in payload["warnings"])
    auth_step = next(step for step in captured_plan["plan"].steps if step.id == "wsl.opencode.auth")
    assert "текущего пользователя" in auth_step.title
    assert "customer-secrets" not in json.dumps(payload, ensure_ascii=False)


def test_export_gpt_auth_requires_confirm(tmp_path: Path, capsys) -> None:
    source = tmp_path / "auth.json"
    source.write_text("{}", encoding="utf-8")

    assert (
        main(["export-gpt-auth", "--source", str(source), "--output", str(tmp_path / "out.json")])
        == 2
    )
    assert "--confirm" in capsys.readouterr().err


def test_export_gpt_auth_copies_secret_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "auth.json"
    output = tmp_path / "private-migration" / "open-code-auth.json"
    source.write_text(
        '{"openai": {"type": "oauth"}, "anthropic": {"type": "oauth"}}',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "export-gpt-auth",
                "--source",
                str(source),
                "--output",
                str(output),
                "--confirm",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == {"openai": {"type": "oauth"}}
    assert "Не коммитьте" in payload["message"]
    assert "не передавайте" in payload["message"]


def test_setup_windows_yes_executes_selected_profile(monkeypatch) -> None:
    called = {}

    def fake_execute(plan):
        called["profile"] = plan.profile
        called["execute"] = plan.execute
        return 0

    monkeypatch.setattr("spec_pipeline.cli.execute_install_plan", fake_execute)

    assert main(["setup-windows", "--profile", "qwen36-max", "--yes"]) == 0
    assert called == {"profile": "qwen36-max", "execute": True}


def test_setup_wsl_execute_requires_confirm(capsys) -> None:
    assert main(["setup-wsl", "--execute"]) == 2
    captured = capsys.readouterr()
    assert "--execute requires --confirm" in captured.out
    assert "--execute requires --confirm" in captured.err


def test_setup_wsl_runtime_mismatch_returns_error(capsys) -> None:
    assert main(["setup-wsl", "--runtime", "sglang", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("cannot install runtime 'sglang'" in error for error in payload["errors"])


def test_setup_wsl_confirm_without_execute_stays_dry_run(capsys) -> None:
    assert main(["setup-wsl", "--confirm", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["warnings"]


def test_setup_wsl_execute_runs_plan(monkeypatch) -> None:
    called = {}

    def fake_execute(plan):
        called["profile"] = plan.profile
        called["execute"] = plan.execute
        return 0

    monkeypatch.setattr("spec_pipeline.cli.execute_install_plan", fake_execute)

    assert main(["setup-wsl", "--execute", "--confirm", "--profile", "fast"]) == 0
    assert called == {"profile": "fast", "execute": True}


def _install_default_ready_model(cache_dir: Path, *, key: str, hf_id: str, path: str) -> Path:
    model_dir = cache_dir / path
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    lock_path = cache_dir / "manifest.lock.json"
    lock = {"version": 1, "models": {}}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["models"][key] = {
        "hf_id": hf_id,
        "revision": "rev-111",
        "path": path,
        "status": "ready",
        "downloaded_at": "2026-04-26T12:00:00Z",
        "size_bytes": 2,
        "files_checked": True,
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
    return model_dir


def test_serve_local_vllm_dry_run_json_returns_command(tmp_path: Path, capsys) -> None:
    model_dir = _install_default_ready_model(
        tmp_path,
        key="qwen3_8b_cuda",
        hf_id="Qwen/Qwen3-8B",
        path="huggingface/Qwen--Qwen3-8B",
    )

    assert (
        main(
            [
                "serve-local",
                "qwen3-8b-cuda",
                "--cache-dir",
                str(tmp_path),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--extra-arg=--max-model-len",
                "--extra-arg",
                "8192",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "vllm"
    assert payload["model_path"] == str(model_dir.resolve())
    assert payload["base_url"] == "http://0.0.0.0:9000/v1"
    assert payload["command"][2] == "vllm.entrypoints.openai.api_server"
    assert payload["command"][-2:] == ["--max-model-len", "8192"]
    assert payload["env"]["HF_HUB_OFFLINE"] == "1"


def test_serve_local_sglang_dry_run_json_returns_command(tmp_path: Path, capsys) -> None:
    model_dir = _install_default_ready_model(
        tmp_path,
        key="qwen3_vl_8b_cuda",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        path="huggingface/Qwen--Qwen3-VL-8B-Instruct",
    )

    assert (
        main(
            [
                "serve-local",
                "qwen3-vl-8b",
                "--cache-dir",
                str(tmp_path),
                "--served-model-name",
                "qwen3-vl-local",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "sglang"
    assert payload["model_path"] == str(model_dir.resolve())
    assert payload["served_model_name"] == "qwen3-vl-local"
    assert payload["command"][2] == "sglang.launch_server"
    assert "--model-path" in payload["command"]


def test_serve_local_missing_model_returns_download_hint(tmp_path: Path, capsys) -> None:
    assert main(["serve-local", "qwen3-8b-cuda", "--cache-dir", str(tmp_path), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "models download qwen3-8b-cuda" in captured.err


def test_serve_local_rejects_non_server_backend(tmp_path: Path, capsys) -> None:
    assert main(["serve-local", "qwen3-8b", "--cache-dir", str(tmp_path), "--dry-run"]) == 1
    assert "not a supported local server backend" in capsys.readouterr().err


def test_serve_local_backend_override_mismatch_returns_error(tmp_path: Path, capsys) -> None:
    _install_default_ready_model(
        tmp_path,
        key="qwen3_8b_cuda",
        hf_id="Qwen/Qwen3-8B",
        path="huggingface/Qwen--Qwen3-8B",
    )

    assert (
        main(
            [
                "serve-local",
                "qwen3-8b-cuda",
                "--cache-dir",
                str(tmp_path),
                "--backend",
                "sglang",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "cannot launch" in capsys.readouterr().err


def test_serve_local_extra_args_cannot_override_model(tmp_path: Path, capsys) -> None:
    _install_default_ready_model(
        tmp_path,
        key="qwen3_8b_cuda",
        hf_id="Qwen/Qwen3-8B",
        path="huggingface/Qwen--Qwen3-8B",
    )

    assert (
        main(
            [
                "serve-local",
                "qwen3-8b-cuda",
                "--cache-dir",
                str(tmp_path),
                "--extra-arg=--model",
                "--extra-arg",
                "Qwen/Qwen3-8B",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "reserved server option" in capsys.readouterr().err


def test_map_ocr_qwen_mlx_missing_model_returns_actionable_error(tmp_path: Path, capsys) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "qwen-mlx",
                "--model-cache-dir",
                str(tmp_path / "models"),
            ]
        )
        == 1
    )
    assert "models download qwen3-8b" in capsys.readouterr().err


def test_map_ocr_qwen_mlx_missing_selected_model_hint_uses_alias(tmp_path: Path, capsys) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "qwen-mlx",
                "--ai-model",
                "qwen3-14b",
                "--model-cache-dir",
                str(tmp_path / "models"),
            ]
        )
        == 1
    )
    assert "models download qwen3-14b" in capsys.readouterr().err


def test_map_ocr_qwen_mlx_unknown_model_returns_error(tmp_path: Path, capsys) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "qwen-mlx",
                "--ai-model",
                "unknown-local-model",
                "--model-cache-dir",
                str(tmp_path / "models"),
            ]
        )
        == 1
    )
    assert "Unknown local model" in capsys.readouterr().err


def test_map_ocr_qwen_mlx_passes_model_and_image_flag(tmp_path: Path, monkeypatch) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)
    output_path = tmp_path / "mapped.jsonl"
    called = {}

    class FakeQwenClient:
        def __init__(self, *, config):
            called["config"] = config

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "1",
                            "name": "Диффузор",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr("spec_pipeline.cli.QwenMlxExtractionClient", FakeQwenClient)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "qwen-mlx",
                "--ai-model",
                "qwen3.6-35b-a3b",
                "--no-image-attachments",
            ]
        )
        == 0
    )
    assert called["config"].model == "qwen3.6-35b-a3b"
    assert called["config"].attach_images is False


def _write_minimal_ocr_json(tmp_path: Path) -> Path:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ocr_path


def test_map_ocr_without_explicit_ai_client_fails_closed(tmp_path: Path, capsys) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)

    assert (
        main(
            [
                "map-ocr",
                "safe_default",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "AI is disabled" in captured.err
    assert "--ai-client" in captured.err


def test_benchmark_cli_wires_args(tmp_path: Path, monkeypatch, capsys) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)
    called = {}

    def fake_runner(config, *, sample=None):
        called["config"] = config
        return {
            "sample_id": config.sample_id,
            "report_json": str(config.output_dir / "report.json"),
            "results": [
                {"backend": "qwen-mlx", "ok": True, "runtime_sec": 1.0},
                {"backend": "opencode-gpt", "ok": True, "runtime_sec": 2.0},
            ],
            "quality_delta": {"runtime_sec_qwen_minus_gpt": -1.0},
        }

    monkeypatch.setattr("spec_pipeline.cli.run_local_vs_gpt_benchmark", fake_runner)

    assert (
        main(
            [
                "benchmark",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output-dir",
                str(tmp_path / "bench"),
                "--qwen-model",
                "qwen3-8b",
                "--gpt-model",
                "openai/gpt-5.5",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_id"] == "client_sample"
    assert called["config"].local_ai_client == "qwen-mlx"
    assert called["config"].qwen_model == "qwen3-8b"
    assert called["config"].gpt_model == "openai/gpt-5.5"
    assert called["config"].attach_images is True


def test_benchmark_cli_can_disable_local_images(tmp_path: Path, monkeypatch) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)
    called = {}

    def fake_runner(config, *, sample=None):
        called["config"] = config
        return {
            "sample_id": config.sample_id,
            "report_json": str(config.output_dir / "report.json"),
            "results": [
                {"backend": "qwen-mlx", "ok": True, "runtime_sec": 1.0},
                {"backend": "opencode-gpt", "ok": True, "runtime_sec": 2.0},
            ],
        }

    monkeypatch.setattr("spec_pipeline.cli.run_local_vs_gpt_benchmark", fake_runner)

    assert (
        main(
            [
                "benchmark",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output-dir",
                str(tmp_path / "bench"),
                "--qwen-model",
                "qwen3.6-35b-a3b",
                "--no-image-attachments",
            ]
        )
        == 0
    )
    assert called["config"].qwen_model == "qwen3.6-35b-a3b"
    assert called["config"].attach_images is False


def test_benchmark_cli_supports_windows_openai_compatible_local_backend(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)
    called = {}

    def fake_runner(config, *, sample=None):
        called["config"] = config
        return {
            "sample_id": config.sample_id,
            "report_json": str(config.output_dir / "report.json"),
            "results": [
                {"backend": "openai-compatible-local", "ok": True, "runtime_sec": 1.0},
                {"backend": "opencode-gpt", "ok": True, "runtime_sec": 2.0},
            ],
            "quality_delta": {"runtime_sec_local_minus_gpt": -1.0},
        }

    monkeypatch.setattr("spec_pipeline.cli.run_local_vs_gpt_benchmark", fake_runner)

    assert (
        main(
            [
                "benchmark",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--local-ai-client",
                "openai-compatible",
                "--local-base-url",
                "http://127.0.0.1:8000/v1",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert called["config"].local_ai_client == "openai-compatible"
    assert called["config"].qwen_model == "qwen3.6-safe"
    assert called["config"].local_served_model == "qwen3.6-safe-local"


def test_benchmark_openai_compatible_secondary_alias_served_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_ocr_json(tmp_path)
    called = {}

    def fake_runner(config, *, sample=None):
        called["config"] = config
        return {
            "sample_id": config.sample_id,
            "report_json": str(config.output_dir / "report.json"),
            "results": [
                {"backend": "openai-compatible-local", "ok": True, "runtime_sec": 1.0},
                {"backend": "opencode-gpt", "ok": True, "runtime_sec": 2.0},
            ],
        }

    monkeypatch.setattr("spec_pipeline.cli.run_local_vs_gpt_benchmark", fake_runner)

    assert (
        main(
            [
                "benchmark",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--local-ai-client",
                "openai-compatible",
                "--local-model",
                "qwen3.6-27b-safe",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert called["config"].qwen_model == "qwen3.6-27b-safe"
    assert called["config"].local_served_model == "qwen3.6-safe-local"


def test_openai_compatible_served_model_defaults_cover_qwen36_aliases() -> None:
    assert _default_local_served_model("qwen36-safe", "openai-compatible") == "qwen3.6-safe-local"
    assert (
        _default_local_served_model("qwen3_6_27b_gguf_iq3_xxs", "openai-compatible")
        == "qwen3.6-safe-local"
    )
    assert (
        _default_local_served_model("qwen36-balanced", "openai-compatible")
        == "qwen3.6-balanced-local"
    )
    assert (
        _default_local_served_model("qwen3_6_27b_gguf_q4_k_s", "openai-compatible")
        == "qwen3.6-balanced-local"
    )
    assert (
        _default_local_served_model("qwen36-quality", "openai-compatible")
        == "qwen3.6-quality-local"
    )
    assert (
        _default_local_served_model("qwen3_6_27b_gguf_q4_k_m", "openai-compatible")
        == "qwen3.6-quality-local"
    )
    assert _default_local_served_model("qwen36-max", "openai-compatible") == "qwen36-max-local"
    assert _default_local_served_model("qwen36-max-vllm", "openai-compatible") == "qwen36-max-local"
    assert (
        _default_local_served_model("qwen3.6-vllm-experimental", "openai-compatible")
        == "qwen36-max-local"
    )
    assert (
        _default_local_served_model("qwen3_6_27b_gptq_cuda", "openai-compatible")
        == "qwen36-max-local"
    )


def test_samples_json_returns_success() -> None:
    assert main(["samples", "--json"]) == 0


def test_samples_validate_returns_success(tmp_path: Path) -> None:
    manifest_path = _write_cli_manifest(tmp_path)
    assert main(["samples", "--manifest", str(manifest_path), "--validate", "--json"]) == 0


def test_rows_command_returns_success(tmp_path: Path) -> None:
    manifest_path = _write_cli_manifest(tmp_path)
    assert (
        main(["rows", "sample_1", "--manifest", str(manifest_path), "--limit", "1", "--json"]) == 0
    )


def test_normalize_command_returns_success(tmp_path: Path) -> None:
    manifest_path = _write_cli_manifest(tmp_path)
    output_path = tmp_path / "normalized.jsonl"
    assert (
        main(
            [
                "normalize",
                "sample_1",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert output_path.exists()


def test_export_xlsx_command_returns_success(tmp_path: Path) -> None:
    manifest_path = _write_cli_manifest(tmp_path)
    output_path = tmp_path / "exported.xlsx"
    assert (
        main(
            [
                "export-xlsx",
                "sample_1",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert output_path.exists()


def test_export_jsonl_xlsx_command_returns_success(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "rows.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": "Котел",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "1",
                "mass_kg": None,
                "note": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "rows.xlsx"

    assert main(["export-jsonl-xlsx", str(input_jsonl), "--output", str(output_path)]) == 0
    assert output_path.exists()


def test_export_kp_xlsx_command_returns_success(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    workbook = Workbook()
    worksheet = cast(Worksheet, workbook.active)
    worksheet.title = "Quote"
    workbook.create_sheet("Лист2")
    worksheet["B11"] = "Система вентиляции"
    worksheet["B12"] = "ПВ7"
    worksheet["B13"] = "Шаблон"
    worksheet["C13"] = "шт"
    worksheet["C20"] = "Итого стоимость оборудования и материалов:"
    worksheet["C21"] = "Итого стоимость оборудования:"
    worksheet["C22"] = "Итого стоимость материалов:"
    worksheet["C23"] = "Стоимость монтажных работ:"
    worksheet["C24"] = "Расходные и крепежные материалы:"
    worksheet["C25"] = "Транспортные и накладные расходы:"
    worksheet["C26"] = "ИТОГО:"
    worksheet["C27"] = "в т.ч. НДС 22%"
    workbook.save(template)

    input_jsonl = tmp_path / "rows.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "page_ref": "1",
                "section": "ПВ7",
                "subsection": None,
                "pos": "1",
                "name": "Котел",
                "type_mark": "ALIXIA",
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "1",
                "mass_kg": None,
                "note": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "kp.xlsx"

    assert (
        main(
            [
                "export-kp-xlsx",
                str(input_jsonl),
                "--output",
                str(output_path),
                "--template",
                str(template),
            ]
        )
        == 0
    )
    assert output_path.exists()
    exported = load_workbook(output_path)
    assert exported.sheetnames == ["Quote"]


def test_export_kp_xlsx_finds_template_in_output_ancestor_examples(
    tmp_path: Path, monkeypatch
) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    template = examples_dir / "kp-template.xlsx"
    workbook = Workbook()
    worksheet = cast(Worksheet, workbook.active)
    worksheet.title = "Quote"
    worksheet["B11"] = "Система вентиляции"
    worksheet["B12"] = "ПВ7"
    worksheet["B13"] = "Шаблон"
    worksheet["C13"] = "шт"
    worksheet["C20"] = "Итого стоимость оборудования и материалов:"
    worksheet["C21"] = "Итого стоимость оборудования:"
    worksheet["C22"] = "Итого стоимость материалов:"
    worksheet["C23"] = "Стоимость монтажных работ:"
    worksheet["C24"] = "Расходные и крепежные материалы:"
    worksheet["C25"] = "Транспортные и накладные расходы:"
    worksheet["C26"] = "ИТОГО:"
    worksheet["C27"] = "в т.ч. НДС 22%"
    workbook.save(template)

    monkeypatch.setattr("spec_pipeline.cli.PROJECT_ROOT", tmp_path / "missing_system")
    input_jsonl = tmp_path / "rows.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "page_ref": "1",
                "section": "ПВ7",
                "subsection": None,
                "pos": "1",
                "name": "Котел",
                "type_mark": "ALIXIA",
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "1",
                "mass_kg": None,
                "note": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "OUTPUT_RESULTS" / "run" / "result.kp.xlsx"
    output_path.parent.mkdir(parents=True)

    assert main(["export-kp-xlsx", str(input_jsonl), "--output", str(output_path)]) == 0
    assert output_path.exists()


def test_compare_command_returns_success(tmp_path: Path) -> None:
    manifest_path = _write_cli_manifest(tmp_path)
    predicted_path = tmp_path / "predicted.jsonl"
    predicted_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "page_ref": "001",
                "section": "ПВ7",
                "subsection": None,
                "pos": "1",
                "name": "Установка",
                "type_mark": None,
                "code": None,
                "manufacturer": "СВОК",
                "unit": "шт",
                "quantity": "1",
                "mass_kg": None,
                "note": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["compare", "sample_1", str(predicted_path), "--manifest", str(manifest_path)]) == 0


def test_probe_pdf_command_returns_success(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_path = _write_cli_manifest(tmp_path)

    def fake_probe(sample):
        return PdfTextLayerProbe(
            backend="pdfminer-text-probe",
            sample_id=sample.sample_id,
            source_pdf=str(sample.source_pdf),
            page_count=1,
            nonempty_pages=0,
            text_line_count=0,
            text_char_count=0,
            garbled_text_char_count=0,
            garbled_text_ratio=0.0,
            average_chars_per_page=0.0,
            classification="ocr_required",
            needs_ocr=True,
        )

    monkeypatch.setattr("spec_pipeline.cli.probe_pdf_text_layer", fake_probe)

    assert main(["probe-pdf", "sample_1", "--manifest", str(manifest_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_id"] == "sample_1"
    assert payload["classification"] == "ocr_required"
    assert payload["needs_ocr"] is True


def test_extract_ocr_command_returns_success(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "ocr.json"

    class FakeTesseractProvider:
        provider_name = "tesseract"

        def __init__(self, *, config):
            self.config = config

        def extract_pages(self, path, *, pages=None, output_dir=None):
            assert pages == [1, 2]
            return [
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="Позиция Наименование\n1 Диффузор",
                    tokens=[
                        OcrToken(
                            text="Диффузор",
                            page_ref="1",
                            x0=10,
                            y0=20,
                            x1=80,
                            y1=35,
                            confidence=90.0,
                        )
                    ],
                    provider="tesseract",
                    image_path="page_001.png",
                )
            ]

    monkeypatch.setattr("spec_pipeline.cli.TesseractOcrProvider", FakeTesseractProvider)

    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "tesseract",
                "--pages",
                "1-2",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["page_count"] == 1
    assert payload["provider"] == "tesseract"
    assert payload["pages"][0]["tokens"][0]["text"] == "Диффузор"


def test_extract_ocr_command_supports_paddleocr_vl(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "ocr.json"
    local_model_path = tmp_path / "paddle-model"
    local_model_path.mkdir()

    class FakePaddleProvider:
        provider_name = "paddleocr-vl"

        def __init__(self, *, config):
            assert config.model_name == str(local_model_path)
            assert config.device == "gpu:0"
            assert config.dpi == 300
            assert config.progress is True
            assert config.pipeline_version == "v1"
            assert config.vl_rec_backend == "native"
            assert config.require_cuda is True
            assert config.enable_hpi is True
            assert config.use_tensorrt is False
            assert config.precision == "fp16"
            assert config.max_new_tokens == 512
            assert config.repetition_penalty == 1.08
            assert config.temperature == 0.0
            assert config.top_p == 0.95
            assert config.split_wide_pages is True
            assert config.split_band_height == 520
            assert config.split_band_overlap == 80
            assert config.use_layout_detection is False

        def extract_pages(self, path, *, pages=None, output_dir=None):
            assert pages == [1]
            return [
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="[10.0,20.0,80.0,35.0] Диффузор",
                    tokens=[
                        OcrToken(
                            text="Диффузор",
                            page_ref="1",
                            x0=10,
                            y0=20,
                            x1=80,
                            y1=35,
                            confidence=0.9,
                        )
                    ],
                    provider="paddleocr-vl",
                    image_path="page_001.png",
                )
            ]

    monkeypatch.setattr("spec_pipeline.cli.PaddleOcrVlProvider", FakePaddleProvider)

    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "paddleocr-vl",
                "--pages",
                "1",
                "--dpi",
                "300",
                "--paddle-model",
                str(local_model_path),
                "--paddle-device",
                "gpu:0",
                "--paddle-require-cuda",
                "--paddle-enable-hpi",
                "true",
                "--paddle-use-tensorrt",
                "false",
                "--paddle-precision",
                "fp16",
                "--paddle-vl-backend",
                "native",
                "--paddle-max-new-tokens",
                "512",
                "--paddle-repetition-penalty",
                "1.08",
                "--paddle-temperature",
                "0.0",
                "--paddle-top-p",
                "0.95",
                "--paddle-split-wide-pages",
                "true",
                "--paddle-band-height",
                "520",
                "--paddle-band-overlap",
                "80",
                "--paddle-use-layout-detection",
                "false",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "paddleocr-vl"
    assert payload["pages"][0]["tokens"][0]["text"] == "Диффузор"


def test_extract_ocr_command_supports_dots_mocr(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "ocr.json"

    class FakeDotsProvider:
        provider_name = "dots-mocr"

        def __init__(self, *, config):
            assert config.model_name == "test-dots"
            assert config.backend == "transformers"
            assert config.device == "cuda"
            assert config.device_map == "auto"
            assert config.torch_dtype == "float16"
            assert config.attn_implementation == "sdpa"
            assert config.trust_model_code is True
            assert config.model_cache_dir is None
            assert config.dpi == 240
            assert config.upscale == 2.0
            assert config.max_image_edge == 2600
            assert config.max_tokens == 2048
            assert config.crop_to_content is True
            assert config.progress is True

        def extract_pages(self, path, *, pages=None, output_dir=None):
            assert pages == [1]
            return [
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="DOTS_MOCR_HTML:\n<table><tr><td>1.1</td></tr></table>",
                    tokens=[],
                    provider="dots-mocr",
                    image_path="page_001_crop_2x.png",
                )
            ]

    monkeypatch.setattr("spec_pipeline.cli.DotsMocrProvider", FakeDotsProvider)

    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "dots-mocr",
                "--pages",
                "1",
                "--dpi",
                "240",
                "--dots-model",
                "test-dots",
                "--dots-backend",
                "transformers",
                "--dots-device",
                "cuda",
                "--dots-device-map",
                "auto",
                "--dots-torch-dtype",
                "float16",
                "--dots-attn-implementation",
                "sdpa",
                "--dots-trust-model-code",
                "--dots-upscale",
                "2.0",
                "--dots-max-image-edge",
                "2600",
                "--dots-max-tokens",
                "2048",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "dots-mocr"
    assert payload["pages"][0]["image_path"] == "page_001_crop_2x.png"
    assert "DOTS_MOCR_HTML" in payload["pages"][0]["text"]


def test_extract_ocr_auto_uses_pdf_table_layer_without_loading_paddle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_path = tmp_path / "ocr.json"
    table_rows = [
        "PDF_TABLE_ROW "
        + json.dumps(
            {
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "status": None,
                "pos": str(index),
                "name": f"Строка {index}",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "1",
                "mass_kg": None,
                "note": None,
            },
            ensure_ascii=False,
        )
        for index in range(1, 6)
    ]

    class FakePdfTextProvider:
        provider_name = "pdf-text"

        def __init__(self, **kwargs):
            assert kwargs["render_dpi"] == 200

        def extract_pages(self, path, *, pages=None, output_dir=None):
            assert output_dir is None
            return [
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="PDF_TABLE_LAYER:\n" + "\n".join(table_rows),
                    tokens=[],
                    provider="pdf-text",
                    image_path=None,
                )
            ]

    def fail_resolve(*args, **kwargs):
        raise AssertionError("PaddleOCR-VL must not be resolved for searchable PDF")

    monkeypatch.setattr("spec_pipeline.cli.PdfTextOcrProvider", FakePdfTextProvider)
    monkeypatch.setattr("spec_pipeline.cli.resolve_paddleocr_vl_model", fail_resolve)

    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "auto",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "pdf-text"
    assert "PDF_TABLE_LAYER" in payload["pages"][0]["text"]
    assert "skipping PaddleOCR-VL" in capsys.readouterr().out


def test_extract_ocr_auto_logs_pdf_table_route_count(tmp_path: Path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "ocr.json"
    table_row = json.dumps(
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": "1",
            "name": "Строка",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        ensure_ascii=False,
    )

    class FakePdfTextProvider:
        provider_name = "pdf-text"

        def __init__(self, **kwargs):
            assert kwargs["render_dpi"] == 200

        def extract_pages(self, path, *, pages=None, output_dir=None):
            assert output_dir is None
            return [
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="PDF_TABLE_LAYER:\n" + "\n".join(["PDF_TABLE_ROW " + table_row] * 5),
                    tokens=[],
                    provider="pdf-text",
                    image_path=None,
                )
            ]

    monkeypatch.setattr("spec_pipeline.cli.PdfTextOcrProvider", FakePdfTextProvider)

    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "auto",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "auto route PDF_TABLE_ROW count=5" in output
    assert "skipping PaddleOCR-VL" in output


def test_extract_ocr_paddleocr_vl_requires_managed_model(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "extract-ocr",
                str(tmp_path / "input.pdf"),
                "--provider",
                "paddleocr-vl",
                "--model-cache-dir",
                str(tmp_path / "models"),
            ]
        )
        == 1
    )
    assert "models download paddleocr-vl" in capsys.readouterr().err


def test_map_ocr_command_returns_success(tmp_path: Path, monkeypatch, capsys) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    quality_path = tmp_path / "quality.json"
    performance_path = tmp_path / "performance.json"

    class FakeOpenCodeClient:
        def __init__(self, *, config):
            self.config = config

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "1",
                            "name": "Диффузор",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeClient)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(output_path),
                "--quality-json",
                str(quality_path),
                "--performance-json",
                str(performance_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["sample_id"] == "client_sample"
    assert row["name"] == "Диффузор"
    assert row["quantity"] == "2"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["row_count"] == 1
    assert quality["issue_count"] == 0
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    assert performance["sample_id"] == "client_sample"
    assert performance["ai_client"] == "opencode"
    assert performance["ai_model"] == "openai/gpt-5.5"
    assert performance["ai_agent"] == "build"
    assert performance["extraction_mode"] == "ai"
    assert performance["page_count"] == 1
    assert performance["row_count"] == 1
    assert performance["quality_errors"] == 0
    assert performance["quality_warnings"] == 0
    assert performance["runtime_sec"] >= 0
    assert performance["performance_json"] == str(performance_path)
    assert "map-ocr performance:" in output
    assert "extraction_mode:" in output
    assert "rows_per_min:" in output


def _write_minimal_map_ocr_input(tmp_path: Path) -> Path:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ocr_path


def _patch_opencode_rows(monkeypatch, rows: list[dict[str, object]]) -> None:
    class FakeOpenCodeClient:
        def __init__(self, *, config):
            self.config = config

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {"rows": rows, "warnings": [], "raw_response": {}},
            )()

    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeClient)


def test_map_ocr_opencode_failure_returns_clean_error(tmp_path: Path, monkeypatch, capsys) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)

    class FakeOpenCodeClient:
        def __init__(self, *, config):
            self.config = config

        def extract_rows(self, request):
            raise ValueError(
                "OpenCode response did not contain the requested JSON object. "
                "stderr='[multi-auth] No available accounts'"
            )

    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeClient)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "map-ocr: OpenCode/GPT extraction failed" in captured.err
    assert "opencode auth login" in captured.err
    assert "собственный OpenCode/OpenAI аккаунт" in captured.err
    assert "customer-secrets" not in captured.err
    assert "Traceback" not in captured.err


def test_map_ocr_opencode_forbidden_returns_account_network_hint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)

    class FakeOpenCodeClient:
        def __init__(self, *, config):
            self.config = config

        def extract_rows(self, request):
            raise RuntimeError(
                "opencode extraction failed: returncode=1 stderr=Error: Forbidden: "
                "request was blocked by a gateway or proxy. You may not have permission "
                "to access this resource - check your account and provider settings."
            )

    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeClient)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "map-ocr: OpenCode/GPT extraction failed" in captured.err
    assert "запрос отклонен" in captured.err
    assert "сетью/VPN/proxy" in captured.err
    assert "opencode auth login" in captured.err
    assert "текущего пользователя WSL" in captured.err
    assert "customer-secrets" not in captured.err
    assert "Traceback" not in captured.err


def test_map_ocr_quality_gate_fails_on_errors_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    output_path = tmp_path / "mapped.jsonl"
    quality_path = tmp_path / "quality.json"
    performance_path = tmp_path / "performance.json"
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": None,
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "2",
                "mass_kg": None,
                "note": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(output_path),
                "--quality-json",
                str(quality_path),
                "--performance-json",
                str(performance_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()

    assert output_path.exists()
    assert performance_path.exists()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["error_count"] == 1
    assert "map-ocr quality gate failed" in captured.err


def test_map_ocr_quality_gate_can_fail_on_warnings(tmp_path: Path, monkeypatch, capsys) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": "057x3.5",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "wm",
                "quantity": "2",
                "mass_kg": None,
                "note": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "default.jsonl"),
                "--quality-json",
                str(tmp_path / "default.quality.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "strict.jsonl"),
                "--quality-json",
                str(tmp_path / "strict.quality.json"),
                "--quality-fail-on",
                "warning",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    quality = json.loads((tmp_path / "strict.quality.json").read_text(encoding="utf-8"))

    assert quality["error_count"] == 0
    assert quality["warning_count"] == 2
    assert "map-ocr quality gate failed" in captured.err


def test_map_ocr_quality_gate_allows_export_normalized_warnings(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": "057x3.5",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "wm",
                "quantity": "2",
                "mass_kg": None,
                "note": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "mapped.jsonl"),
                "--quality-json",
                str(tmp_path / "quality.json"),
                "--quality-fail-on",
                "uncorrected-warning",
            ]
        )
        == 0
    )
    quality = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert quality["warning_count"] == 2


def test_map_ocr_quality_gate_fails_on_uncorrected_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "PAGE 1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": "Диффузор",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "шт",
                "quantity": "2",
                "mass_kg": None,
                "note": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "mapped.jsonl"),
                "--quality-json",
                str(tmp_path / "quality.json"),
                "--quality-fail-on",
                "uncorrected-warning",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "map-ocr quality gate failed" in captured.err


def test_map_ocr_quality_gate_fails_when_repairable_and_uncorrected_warnings_mix(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "PAGE 1",
                "section": None,
                "subsection": None,
                "pos": "1",
                "name": "Кран $ \\Phi $15",
                "type_mark": None,
                "code": None,
                "manufacturer": None,
                "unit": "$ \\omega $m",
                "quantity": "2",
                "mass_kg": None,
                "note": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "mapped.jsonl"),
                "--quality-json",
                str(tmp_path / "quality.json"),
                "--quality-fail-on",
                "uncorrected-warning",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    quality = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))

    assert quality["error_count"] == 0
    assert quality["warning_count"] == 3
    assert "map-ocr quality gate failed" in captured.err


def test_map_ocr_pdf_table_layer_repairable_warnings_export_clean_excel(
    tmp_path: Path,
) -> None:
    ocr_path = tmp_path / "ocr.json"
    output_jsonl = tmp_path / "mapped.jsonl"
    quality_path = tmp_path / "quality.json"
    output_xlsx = tmp_path / "mapped.xlsx"
    table_rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1",
            "name": "Кран $ \\Phi $15",
            "type_mark": "BVR-R- $ \\Phi $15",
            "code": None,
            "manufacturer": "Danfoss",
            "unit": "$ \\omega $m",
            "quantity": "4",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "2",
            "name": "₽50₽25",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "wm",
            "quantity": "168",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "3",
            "name": "Диффузор",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4",
            "name": "Клапан",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "5",
            "name": "Решетка",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
    ]
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "pdf-text",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "PDF_TABLE_LAYER:\n"
                        + "\n".join(
                            "PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False)
                            for row in table_rows
                        ),
                        "tokens": [],
                        "provider": "pdf-text",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_jsonl),
                "--quality-json",
                str(quality_path),
                "--quality-fail-on",
                "uncorrected-warning",
            ]
        )
        == 0
    )
    assert main(["export-jsonl-xlsx", str(output_jsonl), "--output", str(output_xlsx)]) == 0

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    worksheet = cast(Worksheet, load_workbook(output_xlsx).active)

    assert quality["error_count"] == 0
    assert quality["warning_count"] == 5
    assert worksheet[2][1].value == "Кран Ø15 BVR-R-Ø15 Danfoss"
    assert worksheet[2][2].value == "шт"
    assert worksheet[3][1].value == "Ø50x25"
    assert worksheet[3][2].value == "шт"


def test_pdf_table_layer_merges_split_quantity_tail_and_note() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1.1",
            "name": "100 (δ=0,5 мм)",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "м/м²",
            "quantity": "55/17,2",
            "mass_kg": None,
            "note": "Воздушные",
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": "7",
            "mass_kg": None,
            "note": "затворы",
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["quantity"] == "55/17,27"
    assert extracted[0]["note"] == "Воздушные затворы"


def test_pdf_table_layer_merges_note_only_tail() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4.1",
            "name": "Воздухозаборная решетка наружная 900х400",
            "type_mark": "РН",
            "code": None,
            "manufacturer": None,
            "unit": "шт.",
            "quantity": "1",
            "mass_kg": None,
            "note": "покрасить в RAL",
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": "см. фасад задния",
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["note"] == "покрасить в RAL см. фасад задния"


def test_pdf_table_layer_keeps_positioned_composition_header_without_quantity() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "В9.1",
            "name": "Вытяжная установка, в составе:",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "- вентилятор, 0,07 кВт",
            "type_mark": "Канал-ВЕНТ-100",
            "code": None,
            "manufacturer": "Веза",
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 2
    assert extracted[0]["pos"] == "В9.1"
    assert extracted[0]["name"] == "Вытяжная установка, в составе:"
    assert extracted[0]["unit"] is None
    assert extracted[0]["quantity"] is None
    assert extracted[1]["name"] == "- вентилятор, 0,07 кВт"


def test_pdf_table_layer_expands_ditto_note_from_previous_row() -> None:
    rows = [
        {
            "page_ref": "11",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4.1 Воздухозаборная решетка наружная 900х400",
            "name": None,
            "type_mark": "РН",
            "code": None,
            "manufacturer": None,
            "unit": "шт.",
            "quantity": "1",
            "mass_kg": None,
            "note": "покрасить в RAL",
        },
        {
            "page_ref": "11",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4.2 Воздухозаборная решетка наружная 1650х1200",
            "name": None,
            "type_mark": "РН",
            "code": None,
            "manufacturer": None,
            "unit": "шт.",
            "quantity": "4",
            "mass_kg": None,
            "note": "-//-",
        },
    ]
    page = OcrPage(
        page_index=10,
        page_ref="11",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 2
    assert extracted[0]["note"] == "покрасить в RAL"
    assert extracted[1]["note"] == "покрасить в RAL"


def test_pdf_table_layer_repairs_unit_embedded_quantity_and_mass_tail() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "160 (δ=1 мм)",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт./м² 455/56,",
            "quantity": "14",
            "mass_kg": None,
            "note": None,
        }
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["unit"] == "шт./м²"
    assert extracted[0]["quantity"] == "455"
    assert extracted[0]["mass_kg"] == "56,14"


def test_pdf_table_layer_repairs_quantity_shifted_into_mass_column() -> None:
    row = {
        "page_ref": "1",
        "section": None,
        "subsection": None,
        "status": None,
        "pos": '11 Стальной панельный радиатор "Лидея"',
        "name": None,
        "type_mark": "ЛУ22-506",
        "code": None,
        "manufacturer": None,
        "unit": "шт",
        "quantity": None,
        "mass_kg": "894",
        "note": None,
    }
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\nPDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["pos"] == "11"
    assert extracted[0]["name"] == 'Стальной панельный радиатор "Лидея"'
    assert extracted[0]["quantity"] == "894"
    assert extracted[0]["mass_kg"] is None


def test_pdf_table_layer_uses_code_like_pos_as_name_when_item_cell_shifted() -> None:
    row = {
        "page_ref": "1",
        "section": None,
        "subsection": None,
        "status": None,
        "pos": "КПУ-1Н-О-Н-100-MV220",
        "name": None,
        "type_mark": None,
        "code": None,
        "manufacturer": None,
        "unit": "шт.",
        "quantity": "2",
        "mass_kg": None,
        "note": None,
    }
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\nPDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["pos"] is None
    assert extracted[0]["name"] == "КПУ-1Н-О-Н-100-MV220"
    assert extracted[0]["unit"] == "шт."
    assert extracted[0]["quantity"] == "2"


def test_pdf_table_layer_splits_leading_decimal_pos_when_name_has_tail() -> None:
    row = {
        "page_ref": "16",
        "section": None,
        "subsection": None,
        "status": None,
        "pos": "3.34 Теплоогнезащитная изоляция с покрытием алюминиевой фольгой",
        "name": "s=40 мм",
        "type_mark": "PRO-VENT-40-1НФ",
        "code": None,
        "manufacturer": None,
        "unit": "м²",
        "quantity": "119,04",
        "mass_kg": None,
        "note": None,
    }
    page = OcrPage(
        page_index=15,
        page_ref="16",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\nPDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["pos"] == "3.34"
    assert extracted[0]["name"] == (
        "Теплоогнезащитная изоляция с покрытием алюминиевой фольгой s=40 мм"
    )


def test_pdf_table_layer_merges_type_mark_tail_without_blank_item() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1",
            "name": "Клапан",
            "type_mark": "КЛАБ-200-0*ф-Н-LF24-SR-",
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "2",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "При монтаже количество труб систем",
            "type_mark": None,
            "code": "Изм. Кол.уч. Лист",
            "manufacturer": "№док. Подп. Дата",
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "отопления может отличатся от",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": "Стадия Лист",
            "note": "Листов",
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "проектного значения.",
            "type_mark": None,
            "code": "ГИП",
            "manufacturer": None,
            "unit": "II этап строительства Жилой дом №2",
            "quantity": None,
            "mass_kg": None,
            "note": "Р 1 2",
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": None,
            "type_mark": "V-0-0",
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["type_mark"] == "КЛАБ-200-0*ф-Н-LF24-SR-V-0-0"


def test_pdf_table_layer_extracts_leading_decimal_position_from_name() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "1.4 Отводы прямоугольного сечения",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт./м²",
            "quantity": "455",
            "mass_kg": None,
            "note": None,
        }
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["pos"] == "1.4"
    assert extracted[0]["name"] == "Отводы прямоугольного сечения"


def test_pdf_table_layer_extracts_leading_integer_position_from_name() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": (
                "4 1200х550 в комплекте с электроприводом Siemens на 220В "
                "Клапан противопожарный канальный нормально закрытый морозостойкий EI90"
            ),
            "type_mark": "LKF-1-90-НЗ-SR220-1200x550-MZ",
            "code": None,
            "manufacturer": "LUFTKON",
            "unit": "шт.",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": (
                "10 двигателя вентилятора от внешнего воздействия на кровле "
                "Кожух из оцинкованной стали толщиной δ=0,7мм для защиты воздуховодов и"
            ),
            "type_mark": "ГОСТ 14918-80",
            "code": None,
            "manufacturer": None,
            "unit": "м²",
            "quantity": "18.0",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "12 воздуховодов Забивка строительных отверстий минеральной ватой после монтажа",
            "type_mark": "Техно",
            "code": None,
            "manufacturer": "Технониколь",
            "unit": "м³",
            "quantity": "1.5",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["pos"] == "4"
    assert extracted[0]["name"] == (
        "Клапан противопожарный канальный нормально закрытый морозостойкий EI90 "
        "1200х550 в комплекте с электроприводом Siemens на 220В"
    )
    assert extracted[1]["pos"] == "10"
    assert extracted[1]["name"] == (
        "Кожух из оцинкованной стали толщиной δ=0,7мм для защиты воздуховодов и "
        "двигателя вентилятора от внешнего воздействия на кровле"
    )
    assert extracted[2]["pos"] == "12"
    assert extracted[2]["name"] == (
        "Забивка строительных отверстий минеральной ватой после монтажа воздуховодов"
    )


def test_pdf_table_layer_repairs_tail_text_before_item_name() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "2",
            "name": ("с электродвигателем N=2,2квт, n=1390об/мин Наружная решетка сеч.1000х500"),
            "type_mark": "A90L4 GA1000x500",
            "code": None,
            "manufacturer": "РусКлимат",
            "unit": "шт",
            "quantity": "2",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": (
                "дефлекторов на шахту 3500х3500 по наружному размеру "
                "Переход прямоугольно-треугольного сечения с утеплением для монтажа"
            ),
            "type_mark": "ППТС(У)3-9(800) 3500*3500",
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "2",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[0]["name"] == (
        "Наружная решетка сеч.1000х500 с электродвигателем N=2,2квт, n=1390об/мин"
    )
    assert extracted[1]["name"] == (
        "Переход прямоугольно-треугольного сечения с утеплением для монтажа "
        "дефлекторов на шахту 3500х3500 по наружному размеру"
    )


def test_pdf_table_layer_keeps_to_same_rows_with_compound_quantity() -> None:
    rows = [
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "8 Воздуховод из оцинкованной стали толщиной δ=0,9мм, 1200х550",
            "name": None,
            "type_mark": "ГОСТ 14918-80*",
            "code": None,
            "manufacturer": "Россия",
            "unit": "п.м/м²",
            "quantity": "85,0/ 289,0",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "то же, δ=0,9мм, ∅710",
            "name": None,
            "type_mark": "ГОСТ 14918-80*",
            "code": None,
            "manufacturer": "Россия",
            "unit": "п.м/м²",
            "quantity": "4,0/ 8,92",
            "mass_kg": None,
            "note": "с отводом ∅710",
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="3",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 2
    assert extracted[1]["pos"] is None
    assert extracted[1]["name"] == "то же, δ=0,9мм, ∅710"
    assert extracted[1]["quantity"] == "4,0/ 8,92"
    assert extracted[1]["note"] == "с отводом ∅710"


def test_pdf_table_layer_uses_pos_only_system_rows_as_sections_and_lowercase_as_continuation() -> (
    None
):
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "Вентиляция дома",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1 Канальный вентилятор",
            "name": None,
            "type_mark": "LKP-600x350",
            "code": None,
            "manufacturer": "Люфткон",
            "unit": "шт",
            "quantity": "18+(18-резерв)",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "с регулятором скорости",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "Система ВЕпуи",
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "Воздуховод из тонколистовой оц. стали",
            "name": None,
            "type_mark": "100х200",
            "code": "ГОСТ 14918-2020",
            "manufacturer": None,
            "unit": "п.м.",
            "quantity": "61",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert extracted[1]["name"] == "Канальный вентилятор с регулятором скорости"
    assert extracted[1]["section"] == "Вентиляция дома"
    assert any(row["section"] == "Система ВЕпуи" for row in extracted)
    assert extracted[-1]["section"] == "Система ВЕпуи"
    assert extracted[-1]["name"] == "Воздуховод из тонколистовой оц. стали"


def test_pdf_table_layer_merges_prefix_description_into_next_quantity_row() -> None:
    rows = [
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "Pс=350Па, с электродвигателем N=11кВт Вентилятор радиальный дымоудаления",
            "name": None,
            "type_mark": "VR 80-75-710-D-2h/400-",
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": "КП №DEMO-0042",
        },
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1 управления и автоматики",
            "name": None,
            "type_mark": "11/1500-ПР0-У1 А132М4",
            "code": None,
            "manufacturer": "LUFTKON",
            "unit": "компл.",
            "quantity": "1",
            "mass_kg": None,
            "note": "тел. DEMO-PHONE-0000",
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="3",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["pos"] == "1"
    assert extracted[0]["name"] == (
        "Вентилятор радиальный дымоудаления управления и автоматики "
        "Pс=350Па, с электродвигателем N=11кВт"
    )
    assert extracted[0]["type_mark"] == "VR 80-75-710-D-2h/400-11/1500-ПР0-У1 А132М4"
    assert extracted[0]["note"] == "КП №DEMO-0042 тел. DEMO-PHONE-0000"


def test_pdf_table_layer_merges_numbered_prefix_description_into_next_quantity_row() -> None:
    rows = [
        {
            "page_ref": "6",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "22 Дымовая труба из нержавеющей стали",
            "name": "Ø400  В тепловой изоляции б =60 мм с",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "6",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "покровным слоем из тонколистовой стали б",
            "name": "=1.0 мм Н =82.60 м",
            "type_mark": None,
            "code": None,
            "manufacturer": 'ООО"Демо Канал"',
            "unit": "шт",
            "quantity": "18",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=5,
        page_ref="6",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["pos"] == "22"
    assert extracted[0]["name"] == (
        "Дымовая труба из нержавеющей стали Ø400 В тепловой изоляции б =60 мм с "
        "покровным слоем из тонколистовой стали б =1.0 мм Н =82.60 м"
    )
    assert extracted[0]["manufacturer"] == 'ООО"Демо Канал"'
    assert extracted[0]["unit"] == "шт"
    assert extracted[0]["quantity"] == "18"


def test_pdf_table_layer_keeps_numbered_section_before_child_item() -> None:
    rows = [
        {
            "page_ref": "11",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4",
            "name": "Решетки",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "11",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "4.1 Воздухозаборная решетка наружная 900х400",
            "name": None,
            "type_mark": "РН",
            "code": None,
            "manufacturer": None,
            "unit": "шт.",
            "quantity": "1",
            "mass_kg": None,
            "note": "покрасить в RAL",
        },
    ]
    page = OcrPage(
        page_index=10,
        page_ref="11",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 2
    assert extracted[0]["section"] == "Решетки"
    assert extracted[0]["name"] is None
    assert extracted[1]["section"] == "Решетки"
    assert extracted[1]["pos"] == "4.1"
    assert extracted[1]["name"] == "Воздухозаборная решетка наружная 900х400"
    assert extracted[1]["unit"] == "шт."
    assert extracted[1]["quantity"] == "1"


def test_pdf_table_layer_keeps_family_position_when_merging_dimension_row() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1.1 Воздуховод из оцинкованной стали круглого сечения:",
            "name": None,
            "type_mark": "ГОСТ 14918-80*",
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "100 (δ=0,5 мм)",
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "м/м²",
            "quantity": "210/65,94",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["pos"] == "1.1"
    assert extracted[0]["name"] == (
        "Воздуховод из оцинкованной стали круглого сечения: 100 (δ=0,5 мм)"
    )


def test_pdf_table_layer_drops_project_footer_noise_rows() -> None:
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "Проектом допускается замена указанного в спецификации оборудования",
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": "15-ПД",
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "аналогичные по своим характеристикам оборудование и материалы других производителей",
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": None,
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": "Строительство жилого дома по адресу:",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "1 Клапан",
            "name": None,
            "type_mark": "КЛАБ",
            "code": None,
            "manufacturer": None,
            "unit": "шт.",
            "quantity": "2",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="1",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["name"] == "Клапан"


def test_pdf_table_layer_merges_dangling_installation_continuation() -> None:
    rows = [
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "2.1 Приточно-вытяжная установка П1В1 (П1: 400 м³/ч, 220 Па, В1:",
            "name": None,
            "type_mark": "Канал-ВЕНТ-200 +",
            "code": None,
            "manufacturer": "ВЕЗА",
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": "П1В1",
        },
        {
            "page_ref": "3",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "400 м³/ч, 240 Па) в составе:",
            "name": None,
            "type_mark": "Канал-ВЕНТ-200",
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="3",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["pos"] == "2.1"
    assert "В1: 400 м³/ч, 240 Па) в составе:" in extracted[0]["name"]


def test_pdf_table_layer_merges_item_family_header_into_first_item() -> None:
    rows = [
        {
            "page_ref": "2",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": None,
            "name": "Стальной панельный радиатор нижним подключением",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "2",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "правое подключение тип 22, высота 300 мм",
            "name": None,
            "type_mark": "VC 22 300х400",
            "code": None,
            "manufacturer": "Royal Thermo",
            "unit": "шт.",
            "quantity": "47",
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="2",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["name"] == (
        "Стальной панельный радиатор нижним подключением правое подключение тип 22, высота 300 мм"
    )


def test_pdf_table_layer_merges_lowercase_suffix_into_previous_quantity_row() -> None:
    rows = [
        {
            "page_ref": "4",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "ДУ1 Вентилятор дымоудаления крышный с вертикальным",
            "name": None,
            "type_mark": "УКРОС91-071-ДУ400-Н-",
            "code": None,
            "manufacturer": "ВЕЗА",
            "unit": "шт.",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
        },
        {
            "page_ref": "4",
            "section": None,
            "subsection": None,
            "status": None,
            "pos": "выбросом",
            "name": None,
            "type_mark": "00300/6-У1",
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
        },
    ]
    page = OcrPage(
        page_index=0,
        page_ref="4",
        width=1191,
        height=842,
        text="PDF_TABLE_LAYER:\n"
        + "\n".join("PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows),
        tokens=[],
        provider="pdf-text",
        image_path=None,
    )

    extracted = _ai_rows_from_pdf_table_layer([page])

    assert len(extracted) == 1
    assert extracted[0]["pos"] == "ДУ1"
    assert extracted[0]["name"] == "Вентилятор дымоудаления крышный с вертикальным выбросом"
    assert extracted[0]["type_mark"] == "УКРОС91-071-ДУ400-Н-00300/6-У1"


def test_map_ocr_applies_third_layer_patch_for_scan_semantic_garbage(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "paddleocr-vl",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "Автоматический воздухомобюксир 015 Pugah",
                        "tokens": [],
                        "provider": "paddleocr-vl",
                        "image_path": str(image_path),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls = {"review": 0}

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            self.config = config

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "3",
                            "name": "Автоматический воздухомобюксир 015",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": "Pugah",
                            "unit": "шт",
                            "quantity": "84",
                            "mass_kg": None,
                            "note": None,
                            "status": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    class FakeOpenCodeThirdLayer:
        def __init__(self, *, config):
            self.config = config

        def review_row_patches(self, *, request, rows, issues):
            calls["review"] += 1
            return type(
                "FakePatchResult",
                (),
                {
                    "patches": [
                        {
                            "row_index": 1,
                            "page_ref": "1",
                            "field": "name",
                            "old_value": "Автоматический воздухомобюксир 015",
                            "new_value": "Автоматический воздухоотводчик Ø15",
                            "confidence": "high",
                            "reason": "visible on page",
                        },
                        {
                            "row_index": 1,
                            "page_ref": "1",
                            "field": "manufacturer",
                            "old_value": "Pugah",
                            "new_value": "Bugatti",
                            "confidence": "high",
                            "reason": "visible on page",
                        },
                    ],
                    "warnings": [],
                    "raw_response": {"third_layer": True},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )
    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeThirdLayer)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(tmp_path / "mapped.jsonl"),
                "--quality-json",
                str(tmp_path / "quality.json"),
                "--ai-client",
                "openai-compatible",
                "--quality-fail-on",
                "uncorrected-warning",
                "--third-layer-reviewer",
                "opencode-gpt",
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "mapped.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    quality = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))

    assert calls["review"] == 1
    assert rows[0]["name"] == "Автоматический воздухоотводчик Ø15"
    assert rows[0]["manufacturer"] == "Bugatti"
    assert quality["warning_count"] == 0


def test_map_ocr_blocks_semantic_scan_garbage_when_vision_repair_unavailable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "paddleocr-vl",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "Автоматический воздухомобюксир 015 Pugah",
                        "tokens": [],
                        "provider": "paddleocr-vl",
                        "image_path": str(image_path),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _patch_opencode_rows(
        monkeypatch,
        [
            {
                "page_ref": "1",
                "section": None,
                "subsection": None,
                "pos": "3",
                "name": "Автоматический воздухомобюксир 015",
                "type_mark": None,
                "code": None,
                "manufacturer": "Pugah",
                "unit": "шт",
                "quantity": "84",
                "mass_kg": None,
                "note": None,
                "status": None,
            }
        ],
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(tmp_path / "mapped.jsonl"),
                "--quality-json",
                str(tmp_path / "quality.json"),
                "--quality-fail-on",
                "uncorrected-warning",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    quality = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))

    assert quality["warning_count"] == 2
    assert "map-ocr quality gate failed" in captured.err


def test_map_ocr_openai_compatible_defaults_to_qwen36_safe_model(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            assert config.model == "qwen3.6-safe"
            assert config.base_url == "http://127.0.0.1:8000/v1"
            assert config.served_model_name == "qwen3.6-safe-local"

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "1",
                            "name": "Диффузор",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
            ]
        )
        == 0
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["name"] == "Диффузор"


def test_map_ocr_openai_compatible_batches_pages_and_limits_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 2,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                    {
                        "page_index": 1,
                        "page_ref": "2",
                        "width": 100,
                        "height": 200,
                        "text": "2 Клапан шт 3",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    calls = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            assert config.max_tokens == 1024

        def extract_rows(self, request):
            calls.append([page.page_ref for page in request.pages])
            page_ref = request.pages[0].page_ref
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": page_ref,
                            "section": None,
                            "subsection": None,
                            "pos": page_ref,
                            "name": "Диффузор" if page_ref == "1" else "Клапан",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2" if page_ref == "1" else "3",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {"page_ref": page_ref},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-max-tokens",
                "1024",
                "--ai-page-batch-size",
                "1",
            ]
        )
        == 0
    )
    assert calls == [["1"], ["2"]]
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["name"] for row in rows] == ["Диффузор", "Клапан"]


def test_map_ocr_uses_pdf_table_fallback_when_local_ai_returns_too_few_rows(
    tmp_path: Path, monkeypatch
) -> None:
    table_rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "Общеобменная вентиляция",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "Жилая часть",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": "В1.1",
            "name": "Вентилятор",
            "type_mark": "IN 9/3,5",
            "code": None,
            "manufacturer": "O.ERRE",
            "unit": "шт",
            "quantity": "18",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": "В2.1",
            "name": "Вентилятор",
            "type_mark": "IN 10/4",
            "code": None,
            "manufacturer": "O.ERRE",
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "Клапан приточный",
            "type_mark": "КИВ-125",
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "107",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "Решетка",
            "type_mark": "АЛН 100х150",
            "code": None,
            "manufacturer": "Арктос",
            "unit": "шт",
            "quantity": "62",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "Воздуховод из тонколистовой оцинкованной стали",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": None,
            "quantity": None,
            "mass_kg": None,
            "note": None,
            "status": None,
        },
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": None,
            "name": "класса герметичности В, Ф125",
            "type_mark": "ГОСТ 24751-81",
            "code": None,
            "manufacturer": None,
            "unit": "м",
            "quantity": "281",
            "mass_kg": None,
            "note": None,
            "status": None,
        },
    ]
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "PDF_TABLE_LAYER:\n"
                        + "\n".join(
                            "PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False)
                            for row in table_rows
                        ),
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            pass

        def extract_rows(self, request):
            return type(
                "FakeResult",
                (),
                {
                    "rows": [table_rows[2]],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 7
    assert rows[0]["section"] == "Общеобменная вентиляция"
    assert rows[0]["name"] is None
    assert rows[1]["section"] == "Общеобменная вентиляция"
    assert rows[1]["subsection"] == "Жилая часть"
    assert rows[2]["section"] == "Общеобменная вентиляция"
    assert rows[2]["subsection"] == "Жилая часть"
    assert (
        rows[-1]["name"]
        == "Воздуховод из тонколистовой оцинкованной стали класса герметичности В, Ф125"
    )


def test_map_ocr_force_ai_on_pdf_table_layer_runs_ai(tmp_path: Path, monkeypatch) -> None:
    table_rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": str(index),
            "name": f"PDF row {index}",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": "1",
            "mass_kg": None,
            "note": None,
            "status": None,
        }
        for index in range(1, 6)
    ]
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "PDF_TABLE_LAYER:\n"
                        + "\n".join(
                            "PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False)
                            for row in table_rows
                        ),
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    calls: list[str] = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            pass

        def extract_rows(self, request):
            calls.append(request.sample_id)
            ai_rows = [dict(row) for row in table_rows]
            ai_rows[0]["name"] = "AI corrected row"
            return type(
                "FakeResult",
                (),
                {
                    "rows": ai_rows,
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-force-on-pdf-table-layer",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert calls == ["client_sample-pages-1"]
    assert len(rows) == 5
    assert rows[0]["name"] == "AI corrected row"


def test_map_ocr_pdf_table_layer_can_use_third_layer_patch_reviewer(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake")
    rows = [
        {
            "page_ref": "1",
            "section": None,
            "subsection": None,
            "pos": str(index),
            "name": "Диффузор",
            "type_mark": None,
            "code": None,
            "manufacturer": None,
            "unit": "шт",
            "quantity": None if index == 1 else "1",
            "mass_kg": None,
            "note": None,
            "status": None,
        }
        for index in range(1, 6)
    ]
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "PDF_TABLE_LAYER:\n"
                        + "\n".join(
                            "PDF_TABLE_ROW " + json.dumps(row, ensure_ascii=False) for row in rows
                        ),
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": str(image_path),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"

    class FakeOpenCodeClient:
        def __init__(self, *, config):
            pass

        def review_row_patches(self, *, request, rows, issues):
            assert request.pages[0].image_path == str(image_path)
            assert issues[0].message == "row has unit but no quantity"
            return type(
                "FakePatchResult",
                (),
                {
                    "patches": [
                        {
                            "row_index": 1,
                            "page_ref": "1",
                            "field": "quantity",
                            "old_value": None,
                            "new_value": "2",
                            "confidence": "high",
                            "reason": "visible on page",
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr("spec_pipeline.cli.OpenCodeExtractionClient", FakeOpenCodeClient)

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--third-layer-reviewer",
                "opencode-gpt",
                "--third-layer-model",
                "openai/gpt-5.5",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["quantity"] == "2"


def test_map_ocr_openai_compatible_dynamic_batches_multiple_pages(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 2,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                    {
                        "page_index": 1,
                        "page_ref": "2",
                        "width": 100,
                        "height": 200,
                        "text": "2 Клапан шт 3",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    calls = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            assert config.max_tokens == 2048

        def extract_rows(self, request):
            page_refs = [page.page_ref for page in request.pages]
            calls.append(page_refs)
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": page_ref,
                            "section": None,
                            "subsection": None,
                            "pos": page_ref,
                            "name": f"Item {page_ref}",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "1",
                            "mass_kg": None,
                            "note": None,
                        }
                        for page_ref in page_refs
                    ],
                    "warnings": [],
                    "raw_response": {"page_refs": page_refs},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-max-tokens",
                "2048",
                "--ai-context-tokens",
                "12288",
            ]
        )
        == 0
    )
    assert calls == [["1", "2"]]


def test_map_ocr_openai_compatible_splits_batch_on_context_error(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 2,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                    {
                        "page_index": 1,
                        "page_ref": "2",
                        "width": 100,
                        "height": 200,
                        "text": "2 Клапан шт 3",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    calls = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            pass

        def extract_rows(self, request):
            page_refs = [page.page_ref for page in request.pages]
            calls.append(page_refs)
            if len(page_refs) > 1:
                raise RuntimeError("context_length_exceeded")
            page_ref = page_refs[0]
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": page_ref,
                            "section": None,
                            "subsection": None,
                            "pos": page_ref,
                            "name": f"Item {page_ref}",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "1",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {"page_refs": page_refs},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-max-tokens",
                "2048",
                "--ai-context-tokens",
                "12288",
            ]
        )
        == 0
    )
    assert calls == [["1", "2"], ["1"], ["2"]]


def test_map_ocr_openai_compatible_splits_batch_on_timeout(tmp_path: Path, monkeypatch) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps(
            {
                "page_count": 2,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                    {
                        "page_index": 1,
                        "page_ref": "2",
                        "width": 100,
                        "height": 200,
                        "text": "2 Клапан шт 3",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapped.jsonl"
    calls = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            pass

        def extract_rows(self, request):
            page_refs = [page.page_ref for page in request.pages]
            calls.append(page_refs)
            if len(page_refs) > 1:
                raise RuntimeError("opencode extraction timed out: timeout_seconds=1800")
            page_ref = page_refs[0]
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": page_ref,
                            "section": None,
                            "subsection": None,
                            "pos": page_ref,
                            "name": f"Item {page_ref}",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "1",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {"page_refs": page_refs},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-max-tokens",
                "2048",
                "--ai-context-tokens",
                "12288",
            ]
        )
        == 0
    )
    assert calls == [["1", "2"], ["1"], ["2"]]


def _make_retry_fake_opencode_client(
    calls: list[tuple[str | None, list[str]]], *, succeed_on_call: int
):
    from dataclasses import replace as dataclass_replace

    class FakeOpenCodeClient:
        def __init__(self, *, config):
            self.config = config

        def with_reduced_reasoning(self):
            ladder = ("xhigh", "high", "medium", "low")
            variant = (self.config.variant or "").strip().lower()
            if variant not in ladder or variant == ladder[-1]:
                return None
            reduced = dataclass_replace(self.config, variant=ladder[ladder.index(variant) + 1])
            return FakeOpenCodeClient(config=reduced)

        def extract_rows(self, request):
            calls.append((self.config.variant, [page.page_ref for page in request.pages]))
            if succeed_on_call < 0 or len(calls) < succeed_on_call:
                raise RuntimeError(
                    "opencode extraction timed out: timeout_seconds=1800 command=opencode run"
                )
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "1",
                            "name": "Диффузор",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    return FakeOpenCodeClient


def test_map_ocr_opencode_single_page_timeout_retries_with_reduced_reasoning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    output_path = tmp_path / "mapped.jsonl"
    calls: list[tuple[str | None, list[str]]] = []
    monkeypatch.setattr(
        "spec_pipeline.cli.OpenCodeExtractionClient",
        _make_retry_fake_opencode_client(calls, succeed_on_call=2),
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--output",
                str(output_path),
                "--ai-reasoning-effort",
                "xhigh",
            ]
        )
        == 0
    )
    assert calls == [("xhigh", ["1"]), ("high", ["1"])]
    captured = capsys.readouterr()
    assert "retrying attempt 2/3" in captured.out
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["name"] == "Диффузор"


def test_map_ocr_opencode_single_page_timeout_fails_after_three_attempts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    calls: list[tuple[str | None, list[str]]] = []
    monkeypatch.setattr(
        "spec_pipeline.cli.OpenCodeExtractionClient",
        _make_retry_fake_opencode_client(calls, succeed_on_call=-1),
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--ai-client",
                "opencode",
                "--ai-reasoning-effort",
                "xhigh",
            ]
        )
        == 1
    )
    assert calls == [("xhigh", ["1"]), ("high", ["1"]), ("medium", ["1"])]
    captured = capsys.readouterr()
    assert "timed out after 3 attempts" in captured.err
    assert "Подсказка: страница не успела обработаться" in captured.err
    assert "Traceback" not in captured.err


def test_map_ocr_single_page_timeout_retries_without_reasoning_downgrade_support(
    tmp_path: Path, monkeypatch
) -> None:
    ocr_path = _write_minimal_map_ocr_input(tmp_path)
    output_path = tmp_path / "mapped.jsonl"
    calls = []

    class FakeOpenAiCompatibleClient:
        def __init__(self, *, config):
            pass

        def extract_rows(self, request):
            calls.append([page.page_ref for page in request.pages])
            if len(calls) == 1:
                raise RuntimeError("request timed out after 1800 seconds")
            return type(
                "FakeResult",
                (),
                {
                    "rows": [
                        {
                            "page_ref": "1",
                            "section": None,
                            "subsection": None,
                            "pos": "1",
                            "name": "Диффузор",
                            "type_mark": None,
                            "code": None,
                            "manufacturer": None,
                            "unit": "шт",
                            "quantity": "2",
                            "mass_kg": None,
                            "note": None,
                        }
                    ],
                    "warnings": [],
                    "raw_response": {},
                },
            )()

    monkeypatch.setattr(
        "spec_pipeline.cli.OpenAiCompatibleExtractionClient", FakeOpenAiCompatibleClient
    )

    assert (
        main(
            [
                "map-ocr",
                "client_sample",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(tmp_path / "source.pdf"),
                "--output",
                str(output_path),
                "--ai-client",
                "openai-compatible",
                "--ai-max-tokens",
                "2048",
            ]
        )
        == 0
    )
    assert calls == [["1"], ["1"]]


def test_process_pdf_heredocs_do_not_self_close() -> None:
    """Regression guard for the v83 GPT smoke failure.

    A single-quoted bash heredoc whose FIRST body line equals its delimiter
    closes immediately (empty body); every following message line is then
    parsed as live shell code. That is exactly why the generated
    ``_gpt-opencode-smoke.sh`` blew up with ``line 26: syntax error`` on the
    customer's machine (delimiter ``SPEC_OCR_GPT_AUTH_STALE`` collided with its
    own banner line). This test fails fast if any heredoc in the script ever
    re-introduces that collision.

    Unlike ``test_windows_bootstrap_scripts_are_packaged`` this guard does not
    require the bundled Windows binaries, so it actually runs everywhere.
    """
    import re

    project_root = Path(__file__).resolve().parents[1]
    lines = (project_root / "process-pdf-windows.ps1").read_text(encoding="utf-8").splitlines()
    opener = re.compile(r"<<'([^']+)'")
    heredocs = 0
    for index, line in enumerate(lines):
        match = opener.search(line)
        if match is None:
            continue
        delimiter = match.group(1)
        first_body = lines[index + 1].rstrip("\r") if index + 1 < len(lines) else ""
        heredocs += 1
        assert first_body != delimiter, (
            f"heredoc <<'{delimiter}' at process-pdf-windows.ps1:{index + 1} has a "
            f"first body line identical to its delimiter; bash closes it immediately "
            f"and parses the body as commands (the v83 _gpt-opencode-smoke.sh "
            f"line-26 syntax error). Give the delimiter a unique sentinel."
        )
    assert heredocs >= 2, "expected the FORBIDDEN and AUTH_STALE smoke heredocs to be present"
