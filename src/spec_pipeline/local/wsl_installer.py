from __future__ import annotations

import base64
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CommandRunner = Callable[[str], int]
Timer = Callable[[], float]

PADDLE_CUDA_PACKAGE = "paddlepaddle-gpu==3.2.1"
PADDLE_CUDA_INDEX_URL = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
PADDLEOCR_PACKAGE = "paddleocr[doc-parser]>=3.0.0"
PADDLEOCR_SAFETENSORS_WHEEL = (
    "https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/"
    "safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl"
)
LLAMA_CPP_PACKAGE = "llama-cpp-python[server]"
LLAMA_CPP_CUDA_WHEEL_URL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v0.3.21-cu124/llama_cpp_python-0.3.21-py3-none-linux_x86_64.whl"
)
LLAMA_CPP_CUDA_WHEEL_REQUIREMENT = f"{LLAMA_CPP_PACKAGE} @ {LLAMA_CPP_CUDA_WHEEL_URL}"
LLAMA_CPP_CUDA_RUNTIME_REQUIREMENTS = (
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cublas-cu12==12.6.4.1",
)
LLAMA_CPP_PADDLE_CONSTRAINTS = ("numpy<2.4,>=1.24", "PyYAML<=6.0.2")
LLAMA_CPP_CUDA_LIBRARY_PATH_SCRIPT = (
    'for d in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib '
    "/usr/local/cuda/lib64 /usr/lib/wsl/lib; do "
    '[ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
    "done"
)


@dataclass(frozen=True, slots=True)
class WslInstallConfig:
    distro: str
    profile: str
    project_path: Path
    venv_path: Path
    cache_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str | None = None
    backend: str = "auto"
    runtime: str = "auto"
    install_runtime: bool = True
    install_paddle: bool = True
    execute: bool = False
    confirm: bool = False
    gpt_auth_file: Path | None = None
    gpt_smoke_test: bool = True


@dataclass(frozen=True, slots=True)
class InstallStep:
    id: str
    title: str
    shell: str
    command: str
    mutates: bool = False
    required: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "shell": self.shell,
            "command": self.command,
            "mutates": self.mutates,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class WslInstallPlan:
    ok: bool
    dry_run: bool
    execute: bool
    confirm: bool
    profile: str
    distro: str
    project_path: str
    venv_path: str
    cache_dir: str
    model: str
    served_model_name: str
    base_url: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    steps: tuple[InstallStep, ...]
    next_commands: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "execute": self.execute,
            "confirm": self.confirm,
            "profile": self.profile,
            "distro": self.distro,
            "project_path": self.project_path,
            "venv_path": self.venv_path,
            "cache_dir": self.cache_dir,
            "model": self.model,
            "served_model_name": self.served_model_name,
            "base_url": self.base_url,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "steps": [step.to_dict() for step in self.steps],
            "next_commands": list(self.next_commands),
        }


def build_wsl_install_plan(config: WslInstallConfig) -> WslInstallPlan:
    model = profile_model(config.profile)
    served_model_name = config.served_model_name or default_served_model_name(config.profile)
    project_path = windows_or_posix_project_path_to_wsl(config.project_path)
    venv_path = _path_text(config.venv_path)
    cache_dir = _path_text(config.cache_dir)
    base_url = f"http://{config.host}:{config.port}/v1"
    errors: list[str] = []
    warnings: list[str] = []
    required_runtime = profile_runtime(config.profile)
    if config.execute and not config.confirm:
        errors.append("--execute requires --confirm")
    if config.backend not in {"auto", required_runtime}:
        errors.append(
            f"profile '{config.profile}' uses {required_runtime}; cannot validate serve-local with backend '{config.backend}'"
        )
    if config.runtime not in {"auto", required_runtime}:
        errors.append(
            f"profile '{config.profile}' uses {required_runtime}; cannot install runtime '{config.runtime}'"
        )
    if config.confirm and not config.execute:
        warnings.append("--confirm передан без --execute; остаемся в dry-run режиме")

    if config.profile == "gpt":
        auth_file = config.gpt_auth_file
        if auth_file is None:
            warnings.append(
                "Учетные данные GPT не импортируются: OpenCode использует вход "
                "текущего пользователя в WSL или запустит интерактивный "
                "`opencode auth login`."
            )
        else:
            warnings.append(
                "--gpt-auth-file — расширенный opt-in для переноса ваших собственных "
                "учетных данных. Не коммитьте, не публикуйте и не передавайте этот файл."
            )
            if _can_check_local_path(auth_file) and not auth_file.exists():
                warning = f"Явно указанный --gpt-auth-file не найден: {auth_file}"
                warnings.append(warning)
                if config.execute:
                    errors.append(warning)
        steps = tuple(_build_gpt_steps(config, project_path, venv_path, auth_file))
        next_commands = build_gpt_next_commands(config, project_path, venv_path)
        base_url = "opencode"
    else:
        steps = tuple(
            _build_steps(config, model, served_model_name, project_path, venv_path, cache_dir)
        )
        next_commands = build_next_commands(
            config, model, served_model_name, project_path, venv_path, cache_dir, base_url
        )
    return WslInstallPlan(
        ok=not errors,
        dry_run=not config.execute,
        execute=config.execute,
        confirm=config.confirm,
        profile=config.profile,
        distro=config.distro,
        project_path=project_path,
        venv_path=venv_path,
        cache_dir=cache_dir,
        model=model,
        served_model_name=served_model_name,
        base_url=base_url,
        warnings=tuple(warnings),
        errors=tuple(errors),
        steps=steps,
        next_commands=next_commands,
    )


def execute_install_plan(
    plan: WslInstallPlan,
    *,
    runner: CommandRunner | None = None,
    timer: Timer | None = None,
) -> int:
    if not plan.ok:
        return 2
    if plan.dry_run:
        print("setup-wsl: refusing to execute a dry-run plan")
        return 2
    command_runner = runner or _run_shell_command
    timer_fn = timer or time.perf_counter
    total = len(plan.steps)
    started = timer_fn()
    print(f"setup-wsl: запуск установки ({total} шагов)")
    for index, step in enumerate(plan.steps, start=1):
        step_started = timer_fn()
        start_percent = round((index - 1) / total * 100) if total else 100
        done_percent = round(index / total * 100) if total else 100
        print(
            f"{_progress_bar(index - 1, total)} {index}/{total} {start_percent:>3}% "
            f"{_format_duration(step_started - started)} | {step.title}"
        )
        try:
            return_code = command_runner(step.command)
        except FileNotFoundError as exc:
            missing = exc.filename or "powershell.exe"
            print(f"setup-wsl: не удалось запустить {missing}")
            print("setup-wsl: установка с --yes предназначена для Windows/WSL2.")
            print(
                "setup-wsl: на Mac используйте dry-run без --yes или проверяйте GPT через map-ocr напрямую."
            )
            return 127
        step_elapsed = timer_fn() - step_started
        total_elapsed = timer_fn() - started
        if return_code != 0:
            if step.id in {"windows.wsl.install", "windows.wsl.distro"} and return_code in {
                1641,
                3010,
            }:
                return_code = 3010
                print(
                    "setup-wsl: Windows запросила перезагрузку после включения WSL. "
                    "Перезагрузите компьютер и снова запустите INSTALL_GPT_WINDOWS.cmd."
                )
                print(
                    f"setup-wsl: STOP {step.id} за {_format_duration(step_elapsed)} "
                    f"(всего {_format_duration(total_elapsed)}) exit=3010"
                )
                return return_code
            print(
                f"setup-wsl: FAIL {step.id} за {_format_duration(step_elapsed)} "
                f"(всего {_format_duration(total_elapsed)}) exit={return_code}"
            )
            print(step.command)
            return return_code
        print(
            f"{_progress_bar(index, total)} {index}/{total} {done_percent:>3}% "
            f"OK за {_format_duration(step_elapsed)} "
            f"(всего {_format_duration(total_elapsed)})"
        )
    print(f"setup-wsl: установка завершена за {_format_duration(timer_fn() - started)}")
    return 0


def _progress_bar(completed: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        filled = width
    else:
        filled = max(0, min(width, round(width * completed / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining:02d}s"
    return f"{minutes}m{remaining:02d}s"


def profile_model(profile: str) -> str:
    if profile == "gpt":
        return "openai/gpt-5.5"
    if profile in {"fast", "standard"}:
        return "qwen3-8b-cuda"
    if profile == "hq":
        return "qwen3-14b-cuda"
    if profile == "qwen36-safe":
        return "qwen3.6-safe"
    if profile == "qwen36-balanced":
        return "qwen3.6-balanced"
    if profile == "qwen36-quality":
        return "qwen3.6-quality"
    if profile == "qwen36-max":
        return "qwen36-max"
    raise ValueError(f"Unknown WSL setup profile: {profile}")


def default_served_model_name(profile: str) -> str:
    if profile == "gpt":
        return "opencode-gpt"
    if profile in {"fast", "standard"}:
        return "qwen3-8b-local"
    if profile == "hq":
        return "qwen3-14b-local"
    if profile == "qwen36-safe":
        return "qwen3.6-safe-local"
    if profile == "qwen36-balanced":
        return "qwen3.6-balanced-local"
    if profile == "qwen36-quality":
        return "qwen3.6-quality-local"
    if profile == "qwen36-max":
        return "qwen36-max-local"
    raise ValueError(f"Unknown WSL setup profile: {profile}")


def profile_runtime(profile: str) -> str:
    if profile == "gpt":
        return "gpt"
    if profile in {"fast", "standard", "hq"}:
        return "vllm"
    if profile in {"qwen36-safe", "qwen36-balanced", "qwen36-quality"}:
        return "llama-cpp"
    if profile == "qwen36-max":
        return "vllm"
    raise ValueError(f"Unknown WSL setup profile: {profile}")


def windows_or_posix_project_path_to_wsl(path: Path | str) -> str:
    text = str(path).replace("\\", "/")
    if len(text) >= 3 and text[1] == ":" and text[2] == "/":
        drive = text[0].lower()
        rest = text[3:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return text


def wsl_bash_command(distro: str, script: str) -> str:
    return f"wsl.exe -d {_quote_powershell(distro)} -- bash -lc {_quote_powershell(script)}"


def _wsl_b64_command(distro: str, script: str) -> str:
    """Run a bash script in WSL via a base64 payload, immune to host re-quoting.

    Some steps assign shell variables (e.g. ``marker``/``bundled``) and re-read
    them on the same command line. If any layer between ``powershell.exe
    -Command`` and the WSL ``bash`` subjects the command to an extra expansion
    pass, those variables expand to empty *before* their assignment runs and the
    script collapses — the customer's install log showed ``[ = ]`` and
    ``printf %s > ;`` followed by ``syntax error near ';'``. A base64 payload
    contains no ``$``, quotes or spaces, so no outer pass can alter it; the real
    script is decoded and executed exactly once by the inner bash.
    """
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return wsl_bash_command(distro, f"printf %s {payload} | base64 -d | bash")


def build_next_commands(
    config: WslInstallConfig,
    model: str,
    served_model_name: str,
    project_path: str,
    venv_path: str,
    cache_dir: str,
    base_url: str,
) -> tuple[str, ...]:
    serve = shlex.join(
        [
            "spec-ocr",
            "serve-local",
            model,
            "--cache-dir",
            cache_dir,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--served-model-name",
            served_model_name,
        ]
    )
    map_ocr = shlex.join(
        [
            "spec-ocr",
            "map-ocr",
            "sample_id",
            "--input-ocr",
            "artifacts/ocr/input.paddleocr-vl.json",
            "--source-pdf",
            "input.pdf",
            "--ai-client",
            "openai-compatible",
            "--ai-model",
            model,
            "--ai-served-model",
            served_model_name,
            "--ai-base-url",
            base_url,
            "--ai-timeout",
            "1800",
            "--ai-max-tokens",
            "8192",
            "--ai-page-batch-size",
            "1",
            "--model-cache-dir",
            cache_dir,
            "--output",
            "artifacts/parsed/sample_id.ai.jsonl",
            "--quality-json",
            "artifacts/parsed/sample_id.ai.quality.json",
            "--performance-json",
            "artifacts/parsed/sample_id.ai.performance.json",
        ]
    )
    return (
        _venv_wsl_command(config.distro, project_path, venv_path, serve),
        _venv_wsl_command(config.distro, project_path, venv_path, map_ocr),
    )


def build_gpt_next_commands(
    config: WslInstallConfig, project_path: str, venv_path: str
) -> tuple[str, ...]:
    map_ocr = shlex.join(
        [
            "spec-ocr",
            "map-ocr",
            "sample_id",
            "--input-ocr",
            "artifacts/ocr/input.paddleocr-vl.json",
            "--source-pdf",
            "input.pdf",
            "--ai-client",
            "opencode",
            "--ai-model",
            "openai/gpt-5.5",
            "--ai-agent",
            "build",
            "--ai-reasoning-effort",
            "xhigh",
            "--output",
            "artifacts/parsed/sample_id.ai.jsonl",
            "--quality-json",
            "artifacts/parsed/sample_id.ai.quality.json",
            "--performance-json",
            "artifacts/parsed/sample_id.ai.performance.json",
        ]
    )
    command = 'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"; ' + map_ocr
    return (_venv_wsl_command(config.distro, project_path, venv_path, command),)


def _build_steps(
    config: WslInstallConfig,
    model: str,
    served_model_name: str,
    project_path: str,
    venv_path: str,
    cache_dir: str,
) -> list[InstallStep]:
    steps = [
        InstallStep(
            "windows.wsl.install",
            "Установить/включить WSL при необходимости",
            "powershell",
            _windows_wsl_install_command(config.distro),
            mutates=True,
        ),
        InstallStep(
            "windows.wsl.status", "Проверить состояние WSL", "powershell", "wsl.exe --status"
        ),
        InstallStep(
            "windows.wsl.list",
            "Показать установленные WSL-дистрибутивы",
            "powershell",
            "wsl.exe --list --verbose; if ($LASTEXITCODE -ne 0) { Write-Host 'WSL-дистрибутивы пока не установлены.' }; exit 0",
        ),
        InstallStep(
            "windows.wsl.distro",
            f"Установить {config.distro}, если дистрибутив отсутствует",
            "powershell",
            _windows_wsl_distro_command(config.distro),
            mutates=True,
        ),
        InstallStep(
            "windows.nvidia", "Проверить NVIDIA-драйвер в Windows", "powershell", "nvidia-smi"
        ),
        InstallStep(
            "wsl.uname",
            "Проверить Linux kernel внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro, "set -euo pipefail; uname -a; grep -qi microsoft /proc/version"
            ),
        ),
        InstallStep(
            "wsl.nvidia",
            "Проверить NVIDIA GPU внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro,
                "set -euo pipefail; command -v nvidia-smi; "
                "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
            ),
        ),
        InstallStep(
            "wsl.python.disk",
            "Проверить Python и свободное место внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro, "set -euo pipefail; python3 --version || true; df -h ."
            ),
        ),
        InstallStep(
            "wsl.apt.python",
            "Установить Python/venv/git внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro,
                "set -euo pipefail; sudo apt-get update; "
                "sudo apt-get install -y python3 python3-venv python3-pip git "
                "build-essential cmake python3-dev ninja-build pkg-config",
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.venv.create",
            "Создать Python virtualenv",
            "wsl",
            _project_wsl_command(
                config.distro, project_path, f"python3 -m venv {shlex.quote(venv_path)}"
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.pip.bootstrap",
            "Обновить pip/wheel/setuptools",
            "wsl",
            _venv_wsl_command(
                config.distro,
                project_path,
                venv_path,
                "python -m pip install -U pip wheel setuptools",
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.project.install",
            "Установить зависимости spec-ocr",
            "wsl",
            _venv_wsl_command(
                config.distro,
                project_path,
                venv_path,
                f'python -m pip install -e "{_project_extra(config.install_paddle)}"',
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.paddle.cuda",
            "Проверить/установить CUDA PaddleOCR-VL runtime",
            "wsl",
            _venv_wsl_command(config.distro, project_path, venv_path, paddle_cuda_repair_command()),
            mutates=True,
        ),
    ]
    if config.install_runtime:
        steps.append(
            InstallStep(
                "wsl.runtime.install",
                "Установить runtime локального сервера",
                "wsl",
                _venv_wsl_command(
                    config.distro,
                    project_path,
                    venv_path,
                    _runtime_install_command(config.runtime, model),
                ),
                mutates=True,
            )
        )
    steps.extend(
        [
            InstallStep(
                "wsl.doctor",
                "Запустить диагностику spec-ocr внутри WSL",
                "wsl",
                _venv_wsl_command(config.distro, project_path, venv_path, "spec-ocr doctor --json"),
            ),
            InstallStep(
                "wsl.models.download",
                "Скачать модели выбранного профиля",
                "wsl",
                _venv_wsl_command(
                    config.distro,
                    project_path,
                    venv_path,
                    shlex.join(
                        [
                            "spec-ocr",
                            "setup-local",
                            "--profile",
                            config.profile,
                            "--download",
                            "--json",
                            "--cache-dir",
                            cache_dir,
                        ]
                    ),
                ),
                mutates=True,
            ),
            InstallStep(
                "wsl.serve.validate",
                "Проверить команду запуска локального сервера",
                "wsl",
                _venv_wsl_command(
                    config.distro,
                    project_path,
                    venv_path,
                    shlex.join(
                        [
                            "spec-ocr",
                            "serve-local",
                            model,
                            "--cache-dir",
                            cache_dir,
                            "--served-model-name",
                            served_model_name,
                            "--host",
                            config.host,
                            "--port",
                            str(config.port),
                            "--backend",
                            config.backend,
                            "--dry-run",
                            "--json",
                        ]
                    ),
                ),
            ),
        ]
    )
    return steps


def _build_gpt_steps(
    config: WslInstallConfig,
    project_path: str,
    venv_path: str,
    auth_file: Path | None,
) -> list[InstallStep]:
    opencode_path = 'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"'
    auth_step = _build_gpt_auth_step(config, auth_file, opencode_path)
    steps = [
        InstallStep(
            "windows.wsl.install",
            "Установить/включить WSL при необходимости",
            "powershell",
            _windows_wsl_install_command(config.distro),
            mutates=True,
        ),
        InstallStep(
            "windows.wsl.status", "Проверить состояние WSL", "powershell", "wsl.exe --status"
        ),
        InstallStep(
            "windows.wsl.list",
            "Показать установленные WSL-дистрибутивы",
            "powershell",
            "wsl.exe --list --verbose; if ($LASTEXITCODE -ne 0) { Write-Host 'WSL-дистрибутивы пока не установлены.' }; exit 0",
        ),
        InstallStep(
            "windows.wsl.distro",
            f"Установить {config.distro}, если дистрибутив отсутствует",
            "powershell",
            _windows_wsl_distro_command(config.distro),
            mutates=True,
        ),
        InstallStep(
            "wsl.uname",
            "Проверить Linux kernel внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro, "set -euo pipefail; uname -a; grep -qi microsoft /proc/version"
            ),
        ),
        InstallStep(
            "wsl.python.disk",
            "Проверить Python и свободное место внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro, "set -euo pipefail; python3 --version || true; df -h ."
            ),
        ),
        InstallStep(
            "wsl.apt.python",
            "Установить Python/venv/git/curl внутри WSL",
            "wsl",
            wsl_bash_command(
                config.distro,
                "set -euo pipefail; sudo apt-get update; "
                "sudo apt-get install -y python3 python3-venv python3-pip git curl ca-certificates",
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.venv.create",
            "Создать Python virtualenv",
            "wsl",
            _project_wsl_command(
                config.distro, project_path, f"python3 -m venv {shlex.quote(venv_path)}"
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.pip.bootstrap",
            "Обновить pip/wheel/setuptools",
            "wsl",
            _venv_wsl_command(
                config.distro,
                project_path,
                venv_path,
                "python -m pip install -U pip wheel setuptools",
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.project.install",
            "Установить зависимости spec-ocr",
            "wsl",
            _venv_wsl_command(
                config.distro,
                project_path,
                venv_path,
                'python -m pip install -e ".[local-models]"',
            ),
            mutates=True,
        ),
        InstallStep(
            "wsl.opencode.check",
            "Проверить установленный пользователем OpenCode CLI",
            "wsl",
            wsl_bash_command(
                config.distro,
                f"set -euo pipefail; {opencode_path}; "
                "command -v opencode >/dev/null 2>&1 || { "
                "echo 'OpenCode CLI не найден. Установите его самостоятельно по официальной "
                "документации, затем повторите setup.' >&2; exit 1; }",
            ),
        ),
        auth_step,
        InstallStep(
            "wsl.opencode.config",
            "Настроить модель GPT-5.5 в OpenCode",
            "wsl",
            wsl_bash_command(config.distro, _opencode_gpt55_config_command()),
            mutates=True,
        ),
        InstallStep(
            "wsl.opencode.providers",
            "Проверить подключенные аккаунты OpenCode",
            "wsl",
            wsl_bash_command(
                config.distro,
                f"set -euo pipefail; {opencode_path}; opencode auth list",
            ),
        ),
        InstallStep(
            "wsl.doctor",
            "Запустить диагностику spec-ocr внутри WSL",
            "wsl",
            _venv_wsl_command(config.distro, project_path, venv_path, "spec-ocr doctor --json"),
        ),
    ]
    if config.gpt_smoke_test:
        steps.append(
            InstallStep(
                "wsl.opencode.smoke",
                "Проверить GPT коротким запросом",
                "wsl",
                wsl_bash_command(
                    config.distro,
                    f"set -euo pipefail; {opencode_path}; "
                    "opencode run --model openai/gpt-5.5 --agent build --variant xhigh --format default "
                    "'Ответь ровно OK и больше ничего.'",
                ),
            )
        )
    return steps


def _build_gpt_auth_step(
    config: WslInstallConfig,
    auth_file: Path | None,
    opencode_path: str,
) -> InstallStep:
    if auth_file is None:
        script = (
            f"set -euo pipefail; {opencode_path}; "
            "if opencode auth list 2>/dev/null | grep -qi openai; then "
            "echo 'OpenCode: используется авторизация текущего пользователя WSL.'; "
            "else "
            "echo 'OpenCode: войдите в собственный аккаунт и выберите OpenAI.'; "
            "opencode auth login; "
            "fi; "
            "if ! opencode auth list 2>/dev/null | grep -qi openai; then "
            "echo 'OpenAI не подключен. Выполните opencode auth login в WSL.' >&2; "
            "exit 1; "
            "fi"
        )
        return InstallStep(
            "wsl.opencode.auth",
            "Авторизовать текущего пользователя в OpenCode",
            "wsl",
            _wsl_b64_command(config.distro, script),
            mutates=True,
        )

    auth_path = windows_or_posix_project_path_to_wsl(auth_file)
    script = (
        "set -euo pipefail; "
        f"test -f {shlex.quote(auth_path)}; "
        "mkdir -p $HOME/.local/share/opencode; "
        'marker="$HOME/.local/share/opencode/.spec-ocr-imported-auth.sha256"; '
        f"imported=$(sha256sum {shlex.quote(auth_path)} | cut -d' ' -f1); "
        'if [ -s "$HOME/.local/share/opencode/auth.json" ] && [ -f "$marker" ] '
        '&& [ "$(cat "$marker")" = "$imported" ]; then '
        "echo 'Явно указанный OpenCode credential уже импортирован; "
        "текущая авторизация сохранена.'; "
        "else "
        'if [ -s "$HOME/.local/share/opencode/auth.json" ]; then '
        'cp "$HOME/.local/share/opencode/auth.json" '
        '"$HOME/.local/share/opencode/auth.json.bak"; fi; '
        f"cp {shlex.quote(auth_path)} $HOME/.local/share/opencode/auth.json; "
        "chmod 600 $HOME/.local/share/opencode/auth.json; "
        'printf %s "$imported" > "$marker"; '
        "fi"
    )
    return InstallStep(
        "wsl.opencode.auth",
        "Импортировать явно указанный OpenCode credential",
        "wsl",
        _wsl_b64_command(config.distro, script),
        mutates=True,
    )


def _windows_wsl_install_command(distro: str) -> str:
    distro_arg = _quote_powershell(distro)
    distro_literal = distro.replace("'", "''")
    return (
        "$ErrorActionPreference = 'Continue'; "
        "function Test-SpecOcrRebootText($Output) { return (($Output | Out-String) -match 'перезагруз|reboot|restart') }; "
        "function Write-SpecOcrOutput($Output) { if ($Output) { Write-Host ($Output | Out-String) } }; "
        "function Get-SpecOcrDistros { "
        "$raw = @(& wsl.exe --list --quiet 2>$null); "
        "if ($LASTEXITCODE -ne 0) { return @() }; "
        "return @($raw | ForEach-Object { $_.Trim([char]0).Trim() } | Where-Object { $_ }) }; "
        "function Test-SpecOcrDistroPresent($Name) { return (@(Get-SpecOcrDistros) -contains $Name) }; "
        "$statusOutput = @(& wsl.exe --status 2>&1); "
        "$statusCode = $LASTEXITCODE; "
        "if ($statusCode -eq 0) { exit 0 }; "
        "Write-SpecOcrOutput $statusOutput; "
        "Write-Host 'WSL не установлен или требует первичной настройки. Включаю компоненты Windows...'; "
        "$wslFeatureOutput = @(& dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart 2>&1); "
        "$wslFeatureCode = $LASTEXITCODE; "
        "Write-SpecOcrOutput $wslFeatureOutput; "
        "$vmFeatureOutput = @(& dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart 2>&1); "
        "$vmFeatureCode = $LASTEXITCODE; "
        "Write-SpecOcrOutput $vmFeatureOutput; "
        "if (($wslFeatureCode -ne 0 -and $wslFeatureCode -ne 3010 -and $wslFeatureCode -ne 1641) -or "
        "($vmFeatureCode -ne 0 -and $vmFeatureCode -ne 3010 -and $vmFeatureCode -ne 1641)) { "
        "Write-Host 'Не удалось включить WSL-компоненты Windows.'; "
        "if ($wslFeatureCode -ne 0 -and $wslFeatureCode -ne 3010 -and $wslFeatureCode -ne 1641) { exit $wslFeatureCode }; "
        "exit $vmFeatureCode }; "
        "if ($wslFeatureCode -eq 3010 -or $vmFeatureCode -eq 3010 -or $wslFeatureCode -eq 1641 -or "
        "$vmFeatureCode -eq 1641 -or (Test-SpecOcrRebootText $wslFeatureOutput) -or "
        "(Test-SpecOcrRebootText $vmFeatureOutput)) { "
        "Write-Host 'Windows нужно перезагрузить. После reboot запустите INSTALL_GPT_WINDOWS.cmd еще раз.'; "
        "exit 3010 }; "
        "Write-Host 'Запускаю wsl.exe --install...'; "
        f"$installOutput = @(& wsl.exe --install -d {distro_arg} --no-launch 2>&1); "
        "$code = $LASTEXITCODE; "
        "Write-SpecOcrOutput $installOutput; "
        f"if ($code -eq 0 -and -not (Test-SpecOcrRebootText $installOutput)) {{ "
        f"if (Test-SpecOcrDistroPresent '{distro_literal}') {{ exit 0 }}; "
        "Write-Host 'WSL/Ubuntu установлены, но дистрибутив еще не виден Windows. Перезагрузите компьютер и запустите INSTALL_GPT_WINDOWS.cmd еще раз.'; "
        "exit 3010 }; "
        "if ($code -eq 0 -or $code -eq 1641 -or $code -eq 3010 -or (Test-SpecOcrRebootText $installOutput)) { "
        "Write-Host 'Windows нужно перезагрузить. После reboot запустите INSTALL_GPT_WINDOWS.cmd еще раз.'; "
        "exit 3010 }; "
        "exit $code"
    )


def _windows_wsl_distro_command(distro: str) -> str:
    distro_arg = _quote_powershell(distro)
    distro_literal = distro.replace("'", "''")
    return (
        "$ErrorActionPreference = 'Continue'; "
        "function Test-SpecOcrRebootText($Output) { return (($Output | Out-String) -match 'перезагруз|reboot|restart') }; "
        "function Write-SpecOcrOutput($Output) { if ($Output) { Write-Host ($Output | Out-String) } }; "
        "function Get-SpecOcrDistros { "
        "$raw = @(& wsl.exe --list --quiet 2>$null); "
        "if ($LASTEXITCODE -ne 0) { return @() }; "
        "return @($raw | ForEach-Object { $_.Trim([char]0).Trim() } | Where-Object { $_ }) }; "
        "function Test-SpecOcrDistroPresent($Name) { return (@(Get-SpecOcrDistros) -contains $Name) }; "
        f"if (Test-SpecOcrDistroPresent '{distro_literal}') {{ exit 0 }}; "
        f"Write-Host 'WSL-дистрибутив {distro_literal} не найден. Устанавливаю...'; "
        f"$installOutput = @(& wsl.exe --install -d {distro_arg} --no-launch 2>&1); "
        "$code = $LASTEXITCODE; "
        "Write-SpecOcrOutput $installOutput; "
        f"if ($code -eq 0 -and -not (Test-SpecOcrRebootText $installOutput)) {{ "
        f"if (Test-SpecOcrDistroPresent '{distro_literal}') {{ exit 0 }}; "
        "Write-Host 'WSL сообщил об успешной установке Ubuntu, но дистрибутив еще не виден Windows. Перезагрузите компьютер и запустите INSTALL_GPT_WINDOWS.cmd еще раз.'; "
        "exit 3010 }; "
        "if ($code -eq 0 -or $code -eq 1641 -or $code -eq 3010 -or (Test-SpecOcrRebootText $installOutput)) { "
        "Write-Host 'Windows нужно перезагрузить. После reboot запустите INSTALL_GPT_WINDOWS.cmd еще раз.'; "
        "exit 3010 }; "
        "exit $code"
    )


_OPENCODE_GPT55_CONFIG_JSON = r"""{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai": {
      "models": {
        "gpt-5.5": {
          "name": "GPT-5.5",
          "limit": {
            "context": 400000,
            "output": 128000,
            "input": 400000
          },
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "options": {
            "reasoningEffort": "medium",
            "reasoningSummary": "auto",
            "textVerbosity": "medium",
            "include": ["reasoning.encrypted_content"],
            "store": false
          },
          "id": "gpt-5.5"
        },
        "gpt-5.5-fast": {
          "name": "GPT-5.5 Fast",
          "limit": {
            "context": 400000,
            "output": 128000,
            "input": 400000
          },
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "options": {
            "reasoningEffort": "medium",
            "reasoningSummary": "auto",
            "textVerbosity": "medium",
            "include": ["reasoning.encrypted_content"],
            "store": false
          },
          "id": "gpt-5.5-fast"
        }
      }
    }
  }
}
"""


def _opencode_gpt55_config_command() -> str:
    # base64 keeps the JSON on a single command line: a CRLF-mangled here-doc
    # left a bash warning and a broken opencode.json on Windows hosts.
    payload = base64.b64encode(_OPENCODE_GPT55_CONFIG_JSON.encode("utf-8")).decode("ascii")
    return (
        "set -euo pipefail; "
        'mkdir -p "$HOME/.opencode"; '
        f'printf %s {payload} | base64 -d > "$HOME/.opencode/opencode.json"; '
        'chmod 600 "$HOME/.opencode/opencode.json"'
    )


def _can_check_local_path(path: Path) -> bool:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        return _is_windows_host()
    return True


def _is_windows_host() -> bool:
    return os.name == "nt"


def _project_wsl_command(distro: str, project_path: str, command: str) -> str:
    script = f"set -euo pipefail; cd {shlex.quote(project_path)}; {command}"
    return wsl_bash_command(distro, script)


def _venv_wsl_command(distro: str, project_path: str, venv_path: str, command: str) -> str:
    activate = f". {shlex.quote(venv_path.rstrip('/'))}/bin/activate"
    return _project_wsl_command(distro, project_path, f"{activate}; {command}")


def _project_extra(install_paddle: bool) -> str:
    return ".[local-models,paddleocr-vl]" if install_paddle else ".[local-models]"


def paddle_cuda_repair_command() -> str:
    pip_base = "python -m pip install --disable-pip-version-check --retries 10 --timeout 120"
    check = "spec-ocr paddle-gpu-check --require-cuda --json"
    repair_commands = [
        "python -m pip uninstall -y paddlepaddle paddlepaddle-gpu",
        f"{pip_base} -U {shlex.quote(PADDLEOCR_PACKAGE)}",
        f"{pip_base} -U {shlex.quote(PADDLEOCR_SAFETENSORS_WHEEL)}",
        "python -m pip uninstall -y paddlepaddle paddlepaddle-gpu",
        f"{pip_base} {shlex.quote(PADDLE_CUDA_PACKAGE)} -i {shlex.quote(PADDLE_CUDA_INDEX_URL)}",
        check,
    ]
    return f"{check} || (" + "; ".join(repair_commands) + ")"


def llama_cuda_repair_command() -> str:
    pip_base = "python -m pip install --disable-pip-version-check --retries 10 --timeout 120"
    check = f"{LLAMA_CPP_CUDA_LIBRARY_PATH_SCRIPT}; spec-ocr llama-gpu-check --require-cuda --json"
    runtime_requirements = " ".join(
        shlex.quote(requirement) for requirement in LLAMA_CPP_CUDA_RUNTIME_REQUIREMENTS
    )
    paddle_constraints = " ".join(
        shlex.quote(requirement) for requirement in LLAMA_CPP_PADDLE_CONSTRAINTS
    )
    repair_commands = [
        "python -m pip uninstall -y llama-cpp-python llama_cpp_python",
        (
            f"{pip_base} --upgrade --force-reinstall --no-cache-dir "
            f"--only-binary llama-cpp-python --prefer-binary "
            f"{shlex.quote(LLAMA_CPP_CUDA_WHEEL_REQUIREMENT)} "
            f"{runtime_requirements} {paddle_constraints}"
        ),
        "spec-ocr llama-link-cuda-libs --json",
        check,
    ]
    return f"{check} || (" + "; ".join(repair_commands) + ")"


def _runtime_install_command(runtime: str, model: str) -> str:
    package = _runtime_package(runtime, model)
    if package == LLAMA_CPP_PACKAGE:
        return llama_cuda_repair_command()
    return f"python -m pip install {shlex.quote(package)}"


def _runtime_package(runtime: str, model: str) -> str:
    resolved = _runtime_for_model(model) if runtime == "auto" else runtime
    if resolved == "vllm":
        return "vllm"
    if resolved == "sglang":
        return "sglang[all]"
    if resolved == "llama-cpp":
        return LLAMA_CPP_PACKAGE
    raise ValueError(f"Unknown WSL runtime: {runtime}")


def _runtime_for_model(model: str) -> str:
    if model == "qwen36-max":
        return "vllm"
    if model.startswith("qwen3.6-") and model != "qwen3.6-vllm-experimental":
        return "llama-cpp"
    if "vl" in model:
        return "sglang"
    return "vllm"


def _quote_powershell(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _run_shell_command(command: str) -> int:
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        check=False,
    )
    return int(completed.returncode)
