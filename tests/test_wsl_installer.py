from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from spec_pipeline.local.wsl_installer import (
    WslInstallConfig,
    build_wsl_install_plan,
    execute_install_plan,
    windows_or_posix_project_path_to_wsl,
    wsl_bash_command,
)


def _config(**overrides) -> WslInstallConfig:
    values = {
        "distro": "Ubuntu-24.04",
        "profile": "standard",
        "project_path": Path("/mnt/c/spec-ocr-pipeline"),
        "venv_path": Path(".venv-wsl"),
        "cache_dir": Path("artifacts/models"),
    }
    values.update(overrides)
    return WslInstallConfig(**values)


def _decode_wsl_b64_step(plan, step_id: str) -> str:
    step = next(step for step in plan.steps if step.id == step_id)
    match = re.search(r"printf %s (\S+) \| base64 -d \| bash", step.command)
    assert match is not None
    return base64.b64decode(match.group(1)).decode("utf-8")


def test_build_wsl_install_plan_is_dry_run_by_default() -> None:
    plan = build_wsl_install_plan(_config())

    assert plan.ok is True
    assert plan.dry_run is True
    assert plan.model == "qwen3-8b-cuda"
    commands = "\n".join(step.command for step in plan.steps)
    assert "wsl.exe --status" in commands
    assert "nvidia-smi" in commands
    assert "python3 -m venv .venv-wsl" in commands
    assert "spec-ocr setup-local --profile standard --download" in commands
    assert "spec-ocr serve-local qwen3-8b-cuda" in commands
    assert "--dry-run --json" in commands


def test_windows_wsl_install_commands_detect_reboot_text() -> None:
    plan = build_wsl_install_plan(_config())

    commands = "\n".join(step.command for step in plan.steps)
    assert "Test-SpecOcrRebootText" in commands
    assert "Test-SpecOcrDistroPresent" in commands
    assert "перезагруз|reboot|restart" in commands
    assert "$installOutput = @(& wsl.exe --install" in commands
    assert "дистрибутив еще не виден Windows" in commands
    assert "exit 3010" in commands


def test_hq_profile_selects_qwen3_14b_cuda() -> None:
    plan = build_wsl_install_plan(_config(profile="hq"))

    assert plan.model == "qwen3-14b-cuda"
    assert plan.served_model_name == "qwen3-14b-local"
    assert "qwen3-14b-cuda" in plan.next_commands[0]


def test_qwen36_safe_profile_selects_llama_cpp_gguf_model() -> None:
    plan = build_wsl_install_plan(_config(profile="qwen36-safe"))

    assert plan.model == "qwen3.6-safe"
    assert plan.served_model_name == "qwen3.6-safe-local"
    commands = "\n".join(step.command for step in plan.steps)
    assert "llama-cpp-python[server]" in commands
    assert "llama-gpu-check --require-cuda --json" in commands
    assert "v0.3.21-cu124/llama_cpp_python-0.3.21-py3-none-linux_x86_64.whl" in commands
    assert "--force-reinstall" in commands
    assert "--only-binary llama-cpp-python" in commands
    assert "--prefer-binary" in commands
    assert "nvidia-cuda-runtime-cu12==12.6.77" in commands
    assert "nvidia-cublas-cu12==12.6.4.1" in commands
    assert "llama-link-cuda-libs --json" in commands
    assert "LD_LIBRARY_PATH" in commands
    assert '"$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib' in commands
    assert "numpy<2.4,>=1.24" in commands
    assert "PyYAML<=6.0.2" in commands
    assert "https://abetlen.github.io/llama-cpp-python/whl/cu125" not in commands
    assert "build-essential cmake python3-dev ninja-build pkg-config" in commands
    assert "paddle-gpu-check --require-cuda --json" in commands
    assert "paddlepaddle-gpu==3.2.1" in commands
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu126/" in commands
    assert "safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl" in commands
    assert "spec-ocr serve-local qwen3.6-safe" in plan.next_commands[0]
    assert "--ai-timeout 1800" in plan.next_commands[1]
    assert "--ai-max-tokens 8192" in plan.next_commands[1]
    assert "--ai-page-batch-size 1" in plan.next_commands[1]
    assert "--quality-json artifacts/parsed/sample_id.ai.quality.json" in plan.next_commands[1]
    assert (
        "--performance-json artifacts/parsed/sample_id.ai.performance.json" in plan.next_commands[1]
    )
    assert plan.next_commands[0].startswith("wsl.exe -d")
    assert ". .venv-wsl/bin/activate" in plan.next_commands[0]


def test_qwen36_quality_profile_selects_larger_gguf_model() -> None:
    plan = build_wsl_install_plan(_config(profile="qwen36-quality"))

    assert plan.model == "qwen3.6-quality"
    assert plan.served_model_name == "qwen3.6-quality-local"


def test_qwen36_max_profile_selects_vllm_gptq_model() -> None:
    plan = build_wsl_install_plan(_config(profile="qwen36-max"))

    assert plan.model == "qwen36-max"
    assert plan.served_model_name == "qwen36-max-local"
    commands = "\n".join(step.command for step in plan.steps)
    assert "python -m pip install vllm" in commands
    assert "spec-ocr serve-local qwen36-max" in plan.next_commands[0]


def test_gpt_profile_uses_existing_or_interactive_user_auth_by_default() -> None:
    plan = build_wsl_install_plan(_config(profile="gpt"))

    assert plan.ok is True
    assert plan.model == "openai/gpt-5.5"
    assert plan.served_model_name == "opencode-gpt"
    commands = "\n".join(step.command for step in plan.steps)
    assert "https://opencode.ai/install" not in commands
    assert "curl -fsSL" not in commands
    assert "command -v opencode" in commands
    assert "Установите его самостоятельно" in commands
    assert "models --cache-dir artifacts/models download paddleocr-vl --json" not in commands
    assert "paddle-gpu-check --require-cuda --json" not in commands
    assert "paddlepaddle-gpu==3.2.1" not in commands
    assert 'python -m pip install -e ".[local-models]"' in commands
    auth_script = _decode_wsl_b64_step(plan, "wsl.opencode.auth")
    assert "opencode auth list" in auth_script
    assert "opencode auth login" in auth_script
    assert "текущего пользователя WSL" in auth_script
    assert "auth.json" not in auth_script
    assert "customer-secrets" not in commands
    assert any("Учетные данные GPT не импортируются" in warning for warning in plan.warnings)
    assert "opencode.json" in commands
    config_match = re.search(r"printf %s (\S+) \| base64 -d > ", commands)
    assert config_match is not None
    config_json = base64.b64decode(config_match.group(1)).decode("utf-8")
    assert '"gpt-5.5"' in config_json
    assert '"id": "gpt-5.5"' in config_json
    assert "<<'JSON'" not in commands
    assert "opencode run --model openai/gpt-5.5" in commands
    assert "--variant xhigh" in commands
    assert "--ai-client opencode" in plan.next_commands[0]
    assert "--ai-reasoning-effort xhigh" in plan.next_commands[0]
    assert "--quality-json artifacts/parsed/sample_id.ai.quality.json" in plan.next_commands[0]
    assert (
        "--performance-json artifacts/parsed/sample_id.ai.performance.json" in plan.next_commands[0]
    )
    assert plan.next_commands[0].startswith("wsl.exe -d")
    assert ". .venv-wsl/bin/activate" in plan.next_commands[0]
    assert 'export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"' in plan.next_commands[0]


def test_gpt_profile_explicit_auth_file_is_advanced_opt_in(tmp_path: Path) -> None:
    auth_file = tmp_path / "private-open-code-auth.json"
    auth_file.write_text('{"OpenAI": {}}', encoding="utf-8")

    plan = build_wsl_install_plan(_config(profile="gpt", gpt_auth_file=auth_file))

    assert plan.ok is True
    assert any("расширенный opt-in" in warning for warning in plan.warnings)
    assert any("Не коммитьте" in warning for warning in plan.warnings)
    auth_script = _decode_wsl_b64_step(plan, "wsl.opencode.auth")
    assert windows_or_posix_project_path_to_wsl(auth_file) in auth_script
    assert ".spec-ocr-imported-auth.sha256" in auth_script
    assert "auth.json.bak" in auth_script
    assert "opencode auth login" not in auth_script
    assert "customer-secrets" not in auth_script


def test_gpt_profile_execute_allows_self_auth_without_file() -> None:
    plan = build_wsl_install_plan(_config(profile="gpt", execute=True, confirm=True))

    assert plan.ok is True
    assert not plan.errors
    assert "opencode auth login" in _decode_wsl_b64_step(plan, "wsl.opencode.auth")


def test_gpt_profile_execute_rejects_missing_explicit_auth_file(tmp_path: Path) -> None:
    auth_file = tmp_path / "missing-private-auth.json"
    plan = build_wsl_install_plan(
        _config(
            profile="gpt",
            gpt_auth_file=auth_file,
            execute=True,
            confirm=True,
        )
    )

    assert plan.ok is False
    assert any("Явно указанный --gpt-auth-file не найден" in error for error in plan.errors)


def test_gpt_profile_checks_missing_explicit_auth_for_windows_paths(monkeypatch) -> None:
    monkeypatch.setattr("spec_pipeline.local.wsl_installer._is_windows_host", lambda: True)

    plan = build_wsl_install_plan(
        _config(
            profile="gpt",
            project_path=Path("C:/SpecOCR"),
            gpt_auth_file=Path("C:/private/open-code-auth.json"),
            execute=True,
            confirm=True,
        )
    )

    assert plan.ok is False
    assert any("Явно указанный --gpt-auth-file не найден" in error for error in plan.errors)


def test_confirm_without_execute_stays_dry_run() -> None:
    plan = build_wsl_install_plan(_config(confirm=True))

    assert plan.ok is True
    assert plan.dry_run is True
    assert "dry-run" in plan.warnings[0]


def test_execute_requires_confirm() -> None:
    plan = build_wsl_install_plan(_config(execute=True))

    assert plan.ok is False
    assert plan.dry_run is False
    assert "--execute requires --confirm" in plan.errors


def test_runtime_backend_mismatch_is_rejected() -> None:
    plan = build_wsl_install_plan(_config(runtime="sglang", backend="sglang"))

    assert plan.ok is False
    assert any("cannot install runtime 'sglang'" in error for error in plan.errors)
    assert any(
        "cannot validate serve-local with backend 'sglang'" in error for error in plan.errors
    )


def test_wsl_project_install_includes_paddle_extra_by_default() -> None:
    plan = build_wsl_install_plan(_config())
    commands = "\n".join(step.command for step in plan.steps)

    assert 'python -m pip install -e ".[local-models,paddleocr-vl]"' in commands


def test_wsl_skip_paddle_install_uses_local_models_only() -> None:
    plan = build_wsl_install_plan(_config(install_paddle=False))
    commands = "\n".join(step.command for step in plan.steps)

    assert 'python -m pip install -e ".[local-models]"' in commands


def test_windows_path_converts_to_wsl_path() -> None:
    assert (
        windows_or_posix_project_path_to_wsl(Path("C:\\Users\\Alice\\Spec OCR"))
        == "/mnt/c/Users/Alice/Spec OCR"
    )


def test_wsl_bash_command_quotes_script_safely() -> None:
    command = wsl_bash_command("Ubuntu-24.04", "set -e; cd '/mnt/c/Spec OCR'; echo 'ok'")

    assert command.startswith("wsl.exe -d 'Ubuntu-24.04' -- bash -lc '")
    assert "''/mnt/c/Spec OCR''" in command


def test_execute_install_plan_runs_steps_in_order_and_stops_on_failure() -> None:
    plan = build_wsl_install_plan(_config(execute=True, confirm=True))
    calls: list[str] = []

    def runner(command: str) -> int:
        calls.append(command)
        return 7 if "apt-get update" in command else 0

    assert execute_install_plan(plan, runner=runner) == 7
    assert calls
    assert "apt-get update" in calls[-1]
    assert len(calls) < len(plan.steps)


def test_execute_install_plan_stops_cleanly_on_windows_reboot_request(capsys) -> None:
    plan = build_wsl_install_plan(_config(execute=True, confirm=True))
    calls: list[str] = []

    def runner(command: str) -> int:
        calls.append(command)
        return 3010 if "wsl.exe --install" in command else 0

    assert execute_install_plan(plan, runner=runner) == 3010
    output = capsys.readouterr().out
    assert "Windows запросила перезагрузку" in output
    assert "setup-wsl: STOP windows.wsl.install" in output
    assert "wsl.exe --install" not in output
    assert len(calls) == 1


def test_execute_install_plan_prints_progress_and_timers(capsys) -> None:
    plan = build_wsl_install_plan(_config(execute=True, confirm=True))
    current_time = 0.0

    def runner(command: str) -> int:
        return 0

    def timer() -> float:
        nonlocal current_time
        current_time += 1.0
        return current_time

    assert execute_install_plan(plan, runner=runner, timer=timer) == 0
    output = capsys.readouterr().out
    assert "setup-wsl: запуск установки" in output
    assert "[------------------------] 1/" in output
    assert "OK за" in output
    assert "100%" in output
    assert "setup-wsl: установка завершена за" in output


def test_execute_install_plan_reports_missing_powershell(capsys) -> None:
    plan = build_wsl_install_plan(_config(execute=True, confirm=True))

    def runner(command: str) -> int:
        raise FileNotFoundError(2, "No such file or directory", "powershell.exe")

    assert execute_install_plan(plan, runner=runner) == 127
    output = capsys.readouterr().out
    assert "не удалось запустить powershell.exe" in output
    assert "предназначена для Windows/WSL2" in output
    assert "проверяйте GPT через map-ocr напрямую" in output


def test_execute_install_plan_refuses_dry_run_plan() -> None:
    plan = build_wsl_install_plan(_config())
    calls: list[str] = []

    assert execute_install_plan(plan, runner=lambda command: calls.append(command) or 0) == 2
    assert calls == []


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown WSL setup profile"):
        build_wsl_install_plan(_config(profile="bad"))
