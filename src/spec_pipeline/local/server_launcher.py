from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from spec_pipeline.local.model_manager import ModelCacheManager
from spec_pipeline.local.model_manifest import LocalModelSpec

SERVER_BACKENDS = {"vllm", "sglang", "llama-cpp"}
RESERVED_EXTRA_ARGS = {
    "--model",
    "--model-path",
    "--model_alias",
    "--model-alias",
    "--served-model-name",
    "--host",
    "--port",
}


@dataclass(slots=True)
class LocalServerLaunchConfig:
    model: str
    cache_dir: Path
    backend: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class LocalServerCommand:
    backend: str
    model: str
    model_path: Path
    served_model_name: str
    host: str
    port: int
    base_url: str
    args: list[str]
    env: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "model_path": str(self.model_path),
            "served_model_name": self.served_model_name,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "command": self.args,
            "env": self.env,
        }


def build_local_server_command(
    config: LocalServerLaunchConfig,
    *,
    manager: ModelCacheManager | None = None,
) -> LocalServerCommand:
    model_manager = manager or ModelCacheManager(cache_dir=config.cache_dir)
    spec = model_manager.manifest.resolve(config.model)
    if spec.backend not in SERVER_BACKENDS:
        raise ValueError(
            f"Model '{config.model}' uses backend '{spec.backend}', "
            "not a supported local server backend"
        )

    backend = spec.backend if config.backend == "auto" else config.backend
    if backend not in SERVER_BACKENDS:
        raise ValueError(f"Unknown local server backend '{backend}'")
    if backend != spec.backend:
        raise ValueError(
            f"Model '{config.model}' is configured for backend '{spec.backend}', "
            f"cannot launch with '{backend}'"
        )
    _validate_extra_args(config.extra_args)

    model_path = _server_model_path(spec, model_manager.resolve_for_run(config.model).resolve())
    served_model_name = config.served_model_name or _default_served_model_name(spec, model_path)
    args = _build_backend_args(
        backend=backend,
        model_path=model_path,
        served_model_name=served_model_name,
        host=config.host,
        port=config.port,
        extra_args=spec.server_args + config.extra_args,
    )
    return LocalServerCommand(
        backend=backend,
        model=spec.aliases[0] if spec.aliases else spec.key,
        model_path=model_path,
        served_model_name=served_model_name,
        host=config.host,
        port=config.port,
        base_url=f"http://{config.host}:{config.port}/v1",
        args=args,
        env=_offline_env(),
    )


def launch_local_server(command: LocalServerCommand) -> int:
    env = os.environ.copy()
    env.update(command.env)
    completed = subprocess.run(command.args, env=env, check=False)
    return int(completed.returncode)


def _build_backend_args(
    *,
    backend: str,
    model_path: Path,
    served_model_name: str,
    host: str,
    port: int,
    extra_args: tuple[str, ...],
) -> list[str]:
    if backend == "vllm":
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            served_model_name,
            *extra_args,
        ]
    if backend == "sglang":
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            served_model_name,
            *extra_args,
        ]
    if backend == "llama-cpp":
        return [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "--model_alias",
            served_model_name,
            *extra_args,
        ]
    raise ValueError(f"Unknown local server backend '{backend}'")


def _server_model_path(spec: LocalModelSpec, model_dir: Path) -> Path:
    if spec.backend != "llama-cpp":
        return model_dir
    gguf_files = [file.path for file in spec.files if file.path.endswith(".gguf")]
    if len(gguf_files) != 1:
        raise ValueError(f"llama-cpp model '{spec.key}' must declare exactly one GGUF file")
    model_path = (model_dir / gguf_files[0]).resolve()
    if not model_path.is_file():
        raise ValueError(f"llama-cpp GGUF file is missing: {model_path}")
    return model_path


def _default_served_model_name(spec: LocalModelSpec, model_path: Path) -> str:
    if spec.aliases:
        preferred = {
            "qwen3.6-safe": "qwen3.6-safe-local",
            "qwen3.6-balanced": "qwen3.6-balanced-local",
            "qwen3.6-quality": "qwen3.6-quality-local",
            "qwen36-max": "qwen36-max-local",
            "qwen3-vl-8b": "qwen3-vl-local",
        }.get(spec.aliases[0])
        if preferred is not None:
            return preferred
    if spec.backend == "llama-cpp" and spec.aliases:
        return spec.aliases[0]
    return str(model_path)


def _validate_extra_args(extra_args: tuple[str, ...]) -> None:
    for arg in extra_args:
        option = arg.split("=", 1)[0]
        if option in RESERVED_EXTRA_ARGS:
            raise ValueError(
                f"Extra backend args cannot override reserved server option '{option}'"
            )


def _offline_env() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
