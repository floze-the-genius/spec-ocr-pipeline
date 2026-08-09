from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from spec_pipeline.local.model_manager import ModelCacheManager, ModelNotReadyError
from spec_pipeline.local.model_manifest import LocalModelSpec, ModelFileSpec, ModelManifest
from spec_pipeline.local.server_launcher import (
    LocalServerLaunchConfig,
    build_local_server_command,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen3_8b_cuda",
                aliases=("qwen3-8b-cuda",),
                title="Qwen3 8B CUDA",
                hf_id="Qwen/Qwen3-8B",
                revision="rev-111",
                backend="vllm",
                path="hf/qwen3-8b",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
            LocalModelSpec(
                key="qwen3_vl_8b_cuda",
                aliases=("qwen3-vl-8b",),
                title="Qwen3-VL 8B CUDA",
                hf_id="Qwen/Qwen3-VL-8B-Instruct",
                revision="rev-222",
                backend="sglang",
                path="hf/qwen3-vl-8b",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
            LocalModelSpec(
                key="qwen3_8b_mlx",
                aliases=("qwen3-8b",),
                title="Qwen3 8B MLX",
                hf_id="mlx-community/Qwen3-8B-4bit",
                revision="rev-333",
                backend="mlx",
                path="mlx/qwen3-8b",
                platforms=("macos",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gguf_iq3_xxs",
                aliases=("qwen3.6-safe",),
                title="Qwen3.6 27B GGUF IQ3_XXS",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="rev-444",
                backend="llama-cpp",
                path="gguf/qwen3.6-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("model.gguf"),),
                server_args=("--n_ctx", "8192", "--n_gpu_layers", "0"),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gptq_cuda",
                aliases=("qwen36-max", "qwen36-max-vllm"),
                title="Qwen3.6 Max 27B GPTQ",
                hf_id="groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
                revision="rev-555",
                backend="vllm",
                path="hf/qwen36-max",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
        )
    )


def _install_ready_model(cache_dir: Path, *, key: str, hf_id: str, path: str) -> Path:
    model_dir = cache_dir / path
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
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


def test_build_vllm_command_uses_ready_model_path(tmp_path: Path) -> None:
    model_dir = _install_ready_model(
        tmp_path, key="qwen3_8b_cuda", hf_id="Qwen/Qwen3-8B", path="hf/qwen3-8b"
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    command = build_local_server_command(
        LocalServerLaunchConfig(
            model="qwen3-8b-cuda",
            cache_dir=tmp_path,
            host="0.0.0.0",
            port=9000,
            extra_args=("--max-model-len", "8192"),
        ),
        manager=manager,
    )

    assert command.backend == "vllm"
    assert command.model_path == model_dir.resolve()
    assert command.served_model_name == str(model_dir.resolve())
    assert command.base_url == "http://0.0.0.0:9000/v1"
    assert command.args[:4] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
    ]
    assert command.args[4] == str(model_dir.resolve())
    assert command.args[-2:] == ["--max-model-len", "8192"]
    assert command.env["HF_HUB_OFFLINE"] == "1"
    assert command.env["TRANSFORMERS_OFFLINE"] == "1"


def test_build_sglang_command_uses_model_path_flag(tmp_path: Path) -> None:
    model_dir = _install_ready_model(
        tmp_path,
        key="qwen3_vl_8b_cuda",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        path="hf/qwen3-vl-8b",
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    command = build_local_server_command(
        LocalServerLaunchConfig(
            model="qwen3-vl-8b",
            cache_dir=tmp_path,
            served_model_name="qwen3-vl-local",
        ),
        manager=manager,
    )

    assert command.backend == "sglang"
    assert command.model_path == model_dir.resolve()
    assert command.served_model_name == "qwen3-vl-local"
    assert command.args[:4] == [sys.executable, "-m", "sglang.launch_server", "--model-path"]
    assert command.args[4] == str(model_dir.resolve())


def test_build_llama_cpp_command_uses_gguf_file_and_defaults(tmp_path: Path) -> None:
    model_dir = tmp_path / "gguf" / "qwen3.6-safe"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_bytes(b"fake gguf")
    lock_path = tmp_path / "manifest.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_6_27b_gguf_iq3_xxs": {
                        "hf_id": "bartowski/Qwen_Qwen3.6-27B-GGUF",
                        "revision": "rev-444",
                        "path": "gguf/qwen3.6-safe",
                        "status": "ready",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 9,
                        "files_checked": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    command = build_local_server_command(
        LocalServerLaunchConfig(model="qwen3.6-safe", cache_dir=tmp_path), manager=manager
    )

    assert command.backend == "llama-cpp"
    assert command.model_path == gguf_path.resolve()
    assert command.served_model_name == "qwen3.6-safe-local"
    assert command.args[:4] == [sys.executable, "-m", "llama_cpp.server", "--model"]
    assert command.args[4] == str(gguf_path.resolve())
    assert "--model_alias" in command.args
    assert "--n_ctx" in command.args
    assert "8192" in command.args


def test_default_qwen36_safe_manifest_offloads_all_layers_for_16gb_gpu() -> None:
    from spec_pipeline.local.model_manifest import default_model_manifest

    spec = default_model_manifest().resolve("qwen36-safe")

    assert spec.server_args == (
        "--n_ctx",
        "12288",
        "--n_gpu_layers",
        "-1",
        "--chat_template_kwargs",
        '{"enable_thinking": false}',
    )


def test_build_qwen36_max_command_defaults_to_friendly_served_name(tmp_path: Path) -> None:
    _install_ready_model(
        tmp_path,
        key="qwen3_6_27b_gptq_cuda",
        hf_id="groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
        path="hf/qwen36-max",
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    command = build_local_server_command(
        LocalServerLaunchConfig(model="qwen36-max-vllm", cache_dir=tmp_path), manager=manager
    )

    assert command.backend == "vllm"
    assert command.served_model_name == "qwen36-max-local"


def test_missing_model_does_not_download(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    with pytest.raises(ModelNotReadyError):
        build_local_server_command(
            LocalServerLaunchConfig(model="qwen3-8b-cuda", cache_dir=tmp_path),
            manager=manager,
        )


def test_rejects_non_server_backend(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    with pytest.raises(ValueError, match="not a supported local server backend"):
        build_local_server_command(
            LocalServerLaunchConfig(model="qwen3-8b", cache_dir=tmp_path),
            manager=manager,
        )


def test_backend_override_must_match_manifest(tmp_path: Path) -> None:
    _install_ready_model(tmp_path, key="qwen3_8b_cuda", hf_id="Qwen/Qwen3-8B", path="hf/qwen3-8b")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    with pytest.raises(ValueError, match="cannot launch"):
        build_local_server_command(
            LocalServerLaunchConfig(model="qwen3-8b-cuda", cache_dir=tmp_path, backend="sglang"),
            manager=manager,
        )


def test_extra_args_cannot_override_model_path(tmp_path: Path) -> None:
    _install_ready_model(tmp_path, key="qwen3_8b_cuda", hf_id="Qwen/Qwen3-8B", path="hf/qwen3-8b")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())

    with pytest.raises(ValueError, match="reserved server option"):
        build_local_server_command(
            LocalServerLaunchConfig(
                model="qwen3-8b-cuda",
                cache_dir=tmp_path,
                extra_args=("--model", "Qwen/Qwen3-8B"),
            ),
            manager=manager,
        )

    with pytest.raises(ValueError, match="reserved server option"):
        build_local_server_command(
            LocalServerLaunchConfig(
                model="qwen3-8b-cuda",
                cache_dir=tmp_path,
                extra_args=("--model-path=Qwen/Qwen3-8B",),
            ),
            manager=manager,
        )
