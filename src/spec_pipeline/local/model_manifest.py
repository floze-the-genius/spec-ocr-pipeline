from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelFileSpec:
    path: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LocalModelSpec:
    key: str
    aliases: tuple[str, ...]
    title: str
    hf_id: str
    revision: str
    backend: str
    path: str
    platforms: tuple[str, ...]
    disk_gb: float
    min_ram_gb: int | None = None
    min_vram_gb: int | None = None
    files: tuple[ModelFileSpec, ...] = field(default_factory=tuple)
    file_patterns: tuple[str, ...] = field(default_factory=tuple)
    download_patterns: tuple[str, ...] = field(default_factory=tuple)
    server_args: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized == self.key.lower() or normalized in {
            alias.lower() for alias in self.aliases
        }


@dataclass(frozen=True, slots=True)
class ModelManifest:
    models: tuple[LocalModelSpec, ...]

    def resolve(self, model: str) -> LocalModelSpec:
        for spec in self.models:
            if spec.matches(model):
                return spec
        known = ", ".join(sorted(alias for spec in self.models for alias in spec.aliases))
        raise KeyError(f"Unknown local model '{model}'. Known aliases: {known}")

    def to_dict(self) -> dict[str, object]:
        return {
            "models": [
                {
                    "key": spec.key,
                    "aliases": list(spec.aliases),
                    "title": spec.title,
                    "hf_id": spec.hf_id,
                    "revision": spec.revision,
                    "backend": spec.backend,
                    "path": spec.path,
                    "platforms": list(spec.platforms),
                    "disk_gb": spec.disk_gb,
                    "min_ram_gb": spec.min_ram_gb,
                    "min_vram_gb": spec.min_vram_gb,
                    "files": [
                        {
                            "path": file.path,
                            "size_bytes": file.size_bytes,
                            "sha256": file.sha256,
                        }
                        for file in spec.files
                    ],
                    "file_patterns": list(spec.file_patterns),
                    "download_patterns": list(spec.download_patterns),
                    "server_args": list(spec.server_args),
                }
                for spec in self.models
            ]
        }


def default_model_manifest() -> ModelManifest:
    return ModelManifest(
        models=(
            LocalModelSpec(
                key="paddleocr_vl",
                aliases=("paddleocr-vl", "paddle"),
                title="PaddleOCR-VL",
                hf_id="PaddlePaddle/PaddleOCR-VL",
                revision="main",
                backend="paddle",
                path="paddleocr/PaddleOCR-VL",
                platforms=("macos", "windows-wsl"),
                disk_gb=3.0,
                files=(
                    ModelFileSpec("config.json"),
                    ModelFileSpec("PP-DocLayoutV2/inference.yml"),
                    ModelFileSpec("PP-DocLayoutV2/inference.pdmodel"),
                    ModelFileSpec("PP-DocLayoutV2/inference.pdiparams"),
                ),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_8b_mlx",
                aliases=("qwen3-8b", "qwen3-8b-mlx"),
                title="Qwen3 8B MLX 4-bit",
                hf_id="mlx-community/Qwen3-8B-4bit",
                revision="main",
                backend="mlx",
                path="mlx/mlx-community--Qwen3-8B-4bit",
                platforms=("macos",),
                disk_gb=6.0,
                min_ram_gb=16,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_14b_mlx",
                aliases=("qwen3-14b", "qwen3-14b-mlx"),
                title="Qwen3 14B MLX 4-bit",
                hf_id="mlx-community/Qwen3-14B-4bit",
                revision="main",
                backend="mlx",
                path="mlx/mlx-community--Qwen3-14B-4bit",
                platforms=("macos",),
                disk_gb=10.0,
                min_ram_gb=32,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_6_35b_a3b_mlx_vlm",
                aliases=(
                    "qwen3.6-35b-a3b",
                    "qwen3.6-35b-a3b-mlx-vlm",
                    "qwen3.6-35b-a3b-4bit",
                    "qwen3.6-35b",
                ),
                title="Qwen3.6 35B A3B MLX-VLM 4-bit",
                hf_id="mlx-community/Qwen3.6-35B-A3B-4bit",
                revision="main",
                backend="mlx-vlm",
                path="mlx-vlm/mlx-community--Qwen3.6-35B-A3B-4bit",
                platforms=("macos",),
                disk_gb=21.0,
                min_ram_gb=64,
                files=(
                    ModelFileSpec("config.json"),
                    ModelFileSpec("tokenizer.json"),
                    ModelFileSpec("processor_config.json"),
                    ModelFileSpec("preprocessor_config.json"),
                ),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gguf_iq3_xxs",
                aliases=(
                    "qwen3.6-safe",
                    "qwen36-safe",
                    "qwen3.6-27b-safe",
                    "qwen3.6-27b-iq3-gguf",
                    "qwen3.6-27b-iq3-xxs-gguf",
                ),
                title="Qwen3.6 27B GGUF IQ3_XXS (safe 16GB)",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="main",
                backend="llama-cpp",
                path="huggingface/bartowski--Qwen_Qwen3.6-27B-GGUF-IQ3_XXS",
                platforms=("windows-wsl", "linux"),
                disk_gb=12.5,
                min_ram_gb=32,
                files=(ModelFileSpec("Qwen_Qwen3.6-27B-IQ3_XXS.gguf", 12387788352),),
                download_patterns=("Qwen_Qwen3.6-27B-IQ3_XXS.gguf", "README.md"),
                server_args=(
                    "--n_ctx",
                    "12288",
                    "--n_gpu_layers",
                    "-1",
                    "--chat_template_kwargs",
                    '{"enable_thinking": false}',
                ),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gguf_q4_k_s",
                aliases=(
                    "qwen3.6-balanced",
                    "qwen36-balanced",
                    "qwen3.6-27b-balanced",
                    "qwen3.6-27b-q4ks-gguf",
                    "qwen3.6-27b-q4-k-s-gguf",
                ),
                title="Qwen3.6 27B GGUF Q4_K_S (balanced)",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="main",
                backend="llama-cpp",
                path="huggingface/bartowski--Qwen_Qwen3.6-27B-GGUF-Q4_K_S",
                platforms=("windows-wsl", "linux"),
                disk_gb=16.5,
                min_ram_gb=48,
                files=(ModelFileSpec("Qwen_Qwen3.6-27B-Q4_K_S.gguf", 16474162752),),
                download_patterns=("Qwen_Qwen3.6-27B-Q4_K_S.gguf", "README.md"),
                server_args=(
                    "--n_ctx",
                    "8192",
                    "--n_gpu_layers",
                    "0",
                    "--chat_template_kwargs",
                    '{"enable_thinking": false}',
                ),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gguf_q4_k_m",
                aliases=(
                    "qwen3.6-quality",
                    "qwen36-quality",
                    "qwen3.6-27b-quality",
                    "qwen3.6-27b-q4km-gguf",
                    "qwen3.6-27b-q4-k-m-gguf",
                ),
                title="Qwen3.6 27B GGUF Q4_K_M (quality)",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="main",
                backend="llama-cpp",
                path="huggingface/bartowski--Qwen_Qwen3.6-27B-GGUF-Q4_K_M",
                platforms=("windows-wsl", "linux"),
                disk_gb=17.6,
                min_ram_gb=48,
                files=(ModelFileSpec("Qwen_Qwen3.6-27B-Q4_K_M.gguf", 17533552192),),
                download_patterns=("Qwen_Qwen3.6-27B-Q4_K_M.gguf", "README.md"),
                server_args=(
                    "--n_ctx",
                    "8192",
                    "--n_gpu_layers",
                    "0",
                    "--chat_template_kwargs",
                    '{"enable_thinking": false}',
                ),
            ),
            LocalModelSpec(
                key="qwen3_8b_cuda",
                aliases=("qwen3-8b-cuda", "qwen3-8b-windows"),
                title="Qwen3 8B CUDA",
                hf_id="Qwen/Qwen3-8B",
                revision="main",
                backend="vllm",
                path="huggingface/Qwen--Qwen3-8B",
                platforms=("windows-wsl", "linux"),
                disk_gb=8.0,
                min_vram_gb=12,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_14b_cuda",
                aliases=("qwen3-14b-cuda", "qwen3-14b-windows"),
                title="Qwen3 14B CUDA Quantized",
                hf_id="Qwen/Qwen3-14B",
                revision="main",
                backend="vllm",
                path="huggingface/Qwen--Qwen3-14B",
                platforms=("windows-wsl", "linux"),
                disk_gb=12.0,
                min_vram_gb=16,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_vl_8b_cuda",
                aliases=("qwen3-vl-8b", "qwen3-vl-8b-cuda"),
                title="Qwen3-VL 8B CUDA fallback",
                hf_id="Qwen/Qwen3-VL-8B-Instruct",
                revision="main",
                backend="sglang",
                path="huggingface/Qwen--Qwen3-VL-8B-Instruct",
                platforms=("windows-wsl", "linux"),
                disk_gb=10.0,
                min_vram_gb=16,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gptq_cuda",
                aliases=(
                    "qwen36-max",
                    "qwen36-max-vllm",
                    "qwen3.6-vllm-experimental",
                ),
                title="Qwen3.6 Max 27B GPTQ 4-bit vLLM (experimental)",
                hf_id="groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
                revision="main",
                backend="vllm",
                path="huggingface/groxaxo--Qwen3.6-27B-GPTQ-Pro-4bit",
                platforms=("windows-wsl", "linux"),
                disk_gb=19.0,
                min_ram_gb=48,
                min_vram_gb=24,
                files=(ModelFileSpec("config.json"), ModelFileSpec("tokenizer.json")),
                file_patterns=("*.safetensors",),
                server_args=(
                    "--dtype",
                    "float16",
                    "--quantization",
                    "gptq_marlin",
                    "--max-model-len",
                    "8192",
                    "--gpu-memory-utilization",
                    "0.90",
                    "--language-model-only",
                ),
            ),
        )
    )
