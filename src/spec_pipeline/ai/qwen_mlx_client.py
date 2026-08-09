from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult
from spec_pipeline.ai.json_utils import extract_json_payload
from spec_pipeline.ai.prompt import build_extraction_prompt
from spec_pipeline.ai.schema import normalize_ai_rows_values, validate_ai_rows
from spec_pipeline.ai.vision_repair import build_vision_repair_prompt
from spec_pipeline.local.model_manager import ModelCacheManager, default_model_cache_dir


class MlxGenerator(Protocol):
    def generate(
        self,
        *,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
        image_paths: list[Path] | None = None,
    ) -> str: ...


@dataclass(slots=True)
class QwenMlxClientConfig:
    model: str = "qwen3-8b"
    max_tokens: int = 8192
    temperature: float = 0.0
    attach_images: bool = True
    model_cache_dir: Path | None = None
    project_root: Path | None = None


class QwenMlxExtractionClient:
    client_name = "qwen-mlx"

    def __init__(
        self,
        *,
        config: QwenMlxClientConfig | None = None,
        model_cache_manager: ModelCacheManager | None = None,
        generator: MlxGenerator | None = None,
    ) -> None:
        self.config = config or QwenMlxClientConfig()
        cache_dir = self.config.model_cache_dir or default_model_cache_dir(
            self.config.project_root or Path.cwd()
        )
        self.model_cache_manager = model_cache_manager or ModelCacheManager(cache_dir=cache_dir)
        self.generator = generator

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult:
        prompt = build_extraction_prompt(sample_id=request.sample_id, pages=request.pages)
        return self._generate_rows(prompt=prompt, request=request, purpose="extract")

    def supports_vision_repair(self, request: AiExtractionRequest) -> bool:
        spec = self.model_cache_manager.manifest.resolve(self.config.model)
        return (
            self.config.attach_images
            and spec.backend == "mlx-vlm"
            and bool(_image_paths_for_request(request))
        )

    def repair_rows(
        self,
        *,
        request: AiExtractionRequest,
        rows: list[dict[str, object]],
        issues: object,
    ) -> AiExtractionResult:
        prompt = build_vision_repair_prompt(
            sample_id=request.sample_id,
            pages=request.pages,
            rows=rows,
            issues=list(issues) if isinstance(issues, list | tuple) else [],
        )
        return self._generate_rows(prompt=prompt, request=request, purpose="vision_repair")

    def _generate_rows(
        self, *, prompt: str, request: AiExtractionRequest, purpose: str
    ) -> AiExtractionResult:
        spec = self.model_cache_manager.manifest.resolve(self.config.model)
        if spec.backend not in {"mlx", "mlx-vlm"}:
            raise ValueError(
                f"Model '{self.config.model}' uses backend '{spec.backend}', not an MLX backend"
            )
        model_path = self.model_cache_manager.resolve_for_run(self.config.model)
        generator = self.generator or _default_generator_for_backend(spec.backend)
        image_paths = _image_paths_for_request(request) if spec.backend == "mlx-vlm" else []
        if not self.config.attach_images:
            image_paths = []
        text = generator.generate(
            model_path=model_path,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            image_paths=image_paths,
        )
        payload = extract_json_payload(text)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Qwen MLX response JSON must contain a rows array")
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Qwen MLX response rows must all be JSON objects")
        normalized_rows = normalize_ai_rows_values(cast(list[dict[str, object]], rows))
        validate_ai_rows(normalized_rows)
        warnings = payload.get("warnings")
        return AiExtractionResult(
            rows=normalized_rows,
            warnings=[item for item in warnings if isinstance(item, str)]
            if isinstance(warnings, list)
            else [],
            raw_response={
                "stdout": text,
                "model": self.config.model,
                "model_path": str(model_path),
                "client": self.client_name,
                "backend": spec.backend,
                "purpose": purpose,
                "image_paths": [str(path) for path in image_paths],
            },
        )


class DefaultMlxTextGenerator:
    def __init__(self) -> None:
        self._loaded: tuple[Path, Any, Any] | None = None

    def generate(
        self,
        *,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
        image_paths: list[Path] | None = None,
    ) -> str:
        try:
            from mlx_lm.generate import generate
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install the macos-local extra before using qwen-mlx."
            ) from exc

        if self._loaded is None or self._loaded[0] != model_path:
            model, tokenizer = load(str(model_path))
            self._loaded = (model_path, model, tokenizer)
        _, model, tokenizer = self._loaded
        sampler = make_sampler(temp=temperature)
        formatted_prompt = _format_chat_prompt(tokenizer, prompt)
        return generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )


class DefaultMlxVlmGenerator:
    def __init__(self) -> None:
        self._loaded: tuple[Path, Any, Any] | None = None

    def generate(
        self,
        *,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
        image_paths: list[Path] | None = None,
    ) -> str:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise RuntimeError(
                "mlx-vlm is not installed. Install the macos-local extra before using Qwen3.6."
            ) from exc

        if self._loaded is None or self._loaded[0] != model_path:
            model, processor = load(str(model_path))
            self._loaded = (model_path, model, processor)
        _, model, processor = self._loaded
        images = [str(path) for path in image_paths or []]
        formatted_prompt = apply_chat_template(
            processor,
            getattr(model, "config", {}),
            prompt,
            add_generation_prompt=True,
            num_images=len(images),
            enable_thinking=False,
        )
        result = generate(
            model,
            processor,
            formatted_prompt,
            image=images or None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        return str(getattr(result, "text", result))


def _default_generator_for_backend(backend: str) -> MlxGenerator:
    if backend == "mlx-vlm":
        return DefaultMlxVlmGenerator()
    return DefaultMlxTextGenerator()


def _image_paths_for_request(request: AiExtractionRequest) -> list[Path]:
    return [
        Path(page.image_path)
        for page in request.pages
        if page.image_path is not None and Path(page.image_path).exists()
    ]


def _format_chat_prompt(tokenizer: object, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt
    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only specification extraction engine. Return only valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return str(
            apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
