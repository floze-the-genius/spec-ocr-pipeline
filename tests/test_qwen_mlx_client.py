from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from spec_pipeline.ai.base import AiExtractionRequest
from spec_pipeline.ai.qwen_mlx_client import (
    DefaultMlxTextGenerator,
    DefaultMlxVlmGenerator,
    QwenMlxClientConfig,
    QwenMlxExtractionClient,
)
from spec_pipeline.local.model_manager import ModelCacheManager, ModelNotReadyError
from spec_pipeline.local.model_manifest import LocalModelSpec, ModelFileSpec, ModelManifest
from spec_pipeline.ocr.base import OcrPage


def _manifest() -> ModelManifest:
    return ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen3_8b_mlx",
                aliases=("qwen3-8b",),
                title="Qwen3 8B MLX 4-bit",
                hf_id="mlx-community/Qwen3-8B-4bit",
                revision="rev-111",
                backend="mlx",
                path="mlx/qwen3-8b",
                platforms=("macos",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
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
                key="qwen3_6_35b_a3b_mlx_vlm",
                aliases=("qwen3.6-35b-a3b",),
                title="Qwen3.6 35B A3B MLX-VLM 4-bit",
                hf_id="mlx-community/Qwen3.6-35B-A3B-4bit",
                revision="rev-222",
                backend="mlx-vlm",
                path="mlx-vlm/qwen3.6-35b-a3b",
                platforms=("macos",),
                disk_gb=1,
                files=(
                    ModelFileSpec("config.json"),
                    ModelFileSpec("tokenizer.json"),
                    ModelFileSpec("processor_config.json"),
                    ModelFileSpec("preprocessor_config.json"),
                ),
                file_patterns=("*.safetensors",),
            ),
        )
    )


def _install_ready_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "mlx" / "qwen3-8b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_8b_mlx": {
                        "hf_id": "mlx-community/Qwen3-8B-4bit",
                        "revision": "rev-111",
                        "path": "mlx/qwen3-8b",
                        "status": "ready",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 2,
                        "files_checked": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _install_ready_vlm_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "mlx-vlm" / "qwen3.6-35b-a3b"
    model_dir.mkdir(parents=True)
    for filename in [
        "config.json",
        "tokenizer.json",
        "processor_config.json",
        "preprocessor_config.json",
    ]:
        (model_dir / filename).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_6_35b_a3b_mlx_vlm": {
                        "hf_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                        "revision": "rev-222",
                        "path": "mlx-vlm/qwen3.6-35b-a3b",
                        "status": "ready",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 2,
                        "files_checked": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class FakeGenerator:
    def __init__(self, response: str | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.response = response

    def generate(
        self,
        *,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
        image_paths: list[Path] | None = None,
    ) -> str:
        self.calls.append(
            {
                "model_path": model_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "image_paths": image_paths or [],
            }
        )
        return self.response or (
            '```json\n{"rows":[{"page_ref":"1","section":null,"subsection":null,'
            '"pos":"1","name":"Диффузор","type_mark":null,"code":null,'
            '"manufacturer":null,"unit":"шт","quantity":"2","mass_kg":null,'
            '"note":null}],"warnings":[]}\n```'
        )


def test_qwen_mlx_client_uses_ready_local_model_and_parses_rows(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    generator = FakeGenerator()
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3-8b", max_tokens=1234, temperature=0.0),
        model_cache_manager=manager,
        generator=generator,
    )

    result = client.extract_rows(
        AiExtractionRequest(
            sample_id="sample",
            source_pdf=tmp_path / "input.pdf",
            pages=[
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="1 Диффузор шт 2",
                    tokens=[],
                    provider="test",
                    image_path=None,
                )
            ],
        )
    )

    assert result.rows[0]["name"] == "Диффузор"
    assert result.raw_response["client"] == "qwen-mlx"
    assert generator.calls[0]["model_path"] == tmp_path / "mlx" / "qwen3-8b"
    assert generator.calls[0]["max_tokens"] == 1234
    assert "Return exactly this JSON shape" in str(generator.calls[0]["prompt"])


def test_qwen_mlx_client_does_not_download_missing_model(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3-8b"),
        model_cache_manager=manager,
        generator=FakeGenerator(),
    )

    try:
        client.extract_rows(
            AiExtractionRequest(
                sample_id="sample",
                source_pdf=tmp_path / "input.pdf",
                pages=[],
            )
        )
    except ModelNotReadyError:
        pass
    else:
        raise AssertionError("missing local model must fail instead of downloading")


def test_qwen_mlx_client_rejects_non_mlx_model_alias(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3-8b-cuda"),
        model_cache_manager=manager,
        generator=FakeGenerator(),
    )

    try:
        client.extract_rows(
            AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
        )
    except ValueError as exc:
        assert "not an MLX backend" in str(exc)
    else:
        raise AssertionError("qwen-mlx must reject non-MLX model aliases")


def test_qwen_mlx_client_passes_page_images_for_vlm_model(tmp_path: Path) -> None:
    _install_ready_vlm_model(tmp_path)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    generator = FakeGenerator()
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3.6-35b-a3b"),
        model_cache_manager=manager,
        generator=generator,
    )

    client.extract_rows(
        AiExtractionRequest(
            sample_id="sample",
            source_pdf=tmp_path / "input.pdf",
            pages=[
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="1 Диффузор шт 2",
                    tokens=[],
                    provider="test",
                    image_path=str(image_path),
                )
            ],
        )
    )

    assert generator.calls[0]["model_path"] == tmp_path / "mlx-vlm" / "qwen3.6-35b-a3b"
    assert generator.calls[0]["image_paths"] == [image_path]


def test_qwen_mlx_client_can_disable_vlm_images(tmp_path: Path) -> None:
    _install_ready_vlm_model(tmp_path)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    generator = FakeGenerator()
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3.6-35b-a3b", attach_images=False),
        model_cache_manager=manager,
        generator=generator,
    )

    client.extract_rows(
        AiExtractionRequest(
            sample_id="sample",
            source_pdf=tmp_path / "input.pdf",
            pages=[
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="1 Диффузор шт 2",
                    tokens=[],
                    provider="test",
                    image_path=str(image_path),
                )
            ],
        )
    )

    assert generator.calls[0]["image_paths"] == []


def test_qwen_mlx_client_rejects_non_object_rows(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3-8b"),
        model_cache_manager=manager,
        generator=FakeGenerator('{"rows":["bad"],"warnings":[]}'),
    )

    try:
        client.extract_rows(
            AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
        )
    except ValueError as exc:
        assert "rows must all be JSON objects" in str(exc)
    else:
        raise AssertionError("non-object rows must fail validation")


def test_qwen_mlx_client_rejects_schema_invalid_rows(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = QwenMlxExtractionClient(
        config=QwenMlxClientConfig(model="qwen3-8b"),
        model_cache_manager=manager,
        generator=FakeGenerator('{"rows":[{"page_ref":"1"}],"warnings":[]}'),
    )

    try:
        client.extract_rows(
            AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
        )
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("schema-invalid rows must fail in qwen client")


def test_default_mlx_text_generator_uses_sampler_api(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}
    mlx_package = types.ModuleType("mlx_lm")
    generate_module = types.ModuleType("mlx_lm.generate")
    sample_utils_module = types.ModuleType("mlx_lm.sample_utils")
    utils_module = types.ModuleType("mlx_lm.utils")

    def fake_load(model_path: str):
        calls["load_path"] = model_path
        return "model", "tokenizer"

    def fake_make_sampler(*, temp: float):
        calls["sampler_temp"] = temp
        return "sampler"

    def fake_generate(model, tokenizer, **kwargs):
        calls["generate_kwargs"] = kwargs
        assert "temp" not in kwargs
        assert kwargs["sampler"] == "sampler"
        return '{"rows": [], "warnings": []}'

    setattr(utils_module, "load", fake_load)
    setattr(sample_utils_module, "make_sampler", fake_make_sampler)
    setattr(generate_module, "generate", fake_generate)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils_module)

    generator = DefaultMlxTextGenerator()
    output = generator.generate(
        model_path=tmp_path / "model",
        prompt="prompt",
        max_tokens=10,
        temperature=0.2,
    )

    assert json.loads(output) == {"rows": [], "warnings": []}
    assert calls["load_path"] == str(tmp_path / "model")
    assert calls["sampler_temp"] == 0.2
    assert calls["generate_kwargs"] == {
        "prompt": "prompt",
        "max_tokens": 10,
        "sampler": "sampler",
    }


def test_default_mlx_text_generator_uses_chat_template_without_thinking(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, object] = {}
    mlx_package = types.ModuleType("mlx_lm")
    generate_module = types.ModuleType("mlx_lm.generate")
    sample_utils_module = types.ModuleType("mlx_lm.sample_utils")
    utils_module = types.ModuleType("mlx_lm.utils")

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "formatted prompt"

    def fake_load(model_path: str):
        return "model", FakeTokenizer()

    def fake_make_sampler(*, temp: float):
        return "sampler"

    def fake_generate(model, tokenizer, **kwargs):
        calls["generate_kwargs"] = kwargs
        return '{"rows": [], "warnings": []}'

    setattr(utils_module, "load", fake_load)
    setattr(sample_utils_module, "make_sampler", fake_make_sampler)
    setattr(generate_module, "generate", fake_generate)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils_module)

    DefaultMlxTextGenerator().generate(
        model_path=tmp_path / "model",
        prompt="prompt",
        max_tokens=10,
        temperature=0.0,
    )

    assert calls["template_kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert calls["generate_kwargs"] == {
        "prompt": "formatted prompt",
        "max_tokens": 10,
        "sampler": "sampler",
    }


def test_default_mlx_vlm_generator_uses_images_and_disables_thinking(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, object] = {}
    mlx_vlm_module = types.ModuleType("mlx_vlm")
    prompt_utils_module = types.ModuleType("mlx_vlm.prompt_utils")

    class FakeModel:
        config = {"model_type": "qwen3_5_moe"}

    class FakeResult:
        text = '{"rows": [], "warnings": []}'

    def fake_load(model_path: str):
        calls["load_path"] = model_path
        return FakeModel(), "processor"

    def fake_apply_chat_template(processor, config, prompt, **kwargs):
        calls["template"] = {
            "processor": processor,
            "config": config,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        return "formatted vlm prompt"

    def fake_generate(model, processor, prompt, **kwargs):
        calls["generate"] = {
            "model": model,
            "processor": processor,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        return FakeResult()

    setattr(mlx_vlm_module, "load", fake_load)
    setattr(mlx_vlm_module, "generate", fake_generate)
    setattr(prompt_utils_module, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils_module)

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    output = DefaultMlxVlmGenerator().generate(
        model_path=tmp_path / "model",
        prompt="prompt",
        max_tokens=10,
        temperature=0.7,
        image_paths=[image_path],
    )

    assert json.loads(output) == {"rows": [], "warnings": []}
    assert calls["load_path"] == str(tmp_path / "model")
    assert calls["template"] == {
        "processor": "processor",
        "config": {"model_type": "qwen3_5_moe"},
        "prompt": "prompt",
        "kwargs": {
            "add_generation_prompt": True,
            "num_images": 1,
            "enable_thinking": False,
        },
    }
    generate_call = calls["generate"]
    assert isinstance(generate_call, dict)
    assert generate_call["prompt"] == "formatted vlm prompt"
    assert generate_call["kwargs"] == {
        "image": [str(image_path)],
        "max_tokens": 10,
        "temperature": 0.7,
        "verbose": False,
    }
