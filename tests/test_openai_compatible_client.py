from __future__ import annotations

import json
from pathlib import Path

import pytest

from spec_pipeline.ai.base import AiExtractionRequest
from spec_pipeline.ai.openai_compatible_client import (
    OpenAiCompatibleClientConfig,
    OpenAiCompatibleExtractionClient,
    UrlLibOpenAiCompatibleHttpClient,
)
from spec_pipeline.local.model_manager import ModelCacheManager, ModelNotReadyError
from spec_pipeline.local.model_manifest import LocalModelSpec, ModelFileSpec, ModelManifest
from spec_pipeline.ocr.base import OcrPage


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
                key="qwen3_8b_mlx",
                aliases=("qwen3-8b",),
                title="Qwen3 8B MLX",
                hf_id="mlx-community/Qwen3-8B-4bit",
                revision="rev-111",
                backend="mlx",
                path="mlx/qwen3-8b",
                platforms=("macos",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gguf_iq3_xxs",
                aliases=("qwen3.6-safe",),
                title="Qwen3.6 Safe GGUF",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="rev-111",
                backend="llama-cpp",
                path="gguf/qwen3.6-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("model.gguf"),),
            ),
            LocalModelSpec(
                key="qwen3_6_27b_gptq_cuda",
                aliases=("qwen36-max", "qwen36-max-vllm"),
                title="Qwen3.6 Max GPTQ",
                hf_id="groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
                revision="rev-111",
                backend="vllm",
                path="hf/qwen36-max",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
            LocalModelSpec(
                key="qwen3_vl_8b_cuda",
                aliases=("qwen3-vl-8b", "qwen3-vl-8b-cuda"),
                title="Qwen3-VL 8B CUDA",
                hf_id="Qwen/Qwen3-VL-8B-Instruct",
                revision="rev-111",
                backend="sglang",
                path="hf/qwen3-vl-8b",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
            ),
        )
    )


def _install_ready_cuda_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "hf" / "qwen3-8b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_8b_cuda": {
                        "hf_id": "Qwen/Qwen3-8B",
                        "revision": "rev-111",
                        "path": "hf/qwen3-8b",
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


def _install_ready_llama_cpp_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "gguf" / "qwen3.6-safe"
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_text("model", encoding="utf-8")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_6_27b_gguf_iq3_xxs": {
                        "hf_id": "bartowski/Qwen_Qwen3.6-27B-GGUF",
                        "revision": "rev-111",
                        "path": "gguf/qwen3.6-safe",
                        "status": "ready",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 5,
                        "files_checked": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _install_ready_qwen36_max_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "hf" / "qwen36-max"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_6_27b_gptq_cuda": {
                        "hf_id": "groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
                        "revision": "rev-111",
                        "path": "hf/qwen36-max",
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


def _install_ready_vl_model(cache_dir: Path) -> None:
    model_dir = cache_dir / "hf" / "qwen3-vl-8b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_vl_8b_cuda": {
                        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
                        "revision": "rev-111",
                        "path": "hf/qwen3-vl-8b",
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


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post_json(
        self, url: str, payload: dict[str, object], timeout_seconds: int
    ) -> dict[str, object]:
        self.calls.append({"url": url, "payload": payload, "timeout_seconds": timeout_seconds})
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"rows":[{"page_ref":"1","section":null,"subsection":null,'
                            '"pos":1,"name":"Диффузор","type_mark":null,"code":null,'
                            '"manufacturer":null,"unit":"шт","quantity":2,'
                            '"mass_kg":null,"note":null}],"warnings":[]}'
                        )
                    }
                }
            ]
        }


def _valid_ai_row_payload() -> dict[str, object]:
    return {
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


class FakeShapeHttpClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def post_json(
        self, url: str, payload: dict[str, object], timeout_seconds: int
    ) -> dict[str, object]:
        self.calls.append({"url": url, "payload": payload, "timeout_seconds": timeout_seconds})
        return {"choices": [{"message": {"content": json.dumps(self.payload, ensure_ascii=False)}}]}


def test_openai_compatible_client_uses_ready_cuda_model_path(tmp_path: Path) -> None:
    _install_ready_cuda_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="qwen3-8b-cuda",
            base_url="http://localhost:8000/v1",
            max_tokens=123,
            timeout_seconds=10,
        ),
        model_cache_manager=manager,
        http_client=http_client,
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
    assert http_client.calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == str(tmp_path / "hf" / "qwen3-8b")
    assert payload["max_tokens"] == 123


def test_openai_compatible_client_does_not_download_missing_model(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3-8b-cuda"),
        model_cache_manager=manager,
        http_client=FakeHttpClient(),
    )

    try:
        client.extract_rows(AiExtractionRequest(sample_id="sample", source_pdf=tmp_path, pages=[]))
    except ModelNotReadyError:
        pass
    else:
        raise AssertionError("missing CUDA model must fail instead of downloading")


def test_openai_compatible_client_accepts_llama_cpp_backend(tmp_path: Path) -> None:
    _install_ready_llama_cpp_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="qwen3.6-safe",
            served_model_name="qwen3.6-safe-local",
            base_url="http://localhost:8000/v1",
        ),
        model_cache_manager=manager,
        http_client=http_client,
    )

    result = client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    assert result.rows[0]["name"] == "Диффузор"
    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3.6-safe-local"
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_compatible_client_defaults_llama_cpp_to_local_served_name(tmp_path: Path) -> None:
    _install_ready_llama_cpp_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="qwen3.6-safe",
            base_url="http://localhost:8000/v1",
        ),
        model_cache_manager=manager,
        http_client=http_client,
    )

    client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3.6-safe-local"


def test_openai_compatible_client_accepts_items_array_response(tmp_path: Path) -> None:
    _install_ready_llama_cpp_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3.6-safe"),
        model_cache_manager=manager,
        http_client=FakeShapeHttpClient({"items": [_valid_ai_row_payload()]}),
    )

    result = client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    assert result.rows[0]["name"] == "Диффузор"


def test_openai_compatible_client_accepts_nested_rows_response(tmp_path: Path) -> None:
    _install_ready_llama_cpp_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3.6-safe"),
        model_cache_manager=manager,
        http_client=FakeShapeHttpClient({"result": {"rows": [_valid_ai_row_payload()]}}),
    )

    result = client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    assert result.rows[0]["quantity"] == "2"


def test_openai_compatible_client_accepts_single_row_object_response(tmp_path: Path) -> None:
    _install_ready_llama_cpp_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3.6-safe"),
        model_cache_manager=manager,
        http_client=FakeShapeHttpClient(_valid_ai_row_payload()),
    )

    result = client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    assert len(result.rows) == 1


def test_openai_compatible_client_defaults_qwen36_max_to_local_served_name(
    tmp_path: Path,
) -> None:
    _install_ready_qwen36_max_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen36-max-vllm"),
        model_cache_manager=manager,
        http_client=http_client,
    )

    client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen36-max-local"


def test_openai_compatible_client_can_use_external_openai_compatible_api(tmp_path: Path) -> None:
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="accounts/fireworks/routers/kimi-k2p6-turbo",
            served_model_name="accounts/fireworks/routers/kimi-k2p6-turbo",
            base_url="https://api.fireworks.ai/inference/v1",
            require_local_model=False,
            api_key_env="FIREWORKS_API_KEY",
            reasoning_effort="none",
            max_tokens=4096,
        ),
        http_client=http_client,
    )

    result = client.extract_rows(
        AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
    )

    assert result.rows[0]["name"] == "Диффузор"
    assert http_client.calls[0]["url"] == "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "accounts/fireworks/routers/kimi-k2p6-turbo"
    assert payload["reasoning_effort"] == "none"
    assert "response_format" not in payload
    assert result.raw_response["external"] is True


def test_openai_compatible_client_rejects_cleartext_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="Refusing cleartext"):
        OpenAiCompatibleExtractionClient(
            config=OpenAiCompatibleClientConfig(
                base_url="http://api.example.invalid/v1",
                require_local_model=False,
                api_key_env="EXAMPLE_API_KEY",
            )
        )


def test_openai_compatible_client_allows_cleartext_loopback_endpoint() -> None:
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(base_url="http://[::1]:8000/v1")
    )

    assert client.config.base_url == "http://[::1]:8000/v1"


def test_openai_compatible_external_extraction_can_attach_images(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="accounts/fireworks/routers/kimi-k2p6-turbo",
            base_url="https://api.fireworks.ai/inference/v1",
            served_model_name="accounts/fireworks/routers/kimi-k2p6-turbo",
            require_local_model=False,
            api_key_env="FIREWORKS_API_KEY",
            external_supports_images=True,
            attach_images=True,
            reasoning_effort="none",
            timeout_seconds=10,
        ),
        http_client=http_client,
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
                    provider="paddleocr-vl",
                    image_path=str(image_path),
                )
            ],
        )
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
    assert payload["reasoning_effort"] == "none"


def test_openai_compatible_client_reviews_third_layer_patches_with_external_images(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    http_client = FakeShapeHttpClient(
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
                }
            ],
            "warnings": [],
        }
    )
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(
            model="accounts/fireworks/routers/kimi-k2p6-turbo",
            base_url="https://api.fireworks.ai/inference/v1",
            served_model_name="accounts/fireworks/routers/kimi-k2p6-turbo",
            require_local_model=False,
            api_key_env="FIREWORKS_API_KEY",
            external_supports_images=True,
            attach_images=True,
            reasoning_effort="none",
            timeout_seconds=10,
        ),
        http_client=http_client,
    )

    result = client.review_row_patches(
        request=AiExtractionRequest(
            sample_id="sample",
            source_pdf=tmp_path / "input.pdf",
            pages=[
                OcrPage(
                    page_index=0,
                    page_ref="1",
                    width=100,
                    height=200,
                    text="OCR",
                    tokens=[],
                    provider="paddleocr-vl",
                    image_path=str(image_path),
                )
            ],
        ),
        rows=[_valid_ai_row_payload() | {"name": "Автоматический воздухомобюксир 015"}],
        issues=[],
    )

    call = http_client.calls[0]
    payload = call["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    content = message["content"]

    assert result.patches[0]["new_value"] == "Автоматический воздухоотводчик Ø15"
    assert payload["reasoning_effort"] == "none"
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
    assert result.raw_response["model_path"] is None


def test_urlib_openai_compatible_client_requires_configured_api_key_env(monkeypatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    client = UrlLibOpenAiCompatibleHttpClient(api_key_env="FIREWORKS_API_KEY")

    try:
        client.post_json("https://api.fireworks.ai/inference/v1/chat/completions", {}, 10)
    except RuntimeError as exc:
        assert "FIREWORKS_API_KEY" in str(exc)
    else:
        raise AssertionError("missing API key env var must fail before HTTP request")


def test_openai_compatible_client_sends_image_content_for_vl_model(tmp_path: Path) -> None:
    _install_ready_vl_model(tmp_path)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3-vl-8b"),
        model_cache_manager=manager,
        http_client=http_client,
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
                    provider="paddleocr-vl",
                    image_path=str(image_path),
                )
            ],
        )
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3-vl-local"
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_compatible_client_sends_text_only_when_images_disabled(tmp_path: Path) -> None:
    _install_ready_vl_model(tmp_path)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3-vl-8b", attach_images=False),
        model_cache_manager=manager,
        http_client=http_client,
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
                    provider="paddleocr-vl",
                    image_path=str(image_path),
                )
            ],
        )
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0]["content"], str)


def test_openai_compatible_client_text_model_ignores_page_images(tmp_path: Path) -> None:
    _install_ready_cuda_model(tmp_path)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    http_client = FakeHttpClient()
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3-8b-cuda"),
        model_cache_manager=manager,
        http_client=http_client,
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
                    provider="paddleocr-vl",
                    image_path=str(image_path),
                )
            ],
        )
    )

    payload = http_client.calls[0]["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0]["content"], str)


def test_openai_compatible_client_rejects_non_server_backend(tmp_path: Path) -> None:
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_manifest())
    client = OpenAiCompatibleExtractionClient(
        config=OpenAiCompatibleClientConfig(model="qwen3-8b"),
        model_cache_manager=manager,
        http_client=FakeHttpClient(),
    )

    try:
        client.extract_rows(AiExtractionRequest(sample_id="sample", source_pdf=tmp_path, pages=[]))
    except ValueError as exc:
        assert "OpenAI-compatible" in str(exc)
    else:
        raise AssertionError("OpenAI-compatible backend must reject MLX aliases")


def test_urlib_openai_compatible_client_wraps_timeout(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "spec_pipeline.ai.openai_compatible_client.urllib.request.urlopen",
        raise_timeout,
    )
    client = UrlLibOpenAiCompatibleHttpClient()

    try:
        client.post_json("http://127.0.0.1:8000/v1/chat/completions", {}, 600)
    except RuntimeError as exc:
        message = str(exc)
        assert "timed out after 600 seconds" in message
        assert "increase --ai-timeout" in message
    else:
        raise AssertionError("timeout must be reported as a RuntimeError")
