from __future__ import annotations

import base64
import ipaddress
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult, AiPatchReviewResult
from spec_pipeline.ai.json_utils import extract_json_payload
from spec_pipeline.ai.prompt import build_extraction_prompt
from spec_pipeline.ai.schema import normalize_ai_rows_values, validate_ai_rows
from spec_pipeline.local.model_manager import ModelCacheManager, default_model_cache_dir
from spec_pipeline.local.model_manifest import LocalModelSpec

try:
    from spec_pipeline.ai.vision_repair import build_vision_patch_prompt, build_vision_repair_prompt
except ModuleNotFoundError:  # Stage-1 packages can omit the optional vision layer.

    def build_vision_repair_prompt(*args, **kwargs) -> str:
        raise RuntimeError("Vision repair layer is not included in this package")

    def build_vision_patch_prompt(*args, **kwargs) -> str:
        raise RuntimeError("Vision repair layer is not included in this package")


class OpenAiCompatibleHttpClient(Protocol):
    def post_json(
        self, url: str, payload: dict[str, object], timeout_seconds: int
    ) -> dict[str, object]: ...


@dataclass(slots=True)
class OpenAiCompatibleClientConfig:
    model: str = "qwen3.6-safe"
    base_url: str = "http://127.0.0.1:8000/v1"
    served_model_name: str | None = None
    require_local_model: bool = True
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    external_supports_images: bool = False
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout_seconds: int = 600
    attach_images: bool = True
    model_cache_dir: Path | None = None
    project_root: Path | None = None


class OpenAiCompatibleExtractionClient:
    client_name = "openai-compatible"

    def __init__(
        self,
        *,
        config: OpenAiCompatibleClientConfig | None = None,
        model_cache_manager: ModelCacheManager | None = None,
        http_client: OpenAiCompatibleHttpClient | None = None,
    ) -> None:
        self.config = config or OpenAiCompatibleClientConfig()
        _validate_base_url(self.config.base_url)
        cache_dir = self.config.model_cache_dir or default_model_cache_dir(
            self.config.project_root or Path.cwd()
        )
        self.model_cache_manager = model_cache_manager or ModelCacheManager(cache_dir=cache_dir)
        self.http_client = http_client or UrlLibOpenAiCompatibleHttpClient(
            api_key_env=self.config.api_key_env
        )

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult:
        prompt = build_extraction_prompt(sample_id=request.sample_id, pages=request.pages)
        return self._complete_rows(prompt=prompt, request=request, purpose="extract")

    def supports_vision_repair(self, request: AiExtractionRequest) -> bool:
        if not self.config.require_local_model:
            return False
        spec = self.model_cache_manager.manifest.resolve(self.config.model)
        return (
            self.config.attach_images
            and _supports_openai_images(spec)
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
        return self._complete_rows(prompt=prompt, request=request, purpose="vision_repair")

    def review_row_patches(
        self,
        *,
        request: AiExtractionRequest,
        rows: list[dict[str, object]],
        issues: object,
    ) -> AiPatchReviewResult:
        prompt = build_vision_patch_prompt(
            sample_id=request.sample_id,
            pages=request.pages,
            rows=rows,
            issues=list(issues) if isinstance(issues, list | tuple) else [],
        )
        return self._complete_patch_review(
            prompt=prompt, request=request, purpose="third_layer_review"
        )

    def _complete_rows(
        self, *, prompt: str, request: AiExtractionRequest, purpose: str
    ) -> AiExtractionResult:
        spec: LocalModelSpec | None = None
        model_path: Path | None = None
        if self.config.require_local_model:
            spec = self.model_cache_manager.manifest.resolve(self.config.model)
            if spec.backend not in {"vllm", "sglang", "llama-cpp"}:
                raise ValueError(
                    f"Model '{self.config.model}' uses backend '{spec.backend}', "
                    "not an OpenAI-compatible local server backend"
                )
            model_path = self.model_cache_manager.resolve_for_run(self.config.model)
            preferred_served_model = (
                _default_named_served_model(spec.aliases[0]) if spec.aliases else None
            )
            served_model = (
                self.config.served_model_name
                or preferred_served_model
                or (
                    spec.aliases[0]
                    if spec.backend == "llama-cpp" and spec.aliases
                    else str(model_path)
                )
            )
            attach_images = self.config.attach_images and _supports_openai_images(spec)
        else:
            served_model = self.config.served_model_name or self.config.model
            attach_images = self.config.attach_images and self.config.external_supports_images
        messages = _build_chat_messages(
            prompt,
            request,
            attach_images=attach_images,
        )
        request_payload: dict[str, object] = {
            "model": served_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.reasoning_effort:
            request_payload["reasoning_effort"] = self.config.reasoning_effort
        if spec is not None and spec.backend == "llama-cpp":
            request_payload["response_format"] = {"type": "json_object"}
        response_payload = self.http_client.post_json(
            _chat_completions_url(self.config.base_url),
            request_payload,
            self.config.timeout_seconds,
        )
        text = _extract_chat_content(response_payload)
        payload = extract_json_payload(text)
        rows = _extract_rows_from_payload(payload)
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("OpenAI-compatible response rows must all be JSON objects")
        normalized_rows = normalize_ai_rows_values(cast(list[dict[str, object]], rows))
        validate_ai_rows(normalized_rows)
        warnings = payload.get("warnings")
        return AiExtractionResult(
            rows=normalized_rows,
            warnings=[item for item in warnings if isinstance(item, str)]
            if isinstance(warnings, list)
            else [],
            raw_response={
                "response": response_payload,
                "model": self.config.model,
                "served_model": served_model,
                "model_path": str(model_path) if model_path is not None else None,
                "base_url": self.config.base_url,
                "client": self.client_name,
                "external": not self.config.require_local_model,
                "purpose": purpose,
                "image_paths": [str(path) for path in _image_paths_for_request(request)]
                if attach_images
                else [],
            },
        )

    def _complete_patch_review(
        self, *, prompt: str, request: AiExtractionRequest, purpose: str
    ) -> AiPatchReviewResult:
        spec: LocalModelSpec | None = None
        model_path: Path | None = None
        if self.config.require_local_model:
            spec = self.model_cache_manager.manifest.resolve(self.config.model)
            if spec.backend not in {"vllm", "sglang", "llama-cpp"}:
                raise ValueError(
                    f"Model '{self.config.model}' uses backend '{spec.backend}', "
                    "not an OpenAI-compatible local server backend"
                )
            model_path = self.model_cache_manager.resolve_for_run(self.config.model)
            preferred_served_model = (
                _default_named_served_model(spec.aliases[0]) if spec.aliases else None
            )
            served_model = (
                self.config.served_model_name
                or preferred_served_model
                or (
                    spec.aliases[0]
                    if spec.backend == "llama-cpp" and spec.aliases
                    else str(model_path)
                )
            )
            attach_images = self.config.attach_images and _supports_openai_images(spec)
        else:
            served_model = self.config.served_model_name or self.config.model
            attach_images = self.config.attach_images and self.config.external_supports_images
        messages = _build_chat_messages(prompt, request, attach_images=attach_images)
        request_payload: dict[str, object] = {
            "model": served_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.reasoning_effort:
            request_payload["reasoning_effort"] = self.config.reasoning_effort
        if spec is not None and spec.backend == "llama-cpp":
            request_payload["response_format"] = {"type": "json_object"}
        response_payload = self.http_client.post_json(
            _chat_completions_url(self.config.base_url),
            request_payload,
            self.config.timeout_seconds,
        )
        text = _extract_chat_content(response_payload)
        payload = extract_json_payload(text)
        patches = payload.get("patches")
        if not isinstance(patches, list):
            raise ValueError("OpenAI-compatible third-layer response must contain a patches array")
        warnings = payload.get("warnings")
        return AiPatchReviewResult(
            patches=[patch for patch in patches if isinstance(patch, dict)],
            warnings=[item for item in warnings if isinstance(item, str)]
            if isinstance(warnings, list)
            else [],
            raw_response={
                "response": response_payload,
                "model": self.config.model,
                "served_model": served_model,
                "model_path": str(model_path) if model_path is not None else None,
                "base_url": self.config.base_url,
                "client": self.client_name,
                "external": not self.config.require_local_model,
                "purpose": purpose,
                "image_paths": [str(path) for path in _image_paths_for_request(request)]
                if attach_images
                else [],
            },
        )


class UrlLibOpenAiCompatibleHttpClient:
    def __init__(self, *, api_key_env: str | None = None) -> None:
        self.api_key_env = api_key_env

    def post_json(
        self, url: str, payload: dict[str, object], timeout_seconds: int
    ) -> dict[str, object]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible server request failed: HTTP {exc.code} {body[:500]}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "OpenAI-compatible server request timed out after "
                f"{timeout_seconds} seconds: {url}. "
                "The local model may still be generating; increase --ai-timeout "
                "or use smaller AI page batches."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeError(
                    "OpenAI-compatible server request timed out after "
                    f"{timeout_seconds} seconds: {url}. "
                    "The local model may still be generating; increase --ai-timeout "
                    "or use smaller AI page batches."
                ) from exc
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible server at {url}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI-compatible server returned non-JSON response") from exc
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible server response must be a JSON object")
        return parsed


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _validate_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OpenAI-compatible base URL must use http:// or https:// with a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenAI-compatible base URL must not contain credentials")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "Refusing cleartext OpenAI-compatible endpoint outside localhost; use HTTPS so "
            "credentials and document data are encrypted in transit"
        )


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").casefold()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _default_named_served_model(alias: str) -> str | None:
    return {
        "qwen3.6-safe": "qwen3.6-safe-local",
        "qwen3.6-balanced": "qwen3.6-balanced-local",
        "qwen3.6-quality": "qwen3.6-quality-local",
        "qwen36-max": "qwen36-max-local",
        "qwen3-vl-8b": "qwen3-vl-local",
    }.get(alias)


def _supports_openai_images(spec: LocalModelSpec) -> bool:
    marker = " ".join([spec.key, spec.title, spec.hf_id, *spec.aliases]).lower()
    return spec.backend == "sglang" and ("vl" in marker or "vision" in marker)


def _build_chat_messages(
    prompt: str,
    request: AiExtractionRequest,
    *,
    attach_images: bool,
) -> list[dict[str, object]]:
    image_paths = _image_paths_for_request(request) if attach_images else []
    if not image_paths:
        return [{"role": "user", "content": prompt}]
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    return [{"role": "user", "content": content}]


def _image_paths_for_request(request: AiExtractionRequest) -> list[Path]:
    return [
        Path(page.image_path)
        for page in request.pages
        if page.image_path is not None and Path(page.image_path).exists()
    ]


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_chat_content(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                text = _content_to_text(content)
                if text is not None:
                    return text
            text = _content_to_text(first.get("text"))
            if text is not None:
                return text
    text = _content_to_text(payload.get("content"))
    if text is not None:
        return text
    raise ValueError("OpenAI-compatible response does not contain message content")


def _extract_rows_from_payload(payload: dict[str, object]) -> list[object]:
    rows = _find_rows_in_payload(payload)
    if rows is not None:
        return rows

    keys = ", ".join(sorted(payload.keys())) or "no keys"
    raise ValueError(
        "OpenAI-compatible response JSON must contain a rows array "
        f"or recognizable row list; got keys: {keys}"
    )


def _find_rows_in_payload(payload: dict[str, object]) -> list[object] | None:
    rows = payload.get("rows")
    if isinstance(rows, list):
        return rows

    for key in ("items", "records", "result", "results", "data", "output"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested_rows = _find_rows_in_payload(cast(dict[str, object], value))
            if nested_rows is not None:
                return nested_rows

    if _looks_like_ai_row(payload):
        return [payload]
    return None


def _looks_like_ai_row(payload: dict[str, object]) -> bool:
    row_markers = {"page_ref", "pos", "name", "type_mark", "code", "unit", "quantity"}
    return len(row_markers.intersection(payload.keys())) >= 3


def _content_to_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    return None
