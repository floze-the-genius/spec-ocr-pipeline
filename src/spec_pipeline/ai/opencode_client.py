from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult, AiPatchReviewResult
from spec_pipeline.ai.json_utils import extract_json_payload
from spec_pipeline.ai.prompt import build_extraction_prompt
from spec_pipeline.ai.schema import normalize_ai_rows_values
from spec_pipeline.ai.vision_repair import build_vision_patch_prompt

CommandRunner = Callable[[list[str], int, Path | None], subprocess.CompletedProcess[str]]

REASONING_VARIANT_LADDER = ("xhigh", "high", "medium", "low")


@dataclass(slots=True)
class OpenCodeClientConfig:
    command: str = "opencode"
    model: str = "openai/gpt-5.5"
    agent: str = "build"
    variant: str | None = None
    timeout_seconds: int = 600
    attach_images: bool = True
    cache_dir: Path | None = None
    cwd: Path | None = None


class OpenCodeExtractionClient:
    client_name = "opencode-codex"

    def __init__(
        self,
        *,
        config: OpenCodeClientConfig | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config or OpenCodeClientConfig()
        self.runner = runner or _run_command

    def with_reduced_reasoning(self) -> "OpenCodeExtractionClient | None":
        variant = (self.config.variant or "").strip().lower()
        if variant not in REASONING_VARIANT_LADDER:
            return None
        index = REASONING_VARIANT_LADDER.index(variant)
        if index == len(REASONING_VARIANT_LADDER) - 1:
            return None
        reduced_config = replace(self.config, variant=REASONING_VARIANT_LADDER[index + 1])
        return OpenCodeExtractionClient(config=reduced_config, runner=self.runner)

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult:
        prompt = build_extraction_prompt(sample_id=request.sample_id, pages=request.pages)
        prompt_path = self._write_prompt(request.sample_id, prompt)
        attached_files = [prompt_path]
        if self.config.attach_images:
            attached_files.extend(
                Path(page.image_path)
                for page in request.pages
                if page.image_path is not None and Path(page.image_path).exists()
            )

        command = [
            self.config.command,
            "run",
            "--model",
            self.config.model,
            "--agent",
            self.config.agent,
            "--format",
            "default",
            "Use the attached extraction prompt and page images if present. "
            "Return only the JSON object requested by the prompt. Do not edit files.",
        ]
        if self.config.variant:
            command.extend(["--variant", self.config.variant])
        for file_path in attached_files:
            command.extend(["--file", str(file_path)])

        try:
            completed = self.runner(command, self.config.timeout_seconds, self.config.cwd)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "opencode extraction timed out: "
                f"timeout_seconds={self.config.timeout_seconds} "
                f"command={_format_command_for_error(command)}"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "opencode extraction failed: "
                f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
            )

        try:
            payload = extract_json_payload(completed.stdout)
        except ValueError as exc:
            raise ValueError(
                "OpenCode response did not contain the requested JSON object. "
                f"stdout={_stream_excerpt(completed.stdout)} "
                f"stderr={_stream_excerpt(completed.stderr)}"
            ) from exc
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("OpenCode response JSON must contain a rows array")
        warnings = payload.get("warnings")
        rows = normalize_ai_rows_values([row for row in rows if isinstance(row, dict)])
        return AiExtractionResult(
            rows=rows,
            warnings=[item for item in warnings if isinstance(item, str)]
            if isinstance(warnings, list)
            else [],
            raw_response={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "prompt_path": str(prompt_path),
                "model": self.config.model,
                "agent": self.config.agent,
                "variant": self.config.variant,
            },
        )

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
        prompt_path = self._write_prompt(f"{request.sample_id}.third-layer", prompt)
        attached_files = [prompt_path]
        if self.config.attach_images:
            attached_files.extend(
                Path(page.image_path)
                for page in request.pages
                if page.image_path is not None and Path(page.image_path).exists()
            )

        command = [
            self.config.command,
            "run",
            "--model",
            self.config.model,
            "--agent",
            self.config.agent,
            "--format",
            "default",
            "Use the attached third-layer patch-review prompt and page images if present. "
            "Return only the JSON object requested by the prompt. Do not edit files.",
        ]
        if self.config.variant:
            command.extend(["--variant", self.config.variant])
        for file_path in attached_files:
            command.extend(["--file", str(file_path)])

        try:
            completed = self.runner(command, self.config.timeout_seconds, self.config.cwd)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "opencode third-layer review timed out: "
                f"timeout_seconds={self.config.timeout_seconds} "
                f"command={_format_command_for_error(command)}"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "opencode third-layer review failed: "
                f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
            )

        try:
            payload = extract_json_payload(completed.stdout)
        except ValueError as exc:
            raise ValueError(
                "OpenCode third-layer response did not contain the requested JSON object. "
                f"stdout={_stream_excerpt(completed.stdout)} "
                f"stderr={_stream_excerpt(completed.stderr)}"
            ) from exc
        patches = payload.get("patches")
        if not isinstance(patches, list):
            raise ValueError("OpenCode third-layer response JSON must contain a patches array")
        warnings = payload.get("warnings")
        return AiPatchReviewResult(
            patches=[patch for patch in patches if isinstance(patch, dict)],
            warnings=[item for item in warnings if isinstance(item, str)]
            if isinstance(warnings, list)
            else [],
            raw_response={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "prompt_path": str(prompt_path),
                "model": self.config.model,
                "agent": self.config.agent,
                "variant": self.config.variant,
                "purpose": "third_layer_review",
            },
        )

    def _write_prompt(self, sample_id: str, prompt: str) -> Path:
        base_dir = self.config.cache_dir or Path("artifacts") / "ai" / "opencode_requests"
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{sample_id}.{int(time.time() * 1000)}.prompt.md"
        path.write_text(prompt, encoding="utf-8")
        return path


def _run_command(
    command: list[str], timeout_seconds: int, cwd: Path | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        cwd=str(cwd) if cwd is not None else None,
        env=_clean_opencode_environment(),
        check=False,
    )


def _clean_opencode_environment() -> dict[str, str]:
    preserved = {}
    for key in [
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "SHELL",
        "SYSTEMROOT",
        "WINDIR",
        "PATHEXT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
    ]:
        value = os.environ.get(key)
        if value is not None:
            preserved[key] = value
    return preserved


def _stream_excerpt(text: str, *, max_chars: int = 1200) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return repr(stripped)
    return repr(stripped[:max_chars] + "...<truncated>")


def _format_command_for_error(command: list[str], *, max_chars: int = 1200) -> str:
    text = " ".join(command)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"
