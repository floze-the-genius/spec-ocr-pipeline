from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spec_pipeline.ai.base import AiExtractionRequest
from spec_pipeline.ai.opencode_client import (
    OpenCodeClientConfig,
    OpenCodeExtractionClient,
    _clean_opencode_environment,
    extract_json_payload,
)
from spec_pipeline.ai.prompt import build_extraction_prompt
from spec_pipeline.ocr.base import OcrPage


def test_extract_json_payload_accepts_fenced_json() -> None:
    payload = extract_json_payload('noise\n```json\n{"rows": [], "warnings": []}\n```')

    assert payload == {"rows": [], "warnings": []}


def test_extract_json_payload_skips_invalid_brace_block() -> None:
    payload = extract_json_payload('not json: {oops}\nreal: {"rows": [], "warnings": []}')

    assert payload == {"rows": [], "warnings": []}


def test_extraction_prompt_disables_reasoning_output() -> None:
    prompt = build_extraction_prompt(sample_id="sample", pages=[])

    assert "Do not output reasoning" in prompt
    assert "<think>" in prompt
    assert "deduplicate repeated visual rows" in prompt
    assert "crossed_out" in prompt
    assert "PDF_TABLE_LAYER" in prompt


def test_opencode_client_builds_command_and_parses_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake")
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], timeout: int, cwd: Path | None):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"rows":[{"page_ref":"1","section":null,"subsection":null,'
                '"pos":1,"name":"Диффузор","type_mark":null,"code":null,'
                '"manufacturer":null,"unit":"шт","quantity":"2","mass_kg":null,'
                '"note":null}],"warnings":[]}'
            ),
            stderr="",
        )

    client = OpenCodeExtractionClient(
        config=OpenCodeClientConfig(
            model="openai/gpt-5.5",
            agent="codex",
            variant="xhigh",
            timeout_seconds=123,
            cache_dir=tmp_path,
            cwd=tmp_path,
        ),
        runner=fake_runner,
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
                    image_path=str(image_path),
                )
            ],
        )
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:4] == ["opencode", "run", "--model", "openai/gpt-5.5"]
    assert "--agent" in command
    assert "codex" in command
    assert "--variant" in command
    assert "xhigh" in command
    assert str(image_path) in command
    assert captured["timeout"] == 123
    assert captured["cwd"] == tmp_path
    assert result.rows[0]["pos"] == "1"
    assert result.rows[0]["name"] == "Диффузор"
    assert Path(str(result.raw_response["prompt_path"])).exists()


def test_opencode_client_no_json_error_includes_process_output(tmp_path: Path) -> None:
    def fake_runner(command: list[str], timeout: int, cwd: Path | None):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="[multi-auth] Refreshing token for new_account\n",
            stderr="[multi-auth] No available accounts\n",
        )

    client = OpenCodeExtractionClient(
        config=OpenCodeClientConfig(cache_dir=tmp_path, cwd=tmp_path),
        runner=fake_runner,
    )

    with pytest.raises(ValueError, match="No available accounts"):
        client.extract_rows(
            AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
        )


def test_opencode_client_timeout_raises_short_runtime_error(tmp_path: Path) -> None:
    def fake_runner(command: list[str], timeout: int, cwd: Path | None):
        raise subprocess.TimeoutExpired(command, timeout)

    client = OpenCodeExtractionClient(
        config=OpenCodeClientConfig(cache_dir=tmp_path, cwd=tmp_path, timeout_seconds=12),
        runner=fake_runner,
    )

    with pytest.raises(RuntimeError, match="opencode extraction timed out: timeout_seconds=12"):
        client.extract_rows(
            AiExtractionRequest(sample_id="sample", source_pdf=tmp_path / "input.pdf", pages=[])
        )


def test_opencode_client_reviews_third_layer_patches(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake")
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], timeout: int, cwd: Path | None):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"patches":[{"row_index":1,"page_ref":"1","field":"name",'
                '"old_value":"Автоматический воздухомобюксир 015",'
                '"new_value":"Автоматический воздухоотводчик Ø15",'
                '"confidence":"high","reason":"visible"}],"warnings":[]}'
            ),
            stderr="",
        )

    client = OpenCodeExtractionClient(
        config=OpenCodeClientConfig(cache_dir=tmp_path, cwd=tmp_path, variant="xhigh"),
        runner=fake_runner,
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
        rows=[],
        issues=[],
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert any("Do not edit files." in part for part in command)
    assert "--variant" in command
    assert "xhigh" in command
    assert str(image_path) in command
    assert result.patches[0]["new_value"] == "Автоматический воздухоотводчик Ø15"


def test_opencode_client_with_reduced_reasoning_steps_down_ladder() -> None:
    sentinel_runner = object()
    client = OpenCodeExtractionClient(
        config=OpenCodeClientConfig(variant="xhigh"), runner=sentinel_runner
    )

    reduced = client.with_reduced_reasoning()
    assert reduced is not None
    assert reduced.config.variant == "high"
    assert reduced.runner is sentinel_runner
    assert reduced.config.model == client.config.model

    medium = reduced.with_reduced_reasoning()
    low = medium.with_reduced_reasoning()
    assert medium.config.variant == "medium"
    assert low.config.variant == "low"
    assert low.with_reduced_reasoning() is None


def test_opencode_client_with_reduced_reasoning_without_variant_returns_none() -> None:
    assert (
        OpenCodeExtractionClient(config=OpenCodeClientConfig(variant=None)).with_reduced_reasoning()
        is None
    )
    assert (
        OpenCodeExtractionClient(
            config=OpenCodeClientConfig(variant="turbo")
        ).with_reduced_reasoning()
        is None
    )


def test_clean_opencode_environment_preserves_windows_runtime_paths(monkeypatch) -> None:
    expected = {
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "TEMP": r"C:\Users\Alice\AppData\Local\Temp",
        "TMP": r"C:\Users\Alice\AppData\Local\Temp",
        "USERPROFILE": r"C:\Users\Alice",
        "APPDATA": r"C:\Users\Alice\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\Alice\AppData\Local",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPENCODE_SESSION", "must-not-leak")

    environment = _clean_opencode_environment()

    assert {key: environment[key] for key in expected} == expected
    assert "OPENCODE_SESSION" not in environment
