from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from spec_pipeline.ai.base import AiExtractionClient, AiExtractionRequest
from spec_pipeline.ai.openai_compatible_client import (
    OpenAiCompatibleClientConfig,
    OpenAiCompatibleExtractionClient,
)
from spec_pipeline.ai.opencode_client import OpenCodeClientConfig, OpenCodeExtractionClient
from spec_pipeline.ai.quality import assess_extracted_rows
from spec_pipeline.ai.qwen_mlx_client import QwenMlxClientConfig, QwenMlxExtractionClient
from spec_pipeline.ai.schema import ai_row_to_specification_row, validate_ai_rows
from spec_pipeline.models import SampleDefinition, SpecificationRow
from spec_pipeline.samples import load_ground_truth_rows
from spec_pipeline.workflows.compare import compare_rows
from spec_pipeline.workflows.ocr import load_ocr_pages_json
from spec_pipeline.workflows.parse import write_parsed_rows_jsonl


@dataclass(frozen=True, slots=True)
class LocalVsGptBenchmarkConfig:
    sample_id: str
    input_ocr: Path
    source_pdf: Path
    output_dir: Path
    project_root: Path
    model_cache_dir: Path | None = None
    local_ai_client: str = "qwen-mlx"
    qwen_model: str = "qwen3-8b"
    local_base_url: str = "http://127.0.0.1:8000/v1"
    local_served_model: str | None = None
    gpt_model: str = "openai/gpt-5.5"
    gpt_agent: str = "build"
    ai_timeout: int = 600
    attach_images: bool = True


def run_local_vs_gpt_benchmark(
    config: LocalVsGptBenchmarkConfig,
    *,
    sample: SampleDefinition | None = None,
    clients: Mapping[str, AiExtractionClient] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    pages = load_ocr_pages_json(config.input_ocr)
    request = AiExtractionRequest(
        sample_id=config.sample_id,
        source_pdf=config.source_pdf,
        pages=pages,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_clients = clients or _default_clients(config)
    ground_truth_rows = _load_optional_ground_truth(sample)

    results: list[dict[str, object]] = []
    rows_by_backend: dict[str, list[SpecificationRow]] = {}
    for backend, client in resolved_clients.items():
        started = timer()
        try:
            extraction = client.extract_rows(request)
            runtime_sec = timer() - started
            validate_ai_rows(extraction.rows)
            rows = [ai_row_to_specification_row(config.sample_id, row) for row in extraction.rows]
            rows_by_backend[backend] = rows

            output_jsonl = config.output_dir / f"{config.sample_id}.{backend}.jsonl"
            raw_json = config.output_dir / f"{config.sample_id}.{backend}.raw.json"
            quality_json = config.output_dir / f"{config.sample_id}.{backend}.quality.json"
            write_parsed_rows_jsonl(
                [cast(dict[str, object], row.to_dict()) for row in rows], output_jsonl
            )
            raw_json.write_text(
                json.dumps(extraction.raw_response, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            quality = assess_extracted_rows(rows)
            quality_json.write_text(
                json.dumps(quality.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            compare = (
                compare_rows(
                    sample_id=config.sample_id,
                    ground_truth_rows=ground_truth_rows,
                    predicted_rows=rows,
                ).to_dict()
                if ground_truth_rows is not None
                else None
            )
            results.append(
                {
                    "backend": backend,
                    "ok": True,
                    "model": _model_for_backend(config, backend),
                    "runtime_sec": runtime_sec,
                    "row_count": len(rows),
                    "quality": quality.to_dict(),
                    "compare": compare,
                    "output_jsonl": str(output_jsonl),
                    "quality_json": str(quality_json),
                    "raw_response_json": str(raw_json),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "backend": backend,
                    "ok": False,
                    "model": _model_for_backend(config, backend),
                    "runtime_sec": timer() - started,
                    "error": str(exc),
                }
            )

    report: dict[str, object] = {
        "sample_id": config.sample_id,
        "input_ocr": str(config.input_ocr),
        "source_pdf": str(config.source_pdf),
        "results": results,
        "quality_delta": _quality_delta(results),
        "local_backend": _local_backend_key(config),
        "pairwise_similarity": _pairwise_similarity(
            rows_by_backend.get(_local_backend_key(config)), rows_by_backend.get("opencode-gpt")
        ),
    }
    report_path = config.output_dir / f"{config.sample_id}.local-vs-gpt.benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def _default_clients(config: LocalVsGptBenchmarkConfig) -> dict[str, AiExtractionClient]:
    local_backend = _local_backend_key(config)
    return {
        local_backend: _local_client(config),
        "opencode-gpt": OpenCodeExtractionClient(
            config=OpenCodeClientConfig(
                model=config.gpt_model,
                agent=config.gpt_agent,
                timeout_seconds=config.ai_timeout,
                attach_images=config.attach_images,
                cache_dir=config.project_root / "artifacts" / "ai" / "opencode_requests",
                cwd=config.project_root,
            )
        ),
    }


def _local_client(config: LocalVsGptBenchmarkConfig) -> AiExtractionClient:
    if config.local_ai_client == "qwen-mlx":
        return QwenMlxExtractionClient(
            config=QwenMlxClientConfig(
                model=config.qwen_model,
                project_root=config.project_root,
                model_cache_dir=config.model_cache_dir,
                attach_images=config.attach_images,
            )
        )
    if config.local_ai_client == "openai-compatible":
        return OpenAiCompatibleExtractionClient(
            config=OpenAiCompatibleClientConfig(
                model=config.qwen_model,
                base_url=config.local_base_url,
                served_model_name=config.local_served_model,
                timeout_seconds=config.ai_timeout,
                attach_images=config.attach_images,
                project_root=config.project_root,
                model_cache_dir=config.model_cache_dir,
            )
        )
    raise ValueError(f"Unknown local benchmark AI client: {config.local_ai_client}")


def _local_backend_key(config: LocalVsGptBenchmarkConfig) -> str:
    if config.local_ai_client == "qwen-mlx":
        return "qwen-mlx"
    if config.local_ai_client == "openai-compatible":
        return "openai-compatible-local"
    return config.local_ai_client


def _load_optional_ground_truth(sample: SampleDefinition | None) -> list[SpecificationRow] | None:
    if sample is None or sample.ground_truth_tsv is None:
        return None
    return load_ground_truth_rows(sample)


def _model_for_backend(config: LocalVsGptBenchmarkConfig, backend: str) -> str:
    return config.gpt_model if backend == "opencode-gpt" else config.qwen_model


def _quality_delta(results: list[dict[str, object]]) -> dict[str, object] | None:
    by_backend = {str(item["backend"]): item for item in results if item.get("ok") is True}
    qwen = by_backend.get("qwen-mlx") or by_backend.get("openai-compatible-local")
    gpt = by_backend.get("opencode-gpt")
    if qwen is None or gpt is None:
        return None
    qwen_quality = cast(dict[str, object], qwen["quality"])
    gpt_quality = cast(dict[str, object], gpt["quality"])
    return {
        "local_backend": str(qwen["backend"]),
        "runtime_sec_qwen_minus_gpt": _float_value(qwen["runtime_sec"])
        - _float_value(gpt["runtime_sec"]),
        "runtime_sec_local_minus_gpt": _float_value(qwen["runtime_sec"])
        - _float_value(gpt["runtime_sec"]),
        "row_count_qwen_minus_gpt": _int_value(qwen["row_count"]) - _int_value(gpt["row_count"]),
        "row_count_local_minus_gpt": _int_value(qwen["row_count"]) - _int_value(gpt["row_count"]),
        "heuristic_errors_qwen_minus_gpt": _int_value(qwen_quality["error_count"])
        - _int_value(gpt_quality["error_count"]),
        "heuristic_errors_local_minus_gpt": _int_value(qwen_quality["error_count"])
        - _int_value(gpt_quality["error_count"]),
        "heuristic_warnings_qwen_minus_gpt": _int_value(qwen_quality["warning_count"])
        - _int_value(gpt_quality["warning_count"]),
        "heuristic_warnings_local_minus_gpt": _int_value(qwen_quality["warning_count"])
        - _int_value(gpt_quality["warning_count"]),
        "field_match_rate_qwen_minus_gpt": _compare_delta(qwen, gpt, "field_match_rate"),
        "field_match_rate_local_minus_gpt": _compare_delta(qwen, gpt, "field_match_rate"),
        "exact_match_rate_qwen_minus_gpt": _compare_delta(qwen, gpt, "exact_match_rate"),
        "exact_match_rate_local_minus_gpt": _compare_delta(qwen, gpt, "exact_match_rate"),
    }


def _compare_delta(qwen: dict[str, object], gpt: dict[str, object], key: str) -> float | None:
    qwen_compare = qwen.get("compare")
    gpt_compare = gpt.get("compare")
    if not isinstance(qwen_compare, dict) or not isinstance(gpt_compare, dict):
        return None
    qwen_value = qwen_compare.get(key)
    gpt_value = gpt_compare.get(key)
    if isinstance(qwen_value, (int, float)) and isinstance(gpt_value, (int, float)):
        return float(qwen_value) - float(gpt_value)
    return None


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _pairwise_similarity(
    qwen_rows: list[SpecificationRow] | None, gpt_rows: list[SpecificationRow] | None
) -> dict[str, object] | None:
    if qwen_rows is None or gpt_rows is None:
        return None
    qwen_positions = {row.pos for row in qwen_rows if row.pos is not None}
    gpt_positions = {row.pos for row in gpt_rows if row.pos is not None}
    qwen_names = {_normalized(row.name) for row in qwen_rows if row.name}
    gpt_names = {_normalized(row.name) for row in gpt_rows if row.name}
    return {
        "common_positions": len(qwen_positions & gpt_positions),
        "qwen_positions": len(qwen_positions),
        "gpt_positions": len(gpt_positions),
        "common_normalized_names": len(qwen_names & gpt_names),
        "qwen_normalized_names": len(qwen_names),
        "gpt_normalized_names": len(gpt_names),
    }


def _normalized(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())
