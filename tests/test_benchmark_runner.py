from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult
from spec_pipeline.benchmark.runner import LocalVsGptBenchmarkConfig, run_local_vs_gpt_benchmark


class FakeClient:
    client_name = "fake"

    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[AiExtractionRequest] = []

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult:
        self.requests.append(request)
        return AiExtractionResult(
            rows=[
                {
                    "page_ref": "1",
                    "section": None,
                    "subsection": None,
                    "pos": "1",
                    "name": f"Диффузор {self.name}",
                    "type_mark": None,
                    "code": None,
                    "manufacturer": None,
                    "unit": "шт",
                    "quantity": "2",
                    "mass_kg": None,
                    "note": None,
                }
            ],
            warnings=[],
            raw_response={"backend": self.name},
        )


def test_benchmark_uses_same_ocr_request_for_qwen_and_gpt(tmp_path: Path) -> None:
    qwen = FakeClient("qwen")
    gpt = FakeClient("gpt")
    ticks = iter([10.0, 12.5, 20.0, 26.0])

    report = run_local_vs_gpt_benchmark(
        LocalVsGptBenchmarkConfig(
            sample_id="sample",
            input_ocr=_write_ocr_json(tmp_path),
            source_pdf=tmp_path / "source.pdf",
            output_dir=tmp_path / "bench",
            project_root=tmp_path,
        ),
        clients={"qwen-mlx": qwen, "opencode-gpt": gpt},
        timer=lambda: next(ticks),
    )

    assert qwen.requests[0].pages is gpt.requests[0].pages
    results = report["results"]
    assert isinstance(results, list)
    assert results[0]["runtime_sec"] == 2.5
    assert results[1]["runtime_sec"] == 6.0
    assert (tmp_path / "bench" / "sample.qwen-mlx.jsonl").exists()
    assert (tmp_path / "bench" / "sample.opencode-gpt.jsonl").exists()
    quality_delta = report["quality_delta"]
    assert isinstance(quality_delta, dict)
    assert quality_delta["runtime_sec_qwen_minus_gpt"] == -3.5
    pairwise = report["pairwise_similarity"]
    assert isinstance(pairwise, dict)
    assert pairwise["common_positions"] == 1


def _write_ocr_json(tmp_path: Path) -> Path:
    path = tmp_path / "ocr.json"
    path.write_text(
        json.dumps(
            {
                "page_count": 1,
                "provider": "fake-ocr",
                "pages": [
                    {
                        "page_index": 0,
                        "page_ref": "1",
                        "width": 100,
                        "height": 200,
                        "text": "1 Диффузор шт 2",
                        "tokens": [],
                        "provider": "fake-ocr",
                        "image_path": None,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
