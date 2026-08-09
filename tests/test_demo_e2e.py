from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.cli import main
from spec_pipeline.samples import DEFAULT_MANIFEST_PATH, load_sample_manifest
from spec_pipeline.workflows.compare import compare_sample_against_jsonl


def test_public_demo_pdf_to_excel_matches_ground_truth(tmp_path: Path) -> None:
    sample = load_sample_manifest(DEFAULT_MANIFEST_PATH).get_sample("demo")
    ocr_path = tmp_path / "ocr.json"
    rows_path = tmp_path / "rows.jsonl"
    xlsx_path = tmp_path / "result.xlsx"

    assert sample.source_pdf.exists()
    assert (
        main(
            [
                "extract-ocr",
                str(sample.source_pdf),
                "--provider",
                "pdf-text",
                "--output",
                str(ocr_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "map-ocr",
                "demo",
                "--input-ocr",
                str(ocr_path),
                "--source-pdf",
                str(sample.source_pdf),
                "--output",
                str(rows_path),
                "--no-image-attachments",
            ]
        )
        == 0
    )
    assert main(["export-jsonl-xlsx", str(rows_path), "--output", str(xlsx_path)]) == 0

    result = compare_sample_against_jsonl(sample=sample, predicted_jsonl_path=rows_path)
    assert result.row_delta == 0
    assert result.exact_match_rate == 1.0
    assert result.field_match_rate == 1.0
    assert xlsx_path.is_file()


def test_process_command_is_safe_reproducible_and_creates_quote(
    tmp_path: Path, monkeypatch
) -> None:
    sample = load_sample_manifest(DEFAULT_MANIFEST_PATH).get_sample("demo")
    output_dir = tmp_path / "result with spaces"

    def fail_external_ai(*args, **kwargs):
        raise AssertionError("The searchable-PDF process command must not call external AI")

    monkeypatch.setattr(
        "spec_pipeline.ai.opencode_client.OpenCodeExtractionClient.extract_rows",
        fail_external_ai,
    )

    assert (
        main(
            [
                "process",
                str(sample.source_pdf),
                "--output-dir",
                str(output_dir),
                "--quote",
            ]
        )
        == 0
    )
    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["table_row_count"] == 6
    assert metadata["external_ai_used"] is False
    assert (output_dir / "rows.jsonl").is_file()
    assert (output_dir / "result.xlsx").is_file()
    assert (output_dir / "quote.xlsx").is_file()

    assert (
        main(
            [
                "process",
                str(sample.source_pdf),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "process",
                str(sample.source_pdf),
                "--output-dir",
                str(output_dir),
                "--force",
            ]
        )
        == 0
    )
