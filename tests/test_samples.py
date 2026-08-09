from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from spec_pipeline.samples import load_ground_truth_rows, load_sample_manifest, summarize_manifest


def _write_manifest(tmp_path: Path, *, page_field: str = "page") -> Path:
    tsv_path = tmp_path / "sample.tsv"
    tsv_path.write_text(
        "\t".join(
            [
                page_field,
                "section",
                "subsection",
                "pos",
                "name",
                "type_mark",
                "code",
                "manufacturer",
                "unit",
                "quantity",
                "mass_kg",
                "note",
            ]
        )
        + "\n"
        + "\t".join(["001", "ПВ7", "", "", "", "", "", "", "", "", "", ""])
        + "\n"
        + "\t".join(
            ["001", "ПВ7", "КИПиА", "1", "Датчик", "ТС-1", "", "СВОК", "шт", "12,5", "", "ok"]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "1",
                "name": "test-manifest",
                "samples": [
                    {
                        "sample_id": "sample_1",
                        "title": "Sample 1",
                        "source_pdf": str(tmp_path / "input.pdf"),
                        "ground_truth_tsv": str(tsv_path),
                        "expected_xlsx": str(tmp_path / "expected.xlsx"),
                        "visual_xlsx": str(tmp_path / "visual.xlsx"),
                        "page_field": page_field,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return manifest_path


def test_load_sample_manifest_reads_sample_definitions(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    manifest = load_sample_manifest(manifest_path)

    assert manifest.name == "test-manifest"
    assert len(manifest.samples) == 1
    assert manifest.samples[0].sample_id == "sample_1"
    assert manifest.samples[0].ground_truth_tsv == tmp_path / "sample.tsv"


def test_summarize_manifest_reports_validation_details(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_sample_manifest(manifest_path)

    summary = summarize_manifest(manifest, validate_paths=True)

    sample_summary = summary["samples"][0]
    assert summary["sample_count"] == 1
    assert sample_summary["exists"]["ground_truth_tsv"] is True
    assert sample_summary["exists"]["source_pdf"] is False
    assert sample_summary["ground_truth_rows"] == 2


def test_load_ground_truth_rows_normalizes_quantity_and_headers(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_sample_manifest(manifest_path)

    rows = load_ground_truth_rows(manifest.samples[0])

    assert rows[0].is_grouping_row() is True
    assert rows[1].page_ref == "001"
    assert rows[1].subsection == "КИПиА"
    assert rows[1].quantity == Decimal("12.5")
