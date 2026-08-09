from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.samples import load_sample_manifest
from spec_pipeline.workflows.normalize import normalize_sample_to_jsonl


def _write_manifest(tmp_path: Path) -> Path:
    tsv_path = tmp_path / "sample.tsv"
    tsv_path.write_text(
        "page\tsection\tsubsection\tpos\tname\ttype_mark\tcode\tmanufacturer\tunit\tquantity\tmass_kg\tnote\n"
        "001\tПВ7\t\t1\tУстановка\t\t\tСВОК\tшт\t1\t\t\n"
        "001\tПВ7\tКИПиА\t2\tДатчик\tТС-1\t\tСВОК\tшт\t12,5\t\tok\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "1",
                "name": "normalize-test",
                "samples": [
                    {
                        "sample_id": "sample_1",
                        "title": "Sample 1",
                        "source_pdf": str(tmp_path / "input.pdf"),
                        "ground_truth_tsv": str(tsv_path),
                        "expected_xlsx": str(tmp_path / "expected.xlsx"),
                        "visual_xlsx": str(tmp_path / "visual.xlsx"),
                        "page_field": "page",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_normalize_sample_to_jsonl_writes_default_artifact(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_sample_manifest(manifest_path)

    output_path, row_count = normalize_sample_to_jsonl(
        sample=manifest.samples[0],
        project_root=tmp_path,
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert row_count == 2
    assert output_path == tmp_path / "artifacts" / "normalized" / "sample_1.jsonl"
    assert len(lines) == 2
    assert json.loads(lines[1])["quantity"] == "12.5"


def test_normalize_sample_to_jsonl_supports_custom_output(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_sample_manifest(manifest_path)
    custom_path = tmp_path / "custom.jsonl"

    output_path, row_count = normalize_sample_to_jsonl(
        sample=manifest.samples[0],
        project_root=tmp_path,
        output_path=custom_path,
    )

    assert row_count == 2
    assert output_path == custom_path
    assert custom_path.exists()
