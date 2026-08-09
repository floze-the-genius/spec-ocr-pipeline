from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.models import SampleDefinition, SpecificationRow
from spec_pipeline.samples import load_ground_truth_rows


def default_output_path(project_root: Path, sample_id: str) -> Path:
    return project_root / "artifacts" / "normalized" / f"{sample_id}.jsonl"


def write_rows_jsonl(rows: list[SpecificationRow], output_path: Path) -> int:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")
    return len(rows)


def normalize_sample_to_jsonl(
    *,
    sample: SampleDefinition,
    project_root: Path,
    output_path: str | Path | None = None,
) -> tuple[Path, int]:
    rows = load_ground_truth_rows(sample)
    resolved_output = (
        Path(output_path).expanduser()
        if output_path is not None
        else default_output_path(project_root, sample.sample_id)
    )

    if output_path is None:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
    elif not resolved_output.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {resolved_output.parent}")

    row_count = write_rows_jsonl(rows, resolved_output)
    return resolved_output, row_count
