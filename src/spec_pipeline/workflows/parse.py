from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.models import SampleDefinition
from spec_pipeline.parsers.base import ParseDiagnostics, ParserAdapter


def default_parsed_output_path(project_root: Path, sample_id: str) -> Path:
    return project_root / "artifacts" / "parsed" / f"{sample_id}.jsonl"


def default_diagnostics_output_path(project_root: Path, sample_id: str, backend: str) -> Path:
    return project_root / "artifacts" / "diagnostics" / f"{sample_id}.{backend}.json"


def write_parsed_rows_jsonl(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return output_path


def parse_sample_to_jsonl(
    *,
    parser: ParserAdapter,
    sample: SampleDefinition,
    project_root: Path,
    output_path: str | Path | None = None,
) -> tuple[Path, int]:
    rows = parser.parse_rows(sample)
    resolved_output = (
        Path(output_path).expanduser()
        if output_path is not None
        else default_parsed_output_path(project_root, sample.sample_id)
    )
    if output_path is not None and not resolved_output.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {resolved_output.parent}")
    write_parsed_rows_jsonl([row.to_dict() for row in rows], resolved_output)
    return resolved_output, len(rows)


def write_diagnostics(diagnostics: ParseDiagnostics, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
