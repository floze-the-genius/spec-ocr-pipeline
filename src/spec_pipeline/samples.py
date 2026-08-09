from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from spec_pipeline.models import SampleDefinition, SampleManifest, SpecificationRow


def _default_workspace_root() -> Path:
    configured = os.environ.get("SPEC_OCR_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / "pyproject.toml").is_file():
        return source_checkout
    return Path.cwd().resolve()


PROJECT_ROOT = _default_workspace_root()
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_MANIFEST_PATH = PROJECT_ROOT / "samples/stage1_samples.json"
DEFAULT_MANIFEST_PATH = (
    SOURCE_MANIFEST_PATH
    if SOURCE_MANIFEST_PATH.is_file()
    else PACKAGE_DATA_DIR / "demo-manifest.json"
)


def _resolve_optional_path(raw_value: str | None, *, base_dir: Path) -> Path | None:
    if not raw_value:
        return None

    raw_path = Path(raw_value).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return (base_dir / raw_path).resolve()


def load_sample_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> SampleManifest:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    samples = [
        SampleDefinition(
            sample_id=item["sample_id"],
            title=item["title"],
            source_pdf=_resolve_optional_path(item["source_pdf"], base_dir=base_dir),
            ground_truth_tsv=_resolve_optional_path(
                item.get("ground_truth_tsv"), base_dir=base_dir
            ),
            expected_xlsx=_resolve_optional_path(item.get("expected_xlsx"), base_dir=base_dir),
            visual_xlsx=_resolve_optional_path(item.get("visual_xlsx"), base_dir=base_dir),
            page_field=item.get("page_field", "page"),
        )
        for item in payload.get("samples", [])
    ]

    return SampleManifest(
        manifest_version=str(payload.get("manifest_version", "1")),
        name=payload.get("name", path.stem),
        manifest_path=path,
        samples=samples,
    )


def count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def summarize_manifest(
    manifest: SampleManifest,
    *,
    validate_paths: bool = False,
) -> dict[str, object]:
    summary_samples: list[dict[str, object]] = []

    for sample in manifest.samples:
        artifacts = sample.artifact_paths()
        item: dict[str, object] = {
            "sample_id": sample.sample_id,
            "title": sample.title,
            "page_field": sample.page_field,
            "artifacts": {
                name: str(path) if path is not None else None for name, path in artifacts.items()
            },
        }
        if validate_paths:
            exists = {name: bool(path and path.exists()) for name, path in artifacts.items()}
            item["exists"] = exists
            if sample.ground_truth_tsv and sample.ground_truth_tsv.exists():
                item["ground_truth_rows"] = count_tsv_rows(sample.ground_truth_tsv)
        summary_samples.append(item)

    return {
        "manifest_version": manifest.manifest_version,
        "name": manifest.name,
        "manifest_path": str(manifest.manifest_path),
        "sample_count": len(manifest.samples),
        "samples": summary_samples,
    }


def load_ground_truth_rows(sample: SampleDefinition) -> list[SpecificationRow]:
    if sample.ground_truth_tsv is None:
        raise ValueError(f"Sample {sample.sample_id} does not declare ground_truth_tsv")

    with sample.ground_truth_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            SpecificationRow.from_tsv_row(
                sample_id=sample.sample_id, page_field=sample.page_field, row=row
            )
            for row in reader
        ]
