from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.ocr.base import OcrPage


def default_ocr_output_path(project_root: Path, pdf_path: Path, provider: str) -> Path:
    return project_root / "artifacts" / "ocr" / f"{pdf_path.stem}.{provider}.json"


def default_ocr_page_dir(project_root: Path, pdf_path: Path, provider: str) -> Path:
    return project_root / "artifacts" / "ocr" / f"{pdf_path.stem}.{provider}.pages"


def write_ocr_pages_json(pages: list[OcrPage], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "page_count": len(pages),
        "provider": pages[0].provider if pages else None,
        "pages": [page.to_dict() for page in pages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_ocr_pages_json(input_path: str | Path) -> list[OcrPage]:
    path = Path(input_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"OCR JSON does not contain a pages array: {path}")
    return [OcrPage.from_dict(page) for page in pages if isinstance(page, dict)]
