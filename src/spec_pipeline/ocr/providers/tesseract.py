from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from spec_pipeline.models import clean_text
from spec_pipeline.ocr.base import OcrPage, OcrToken
from spec_pipeline.ocr.render import PdfRenderer, create_pdf_renderer


@dataclass(slots=True)
class TesseractConfig:
    language: str = "rus+eng"
    page_segmentation_mode: int = 6
    dpi: int = 200
    command: str = "tesseract"


class TesseractOcrProvider:
    provider_name = "tesseract"

    def __init__(
        self,
        *,
        config: TesseractConfig | None = None,
        renderer: PdfRenderer | None = None,
    ) -> None:
        self.config = config or TesseractConfig()
        self.renderer = renderer or create_pdf_renderer(dpi=self.config.dpi)

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        pages: list[int] | None = None,
        output_dir: Path | None = None,
    ) -> list[OcrPage]:
        resolved_output_dir = output_dir or Path("artifacts") / "ocr" / pdf_path.stem
        page_numbers = pages or list(range(1, self.renderer.page_count(pdf_path) + 1))
        extracted: list[OcrPage] = []
        for page_number in page_numbers:
            image_path = resolved_output_dir / f"page_{page_number:03d}.png"
            rendered = self.renderer.render_page(
                pdf_path, page_number=page_number, output_path=image_path
            )
            tsv = self._run_tesseract_tsv(rendered.image_path)
            extracted.append(
                parse_tesseract_tsv(
                    tsv,
                    page_index=page_number - 1,
                    page_ref=str(page_number),
                    width=rendered.width,
                    height=rendered.height,
                    provider=self.provider_name,
                    image_path=str(rendered.image_path),
                )
            )
        return extracted

    def _run_tesseract_tsv(self, image_path: Path) -> str:
        result = subprocess.run(
            [
                self.config.command,
                str(image_path),
                "stdout",
                "-l",
                self.config.language,
                "--psm",
                str(self.config.page_segmentation_mode),
                "tsv",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"tesseract failed for {image_path}: {result.stderr.strip()}")
        return result.stdout


def parse_page_selection(value: str | None) -> list[int] | None:
    if value is None:
        return None
    pages: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid page range: {token}")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(token))
    return sorted(dict.fromkeys(pages))


def parse_tesseract_tsv(
    tsv: str,
    *,
    page_index: int,
    page_ref: str,
    width: int,
    height: int,
    provider: str,
    image_path: str | None = None,
) -> OcrPage:
    tokens: list[OcrToken] = []
    line_parts: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    reader = csv.DictReader(tsv.splitlines(), delimiter="\t")
    for row in reader:
        text = clean_text(row.get("text"))
        if text is None:
            continue
        confidence = _parse_confidence(row.get("conf"))
        x0 = float(row.get("left") or 0)
        y0 = float(row.get("top") or 0)
        w = float(row.get("width") or 0)
        h = float(row.get("height") or 0)
        tokens.append(
            OcrToken(
                text=text,
                page_ref=page_ref,
                x0=x0,
                y0=y0,
                x1=x0 + w,
                y1=y0 + h,
                confidence=confidence,
            )
        )
        key = (
            row.get("block_num") or "0",
            row.get("par_num") or "0",
            row.get("line_num") or "0",
        )
        line_parts[key].append(text)

    lines = [" ".join(parts) for _, parts in sorted(line_parts.items())]
    return OcrPage(
        page_index=page_index,
        page_ref=page_ref,
        width=width,
        height=height,
        text="\n".join(lines),
        tokens=tokens,
        provider=provider,
        image_path=image_path,
    )


def _parse_confidence(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except ValueError:
        return None
    return confidence if confidence >= 0 else None
