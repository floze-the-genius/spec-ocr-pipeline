from __future__ import annotations

import re
from dataclasses import dataclass

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

from spec_pipeline.models import SampleDefinition, clean_text
from spec_pipeline.ocr.providers.pdf_text import repair_pdf_text


@dataclass(slots=True)
class PdfTextLayerProbe:
    backend: str
    sample_id: str
    source_pdf: str
    page_count: int
    nonempty_pages: int
    text_line_count: int
    text_char_count: int
    garbled_text_char_count: int
    garbled_text_ratio: float
    average_chars_per_page: float
    classification: str
    needs_ocr: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sample_id": self.sample_id,
            "source_pdf": self.source_pdf,
            "page_count": self.page_count,
            "nonempty_pages": self.nonempty_pages,
            "text_line_count": self.text_line_count,
            "text_char_count": self.text_char_count,
            "garbled_text_char_count": self.garbled_text_char_count,
            "garbled_text_ratio": self.garbled_text_ratio,
            "average_chars_per_page": self.average_chars_per_page,
            "classification": self.classification,
            "needs_ocr": self.needs_ocr,
        }


CID_MARKER_RE = re.compile(r"\(cid:\d+\)")
MOJIBAKE_MARKER_RE = re.compile(r"[ÏÍÒÊîçèàåíðóÿõ÷äëñòì]{4,}")


def classify_text_layer(
    *,
    page_count: int,
    nonempty_pages: int,
    text_line_count: int,
    text_char_count: int,
    garbled_text_char_count: int = 0,
) -> tuple[str, bool]:
    if page_count <= 0:
        return "no_pages", True

    average_chars = text_char_count / page_count
    nonempty_ratio = nonempty_pages / page_count
    garbled_ratio = garbled_text_char_count / text_char_count if text_char_count else 0.0
    if garbled_ratio >= 0.2:
        return "text_layer_garbled", True
    if average_chars >= 80 and text_line_count >= page_count and nonempty_ratio >= 0.5:
        return "text_layer_usable", False
    return "ocr_required", True


def probe_pdf_text_layer(sample: SampleDefinition) -> PdfTextLayerProbe:
    page_count = 0
    nonempty_pages = 0
    text_line_count = 0
    text_char_count = 0
    garbled_text_char_count = 0

    for page_layout in extract_pages(str(sample.source_pdf)):
        page_count += 1
        page_chars = 0
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                text = clean_text(repair_pdf_text(line.get_text()))
                if text is None:
                    continue
                text_line_count += 1
                page_chars += len(text)
                if is_garbled_text(text):
                    garbled_text_char_count += len(text)
        if page_chars > 0:
            nonempty_pages += 1
            text_char_count += page_chars

    average_chars = round(text_char_count / page_count, 2) if page_count else 0.0
    garbled_ratio = round(garbled_text_char_count / text_char_count, 4) if text_char_count else 0.0
    classification, needs_ocr = classify_text_layer(
        page_count=page_count,
        nonempty_pages=nonempty_pages,
        text_line_count=text_line_count,
        text_char_count=text_char_count,
        garbled_text_char_count=garbled_text_char_count,
    )
    return PdfTextLayerProbe(
        backend="pdfminer-text-probe",
        sample_id=sample.sample_id,
        source_pdf=str(sample.source_pdf),
        page_count=page_count,
        nonempty_pages=nonempty_pages,
        text_line_count=text_line_count,
        text_char_count=text_char_count,
        garbled_text_char_count=garbled_text_char_count,
        garbled_text_ratio=garbled_ratio,
        average_chars_per_page=average_chars,
        classification=classification,
        needs_ocr=needs_ocr,
    )


def is_garbled_text(text: str) -> bool:
    return bool(CID_MARKER_RE.search(text) or MOJIBAKE_MARKER_RE.search(text))
