from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LTTextContainer, LTTextLine

from spec_pipeline.models import clean_text
from spec_pipeline.ocr.base import OcrPage, OcrToken
from spec_pipeline.ocr.render import PdfRenderer, create_pdf_renderer

MOJIBAKE_MARKER_RE = re.compile(r"[ÏÍÒÊîçèàåíðóÿõ÷äëñòì]{4,}")
LATIN1_CP1251_RUN_RE = re.compile(r"[À-ÿ]{2,}")
PDF_TABLE_LAYER_PREFIX = "PDF_TABLE_LAYER:"
PDF_TABLE_ROW_PREFIX = "PDF_TABLE_ROW "
PDF_TEXT_COORDS_PREFIX = "PDF_TEXT_COORDS:"
PDF_TABLE_Y_TOLERANCE = 12.0
PDF_TABLE_REFERENCE_WIDTH = 1191.0
PDF_TABLE_COLUMNS = (
    ("pos", 60.0, 130.0),
    ("name", 130.0, 455.0),
    ("type_mark", 455.0, 600.0),
    ("code", 600.0, 735.0),
    ("manufacturer", 735.0, 860.0),
    ("unit", 860.0, 915.0),
    ("quantity", 915.0, 970.0),
    ("mass_kg", 970.0, 1040.0),
    ("note", 1040.0, 1160.0),
)
PDF_TABLE_HEADER_TEXTS = {
    "Pos",
    "Name",
    "Type / mark",
    "Code",
    "Manufacturer",
    "Unit",
    "Qty",
    "Mass kg",
    "Note",
    "Позиция",
    "Наименование и техническая характеристика",
    "Тип, марка, обозна-",
    "чение документа",
    "опросного листа",
    "Код оборудования,",
    "изделия,",
    "материала",
    "Завод-изготовитель",
    "Единица",
    "измере-",
    "ния",
    "Коли-",
    "чество",
    "Масса",
    "единицы,",
    "кг",
    "Примечание",
}
PDF_TABLE_HEADER_FRAGMENTS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "позиция",
    "наименование",
    "техническая",
    "характеристика",
    "тип",
    "марка",
    "обозначение",
    "документа",
    "опросного",
    "листа",
    "код",
    "оборудования",
    "изделия",
    "материала",
    "завод",
    "изготовитель",
    "единица",
    "измере",
    "рения",
    "ния",
    "коли",
    "чество",
    "масса",
    "единицы",
    "кг",
    "примечание",
    "pos",
    "name",
    "type",
    "mark",
    "code",
    "manufacturer",
    "unit",
    "qty",
    "mass",
    "kg",
    "note",
}
PDF_TITLE_BLOCK_RE = re.compile(
    r"(?:^|\s)(?:изм\.?|кол\.?уч|лист|№док|подп\.?|дата|гип|стадия|листов)(?:\s|$)",
    re.IGNORECASE,
)
PDF_TITLE_BLOCK_MOJIBAKE_RE = re.compile(r"(?:Èçì|Êîë|Ëèñò|äîê|Ïîäï)", re.IGNORECASE)


@dataclass(slots=True)
class PdfTextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class PdfTextOcrProvider:
    """Expose a PDF text layer through the same page/token contract as OCR."""

    provider_name = "pdf-text"

    def __init__(self, *, renderer: PdfRenderer | None = None, render_dpi: int = 200) -> None:
        self._renderer = renderer
        self._render_dpi = render_dpi

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        pages: list[int] | None = None,
        output_dir: Path | None = None,
    ) -> list[OcrPage]:
        selected = set(pages) if pages is not None else None
        extracted: list[OcrPage] = []
        renderer = self._renderer
        if output_dir is not None and renderer is None:
            renderer = create_pdf_renderer(dpi=self._render_dpi)
        for page_number, page_layout in enumerate(extract_pages(str(pdf_path)), start=1):
            if selected is not None and page_number not in selected:
                continue
            image_path = None
            if output_dir is not None and renderer is not None:
                rendered_path = output_dir / f"page_{page_number:03d}.png"
                image_path = renderer.render_page(
                    pdf_path, page_number=page_number, output_path=rendered_path
                ).image_path
            extracted.append(
                _layout_to_page(
                    page_layout,
                    page_number,
                    self.provider_name,
                    image_path=image_path,
                )
            )
        return extracted


def _layout_to_page(
    page_layout: LTPage,
    page_number: int,
    provider: str,
    *,
    image_path: Path | None = None,
) -> OcrPage:
    lines: list[PdfTextLine] = []
    for element in page_layout:
        if not isinstance(element, LTTextContainer):
            continue
        for line in element:
            if not isinstance(line, LTTextLine):
                continue
            text = clean_text(repair_pdf_text(line.get_text()))
            if text is None:
                continue
            lines.append(PdfTextLine(text=text, x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1))

    ordered = sorted(lines, key=lambda item: (-item.y0, item.x0))
    page_ref = str(page_number)
    tokens = [
        OcrToken(
            text=line.text,
            page_ref=page_ref,
            x0=line.x0,
            y0=line.y0,
            x1=line.x1,
            y1=line.y1,
            confidence=None,
        )
        for line in ordered
    ]
    coords_text = "\n".join(
        f"[{line.x0:.1f},{line.y0:.1f},{line.x1:.1f},{line.y1:.1f}] {line.text}" for line in ordered
    )
    table_layer = build_pdf_table_layer(
        ordered,
        page_ref=page_ref,
        page_height=page_layout.height,
        page_width=page_layout.width,
    )
    text = (
        f"{table_layer}\n\n{PDF_TEXT_COORDS_PREFIX}\n{coords_text}" if table_layer else coords_text
    )
    return OcrPage(
        page_index=page_number - 1,
        page_ref=page_ref,
        width=int(page_layout.width),
        height=int(page_layout.height),
        text=text,
        tokens=tokens,
        provider=provider,
        image_path=str(image_path) if image_path is not None else None,
    )


def repair_pdf_text(text: str) -> str:
    """Repair common CP1251 text extracted as Latin-1 mojibake from CAD PDFs."""

    repaired = text
    if any(128 <= ord(char) <= 255 for char in text):
        try:
            candidate = text.encode("latin-1").decode("cp1251")
        except UnicodeError:
            candidate = text
        repaired = candidate if _text_score(candidate) > _text_score(text) else text
    repaired = _repair_cp1251_latin1_runs(repaired)
    return _repair_pdf_glyphs(repaired)


def _repair_cp1251_latin1_runs(text: str) -> str:
    """Repair mojibake words even when a line also contains valid Unicode symbols."""

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        try:
            candidate = source.encode("latin-1").decode("cp1251")
        except UnicodeError:
            return source
        return candidate if _text_score(candidate) > _text_score(source) else source

    return LATIN1_CP1251_RUN_RE.sub(replace, text)


def _repair_pdf_glyphs(text: str) -> str:
    repaired = (
        text.replace("ìì", "мм")
        .replace("ÌÌ", "мм")
        .replace("мІ", "м²")
        .replace("МІ", "м²")
        .replace("мі", "м³")
        .replace("Мі", "м³")
        .replace("м2", "м²")
        .replace("「", "[")
    )
    repaired = re.sub(r"(?<=\d)є\b", "°", repaired)
    return re.sub(r"(?<![A-Za-zА-Яа-я])(?:[ГG][ОO0][СC][ТT])(?![A-Za-zА-Яа-я])", "ГОСТ", repaired)


def _text_score(text: str) -> int:
    cyrillic = sum(1 for char in text if "А" <= char <= "я" or char == "ё" or char == "Ё")
    mojibake = len(MOJIBAKE_MARKER_RE.findall(text))
    replacement = text.count("�")
    return cyrillic * 2 - mojibake * 10 - replacement * 5


def build_pdf_table_layer(
    lines: list[PdfTextLine], *, page_ref: str, page_height: float, page_width: float | None = None
) -> str | None:
    columns = _pdf_table_columns(page_width)
    table_lines = _pdf_table_candidate_lines(lines, page_height=page_height, columns=columns)
    if len(table_lines) < 3:
        return None
    grouped = _group_pdf_table_lines(table_lines)
    rows: list[str] = []
    for group in grouped:
        payload = _pdf_table_group_to_row(group, page_ref=page_ref, columns=columns)
        if payload is None:
            continue
        rows.append(PDF_TABLE_ROW_PREFIX + json.dumps(payload, ensure_ascii=False))
    if len(rows) < 3:
        return None
    return PDF_TABLE_LAYER_PREFIX + "\n" + "\n".join(rows)


def parse_pdf_table_layer_rows(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.startswith(PDF_TABLE_ROW_PREFIX):
            continue
        try:
            payload = json.loads(line[len(PDF_TABLE_ROW_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def compact_pdf_text_overlay(text: str) -> str:
    if text.startswith(PDF_TABLE_LAYER_PREFIX) and f"\n\n{PDF_TEXT_COORDS_PREFIX}" in text:
        return text.split(f"\n\n{PDF_TEXT_COORDS_PREFIX}", 1)[0].strip()
    return text


def _pdf_table_candidate_lines(
    lines: list[PdfTextLine],
    *,
    page_height: float,
    columns: tuple[tuple[str, float, float], ...],
) -> list[tuple[PdfTextLine, str]]:
    candidates: list[tuple[PdfTextLine, str]] = []
    for line in lines:
        if line.y0 < 60 or line.y0 > page_height - 90:
            continue
        column = _pdf_table_column(line.x0, columns)
        if column is None:
            continue
        if line.text in PDF_TABLE_HEADER_TEXTS and not (
            line.text.lower() == "кг" and column == "unit"
        ):
            continue
        candidates.append((line, column))
    return candidates


def _group_pdf_table_lines(
    lines: list[tuple[PdfTextLine, str]],
) -> list[list[tuple[PdfTextLine, str]]]:
    groups: list[list[tuple[PdfTextLine, str]]] = []
    group_y: list[float] = []
    for item in sorted(lines, key=lambda entry: -entry[0].y0):
        line, _column = item
        placed = False
        for index, anchor_y in enumerate(group_y):
            if abs(line.y0 - anchor_y) <= PDF_TABLE_Y_TOLERANCE:
                groups[index].append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
            group_y.append(line.y0)
    return groups


def _pdf_table_group_to_row(
    group: list[tuple[PdfTextLine, str]],
    *,
    page_ref: str,
    columns: tuple[tuple[str, float, float], ...],
) -> dict[str, object] | None:
    fields: dict[str, list[str]] = {column: [] for column, _start, _end in columns}
    for line, column in sorted(group, key=lambda entry: entry[0].x0):
        fields[column].append(line.text)
    row: dict[str, object] = {
        "page_ref": page_ref,
        "section": None,
        "subsection": None,
        "status": None,
    }
    for column, _start, _end in columns:
        value = clean_text(" ".join(fields[column]))
        row[column] = value
    if _is_ignorable_pdf_table_row(row):
        return None
    return row


def _pdf_table_columns(page_width: float | None) -> tuple[tuple[str, float, float], ...]:
    if page_width is None or page_width <= PDF_TABLE_REFERENCE_WIDTH * 1.1:
        return PDF_TABLE_COLUMNS
    scale = page_width / PDF_TABLE_REFERENCE_WIDTH
    return tuple((column, start * scale, end * scale) for column, start, end in PDF_TABLE_COLUMNS)


def _pdf_table_column(x0: float, columns: tuple[tuple[str, float, float], ...]) -> str | None:
    for column, start, end in columns:
        if start <= x0 < end:
            return column
    return None


def _is_ignorable_pdf_table_row(row: dict[str, object]) -> bool:
    fields = (
        "pos",
        "name",
        "type_mark",
        "code",
        "manufacturer",
        "unit",
        "quantity",
        "mass_kg",
        "note",
    )
    values = [str(row.get(field)).strip() for field in fields if row.get(field)]
    if not values:
        return True
    if _is_header_number_row(row):
        return True
    if _is_header_fragment_row(values):
        return True
    if _is_title_block_row(row, values):
        return True
    return False


def _is_header_number_row(row: dict[str, object]) -> bool:
    expected = {
        "pos": "1",
        "name": "2",
        "type_mark": "3",
        "code": "4",
        "manufacturer": "5",
        "unit": "6",
        "quantity": "7",
        "mass_kg": "8",
        "note": "9",
    }
    matches = 0
    populated = 0
    for field, expected_value in expected.items():
        value = clean_text(str(row.get(field))) if row.get(field) is not None else None
        if value is None:
            continue
        populated += 1
        if value == expected_value:
            matches += 1
    return populated >= 4 and matches == populated


def _is_header_fragment_row(values: list[str]) -> bool:
    normalized_parts: list[str] = []
    for value in values:
        normalized_parts.extend(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", value.lower()))
    if not normalized_parts or all(part.isdigit() for part in normalized_parts):
        return False
    return all(part in PDF_TABLE_HEADER_FRAGMENTS for part in normalized_parts)


def _is_title_block_row(row: dict[str, object], values: list[str]) -> bool:
    combined = " ".join(values)
    if PDF_TITLE_BLOCK_MOJIBAKE_RE.search(combined):
        return True
    if PDF_TITLE_BLOCK_RE.search(combined) and not row.get("name"):
        return True
    return False
