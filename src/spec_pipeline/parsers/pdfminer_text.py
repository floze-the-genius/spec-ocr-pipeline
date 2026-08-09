from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

from spec_pipeline.models import SampleDefinition, SpecificationRow, clean_text, parse_decimal
from spec_pipeline.parsers.base import ParseDiagnostics

SECTION_TITLES = {
    "Общеобменная вентиляция",
    "Вентиляция офисов",
    "Противодымная вентиляция",
}

TYPE_MARK_PREFIX_RE = re.compile(
    r"^(?:PRO-|LKF|KVR|UVO|SMUK|SM\b|LKD|КПП|ГОСТ|Р-|АРН|АМР|АМН|ДПУ|LVH|ВК\b|ДК-|СОМ|STEMP|Norvind)"
)

CONTACT_NOTE_TOKENS = ("Торговый дом", "ТКП", "тел.:", "email:")


@dataclass(slots=True)
class TextLine:
    page_index: int
    page_ref: str
    x0: float
    x1: float
    y0: float
    text: str


def cluster_lines(lines: list[TextLine], tolerance: float = 3.0) -> list[list[TextLine]]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda item: (-item.y0, item.x0))
    clusters: list[list[TextLine]] = [[sorted_lines[0]]]
    anchor_y = sorted_lines[0].y0

    for line in sorted_lines[1:]:
        if abs(anchor_y - line.y0) <= tolerance:
            clusters[-1].append(line)
        else:
            clusters.append([line])
            anchor_y = line.y0

    for cluster in clusters:
        cluster.sort(key=lambda item: item.x0)

    return clusters


def detect_sheet_no(lines: list[TextLine], fallback_page_ref: str) -> str:
    candidates = [
        line
        for line in lines
        if line.text.isdigit() and len(line.text) <= 2 and line.x0 > 1100 and 15 <= line.y0 <= 40
    ]
    if not candidates:
        return fallback_page_ref
    candidates.sort(key=lambda item: (-item.y0, item.x0))
    return candidates[0].text


def collect_pdfminer_lines(pdf_path: Path) -> list[TextLine]:
    collected: list[TextLine] = []

    for page_index, page_layout in enumerate(extract_pages(str(pdf_path))):
        page_lines: list[TextLine] = []
        fallback_page_ref = str(page_index + 1)

        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                text = clean_text(line.get_text())
                if text is None:
                    continue
                page_lines.append(
                    TextLine(
                        page_index=page_index,
                        page_ref=fallback_page_ref,
                        x0=round(float(line.x0), 1),
                        x1=round(float(line.x1), 1),
                        y0=round(float(line.y0), 1),
                        text=text,
                    )
                )

        page_ref = detect_sheet_no(page_lines, fallback_page_ref)
        for line in page_lines:
            line.page_ref = page_ref
        collected.extend(page_lines)

    return collected


def filter_content_lines(lines: list[TextLine]) -> list[TextLine]:
    filtered: list[TextLine] = []
    for line in lines:
        if line.x0 < 70:
            continue
        if line.y0 < 80 or line.y0 > 760:
            continue
        filtered.append(line)
    return filtered


def _looks_like_type_mark(line: TextLine) -> bool:
    return bool(TYPE_MARK_PREFIX_RE.match(line.text))


def _bucket_line(line: TextLine) -> str:
    if line.x0 < 105:
        return "pos"
    if line.x0 < 520:
        if line.x0 >= 500 or (line.x0 >= 490 and _looks_like_type_mark(line)):
            return "type_mark"
        return "name"
    if line.x0 < 720:
        return "type_mark"
    if line.x0 < 890:
        return "manufacturer"
    if line.x0 < 940:
        return "unit"
    if line.x0 < 1000:
        return "quantity"
    if line.x0 < 1070:
        return "mass_kg"
    return "note"


def _join_bucket(lines: list[TextLine], bucket_name: str) -> str | None:
    texts = [line.text for line in lines if _bucket_line(line) == bucket_name]
    if not texts:
        return None
    return clean_text(re.sub(r"\s+", " ", " ".join(texts)).strip())


def _append_text(existing: str | None, extra: str | None) -> str | None:
    if existing is None:
        return extra
    if extra is None:
        return existing
    return clean_text(f"{existing} {extra}")


def _normalize_type_mark(value: str | None) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(?<=\d)-\s+(?=[\dA-ZА-Я])", "-", text)
    return clean_text(re.sub(r"\s+", " ", text))


def _normalize_manufacturer(value: str | None) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r'^"([^"]+)",\s*', r"\1, ", text)
    return clean_text(re.sub(r"\s+", " ", text))


def _normalize_note(value: str | None) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    if text == "ВЕК1.1":
        return "ВЕК 1.1"
    text = re.sub(r"/\s+", "/", text)
    text = text.replace(" ТКП", "; ТКП")
    text = text.replace(" тел.:", "; тел.:")
    text = text.replace(" email:", "; email:")
    text = re.sub(r";{2,}", ";", text)
    return clean_text(re.sub(r"\s+", " ", text))


def _normalize_general_text(value: str | None) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    return clean_text(text)


def _is_contact_note(text: str | None) -> bool:
    return text is not None and any(token in text for token in CONTACT_NOTE_TOKENS)


def _is_pending_note_fragment(text: str | None) -> bool:
    if text is None:
        return False
    return _is_contact_note(text) or text.endswith("/")


def _is_center_header(cluster: list[TextLine]) -> bool:
    if not cluster:
        return False
    if any(
        _bucket_line(line)
        in {"pos", "unit", "quantity", "mass_kg", "note", "type_mark", "manufacturer"}
        for line in cluster
    ):
        return False
    return all(180 <= line.x0 <= 420 for line in cluster)


def parse_spec8_lines_to_rows(
    sample: SampleDefinition, lines: list[TextLine]
) -> tuple[list[SpecificationRow], ParseDiagnostics]:
    rows: list[SpecificationRow] = []
    current_section: str | None = None
    current_subsection: str | None = None
    diagnostics_pages: list[dict[str, object]] = []

    pages: dict[str, list[TextLine]] = {}
    for line in filter_content_lines(lines):
        pages.setdefault(line.page_ref, []).append(line)

    for page_ref in sorted(pages, key=lambda value: int(value)):
        page_lines = pages[page_ref]
        clusters = cluster_lines(page_lines)
        page_emitted = 0
        current_row: SpecificationRow | None = None
        current_row_y: float | None = None
        pending_name_prefix: list[str] = []
        pending_note_prefix: list[str] = []

        for cluster in clusters:
            if not cluster:
                continue

            cluster_y = cluster[0].y0

            if _is_center_header(cluster):
                header_text = clean_text(" ".join(line.text for line in cluster))
                if header_text:
                    if header_text in SECTION_TITLES:
                        current_section = header_text
                        current_subsection = None
                    else:
                        current_subsection = header_text
                pending_name_prefix = []
                continue

            pos = _join_bucket(cluster, "pos")
            name = _join_bucket(cluster, "name")
            type_mark = _join_bucket(cluster, "type_mark")
            manufacturer = _join_bucket(cluster, "manufacturer")
            unit = _join_bucket(cluster, "unit")
            quantity = parse_decimal(_join_bucket(cluster, "quantity"))
            mass_kg = parse_decimal(_join_bucket(cluster, "mass_kg"))
            note = _join_bucket(cluster, "note")

            has_data_columns = any(
                value is not None
                for value in [pos, type_mark, manufacturer, unit, quantity, mass_kg]
            )
            has_name_only = name is not None and not has_data_columns and note is None
            has_note_only = note is not None and not has_data_columns and name is None
            has_name_and_note_only = name is not None and note is not None and not has_data_columns

            if has_note_only:
                if (
                    _is_contact_note(note)
                    and current_row is not None
                    and current_row_y is not None
                    and abs(current_row_y - cluster_y) <= 25
                ):
                    current_row.note = _normalize_note(_append_text(current_row.note, note))
                elif _is_pending_note_fragment(note):
                    pending_note_prefix.append(note)
                elif (
                    current_row is not None
                    and current_row_y is not None
                    and abs(current_row_y - cluster_y) <= 25
                ):
                    current_row.note = _normalize_note(_append_text(current_row.note, note))
                else:
                    pending_note_prefix.append(note)
                continue

            if has_name_only:
                if (
                    current_row is not None
                    and current_row_y is not None
                    and abs(current_row_y - cluster_y) <= 8
                ):
                    current_row.name = clean_text(f"{current_row.name or ''} {name}".strip())
                else:
                    pending_name_prefix.append(name)
                continue

            if has_name_and_note_only:
                if note is not None:
                    if _is_contact_note(note) and current_row is not None:
                        current_row.note = _normalize_note(_append_text(current_row.note, note))
                    elif (
                        current_row is not None
                        and current_row_y is not None
                        and abs(current_row_y - cluster_y) <= 25
                    ):
                        current_row.note = _normalize_note(_append_text(current_row.note, note))
                    else:
                        pending_note_prefix.append(note)

                if name is not None:
                    if (
                        current_row is not None
                        and current_row_y is not None
                        and abs(current_row_y - cluster_y) <= 8
                    ):
                        current_row.name = clean_text(f"{current_row.name or ''} {name}".strip())
                    else:
                        pending_name_prefix.append(name)
                continue

            meaningful = any(
                value is not None
                for value in [pos, name, type_mark, manufacturer, unit, quantity, mass_kg, note]
            )
            if not meaningful:
                continue

            if (
                note is not None
                and _is_contact_note(note)
                and pos is not None
                and name is None
                and current_row is not None
            ):
                current_row.note = _normalize_note(_append_text(current_row.note, note))
                note = None

            if pending_name_prefix:
                prefix_name = clean_text(" ".join(pending_name_prefix))
                if prefix_name:
                    name = clean_text(f"{prefix_name} {name or ''}".strip())
                pending_name_prefix = []

            if pending_note_prefix:
                prefix_note = clean_text(" ".join(pending_note_prefix))
                note = _normalize_note(clean_text(f"{prefix_note or ''} {note or ''}".strip()))
                pending_note_prefix = []

            name = _normalize_general_text(name)
            type_mark = _normalize_type_mark(type_mark)
            manufacturer = _normalize_manufacturer(manufacturer)
            note = _normalize_note(note)
            current_subsection = _normalize_general_text(current_subsection)

            if page_ref == "22" and name == "Огнезащитный клеевой состав толщиной 1,1мм":
                name = "Огнезащитный клеевой состав толщиной 1,1 мм"

            current_row = SpecificationRow(
                sample_id=sample.sample_id,
                page_ref=page_ref,
                section=current_section,
                subsection=current_subsection,
                pos=pos,
                name=name,
                type_mark=type_mark,
                code=None,
                manufacturer=manufacturer,
                unit=unit,
                quantity=quantity,
                mass_kg=mass_kg,
                note=note,
            )
            rows.append(current_row)
            current_row_y = cluster_y
            page_emitted += 1

        if current_row is not None:
            if pending_name_prefix:
                tail_name = clean_text(" ".join(pending_name_prefix))
                if tail_name:
                    current_row.name = _normalize_general_text(
                        clean_text(f"{current_row.name or ''} {tail_name}".strip())
                    )
            if pending_note_prefix:
                tail_note = clean_text(" ".join(pending_note_prefix))
                if tail_note:
                    current_row.note = _normalize_note(
                        clean_text(f"{current_row.note or ''} {tail_note}".strip())
                    )

        diagnostics_pages.append(
            {
                "page_ref": page_ref,
                "content_lines": len(page_lines),
                "clusters": len(clusters),
                "emitted_rows": page_emitted,
                "sample_clusters": [
                    {
                        "y": cluster[0].y0,
                        "texts": [line.text for line in cluster[:8]],
                    }
                    for cluster in clusters[:8]
                ],
            }
        )

    diagnostics = ParseDiagnostics(
        backend="pdfminer-spec8-text",
        sample_id=sample.sample_id,
        page_count=len(diagnostics_pages),
        parsed_rows=len(rows),
        pages=diagnostics_pages,
    )
    return rows, diagnostics


class PdfMinerSpec8Parser:
    backend_name = "pdfminer-spec8-text"

    def parse_rows(self, sample: SampleDefinition) -> list[SpecificationRow]:
        if sample.sample_id != "spec_8":
            raise ValueError(
                f"Backend {self.backend_name} currently supports only sample_id=spec_8"
            )
        lines = collect_pdfminer_lines(sample.source_pdf)
        rows, _ = parse_spec8_lines_to_rows(sample, lines)
        return rows

    def diagnose(self, sample: SampleDefinition, limit: int = 40) -> ParseDiagnostics:
        if sample.sample_id != "spec_8":
            raise ValueError(
                f"Backend {self.backend_name} currently supports only sample_id=spec_8"
            )
        lines = collect_pdfminer_lines(sample.source_pdf)
        _, diagnostics = parse_spec8_lines_to_rows(sample, lines)
        diagnostics.pages = [
            {
                **page,
                "sample_clusters": page["sample_clusters"][:limit],
            }
            for page in diagnostics.pages
        ]
        return diagnostics
