from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

POSITION_CODE_RE = re.compile(r"^[A-Za-zА-Яа-я]{0,4}[-_.]?[0-9][A-Za-zА-Яа-я0-9./_-]*$")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
DIMENSION_TEXT_RE = re.compile(r"(?:\d{2,4}\s*[xх×]\s*\d{2,4}|[Øø∅⌀]|\bδ\s*=|мм|ìì)", re.IGNORECASE)
BARE_DIMENSION_VALUE_RE = re.compile(r"^\d{2,4}$")
LATIN1_CP1251_RUN_RE = re.compile(r"[À-ÿ]{2,}")
MOJIBAKE_MARKER_RE = re.compile(r"[ÏÍÒÊîçèàåíðóÿõ÷äëñòì]{4,}")


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = repair_text_encoding_artifacts(str(value).strip())
    return text or None


def repair_text_encoding_artifacts(text: str) -> str:
    """Repair common OCR/PDF encoding fragments without touching normal text."""

    repaired = _repair_cp1251_latin1_runs(text)
    repaired = (
        repaired.replace("ìì", "мм")
        .replace("ÌÌ", "мм")
        .replace("мІ", "м²")
        .replace("МІ", "м²")
        .replace("мі", "м³")
        .replace("Мі", "м³")
        .replace("м2", "м²")
        .replace("「", "[")
    )
    repaired = re.sub(
        r"(?<![A-Za-zА-Яа-я])(?:[ГG][ОO0][СC][ТT])(?![A-Za-zА-Яа-я])",
        "ГОСТ",
        repaired,
    )
    repaired = re.sub(r"\[(SPH-W2)-\[(SPC-F3)\]", r"[\1]-[\2]", repaired)
    repaired = re.sub(r"\[(?:VI|VІ|V1)\.K[L1IІ]35C(?=\()", "[V1.КЦ35C", repaired)
    repaired = re.sub(r"\bКП[ОO][ЛП]-2(?=\()", "КЛОП-2", repaired)
    repaired = re.sub(r"\b(ПВ\d+)\s+Ф[ФЭэ]\s+SUP[MVU]\b", r"\1 фр SUPM", repaired)
    repaired = re.sub(r"\b(ПВ\d+)\s+фр\s+SUP[UV]\b", r"\1 фр SUPM", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\bрекупе\?раторе\b", "рекуператоре", repaired, flags=re.IGNORECASE)
    return re.sub(
        r"(?<![A-Za-zА-Яа-я0-9/.-])(12|24|36|110|220|230|380)B\b",
        r"\1В",
        repaired,
    )


def _repair_cp1251_latin1_runs(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        try:
            candidate = source.encode("latin-1").decode("cp1251")
        except UnicodeError:
            return source
        return candidate if _text_score(candidate) > _text_score(source) else source

    return LATIN1_CP1251_RUN_RE.sub(replace, text)


def _text_score(text: str) -> int:
    cyrillic = sum(1 for char in text if "А" <= char <= "я" or char in {"ё", "Ё"})
    mojibake = len(MOJIBAKE_MARKER_RE.findall(text))
    replacement = text.count("�")
    return cyrillic * 2 - mojibake * 10 - replacement * 5


def parse_decimal(value: str | None) -> Decimal | None:
    text = clean_text(value)
    if text is None:
        return None
    normalized = text.replace(" ", "").replace(",", ".")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def raw_decimal_text(value: str | None, parsed: Decimal | None) -> str | None:
    text = clean_text(value)
    if text is None or parsed is not None:
        return None
    return text


def repair_shifted_pos_name(
    *,
    pos: str | None,
    name: str | None,
    unit: str | None,
    quantity: Decimal | None,
    type_mark: str | None,
    code: str | None,
    manufacturer: str | None,
    mass_kg: Decimal | None,
) -> tuple[str | None, str | None]:
    if name is not None or pos is None:
        return pos, name
    if not any([unit, quantity, type_mark, code, manufacturer, mass_kg]):
        return pos, name
    if looks_like_shifted_item_name(pos):
        return None, pos
    return pos, name


def looks_like_shifted_item_name(value: str) -> bool:
    text = clean_text(value)
    if text is None:
        return False
    if DIMENSION_TEXT_RE.search(text):
        return True
    if BARE_DIMENSION_VALUE_RE.fullmatch(text) and int(text) >= 50:
        return True
    if any(separator in text for separator in (" ", ",", ":", "(", ")", '"')):
        return True
    if POSITION_CODE_RE.fullmatch(text):
        return False
    return bool(WORD_RE.search(text))


@dataclass(slots=True)
class SpecificationRow:
    sample_id: str
    page_ref: str | None
    section: str | None
    subsection: str | None
    pos: str | None
    name: str | None
    type_mark: str | None
    code: str | None
    manufacturer: str | None
    unit: str | None
    quantity: Decimal | None
    mass_kg: Decimal | None
    note: str | None
    status: str | None = None
    quantity_raw: str | None = None
    mass_kg_raw: str | None = None

    @classmethod
    def from_tsv_row(
        cls,
        *,
        sample_id: str,
        page_field: str,
        row: dict[str, str],
    ) -> "SpecificationRow":
        quantity = parse_decimal(row.get("quantity"))
        mass_kg = parse_decimal(row.get("mass_kg"))
        pos, name = repair_shifted_pos_name(
            pos=clean_text(row.get("pos")),
            name=clean_text(row.get("name")),
            unit=clean_text(row.get("unit")),
            quantity=quantity,
            type_mark=clean_text(row.get("type_mark")),
            code=clean_text(row.get("code")),
            manufacturer=clean_text(row.get("manufacturer")),
            mass_kg=mass_kg,
        )
        return cls(
            sample_id=sample_id,
            page_ref=clean_text(row.get(page_field)),
            section=clean_text(row.get("section")),
            subsection=clean_text(row.get("subsection")),
            pos=pos,
            name=name,
            type_mark=clean_text(row.get("type_mark")),
            code=clean_text(row.get("code")),
            manufacturer=clean_text(row.get("manufacturer")),
            unit=clean_text(row.get("unit")),
            quantity=quantity,
            quantity_raw=raw_decimal_text(row.get("quantity"), quantity),
            mass_kg=mass_kg,
            mass_kg_raw=raw_decimal_text(row.get("mass_kg"), mass_kg),
            note=clean_text(row.get("note")),
            status=clean_text(row.get("status")),
        )

    def is_grouping_row(self) -> bool:
        return (
            self.pos is None
            and self.name is None
            and (self.section is not None or self.subsection is not None)
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "sample_id": self.sample_id,
            "page_ref": self.page_ref,
            "section": self.section,
            "subsection": self.subsection,
            "pos": self.pos,
            "name": self.name,
            "type_mark": self.type_mark,
            "code": self.code,
            "manufacturer": self.manufacturer,
            "unit": self.unit,
            "quantity": self.quantity_raw
            if self.quantity_raw is not None
            else str(self.quantity)
            if self.quantity is not None
            else None,
            "mass_kg": self.mass_kg_raw
            if self.mass_kg_raw is not None
            else str(self.mass_kg)
            if self.mass_kg is not None
            else None,
            "note": self.note,
            "status": self.status,
        }


@dataclass(slots=True)
class SampleDefinition:
    sample_id: str
    title: str
    source_pdf: Path
    ground_truth_tsv: Path | None
    expected_xlsx: Path | None
    visual_xlsx: Path | None
    page_field: str = "page"

    def artifact_paths(self) -> dict[str, Path | None]:
        return {
            "source_pdf": self.source_pdf,
            "ground_truth_tsv": self.ground_truth_tsv,
            "expected_xlsx": self.expected_xlsx,
            "visual_xlsx": self.visual_xlsx,
        }


@dataclass(slots=True)
class SampleManifest:
    manifest_version: str
    name: str
    manifest_path: Path
    samples: list[SampleDefinition]

    def get_sample(self, sample_id: str) -> SampleDefinition:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(f"Unknown sample_id: {sample_id}")
