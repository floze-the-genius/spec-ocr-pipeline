from __future__ import annotations

import re

from spec_pipeline.models import SpecificationRow

CONTACT_NOTE_RE = re.compile(r"\b(?:тел\.?|телефон|phone|e-?mail|почта|ткп)\b|@", re.IGNORECASE)
RUBLE_DIMENSION_PAIR_RE = re.compile(r"₽\s*(\d{2,4})\s*₽\s*(\d{2,4})")
RUBLE_DIMENSION_RE = re.compile(r"₽\s*(\d{2,4})")
CURRENCY_CONTEXT_RE = re.compile(r"(?:цена|стоимост[ьи]|руб\.?|ткп|₽)\s*$", re.IGNORECASE)
LATEX_DIAMETER_RE = re.compile(
    r"\$?\s*\\(?:phi|varnothing|emptyset|oslash)\s*\$?\s*(\d{2,4})", re.IGNORECASE
)
LEADING_ZERO_DIMENSION_PAIR_RE = re.compile(
    r"(?<![\w.-])0(\d{2,4})\s*[xх×]\s*0?(\d{2,4})(?=$|\D)", re.IGNORECASE
)
LEADING_ZERO_ROUND_DIMENSION_RE = re.compile(
    r"(?<![\w.-])0(\d{2,4})(?=\s*[xх×]\s*\d)", re.IGNORECASE
)
DIAMETER_SIGN_RE = re.compile(r"(^|[^\w])(?:[фФΦφØø])\s*(\d{2,4})")
DIMENSION_ONLY_LEADING_ZERO_RE = re.compile(r"^0(\d{2,4})$")
OCR_UNIT_REPAIRS = {
    "wm": "шт",
    "w/m": "шт",
    "um": "шт",
    "шм": "шт",
    r"$\omega$m": "шт",
    r"$\omega$м": "шт",
    r"\omegam": "шт",
    r"\omegaм": "шт",
    "ωm": "шт",
    "ωм": "шт",
    "n.m": "п.м.",
    "n.m.": "п.м.",
}
LATIN1_CP1251_RUN_RE = re.compile(r"[À-ÿ]{2,}")
MOJIBAKE_MARKER_RE = re.compile(r"[ÏÍÒÊîçèàåíðóÿõ÷äëñòì]{4,}")


def customer_description(row: SpecificationRow) -> str | None:
    parts = [row.name, row.type_mark, row.code]
    if should_include_auxiliary_table_text(row):
        parts.append(row.manufacturer)
        if is_contact_like_note(row.note):
            parts.append(row.note)
    values = [part.strip() for part in parts if part and part.strip()]
    return normalize_customer_text(" ".join(values)) if values else None


def customer_note(row: SpecificationRow) -> str | None:
    return None if is_contact_like_note(row.note) else row.note


def is_contact_like_note(value: str | None) -> bool:
    return bool(value and CONTACT_NOTE_RE.search(value))


def should_include_auxiliary_table_text(row: SpecificationRow) -> bool:
    return row.pos is not None or row.type_mark is not None or row.code is not None


def normalize_customer_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = repair_cp1251_latin1_runs(text)
    text = repair_customer_glyphs(text)
    text = re.sub(r"(?<=\d)є\b", "°", text)
    text = RUBLE_DIMENSION_PAIR_RE.sub(
        lambda match: replace_ruble_dimension(match, text, pair=True), text
    )
    text = RUBLE_DIMENSION_RE.sub(
        lambda match: replace_ruble_dimension(match, text, pair=False), text
    )
    text = LATEX_DIAMETER_RE.sub(r"Ø\1", text)
    text = LEADING_ZERO_DIMENSION_PAIR_RE.sub(r"Ø\1x\2", text)
    text = LEADING_ZERO_ROUND_DIMENSION_RE.sub(r"Ø\1", text)
    text = DIAMETER_SIGN_RE.sub(r"\1Ø\2", text)
    text = DIMENSION_ONLY_LEADING_ZERO_RE.sub(r"Ø\1", text)
    text = re.sub(r"([-/])\s+(Ø\d{2,4})", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def replace_ruble_dimension(match: re.Match[str], source: str, *, pair: bool) -> str:
    if has_currency_context(source, match.start()):
        return match.group(0)
    if pair:
        return f"Ø{match.group(1)}x{match.group(2)}"
    return f"Ø{match.group(1)}"


def has_currency_context(source: str, match_start: int) -> bool:
    return bool(CURRENCY_CONTEXT_RE.search(source[:match_start][-24:]))


def normalize_customer_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = re.sub(r"\s+", "", unit).strip()
    if not text:
        return None
    text = repair_cp1251_latin1_runs(text)
    text = repair_customer_glyphs(text)
    return OCR_UNIT_REPAIRS.get(text.lower(), text)


def repair_cp1251_latin1_runs(text: str) -> str:
    """Repair CP1251 mojibake fragments without rewriting valid Unicode around them."""

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        try:
            candidate = source.encode("latin-1").decode("cp1251")
        except UnicodeError:
            return source
        return candidate if text_score(candidate) > text_score(source) else source

    return LATIN1_CP1251_RUN_RE.sub(replace, text)


def repair_customer_glyphs(text: str) -> str:
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
    return re.sub(r"(?<![A-Za-zА-Яа-я])(?:[ГG][ОO0][СC][ТT])(?![A-Za-zА-Яа-я])", "ГОСТ", repaired)


def text_score(text: str) -> int:
    cyrillic = sum(1 for char in text if "А" <= char <= "я" or char in {"ё", "Ё"})
    mojibake = len(MOJIBAKE_MARKER_RE.findall(text))
    replacement = text.count("�")
    return cyrillic * 2 - mojibake * 10 - replacement * 5
