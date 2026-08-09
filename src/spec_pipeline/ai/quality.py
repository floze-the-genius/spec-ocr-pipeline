from __future__ import annotations

import re
from dataclasses import dataclass

from spec_pipeline.models import SpecificationRow

try:
    from spec_pipeline.ai.vision_repair import semantic_ocr_garbage_message
except ModuleNotFoundError:  # Stage-1 packages do not include the optional vision layer.

    def semantic_ocr_garbage_message(value: str) -> str | None:
        return None


CONTACT_TOKENS = ("тел", "email", "@", "ТКП", "№")
HEADER_TOKENS = ("Наименование", "характеристика", "Позиция", "Количество")
CURRENCY_CONTEXT_RE = re.compile(r"(?:цена|стоимост[ьи]|руб\.?|ткп|₽)\s*$", re.IGNORECASE)
TEXT_CONTINUATION_PREFIX_RE = re.compile(
    r"^(?:с|со|для|и|или|на|в комплекте|комплектуется|подключается)\b",
    re.IGNORECASE,
)
OCR_DAMAGED_UNITS = {
    "wm",
    "w/m",
    "um",
    "шм",
    r"$\omega$m",
    r"$\omega$м",
    r"\omegam",
    r"\omegaм",
    "ωm",
    "ωм",
    "n.m",
    "n.m.",
}
RUBLE_DIMENSION_WARNING_RE = re.compile(r"₽\s*\d{2,4}")
SUSPICIOUS_DIMENSION_RE = re.compile(
    r"(?:\\(?:phi|varnothing|emptyset|oslash)|(?<![\w.-])0\d{2,4}\s*[xх×]|^0\d{2,4}$)",
    re.IGNORECASE,
)
SUSPICIOUS_TECHNICAL_CODE_RE = re.compile(
    r"(?:\bSUP[UV]\b|\bКП[ОO][ЛП]-2(?=\()|\[SPH-W2-\[SPC-F3\]|\[(?:VI|VІ)\.K[L1IІ]35C)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class QualityIssue:
    severity: str
    row_index: int
    page_ref: str | None
    field: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "row_index": self.row_index,
            "page_ref": self.page_ref,
            "field": self.field,
            "message": self.message,
        }


@dataclass(slots=True)
class QualityReport:
    row_count: int
    issue_count: int
    error_count: int
    warning_count: int
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def assess_extracted_rows(rows: list[SpecificationRow]) -> QualityReport:
    issues: list[QualityIssue] = []
    previous_data_row: SpecificationRow | None = None
    for index, row in enumerate(rows, start=1):
        if row.page_ref and row.page_ref.upper().startswith("PAGE "):
            issues.append(
                _issue(index, row, "warning", "page_ref", "page_ref includes the PAGE label")
            )
        if (
            row.name is None
            and any([row.pos, row.unit, row.quantity, row.type_mark, row.manufacturer])
            and not is_probable_continuation_row(row, previous_data_row)
        ):
            issues.append(
                _issue(
                    index,
                    row,
                    "error" if row.quantity is not None else "warning",
                    "name",
                    "data-like row has no item name",
                )
            )
        if row.pos and looks_like_textual_position(row.pos):
            issues.append(
                _issue(
                    index,
                    row,
                    "error",
                    "pos",
                    "position contains item-description text",
                )
            )
        if (
            row.name
            and row.name.strip().startswith("=")
            and any([row.unit, row.quantity, row.quantity_raw, row.manufacturer])
        ):
            issues.append(
                _issue(
                    index,
                    row,
                    "error",
                    "name",
                    "item name looks like a continuation fragment",
                )
            )
        if is_unmerged_name_only_continuation(row, previous_data_row):
            issues.append(
                _issue(
                    index,
                    row,
                    "warning",
                    "name",
                    "name-only continuation row should be merged into previous item",
                )
            )
        if row.name and all(token in row.name for token in HEADER_TOKENS[:2]):
            issues.append(
                _issue(index, row, "warning", "name", "row may be a repeated table header")
            )
        if row.manufacturer and any(token in row.manufacturer for token in CONTACT_TOKENS):
            issues.append(
                _issue(
                    index,
                    row,
                    "warning",
                    "manufacturer",
                    "manufacturer contains contact/commercial text that should likely be note",
                )
            )
        if row.unit and row.quantity is None and row.quantity_raw is None:
            issues.append(_issue(index, row, "warning", "quantity", "row has unit but no quantity"))
        if row.unit and normalize_ocr_unit_key(row.unit) in OCR_DAMAGED_UNITS:
            issues.append(_issue(index, row, "warning", "unit", "unit looks OCR-damaged"))
        for field, value in (("name", row.name), ("type_mark", row.type_mark)):
            if value and looks_ocr_damaged_dimension_text(value):
                issues.append(
                    _issue(index, row, "warning", field, "dimension text looks OCR-damaged")
                )
            if value and looks_ocr_damaged_technical_code(value):
                issues.append(
                    _issue(index, row, "warning", field, "technical code looks OCR-damaged")
                )
        for field, value in (
            ("section", row.section),
            ("subsection", row.subsection),
            ("name", row.name),
            ("type_mark", row.type_mark),
            ("manufacturer", row.manufacturer),
            ("note", row.note),
        ):
            if (
                value
                and semantic_ocr_garbage_message(value) == "text looks semantically OCR-damaged"
            ):
                issues.append(
                    _issue(index, row, "warning", field, "text looks semantically OCR-damaged")
                )
        if not row.is_grouping_row() and row.name is not None:
            previous_data_row = row

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return QualityReport(
        row_count=len(rows),
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
    )


def is_probable_continuation_row(
    row: SpecificationRow, previous_data_row: SpecificationRow | None
) -> bool:
    return (
        previous_data_row is not None
        and row.pos is None
        and row.name is None
        and previous_data_row.name is not None
        and any([row.type_mark, row.code, row.manufacturer, row.unit, row.quantity])
    )


def is_unmerged_name_only_continuation(
    row: SpecificationRow, previous_data_row: SpecificationRow | None
) -> bool:
    if previous_data_row is None or previous_data_row.name is None:
        return False
    if row.name is None or row.pos is not None:
        return False
    if any(
        [
            row.type_mark,
            row.code,
            row.manufacturer,
            row.unit,
            row.quantity,
            row.quantity_raw,
            row.mass_kg,
            row.mass_kg_raw,
            row.note,
        ]
    ):
        return False
    stripped = row.name.strip()
    first = stripped[:1]
    return bool(
        (first and first.islower())
        or TEXT_CONTINUATION_PREFIX_RE.match(stripped)
        or stripped.startswith(("=", "-", "/", "+"))
    )


def normalize_ocr_unit_key(unit: str) -> str:
    return re.sub(r"\s+", "", unit).strip().lower()


def looks_like_textual_position(value: str) -> bool:
    text = value.strip()
    return bool(len(text) > 25 and re.search(r"\s", text) and re.search(r"[А-Яа-яЁё]{3,}", text))


def looks_ocr_damaged_dimension_text(value: str) -> bool:
    if SUSPICIOUS_DIMENSION_RE.search(value):
        return True
    for match in RUBLE_DIMENSION_WARNING_RE.finditer(value):
        if not has_currency_context(value, match.start()):
            return True
    return False


def looks_ocr_damaged_technical_code(value: str) -> bool:
    return bool(SUSPICIOUS_TECHNICAL_CODE_RE.search(value))


def has_currency_context(source: str, match_start: int) -> bool:
    return bool(CURRENCY_CONTEXT_RE.search(source[:match_start][-24:]))


def _issue(
    index: int, row: SpecificationRow, severity: str, field: str, message: str
) -> QualityIssue:
    return QualityIssue(
        severity=severity,
        row_index=index,
        page_ref=row.page_ref,
        field=field,
        message=message,
    )
