from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spec_pipeline.ai.schema import validate_ai_rows
from spec_pipeline.ocr.base import OcrPage

TEXT_FIELDS = ("section", "subsection", "name", "type_mark", "manufacturer", "note")
PATCHABLE_FIELDS = {
    "section",
    "subsection",
    "pos",
    "name",
    "type_mark",
    "code",
    "manufacturer",
    "unit",
    "quantity",
    "mass_kg",
    "note",
    "status",
}
HIGH_CONFIDENCE_VALUES = {"high", "высокая", "высокий"}
STRUCTURAL_ARTIFACT_RE = re.compile(
    r"(?:�|\\(?:phi|Phi|omega|Omega|varnothing|emptyset|oslash)|[$][^\n]{0,24}[$])",
    re.IGNORECASE,
)
CP1251_MOJIBAKE_RE = re.compile(r"[À-ÿ]{2,}")
CURRENCY_CONTEXT_RE = re.compile(r"(?:цена|стоимост[ьи]|руб\.?|ткп|₽)\s*$", re.IGNORECASE)
RUBLE_ARTIFACT_RE = re.compile(r"₽\s*\d{2,4}")
SEMANTIC_GARBAGE_RE = re.compile(
    r"(?:воздухом[а-я]*|воздухомоб|воздухомор|мемалл|латянн|pugah|объезж|объясн)",
    re.IGNORECASE,
)
UNIT_ARTIFACT_RE = re.compile(
    r"^(?:wm|w/m|um|шм|n\.m\.?|\$?\\omega\$?[mм]|ω[mм])$",
    re.IGNORECASE,
)
MIN_PAGE_ISSUES_FOR_REPAIR = 1


@dataclass(slots=True, frozen=True)
class VisionRepairIssue:
    row_index: int
    page_ref: str | None
    field: str
    value: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "page_ref": self.page_ref,
            "field": self.field,
            "value": self.value,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class VisionRepairPlan:
    page_refs: tuple[str, ...]
    issues: tuple[VisionRepairIssue, ...]

    @property
    def needed(self) -> bool:
        return bool(self.page_refs)


def detect_vision_repair_plan(rows: list[dict[str, object]]) -> VisionRepairPlan:
    issues: list[VisionRepairIssue] = []
    page_issue_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        page_ref = _string_or_none(row.get("page_ref"))
        for field in TEXT_FIELDS:
            value = _string_or_none(row.get(field))
            if value is None:
                continue
            message = semantic_ocr_garbage_message(value)
            if message is None:
                continue
            issues.append(
                VisionRepairIssue(
                    row_index=index,
                    page_ref=page_ref,
                    field=field,
                    value=value,
                    message=message,
                )
            )
            if page_ref:
                page_issue_counts[page_ref] = page_issue_counts.get(page_ref, 0) + 1
        unit = _string_or_none(row.get("unit"))
        if unit is not None and UNIT_ARTIFACT_RE.fullmatch(_compact(unit)):
            issues.append(
                VisionRepairIssue(
                    row_index=index,
                    page_ref=page_ref,
                    field="unit",
                    value=unit,
                    message="unit has OCR symbol garbage",
                )
            )
            if page_ref:
                page_issue_counts[page_ref] = page_issue_counts.get(page_ref, 0) + 1
        if unit is not None and _string_or_none(row.get("quantity")) is None:
            issues.append(
                VisionRepairIssue(
                    row_index=index,
                    page_ref=page_ref,
                    field="quantity",
                    value="",
                    message="row has unit but no quantity",
                )
            )
            if page_ref:
                page_issue_counts[page_ref] = page_issue_counts.get(page_ref, 0) + 1
        name = _string_or_none(row.get("name"))
        if name is None and _row_has_data_like_values(row):
            issues.append(
                VisionRepairIssue(
                    row_index=index,
                    page_ref=page_ref,
                    field="name",
                    value="",
                    message="data-like row has no item name",
                )
            )
            if page_ref:
                page_issue_counts[page_ref] = page_issue_counts.get(page_ref, 0) + 1

    page_refs = tuple(
        page_ref
        for page_ref, count in page_issue_counts.items()
        if count >= MIN_PAGE_ISSUES_FOR_REPAIR
    )
    return VisionRepairPlan(page_refs=page_refs, issues=tuple(issues))


def semantic_ocr_garbage_message(value: str) -> str | None:
    if CP1251_MOJIBAKE_RE.search(value):
        return "text contains CP1251 mojibake"
    if SEMANTIC_GARBAGE_RE.search(value):
        return "text looks semantically OCR-damaged"
    if STRUCTURAL_ARTIFACT_RE.search(value):
        return "text contains OCR symbol garbage"
    for match in RUBLE_ARTIFACT_RE.finditer(value):
        if not CURRENCY_CONTEXT_RE.search(value[: match.start()][-24:]):
            return "text contains OCR symbol garbage"
    return None


def existing_image_pages(pages: Iterable[OcrPage], page_refs: set[str]) -> list[OcrPage]:
    return [
        page
        for page in pages
        if page.page_ref in page_refs
        and page.image_path is not None
        and Path(page.image_path).exists()
    ]


def build_vision_repair_prompt(
    *,
    sample_id: str,
    pages: list[OcrPage],
    rows: list[dict[str, object]],
    issues: list[VisionRepairIssue],
) -> str:
    page_refs = {page.page_ref for page in pages}
    page_rows = [row for row in rows if _string_or_none(row.get("page_ref")) in page_refs]
    page_texts = [
        {
            "page_ref": page.page_ref,
            "ocr_text": page.text[:12000],
        }
        for page in pages
    ]
    payload = {
        "sample_id": sample_id,
        "affected_pages": sorted(page_refs),
        "issues": [issue.to_dict() for issue in issues if issue.page_ref in page_refs],
        "current_rows": page_rows,
        "ocr_pages": page_texts,
    }
    return (
        "Ты исправляешь OCR-ошибки в спецификации по приложенным изображениям страниц.\n"
        "Тебе даны текущие строки, OCR-текст и изображения этих же страниц.\n"
        "Исправляй только то, что явно видно на изображении: названия, марки, производителя, "
        "единицы и размеры. Не придумывай новых позиций. Сохраняй порядок строк, page_ref, "
        "section/subsection и status. Верни все строки для affected_pages, а не только измененные.\n"
        'Ответ должен быть строго JSON без Markdown: {"rows": [...], "warnings": [...]} .\n'
        "Каждая строка обязана иметь поля: page_ref, section, subsection, pos, name, type_mark, "
        "code, manufacturer, unit, quantity, mass_kg, note, status. Значения только string или null.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_vision_patch_prompt(
    *,
    sample_id: str,
    pages: list[OcrPage],
    rows: list[dict[str, object]],
    issues: list[VisionRepairIssue],
) -> str:
    page_refs = {page.page_ref for page in pages}
    page_rows = [
        {"row_index": index, **row}
        for index, row in enumerate(rows, start=1)
        if _string_or_none(row.get("page_ref")) in page_refs
    ]
    payload = {
        "sample_id": sample_id,
        "affected_pages": sorted(page_refs),
        "issues": [issue.to_dict() for issue in issues if issue.page_ref in page_refs],
        "current_rows": page_rows,
        "ocr_pages": [{"page_ref": page.page_ref, "ocr_text": page.text[:12000]} for page in pages],
    }
    return (
        "Ты третий слой контроля качества спецификации. Основной пайплайн уже извлек строки; "
        "твоя задача - проверить только подозрительные места по OCR-тексту и приложенным "
        "изображениям страниц.\n"
        "Не извлекай таблицу заново. Не добавляй строки. Не удаляй строки. Не меняй порядок строк. "
        "Верни только точечные исправления полей, которые явно подтверждаются изображением или OCR. "
        "Если не уверен - не исправляй.\n"
        'Ответ строго JSON без Markdown: {"patches": [...], "warnings": [...]} .\n'
        "Каждый patch: row_index (1-based из current_rows), page_ref, field, old_value, "
        "new_value, confidence, reason. old_value должен в точности совпадать с текущим значением. "
        "confidence ставь high только когда исправление явно видно; иначе не возвращай patch. "
        "Разрешенные field: section, subsection, pos, name, type_mark, code, manufacturer, "
        "unit, quantity, mass_kg, note, status.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def apply_row_patches(
    original_rows: list[dict[str, object]],
    patches: list[dict[str, object]],
    affected_page_refs: set[str],
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]], list[dict[str, object]]]:
    patched_rows = [dict(row) for row in original_rows]
    warnings: list[str] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for patch in patches:
        reason = _reject_patch_reason(patched_rows, patch, affected_page_refs)
        if reason is not None:
            rejected.append({**patch, "rejected_reason": reason})
            continue

        row_index_value = patch["row_index"]
        if not isinstance(row_index_value, int):
            rejected.append({**patch, "rejected_reason": "row_index must be an integer"})
            continue
        row_index = row_index_value
        field = str(patch["field"])
        patched_rows[row_index - 1][field] = patch.get("new_value")
        accepted.append(dict(patch))

    try:
        validate_ai_rows(patched_rows)
    except ValueError as exc:
        warnings.append(f"third-layer patches rejected because patched rows are invalid: {exc}")
        return original_rows, warnings, [], [*rejected, *accepted]

    if len(patched_rows) != len(original_rows):
        warnings.append("third-layer patches rejected because row count changed")
        return original_rows, warnings, [], [*rejected, *accepted]

    if rejected:
        warnings.append(f"third-layer rejected {len(rejected)} unsafe patch(es)")
    if accepted:
        warnings.append(f"third-layer accepted {len(accepted)} patch(es)")
    return patched_rows, warnings, accepted, rejected


def _reject_patch_reason(
    rows: list[dict[str, object]], patch: dict[str, object], affected_page_refs: set[str]
) -> str | None:
    row_index_value = patch.get("row_index")
    if isinstance(row_index_value, bool) or not isinstance(row_index_value, int):
        return "row_index must be an integer"
    if row_index_value < 1 or row_index_value > len(rows):
        return "row_index out of range"

    field = patch.get("field")
    if not isinstance(field, str) or field not in PATCHABLE_FIELDS:
        return "field is not patchable"

    row = rows[row_index_value - 1]
    row_page_ref = _string_or_none(row.get("page_ref"))
    patch_page_ref = _string_or_none(patch.get("page_ref"))
    if affected_page_refs and row_page_ref not in affected_page_refs:
        return "row page is not affected"
    if patch_page_ref is not None and row_page_ref != patch_page_ref:
        return "patch page_ref does not match row page_ref"

    old_value = patch.get("old_value")
    new_value = patch.get("new_value")
    if old_value is not None and not isinstance(old_value, str):
        return "old_value must be string or null"
    if new_value is not None and not isinstance(new_value, str):
        return "new_value must be string or null"
    if row.get(field) != old_value:
        return "old_value does not match current row value"
    if field == "status" and new_value not in {None, "crossed_out"}:
        return "invalid status value"
    if isinstance(new_value, str) and not new_value.strip():
        return "new_value is blank"

    confidence = _string_or_none(patch.get("confidence"))
    if confidence is None or confidence.strip().lower() not in HIGH_CONFIDENCE_VALUES:
        return "confidence is not high"
    return None


def merge_repaired_page_rows(
    original_rows: list[dict[str, object]],
    repaired_rows: list[dict[str, object]],
    repaired_page_refs: set[str],
) -> list[dict[str, object]]:
    validate_ai_rows(repaired_rows)
    repaired_by_page: dict[str, list[dict[str, object]]] = {}
    for row in repaired_rows:
        page_ref = _string_or_none(row.get("page_ref"))
        if page_ref in repaired_page_refs:
            repaired_by_page.setdefault(page_ref, []).append(row)

    merged: list[dict[str, object]] = []
    inserted: set[str] = set()
    for row in original_rows:
        page_ref = _string_or_none(row.get("page_ref"))
        if page_ref in repaired_page_refs:
            if page_ref not in inserted:
                merged.extend(repaired_by_page.get(page_ref, []))
                inserted.add(page_ref)
            continue
        merged.append(row)
    for page_ref in repaired_page_refs - inserted:
        merged.extend(repaired_by_page.get(page_ref, []))
    return merged


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def _row_has_data_like_values(row: dict[str, object]) -> bool:
    return any(
        _string_or_none(row.get(field)) is not None
        for field in ("pos", "type_mark", "code", "manufacturer", "unit", "quantity", "mass_kg")
    )
