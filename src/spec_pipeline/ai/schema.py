from __future__ import annotations

import re
from collections.abc import Mapping

from spec_pipeline.models import (
    SpecificationRow,
    clean_text,
    parse_decimal,
    raw_decimal_text,
    repair_shifted_pos_name,
)

AI_ROW_FIELDS = [
    "page_ref",
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
]

AI_SCHEMA_VERSION = "specification-row-v1"
AI_OPTIONAL_ROW_FIELDS = {"status"}
AI_DATA_FIELDS = (
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
AI_CONTINUATION_PREFIX_RE = re.compile(
    r"^(?:с|со|для|и|или|на|в комплекте|комплектуется|подключается)\b",
    re.IGNORECASE,
)


def normalize_ai_row_values(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized.setdefault("status", None)
    for field in AI_ROW_FIELDS:
        value = normalized.get(field)
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            normalized[field] = str(value)
    return normalized


def normalize_ai_rows_values(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return _merge_ai_name_only_continuation_rows([normalize_ai_row_values(row) for row in rows])


def _merge_ai_name_only_continuation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for row in rows:
        if (
            merged
            and _is_ai_name_only_continuation(row)
            and _can_receive_ai_name_continuation(merged[-1])
        ):
            target = merged[-1]
            target["name"] = _join_ai_text(_string_or_none(target.get("name")), row["name"])
            if target.get("status") is None and row.get("status") is not None:
                target["status"] = row.get("status")
            continue
        merged.append(row)
    return merged


def _is_ai_name_only_continuation(row: Mapping[str, object]) -> bool:
    name = _string_or_none(row.get("name"))
    if name is None:
        return False
    if any(_string_or_none(row.get(field)) for field in AI_DATA_FIELDS if field != "name"):
        return False
    stripped = name.strip()
    first = stripped[:1]
    return bool(
        (first and first.islower())
        or AI_CONTINUATION_PREFIX_RE.match(stripped)
        or stripped.startswith(("=", "-", "/", "+"))
    )


def _can_receive_ai_name_continuation(row: Mapping[str, object]) -> bool:
    if _string_or_none(row.get("name")) is None:
        return False
    return any(
        _string_or_none(row.get(field))
        for field in ("pos", "type_mark", "code", "manufacturer", "unit", "quantity", "mass_kg")
    )


def _join_ai_text(left: str | None, right: object) -> str | None:
    right_text = _string_or_none(right)
    values = [part.strip() for part in (left, right_text) if part and part.strip()]
    return " ".join(values) if values else None


def validate_ai_row(row: Mapping[str, object]) -> None:
    keys = set(row)
    required = set(AI_ROW_FIELDS) - AI_OPTIONAL_ROW_FIELDS
    missing = sorted(required - keys)
    extra = sorted(keys - set(AI_ROW_FIELDS))
    if missing:
        raise ValueError(f"AI row is missing required fields: {missing}")
    if extra:
        raise ValueError(f"AI row contains unknown fields: {extra}")
    for field in AI_ROW_FIELDS:
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"AI row field must be string or null: {field}")
    status = row.get("status")
    if status not in (None, "crossed_out"):
        raise ValueError("AI row status must be null or 'crossed_out'")


def validate_ai_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        validate_ai_row(row)


def ai_row_to_specification_row(sample_id: str, row: Mapping[str, object]) -> SpecificationRow:
    validate_ai_row(row)
    quantity_text = _string_or_none(row["quantity"])
    mass_kg_text = _string_or_none(row["mass_kg"])
    quantity = parse_decimal(quantity_text)
    mass_kg = parse_decimal(mass_kg_text)
    pos, name = repair_shifted_pos_name(
        pos=clean_text(_string_or_none(row["pos"])),
        name=clean_text(_string_or_none(row["name"])),
        unit=clean_text(_string_or_none(row["unit"])),
        quantity=quantity,
        type_mark=clean_text(_string_or_none(row["type_mark"])),
        code=clean_text(_string_or_none(row["code"])),
        manufacturer=clean_text(_string_or_none(row["manufacturer"])),
        mass_kg=mass_kg,
    )
    return SpecificationRow(
        sample_id=sample_id,
        page_ref=clean_text(_string_or_none(row["page_ref"])),
        section=clean_text(_string_or_none(row["section"])),
        subsection=clean_text(_string_or_none(row["subsection"])),
        pos=pos,
        name=name,
        type_mark=clean_text(_string_or_none(row["type_mark"])),
        code=clean_text(_string_or_none(row["code"])),
        manufacturer=clean_text(_string_or_none(row["manufacturer"])),
        unit=clean_text(_string_or_none(row["unit"])),
        quantity=quantity,
        quantity_raw=raw_decimal_text(quantity_text, quantity),
        mass_kg=mass_kg,
        mass_kg_raw=raw_decimal_text(mass_kg_text, mass_kg),
        note=clean_text(_string_or_none(row["note"])),
        status=clean_text(_string_or_none(row.get("status"))),
    )


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
