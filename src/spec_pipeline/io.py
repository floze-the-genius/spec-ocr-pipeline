from __future__ import annotations

import json
from pathlib import Path

from spec_pipeline.models import (
    SpecificationRow,
    clean_text,
    parse_decimal,
    raw_decimal_text,
    repair_shifted_pos_name,
)


def load_rows_from_jsonl(path: str | Path) -> list[SpecificationRow]:
    resolved = Path(path).expanduser().resolve()
    rows: list[SpecificationRow] = []

    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            quantity = parse_decimal(payload.get("quantity"))
            mass_kg = parse_decimal(payload.get("mass_kg"))
            pos, name = repair_shifted_pos_name(
                pos=clean_text(payload.get("pos")),
                name=clean_text(payload.get("name")),
                unit=clean_text(payload.get("unit")),
                quantity=quantity,
                type_mark=clean_text(payload.get("type_mark")),
                code=clean_text(payload.get("code")),
                manufacturer=clean_text(payload.get("manufacturer")),
                mass_kg=mass_kg,
            )
            rows.append(
                SpecificationRow(
                    sample_id=clean_text(payload["sample_id"]) or str(payload["sample_id"]),
                    page_ref=clean_text(payload.get("page_ref")),
                    section=clean_text(payload.get("section")),
                    subsection=clean_text(payload.get("subsection")),
                    pos=pos,
                    name=name,
                    type_mark=clean_text(payload.get("type_mark")),
                    code=clean_text(payload.get("code")),
                    manufacturer=clean_text(payload.get("manufacturer")),
                    unit=clean_text(payload.get("unit")),
                    quantity=quantity,
                    quantity_raw=raw_decimal_text(payload.get("quantity"), quantity),
                    mass_kg=mass_kg,
                    mass_kg_raw=raw_decimal_text(payload.get("mass_kg"), mass_kg),
                    note=clean_text(payload.get("note")),
                    status=clean_text(payload.get("status")),
                )
            )

    return rows
