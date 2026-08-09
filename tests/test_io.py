from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from spec_pipeline.io import load_rows_from_jsonl


def test_load_rows_from_jsonl_reads_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "spec_1",
                        "page_ref": "001",
                        "section": "ПВ7",
                        "subsection": None,
                        "pos": "1",
                        "name": "Установка",
                        "type_mark": None,
                        "code": None,
                        "manufacturer": "СВОК",
                        "unit": "шт",
                        "quantity": "12.5",
                        "mass_kg": None,
                        "note": None,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "sample_id": "spec_1",
                        "page_ref": "001",
                        "section": "ПВ7",
                        "subsection": "КИПиА",
                        "pos": None,
                        "name": None,
                        "type_mark": None,
                        "code": None,
                        "manufacturer": None,
                        "unit": None,
                        "quantity": None,
                        "mass_kg": None,
                        "note": None,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_rows_from_jsonl(path)

    assert len(rows) == 2
    assert rows[0].quantity == Decimal("12.5")
    assert rows[1].is_grouping_row() is True


def test_load_rows_from_jsonl_preserves_compound_quantity_text(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    payload = {
        "sample_id": "spec_1",
        "page_ref": "001",
        "section": None,
        "subsection": None,
        "pos": "1.1",
        "name": "Воздуховод",
        "type_mark": None,
        "code": None,
        "manufacturer": None,
        "unit": "м/м²",
        "quantity": "55/17,27",
        "mass_kg": None,
        "note": None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = load_rows_from_jsonl(path)

    assert rows[0].quantity is None
    assert rows[0].quantity_raw == "55/17,27"
    assert rows[0].to_dict()["quantity"] == "55/17,27"


def test_load_rows_from_jsonl_repairs_text_encoding_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    payload = {
        "sample_id": "spec_1",
        "page_ref": "001",
        "section": None,
        "subsection": None,
        "pos": "1",
        "name": "1.14 Òåïëîèçîëÿöèÿ âîçäóõîâîäîâ δ=25 ìì",
        "type_mark": "ГOCT 14918-2020",
        "code": None,
        "manufacturer": None,
        "unit": "шт",
        "quantity": "1",
        "mass_kg": None,
        "note": "RL/P1-「K1",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = load_rows_from_jsonl(path)

    assert rows[0].name == "1.14 Теплоизоляция воздуховодов δ=25 мм"
    assert rows[0].type_mark == "ГОСТ 14918-2020"
    assert rows[0].note == "RL/P1-[K1"
