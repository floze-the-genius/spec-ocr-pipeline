from __future__ import annotations

from decimal import Decimal

import pytest

from spec_pipeline.ai.schema import (
    AI_ROW_FIELDS,
    ai_row_to_specification_row,
    normalize_ai_row_values,
    normalize_ai_rows_values,
    validate_ai_row,
)


def _valid_ai_row() -> dict[str, object]:
    row: dict[str, object] = {field: None for field in AI_ROW_FIELDS}
    row.update(
        {
            "page_ref": "1",
            "pos": "1",
            "name": "Диффузор",
            "unit": "шт",
            "quantity": "2,5",
        }
    )
    return row


def test_validate_ai_row_accepts_strict_schema() -> None:
    validate_ai_row(_valid_ai_row())


def test_validate_ai_row_rejects_unknown_fields() -> None:
    row = _valid_ai_row()
    row["confidence"] = 0.9

    with pytest.raises(ValueError, match="unknown fields"):
        validate_ai_row(row)


def test_validate_ai_row_rejects_missing_fields() -> None:
    row = _valid_ai_row()
    del row["note"]

    with pytest.raises(ValueError, match="missing required"):
        validate_ai_row(row)


def test_validate_ai_row_allows_missing_optional_status() -> None:
    row = _valid_ai_row()
    del row["status"]

    validate_ai_row(row)


def test_validate_ai_row_rejects_unknown_status() -> None:
    row = _valid_ai_row()
    row["status"] = "deleted"

    with pytest.raises(ValueError, match="status"):
        validate_ai_row(row)


def test_normalize_ai_row_values_coerces_numeric_scalars_to_strings() -> None:
    row = _valid_ai_row()
    row["pos"] = 1
    row["quantity"] = 2.5

    normalized = normalize_ai_row_values(row)

    assert normalized["pos"] == "1"
    assert normalized["quantity"] == "2.5"
    validate_ai_row(normalized)


def test_normalize_ai_rows_values_merges_name_only_continuation_into_previous_item() -> None:
    first = _valid_ai_row()
    first["pos"] = "21"
    first["name"] = "Клапан противопожарный нормально открытый (огнезадерживающий)"
    first["type_mark"] = "КЛОП-2(60)-НО-К-350x200-МВ(220)-Н"
    first["manufacturer"] = "ВИНГС-М"
    continuation = {field: None for field in AI_ROW_FIELDS}
    continuation.update(
        {
            "page_ref": "2",
            "name": "с электромеханическим приводом Belimo 220B",
        }
    )
    child = _valid_ai_row()
    child["pos"] = None
    child["name"] = "Ø100"
    child["unit"] = "шт"
    child["quantity"] = "5"

    normalized = normalize_ai_rows_values([first, continuation, child])

    assert len(normalized) == 2
    assert normalized[0]["name"] == (
        "Клапан противопожарный нормально открытый (огнезадерживающий) "
        "с электромеханическим приводом Belimo 220B"
    )
    assert normalized[1]["name"] == "Ø100"


def test_ai_row_to_specification_row_parses_decimal_quantity() -> None:
    row = ai_row_to_specification_row("sample", _valid_ai_row())

    assert row.sample_id == "sample"
    assert row.name == "Диффузор"
    assert row.quantity == Decimal("2.5")
    assert row.status is None


def test_ai_row_to_specification_row_repairs_mixed_cp1251_mojibake() -> None:
    payload = _valid_ai_row()
    payload["name"] = (
        "1.14 Òåïëîèçîëÿöèÿ âîçäóõîâîäîâ δ=25 ìì "
        "ПВ7 ФЭ SUPU 70-40 -[SPH-W2-[SPC-F3]-[VI.KL35C(1.1/3000)]"
    )
    payload["type_mark"] = "ГOCT 14918-2020 КПОЛ-2(60)-НО-К-500x300-МВ(220)-Н"
    payload["unit"] = "ìì"
    payload["note"] = "Belimo 220B RL/P1-「K1"

    row = ai_row_to_specification_row("sample", payload)

    assert (
        row.name == "1.14 Теплоизоляция воздуховодов δ=25 мм ПВ7 фр SUPM 70-40 "
        "-[SPH-W2]-[SPC-F3]-[V1.КЦ35C(1.1/3000)]"
    )
    assert row.type_mark == "ГОСТ 14918-2020 КЛОП-2(60)-НО-К-500x300-МВ(220)-Н"
    assert row.unit == "мм"
    assert row.note == "Belimo 220В RL/P1-[K1"


def test_ai_row_to_specification_row_repairs_customer_spec1_gpt_artifacts() -> None:
    payload = _valid_ai_row()
    payload["name"] = (
        "Установка (ID 4204937) - ПВ7 фр SUPV 70-40 RL/[P1]-[K1]-[F1]-[R3]-"
        "[SPH-W2]-[SPC-F3]-[V1.КЦ35C(1.1/3000)]-[K1]-[P1]"
    )
    payload["note"] = "Осушествляется подмес воздуха в вытяжном канале перед рекупе?раторе"

    row = ai_row_to_specification_row("sample", payload)

    assert row.name == (
        "Установка (ID 4204937) - ПВ7 фр SUPM 70-40 RL/[P1]-[K1]-[F1]-[R3]-"
        "[SPH-W2]-[SPC-F3]-[V1.КЦ35C(1.1/3000)]-[K1]-[P1]"
    )
    assert row.note == "Осушествляется подмес воздуха в вытяжном канале перед рекуператоре"


def test_ai_row_to_specification_row_preserves_compound_quantity_text() -> None:
    payload = _valid_ai_row()
    payload["quantity"] = "55/17,27"

    row = ai_row_to_specification_row("sample", payload)

    assert row.quantity is None
    assert row.quantity_raw == "55/17,27"
    assert row.to_dict()["quantity"] == "55/17,27"


def test_ai_row_to_specification_row_repairs_name_shifted_into_pos() -> None:
    payload = _valid_ai_row()
    payload["pos"] = "Внутренний блок кондиционера настенный"
    payload["name"] = None
    payload["type_mark"] = "DA70EVQ1-1"
    payload["manufacturer"] = "Daichi"

    row = ai_row_to_specification_row("sample", payload)

    assert row.pos is None
    assert row.name == "Внутренний блок кондиционера настенный"
    assert row.type_mark == "DA70EVQ1-1"


def test_ai_row_to_specification_row_keeps_position_code_without_name() -> None:
    payload = _valid_ai_row()
    payload["pos"] = "В1.1"
    payload["name"] = None

    row = ai_row_to_specification_row("sample", payload)

    assert row.pos == "В1.1"
    assert row.name is None


def test_ai_row_to_specification_row_repairs_dimension_and_same_shifted_into_pos() -> None:
    payload = _valid_ai_row()
    payload["pos"] = "То же, ∅160"
    payload["name"] = None
    payload["type_mark"] = "ДПУ-К-160"

    row = ai_row_to_specification_row("sample", payload)

    assert row.pos is None
    assert row.name == "То же, ∅160"

    payload["pos"] = "200x200 (δ=0,5 мм)"
    payload["name"] = None
    payload["type_mark"] = None

    row = ai_row_to_specification_row("sample", payload)

    assert row.pos is None
    assert row.name == "200x200 (δ=0,5 мм)"

    payload["pos"] = "315"
    payload["name"] = None

    row = ai_row_to_specification_row("sample", payload)

    assert row.pos is None
    assert row.name == "315"
