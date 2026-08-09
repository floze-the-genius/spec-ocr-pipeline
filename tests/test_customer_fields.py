from __future__ import annotations

from decimal import Decimal

import pytest

from spec_pipeline.exporters.customer_fields import (
    customer_description,
    customer_note,
    normalize_customer_text,
    normalize_customer_unit,
)
from spec_pipeline.models import SpecificationRow


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("₽50₽25", "Ø50x25"),
        ("₽ 50 ₽ 25", "Ø50x25"),
        ("₽50", "Ø50"),
        ("057x3.5", "Ø57x3.5"),
        ("057х035", "Ø57x35"),
        ("057×035", "Ø57x35"),
        ("040x3.5", "Ø40x3.5"),
        ("0108x4.0", "Ø108x4.0"),
        ("025", "Ø25"),
        ("ф100", "Ø100"),
        ("Φ100", "Ø100"),
        ("Ø100", "Ø100"),
        ("$ \\Phi $15", "Ø15"),
        (r"\varnothing 100", "Ø100"),
        (r"\emptyset 80", "Ø80"),
        (r"\oslash 63", "Ø63"),
        ("Кран $ \\Phi $15 BVR-R- $ \\Phi $15", "Кран Ø15 BVR-R-Ø15"),
        (
            "1.14 Òåïëîèçîëÿöèÿ âîçäóõîâîäîâ δ=25 ìì",
            "1.14 Теплоизоляция воздуховодов δ=25 мм",
        ),
        ("ГOCT 14918-2020 RL/P1-「K1", "ГОСТ 14918-2020 RL/P1-[K1"),
    ],
)
def test_normalize_customer_text_repairs_common_ocr_dimension_artifacts(
    raw: str, expected: str
) -> None:
    assert normalize_customer_text(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Цена ₽1000",
        "стоимость 1000 ₽",
        "ТКП ₽500",
        "BVR-R-015",
        "A-025",
        "010-ОВ",
        "Airvent 015",
    ],
)
def test_normalize_customer_text_does_not_rewrite_currency_or_identifiers(raw: str) -> None:
    assert normalize_customer_text(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("wm", "шт"),
        ("w/m", "шт"),
        ("um", "шт"),
        ("шм", "шт"),
        ("$ \\omega $m", "шт"),
        ("$ \\omega $м", "шт"),
        (r"\omegam", "шт"),
        (r"\omegaм", "шт"),
        ("ωm", "шт"),
        ("ωм", "шт"),
        ("n.m.", "п.м."),
        ("n.m", "п.м."),
        ("шт", "шт"),
        ("п.м.", "п.м."),
    ],
)
def test_normalize_customer_unit_repairs_common_ocr_unit_artifacts(raw: str, expected: str) -> None:
    assert normalize_customer_unit(raw) == expected


def test_customer_description_uses_shared_ocr_normalization_and_filters_contact_note() -> None:
    row = SpecificationRow(
        sample_id="sample",
        page_ref="1",
        section=None,
        subsection=None,
        pos="4",
        name="Кран $ \\Phi $15",
        type_mark="BVR-R- $ \\Phi $15",
        code=None,
        manufacturer="Danfoss",
        unit="$ \\omega $m",
        quantity=Decimal("4"),
        mass_kg=None,
        note="ТКП № 123, тел. +7",
    )

    assert customer_description(row) == "Кран Ø15 BVR-R-Ø15 Danfoss ТКП № 123, тел. +7"
    assert customer_note(row) is None
