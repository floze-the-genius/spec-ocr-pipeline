from __future__ import annotations

from decimal import Decimal

from spec_pipeline.ai.quality import assess_extracted_rows
from spec_pipeline.models import SpecificationRow


def test_assess_extracted_rows_flags_common_ai_mapping_issues() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="PAGE 1",
            section=None,
            subsection=None,
            pos="1",
            name="Котел",
            type_mark=None,
            code=None,
            manufacturer='"DemoHeat" тел.: DEMO-PHONE-0000',
            unit="шт",
            quantity=Decimal("1"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="2",
            name=None,
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="шт",
            quantity=None,
            mass_kg=None,
            note=None,
        ),
    ]

    report = assess_extracted_rows(rows)

    assert report.row_count == 2
    assert report.issue_count == 4
    assert report.error_count == 0
    assert report.warning_count == 4
    assert {issue.field for issue in report.issues} == {
        "page_ref",
        "manufacturer",
        "name",
        "quantity",
    }


def test_assess_extracted_rows_accepts_variant_continuation_without_name() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="4",
            name="Кран шаровой латунный",
            type_mark="BVR-R-015",
            code=None,
            manufacturer="Giacomini",
            unit="шт",
            quantity=Decimal("2"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=None,
            name=None,
            type_mark="BVR-R-025",
            code=None,
            manufacturer="Giacomini",
            unit="шт",
            quantity=Decimal("4"),
            mass_kg=None,
            note=None,
        ),
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0


def test_assess_extracted_rows_accepts_raw_compound_quantity() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=None,
            name="100 (δ=0,5 мм)",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="м/м²",
            quantity=None,
            quantity_raw="55/17,27",
            mass_kg=None,
            note=None,
        )
    ]

    report = assess_extracted_rows(rows)

    assert report.issue_count == 0


def test_assess_extracted_rows_warns_on_unmerged_name_only_continuation() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="21",
            name="Клапан противопожарный нормально открытый (огнезадерживающий)",
            type_mark="КЛОП-2(60)-НО-К-350x200-МВ(220)-Н",
            code=None,
            manufacturer="ВИНГС-М",
            unit="шт",
            quantity=Decimal("1"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="2",
            section=None,
            subsection=None,
            pos=None,
            name="с электромеханическим приводом Belimo 220B",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit=None,
            quantity=None,
            mass_kg=None,
            note=None,
        ),
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0
    assert report.warning_count == 1
    assert report.issues[0].field == "name"
    assert (
        report.issues[0].message == "name-only continuation row should be merged into previous item"
    )


def test_assess_extracted_rows_flags_textual_pos_and_continuation_name() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="6",
            section=None,
            subsection=None,
            pos="покровным слоем из тонколистовой стали б",
            name="=1.0 мм Н =82.60 м",
            type_mark=None,
            code=None,
            manufacturer='ООО"Демо Канал"',
            unit="шт",
            quantity=Decimal("18"),
            mass_kg=None,
            note=None,
        )
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 2
    assert {(issue.field, issue.message) for issue in report.issues} == {
        ("pos", "position contains item-description text"),
        ("name", "item name looks like a continuation fragment"),
    }


def test_assess_extracted_rows_warns_on_ocr_damaged_scan_text() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=None,
            name="₽50₽25",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="w/m",
            quantity=Decimal("168"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=None,
            name="057x3.5",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="n.m.",
            quantity=Decimal("96"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="4",
            name="Кран $ \\Phi $15",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="$ \\omega $m",
            quantity=Decimal("4"),
            mass_kg=None,
            note=None,
        ),
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0
    assert report.warning_count == 6
    assert {(issue.row_index, issue.field) for issue in report.issues} == {
        (1, "name"),
        (1, "unit"),
        (2, "name"),
        (2, "unit"),
        (3, "name"),
        (3, "unit"),
    }


def test_assess_extracted_rows_warns_on_dimension_only_leading_zero() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=None,
            name="025",
            type_mark=None,
            code=None,
            manufacturer=None,
            unit="шт",
            quantity=Decimal("76"),
            mass_kg=None,
            note=None,
        )
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0
    assert report.warning_count == 1
    assert report.issues[0].field == "name"
    assert report.issues[0].message == "dimension text looks OCR-damaged"


def test_assess_extracted_rows_warns_on_suspicious_technical_code_ocr() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="1",
            name="Установка ПВ7 ФЭ SUPU 70-40 -[SPH-W2-[SPC-F3]-[VI.KL35C(1.1/3000)]",
            type_mark=None,
            code=None,
            manufacturer="СВОК",
            unit="шт",
            quantity=Decimal("1"),
            mass_kg=None,
            note=None,
        ),
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos="18",
            name="Клапан противопожарный",
            type_mark="КПОЛ-2(60)-НО-К-ø100-МВ(220)-Н",
            code=None,
            manufacturer="ВИНГС-М",
            unit="шт",
            quantity=Decimal("1"),
            mass_kg=None,
            note=None,
        ),
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0
    assert report.warning_count == 2
    assert [(issue.row_index, issue.field, issue.message) for issue in report.issues] == [
        (1, "name", "technical code looks OCR-damaged"),
        (2, "type_mark", "technical code looks OCR-damaged"),
    ]


def test_assess_extracted_rows_does_not_warn_on_currency_or_identifiers() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection=None,
            pos=str(index),
            name=name,
            type_mark=type_mark,
            code=None,
            manufacturer=None,
            unit="шт",
            quantity=Decimal("1"),
            mass_kg=None,
            note=None,
        )
        for index, (name, type_mark) in enumerate(
            [
                ("Цена ₽1000", "BVR-R-015"),
                ("стоимость 1000 ₽", "A-025"),
                ("ТКП ₽500", "010-ОВ"),
                ("Airvent 015", None),
            ],
            start=1,
        )
    ]

    report = assess_extracted_rows(rows)

    assert report.issue_count == 0


def test_assess_extracted_rows_warns_on_semantic_scan_garbage() -> None:
    rows = [
        SpecificationRow(
            sample_id="sample",
            page_ref="1",
            section=None,
            subsection="Теплонабжение (объезжая горизонтальных завес Guard pro 150W)",
            pos="3",
            name="Автоматический воздухомобюксир 015",
            type_mark=None,
            code=None,
            manufacturer="Pugah",
            unit="шт",
            quantity=Decimal("84"),
            mass_kg=None,
            note=None,
        )
    ]

    report = assess_extracted_rows(rows)

    assert report.error_count == 0
    assert report.warning_count == 3
    assert {(issue.field, issue.message) for issue in report.issues} == {
        ("subsection", "text looks semantically OCR-damaged"),
        ("name", "text looks semantically OCR-damaged"),
        ("manufacturer", "text looks semantically OCR-damaged"),
    }
