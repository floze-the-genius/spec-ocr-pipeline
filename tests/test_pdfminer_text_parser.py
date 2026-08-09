from __future__ import annotations

from pathlib import Path

from spec_pipeline.models import SampleDefinition
from spec_pipeline.parsers.pdfminer_text import (
    TextLine,
    cluster_lines,
    detect_sheet_no,
    parse_spec8_lines_to_rows,
)


def test_detect_sheet_no_prefers_bottom_right_page_stamp() -> None:
    lines = [
        TextLine(page_index=0, page_ref="1", x0=81.0, x1=85.0, y0=650.0, text="1"),
        TextLine(page_index=0, page_ref="1", x0=1155.8, x1=1168.0, y0=21.7, text="14"),
    ]

    assert detect_sheet_no(lines, "1") == "14"


def test_cluster_lines_groups_close_y_values() -> None:
    lines = [
        TextLine(page_index=0, page_ref="14", x0=80, x1=90, y0=600.0, text="1"),
        TextLine(page_index=0, page_ref="14", x0=120, x1=200, y0=599.1, text="Диффузор"),
        TextLine(page_index=0, page_ref="14", x0=120, x1=200, y0=570.0, text="Следующая строка"),
    ]

    clusters = cluster_lines(lines, tolerance=3.0)

    assert len(clusters) == 2
    assert [line.text for line in clusters[0]] == ["1", "Диффузор"]


def test_parse_spec8_lines_to_rows_parses_basic_rows_and_note_lines() -> None:
    sample = SampleDefinition(
        sample_id="spec_8",
        title="Spec 8",
        source_pdf=Path("/tmp/spec8.pdf"),
        ground_truth_tsv=None,
        expected_xlsx=None,
        visual_xlsx=None,
        page_field="sheet_no",
    )
    lines = [
        TextLine(
            page_index=0,
            page_ref="14",
            x0=230.0,
            x1=360.0,
            y0=740.0,
            text="Общеобменная вентиляция",
        ),
        TextLine(
            page_index=0,
            page_ref="14",
            x0=240.0,
            x1=360.0,
            y0=718.0,
            text="Вентиляция квартир и МОП",
        ),
        TextLine(page_index=0, page_ref="14", x0=81.7, x1=85.0, y0=694.5, text="1"),
        TextLine(
            page_index=0,
            page_ref="14",
            x0=117.0,
            x1=281.8,
            y0=695.6,
            text="Диффузор круглый универсальный ∅125",
        ),
        TextLine(page_index=0, page_ref="14", x0=543.5, x1=588.9, y0=695.1, text="ДПУ-К-125"),
        TextLine(page_index=0, page_ref="14", x0=796.7, x1=832.3, y0=695.6, text="Арктика"),
        TextLine(page_index=0, page_ref="14", x0=899.6, x1=913.7, y0=694.5, text="шт."),
        TextLine(page_index=0, page_ref="14", x0=955.6, x1=970.1, y0=694.5, text="480"),
        TextLine(
            page_index=0, page_ref="14", x0=1085.6, x1=1151.6, y0=540.5, text='ООО "Демо Вент"'
        ),
    ]

    rows, diagnostics = parse_spec8_lines_to_rows(sample, lines)

    assert diagnostics.parsed_rows == 1
    assert rows[0].section == "Общеобменная вентиляция"
    assert rows[0].subsection == "Вентиляция квартир и МОП"
    assert rows[0].pos == "1"
    assert rows[0].type_mark == "ДПУ-К-125"
    assert rows[0].quantity is not None
    assert rows[0].note == 'ООО "Демо Вент"'


def test_parse_spec8_lines_to_rows_keeps_low_page_rows_and_splits_type_mark() -> None:
    sample = SampleDefinition(
        sample_id="spec_8",
        title="Spec 8",
        source_pdf=Path("/tmp/spec8.pdf"),
        ground_truth_tsv=None,
        expected_xlsx=None,
        visual_xlsx=None,
        page_field="sheet_no",
    )
    lines = [
        TextLine(page_index=0, page_ref="22", x0=81.8, x1=86.5, y0=127.6, text="5"),
        TextLine(
            page_index=0,
            page_ref="22",
            x0=116.8,
            x1=423.5,
            y0=127.6,
            text="Воздуховод из оцинкованной стали 600х600 с толщиной стенки δ=0,9мм",
        ),
        TextLine(page_index=0, page_ref="22", x0=534.2, x1=598.2, y0=127.6, text="ГОСТ 14918-80*"),
        TextLine(page_index=0, page_ref="22", x0=800.8, x1=827.8, y0=127.6, text="Россия"),
        TextLine(page_index=0, page_ref="22", x0=903.7, x1=909.0, y0=127.6, text="м"),
        TextLine(page_index=0, page_ref="22", x0=957.4, x1=967.6, y0=127.6, text="1,0"),
        TextLine(
            page_index=0, page_ref="22", x0=116.8, x1=217.9, y0=83.0, text="то же, 900х500 δ=0,9мм"
        ),
        TextLine(page_index=0, page_ref="22", x0=534.2, x1=598.2, y0=82.2, text="ГОСТ 14918-80*"),
        TextLine(page_index=0, page_ref="22", x0=800.8, x1=827.8, y0=82.2, text="Россия"),
        TextLine(page_index=0, page_ref="22", x0=903.7, x1=909.0, y0=82.2, text="м"),
        TextLine(page_index=0, page_ref="22", x0=954.2, x1=970.4, y0=83.0, text="75,0"),
        TextLine(
            page_index=0,
            page_ref="22",
            x0=116.8,
            x1=456.6,
            y0=179.9,
            text="Клапан противопожарный нормально закрытый 800х500 стеновой с реверсивным",
        ),
        TextLine(
            page_index=0,
            page_ref="22",
            x0=495.4,
            x1=636.5,
            y0=172.8,
            text="КПП-90-МНЕ220-НЗ(С)-800х500-К",
        ),
        TextLine(page_index=0, page_ref="22", x0=81.8, x1=86.5, y0=172.8, text="3"),
        TextLine(page_index=0, page_ref="22", x0=799.8, x1=828.8, y0=172.8, text="DemoVent"),
        TextLine(page_index=0, page_ref="22", x0=900.4, x1=912.2, y0=172.8, text="шт"),
        TextLine(page_index=0, page_ref="22", x0=958.4, x1=968.3, y0=172.8, text="24"),
        TextLine(page_index=0, page_ref="22", x0=116.8, x1=178.0, y0=168.0, text="приводом 220В"),
    ]

    rows, diagnostics = parse_spec8_lines_to_rows(sample, lines)

    assert diagnostics.parsed_rows == 3
    assert rows[0].pos == "3"
    assert rows[0].name == (
        "Клапан противопожарный нормально закрытый 800х500 стеновой с реверсивным приводом 220В"
    )
    assert rows[0].type_mark == "КПП-90-МНЕ220-НЗ(С)-800х500-К"
    assert rows[1].pos == "5"
    assert rows[2].pos is None
    assert rows[2].type_mark == "ГОСТ 14918-80*"
    assert rows[2].manufacturer == "Россия"
    assert rows[2].quantity is not None and str(rows[2].quantity) == "75.0"


def test_parse_spec8_lines_to_rows_attaches_contact_note_to_previous_row() -> None:
    sample = SampleDefinition(
        sample_id="spec_8",
        title="Spec 8",
        source_pdf=Path("/tmp/spec8.pdf"),
        ground_truth_tsv=None,
        expected_xlsx=None,
        visual_xlsx=None,
        page_field="sheet_no",
    )
    lines = [
        TextLine(
            page_index=0,
            page_ref="21",
            x0=1082.5,
            x1=1148.0,
            y0=336.7,
            text="Торговый дом DemoVent",
        ),
        TextLine(page_index=0, page_ref="21", x0=81.8, x1=86.5, y0=331.6, text="1"),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=116.8,
            x1=420.0,
            y0=331.6,
            text="Вентилятор осевой крышный в комплекте с защитным зонтом",
        ),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=500.4,
            x1=640.0,
            y0=331.6,
            text="DEMO-710-7,5/3000- 3A32-03-KR",
        ),
        TextLine(page_index=0, page_ref="21", x0=799.8, x1=828.8, y0=331.6, text="DemoVent"),
        TextLine(page_index=0, page_ref="21", x0=898.6, x1=914.2, y0=331.6, text="к-т"),
        TextLine(page_index=0, page_ref="21", x0=960.5, x1=963.8, y0=331.6, text="1"),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=1067.4,
            x1=1152.0,
            y0=327.2,
            text="ТКП №DEMO-0042 от 01.02.2030г.",
        ),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=116.4,
            x1=457.6,
            y0=315.9,
            text="Стакан монтажный со встроенным утепленным нормально-закрытым клапаном с",
        ),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=1083.6,
            x1=1154.1,
            y0=315.9,
            text="тел.:DEMO-PHONE-0000",
        ),
        TextLine(page_index=0, page_ref="21", x0=81.8, x1=86.5, y0=308.9, text="2"),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=509.0,
            x1=623.2,
            y0=308.9,
            text="DEMO-MOUNT-710-нестандарт",
        ),
        TextLine(page_index=0, page_ref="21", x0=799.8, x1=828.8, y0=308.9, text="DemoVent"),
        TextLine(page_index=0, page_ref="21", x0=898.6, x1=914.2, y0=308.9, text="к-т"),
        TextLine(page_index=0, page_ref="21", x0=960.5, x1=963.8, y0=308.9, text="1"),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=1086.7,
            x1=1151.1,
            y0=308.9,
            text="email:demo-contact@example.invalid",
        ),
        TextLine(
            page_index=0,
            page_ref="21",
            x0=116.4,
            x1=307.1,
            y0=304.1,
            text="ответным фланцем для воздуховода 900х900",
        ),
    ]

    rows, diagnostics = parse_spec8_lines_to_rows(sample, lines)

    assert diagnostics.parsed_rows == 2
    assert rows[0].type_mark == "DEMO-710-7,5/3000-3A32-03-KR"
    assert rows[0].note is not None
    assert "Торговый дом DemoVent" in rows[0].note
    assert "тел.:DEMO-PHONE-0000" in rows[0].note
    assert "email:demo-contact@example.invalid" in rows[0].note
    assert rows[1].name == (
        "Стакан монтажный со встроенным утепленным нормально-закрытым клапаном с ответным фланцем для воздуховода 900х900"
    )
    assert rows[1].type_mark == "DEMO-MOUNT-710-нестандарт"
    assert rows[1].note is None
