from __future__ import annotations

from pathlib import Path
from typing import cast

from pdfminer.layout import LTPage

from spec_pipeline.ocr.providers.pdf_text import (
    PdfTextLine,
    PdfTextOcrProvider,
    _layout_to_page,
    build_pdf_table_layer,
    compact_pdf_text_overlay,
    parse_pdf_table_layer_rows,
    repair_pdf_text,
)
from spec_pipeline.ocr.render import RenderedPage


class FakeLine:
    def __init__(self, text: str, x0: float, y0: float, x1: float, y1: float) -> None:
        self._text = text
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def get_text(self) -> str:
        return self._text


class FakeContainer:
    def __init__(self, lines: list[FakeLine]) -> None:
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)


class FakePage:
    width = 1000
    height = 800

    def __iter__(self):
        return iter(
            [
                FakeContainer(
                    [
                        FakeLine("1", 10, 700, 20, 710),
                        FakeLine("Диффузор", 100, 700, 180, 710),
                        FakeLine("Позиция", 10, 750, 80, 760),
                    ]
                )
            ]
        )


def test_layout_to_page_orders_lines_and_preserves_coordinates(monkeypatch) -> None:
    monkeypatch.setattr("spec_pipeline.ocr.providers.pdf_text.LTTextContainer", FakeContainer)
    monkeypatch.setattr("spec_pipeline.ocr.providers.pdf_text.LTTextLine", FakeLine)

    page = _layout_to_page(cast(LTPage, FakePage()), 3, "pdf-text")

    assert page.page_ref == "3"
    assert page.provider == "pdf-text"
    assert page.tokens[0].text == "Позиция"
    assert page.tokens[1].text == "1"
    assert "[10.0,750.0,80.0,760.0] Позиция" in page.text


def test_pdf_text_provider_renders_page_images_when_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("spec_pipeline.ocr.providers.pdf_text.LTTextContainer", FakeContainer)
    monkeypatch.setattr("spec_pipeline.ocr.providers.pdf_text.LTTextLine", FakeLine)
    monkeypatch.setattr(
        "spec_pipeline.ocr.providers.pdf_text.extract_pages",
        lambda path: [cast(LTPage, FakePage())],
    )

    class FakeRenderer:
        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-image")
            return RenderedPage(
                page_number=page_number,
                image_path=output_path,
                width=100,
                height=200,
            )

    provider = PdfTextOcrProvider(renderer=FakeRenderer())

    pages = provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")

    assert pages[0].image_path == str(tmp_path / "pages" / "page_001.png")


def test_repair_pdf_text_fixes_cp1251_latin1_mojibake() -> None:
    assert repair_pdf_text("Äåôëåêòîð Ô1250 øò") == "Дефлектор Ф1250 шт"
    assert repair_pdf_text("100 (δ=0,5 ìì)") == "100 (δ=0,5 мм)"
    assert (
        repair_pdf_text("1.14 Òåïëîèçîëÿöèÿ âîçäóõîâîäîâ δ=25 ìì")
        == "1.14 Теплоизоляция воздуховодов δ=25 мм"
    )
    assert repair_pdf_text("ГOCT 14918-2020 RL/P1-「K1") == "ГОСТ 14918-2020 RL/P1-[K1"
    assert repair_pdf_text("м/мІ") == "м/м²"
    assert repair_pdf_text("Позиция") == "Позиция"


def test_build_pdf_table_layer_reconstructs_coordinate_rows() -> None:
    lines = [
        PdfTextLine("Позиция", 77, 772, 123, 786),
        PdfTextLine("Жилая часть", 251, 696, 331, 713),
        PdfTextLine("ВЕ4.1", 76, 673, 127, 685),
        PdfTextLine("Дефлектор Ф1250", 136, 673, 219, 685),
        PdfTextLine("шт", 882, 673, 900, 685),
        PdfTextLine("2", 943, 672, 952, 685),
        PdfTextLine("В1.1", 95, 650, 113, 663),
        PdfTextLine("Вентилятор", 135, 650, 379, 663),
        PdfTextLine("IN 9/3,5", 510, 650, 549, 662),
        PdfTextLine('"O.ERRE"', 755, 650, 793, 662),
        PdfTextLine("шт", 882, 650, 900, 662),
        PdfTextLine("18", 941, 650, 954, 662),
    ]

    layer = build_pdf_table_layer(lines, page_ref="1", page_height=842)

    assert layer is not None
    rows = parse_pdf_table_layer_rows(layer)
    assert rows[0]["name"] == "Жилая часть"
    assert rows[1]["pos"] == "ВЕ4.1"
    assert rows[1]["quantity"] == "2"
    assert rows[2]["type_mark"] == "IN 9/3,5"


def test_build_pdf_table_layer_scales_columns_for_wide_landscape_pages() -> None:
    lines = [
        PdfTextLine("1", 135, 1040, 142, 1052),
        PdfTextLine("Канальный вентилятор", 184, 1040, 360, 1052),
        PdfTextLine("LKP-600x350-V4/380B", 660, 1040, 820, 1052),
        PdfTextLine("Люфткон", 1083, 1040, 1140, 1052),
        PdfTextLine("шт", 1234, 1040, 1250, 1052),
        PdfTextLine("36", 1328, 1040, 1340, 1052),
        PdfTextLine("2", 135, 1010, 142, 1022),
        PdfTextLine("Клапан", 184, 1010, 250, 1022),
        PdfTextLine("шт", 1234, 1010, 1250, 1022),
        PdfTextLine("24", 1328, 1010, 1340, 1022),
        PdfTextLine("3", 135, 980, 142, 992),
        PdfTextLine("Шумоглушитель", 184, 980, 300, 992),
        PdfTextLine("шт", 1234, 980, 1250, 992),
        PdfTextLine("72", 1328, 980, 1340, 992),
    ]

    layer = build_pdf_table_layer(lines, page_ref="1", page_height=1191, page_width=1684)

    assert layer is not None
    rows = parse_pdf_table_layer_rows(layer)
    assert rows[0]["pos"] == "1"
    assert rows[0]["name"] == "Канальный вентилятор"
    assert rows[0]["type_mark"] == "LKP-600x350-V4/380B"
    assert rows[0]["manufacturer"] == "Люфткон"
    assert rows[0]["unit"] == "шт"
    assert rows[0]["quantity"] == "36"


def test_build_pdf_table_layer_filters_header_number_and_title_block_rows() -> None:
    lines = [
        PdfTextLine("1", 78, 740, 85, 752),
        PdfTextLine("2", 300, 740, 307, 752),
        PdfTextLine("3", 575, 740, 582, 752),
        PdfTextLine("6", 905, 740, 912, 752),
        PdfTextLine("7", 945, 740, 952, 752),
        PdfTextLine("Èçì. Êîë.ó÷ Ëèñò №äîê", 620, 130, 780, 142),
        PdfTextLine("Подп. Дата", 810, 130, 880, 142),
        PdfTextLine("1", 78, 650, 86, 662),
        PdfTextLine("Диффузор", 135, 650, 220, 662),
        PdfTextLine("шт", 882, 650, 900, 662),
        PdfTextLine("2", 943, 650, 952, 662),
        PdfTextLine("2", 78, 625, 86, 637),
        PdfTextLine("Решетка", 135, 625, 220, 637),
        PdfTextLine("шт", 882, 625, 900, 637),
        PdfTextLine("3", 943, 625, 952, 637),
        PdfTextLine("3", 78, 600, 86, 612),
        PdfTextLine("Клапан", 135, 600, 220, 612),
        PdfTextLine("шт", 882, 600, 900, 612),
        PdfTextLine("4", 943, 600, 952, 612),
    ]

    layer = build_pdf_table_layer(lines, page_ref="1", page_height=842, page_width=1191)

    assert layer is not None
    rows = parse_pdf_table_layer_rows(layer)
    assert [row["name"] for row in rows] == ["Диффузор", "Решетка", "Клапан"]


def test_build_pdf_table_layer_keeps_kg_as_body_unit() -> None:
    lines = [
        PdfTextLine("1", 78, 740, 85, 752),
        PdfTextLine("2", 300, 740, 307, 752),
        PdfTextLine("6", 905, 740, 912, 752),
        PdfTextLine("7", 945, 740, 952, 752),
        PdfTextLine("кг", 1025, 740, 1042, 752),
        PdfTextLine("1.13", 76, 637, 99, 653),
        PdfTextLine("Краска огнезащитная", 118, 637, 300, 653),
        PdfTextLine("кг", 906, 637, 921, 652),
        PdfTextLine("85", 962, 638, 977, 651),
        PdfTextLine("1.14", 76, 616, 99, 632),
        PdfTextLine("Теплоизоляция", 118, 616, 300, 632),
        PdfTextLine("м²", 906, 616, 921, 632),
        PdfTextLine("200", 959, 617, 980, 630),
        PdfTextLine("1.15", 76, 595, 99, 611),
        PdfTextLine("Пусконаладка", 118, 595, 300, 611),
        PdfTextLine("сист.", 906, 595, 930, 611),
        PdfTextLine("1", 959, 596, 970, 609),
    ]

    layer = build_pdf_table_layer(lines, page_ref="1", page_height=842, page_width=1191)

    assert layer is not None
    rows = parse_pdf_table_layer_rows(layer)
    assert rows[0]["unit"] == "кг"
    assert rows[0]["quantity"] == "85"
    assert rows[1]["unit"] == "м²"


def test_compact_pdf_text_overlay_keeps_table_layer_only() -> None:
    text = "PDF_TABLE_LAYER:\nPDF_TABLE_ROW {}\n\nPDF_TEXT_COORDS:\n[1,2,3,4] raw"

    assert compact_pdf_text_overlay(text) == "PDF_TABLE_LAYER:\nPDF_TABLE_ROW {}"
