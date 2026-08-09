from __future__ import annotations

import pytest

from spec_pipeline.ocr.providers.tesseract import parse_page_selection, parse_tesseract_tsv


def test_parse_page_selection_supports_ranges_and_deduplicates() -> None:
    assert parse_page_selection("1,3-5,3") == [1, 3, 4, 5]
    assert parse_page_selection(None) is None


def test_parse_page_selection_rejects_reversed_ranges() -> None:
    with pytest.raises(ValueError):
        parse_page_selection("5-3")


def test_parse_tesseract_tsv_returns_tokens_and_lines() -> None:
    tsv = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t91.5\tПозиция",
            "5\t1\t1\t1\t1\t2\t45\t20\t80\t10\t88.0\tНаименование",
            "5\t1\t1\t1\t2\t1\t10\t40\t20\t10\t95.0\t1",
            "5\t1\t1\t1\t2\t2\t45\t40\t60\t10\t90.0\tДиффузор",
        ]
    )

    page = parse_tesseract_tsv(
        tsv,
        page_index=0,
        page_ref="1",
        width=1000,
        height=800,
        provider="tesseract",
        image_path="page_001.png",
    )

    assert page.page_ref == "1"
    assert page.text.splitlines() == ["Позиция Наименование", "1 Диффузор"]
    assert len(page.tokens) == 4
    assert page.tokens[0].x0 == 10
    assert page.tokens[0].x1 == 40
    assert page.tokens[0].confidence == 91.5
