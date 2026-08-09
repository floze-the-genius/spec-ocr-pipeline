from __future__ import annotations

from spec_pipeline.workflows.pdf_probe import classify_text_layer, is_garbled_text


def test_classify_text_layer_marks_dense_text_as_usable() -> None:
    classification, needs_ocr = classify_text_layer(
        page_count=10,
        nonempty_pages=10,
        text_line_count=200,
        text_char_count=50000,
    )

    assert classification == "text_layer_usable"
    assert needs_ocr is False


def test_classify_text_layer_marks_sparse_text_as_ocr_required() -> None:
    classification, needs_ocr = classify_text_layer(
        page_count=204,
        nonempty_pages=2,
        text_line_count=2,
        text_char_count=215,
    )

    assert classification == "ocr_required"
    assert needs_ocr is True


def test_classify_text_layer_marks_garbled_text_as_ocr_required() -> None:
    classification, needs_ocr = classify_text_layer(
        page_count=4,
        nonempty_pages=4,
        text_line_count=200,
        text_char_count=10000,
        garbled_text_char_count=8000,
    )

    assert classification == "text_layer_garbled"
    assert needs_ocr is True


def test_is_garbled_text_detects_cid_and_mojibake() -> None:
    assert is_garbled_text("(cid:585)(cid:616)(cid:609)") is True
    assert is_garbled_text("Ïîçèöèÿ") is True
    assert is_garbled_text("Позиция") is False
