from __future__ import annotations

from spec_pipeline.ai.vision_repair import (
    apply_row_patches,
    build_vision_patch_prompt,
    detect_vision_repair_plan,
    existing_image_pages,
    merge_repaired_page_rows,
    semantic_ocr_garbage_message,
)
from spec_pipeline.ocr.base import OcrPage


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "page_ref": "1",
        "section": None,
        "subsection": None,
        "pos": "1",
        "name": "Диффузор",
        "type_mark": None,
        "code": None,
        "manufacturer": None,
        "unit": "шт",
        "quantity": "2",
        "mass_kg": None,
        "note": None,
        "status": None,
    }
    row.update(overrides)
    return row


def test_detect_vision_repair_plan_triggers_on_semantic_scan_garbage() -> None:
    rows = [
        _row(name="Автоматический воздухомобюксир 015", manufacturer="Pugah"),
        _row(pos="2", name="Мемаллорукав L=400мм 3/4”"),
    ]

    plan = detect_vision_repair_plan(rows)

    assert plan.page_refs == ("1",)
    assert {(issue.field, issue.message) for issue in plan.issues} >= {
        ("name", "text looks semantically OCR-damaged"),
        ("manufacturer", "text looks semantically OCR-damaged"),
    }


def test_detect_vision_repair_plan_ignores_normal_model_codes() -> None:
    rows = [
        _row(name="Кран шаровой латунный", type_mark="BVR-R-015", manufacturer="Bugatti"),
        _row(pos="2", name="Вентилятор", type_mark="IN 10/4", manufacturer="O.ERRE"),
    ]

    plan = detect_vision_repair_plan(rows)

    assert plan.needed is False
    assert plan.issues == ()


def test_detect_vision_repair_plan_triggers_on_missing_quantity_and_name() -> None:
    rows = [
        _row(unit="шт", quantity=None),
        _row(name=None, type_mark="ABC-100", unit="шт", quantity="2"),
    ]

    plan = detect_vision_repair_plan(rows)

    assert plan.page_refs == ("1",)
    assert {(issue.field, issue.message) for issue in plan.issues} >= {
        ("quantity", "row has unit but no quantity"),
        ("name", "data-like row has no item name"),
    }


def test_semantic_ocr_garbage_message_distinguishes_symbol_from_semantic_garbage() -> None:
    assert semantic_ocr_garbage_message("Кран $ \\Phi $15") == "text contains OCR symbol garbage"
    assert (
        semantic_ocr_garbage_message("Автоматический воздухомоборчик Airvent 015")
        == "text looks semantically OCR-damaged"
    )
    assert semantic_ocr_garbage_message("Цена ₽1000") is None


def test_merge_repaired_page_rows_replaces_only_repaired_pages() -> None:
    original = [
        _row(page_ref="1", pos="1", name="Автоматический воздухомобюксир 015"),
        _row(page_ref="1", pos="2", name="Мемаллорукав L=400мм 3/4”"),
        _row(page_ref="2", pos="1", name="Диффузор"),
    ]
    repaired = [
        _row(page_ref="1", pos="1", name="Автоматический воздухоотводчик Ø15"),
        _row(page_ref="1", pos="2", name="Металлорукав L=400мм 3/4”"),
    ]

    merged = merge_repaired_page_rows(original, repaired, {"1"})

    assert [row["name"] for row in merged] == [
        "Автоматический воздухоотводчик Ø15",
        "Металлорукав L=400мм 3/4”",
        "Диффузор",
    ]


def test_build_vision_patch_prompt_requests_patches_not_replacement_rows() -> None:
    prompt = build_vision_patch_prompt(
        sample_id="sample",
        pages=[
            OcrPage(
                page_index=0,
                page_ref="1",
                width=100,
                height=100,
                text="OCR",
                tokens=[],
                provider="paddleocr-vl",
            )
        ],
        rows=[_row(name="Автоматический воздухомобюксир 015")],
        issues=[],
    )

    assert '"patches"' in prompt
    assert "Не добавляй строки" in prompt
    assert "Не удаляй строки" in prompt


def test_existing_image_pages_allows_pdf_table_layer_with_rendered_image(tmp_path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake")
    pages = [
        OcrPage(
            page_index=0,
            page_ref="1",
            width=100,
            height=100,
            text="PDF_TABLE_LAYER:\nPDF_TABLE_ROW {}",
            tokens=[],
            provider="pdf-text",
            image_path=image_path,
        )
    ]

    assert existing_image_pages(pages, {"1"}) == pages


def test_apply_row_patches_preserves_row_count_and_order() -> None:
    rows = [
        _row(pos="1", name="Автоматический воздухомобюксир 015"),
        _row(pos="2", name="Диффузор"),
    ]
    patched, warnings, accepted, rejected = apply_row_patches(
        rows,
        [
            {
                "row_index": 1,
                "page_ref": "1",
                "field": "name",
                "old_value": "Автоматический воздухомобюксир 015",
                "new_value": "Автоматический воздухоотводчик Ø15",
                "confidence": "high",
                "reason": "visible on page",
            }
        ],
        {"1"},
    )

    assert len(patched) == len(rows)
    assert [row["pos"] for row in patched] == ["1", "2"]
    assert patched[0]["name"] == "Автоматический воздухоотводчик Ø15"
    assert accepted and not rejected
    assert "third-layer accepted 1 patch(es)" in warnings


def test_apply_row_patches_rejects_unsafe_patch() -> None:
    rows = [_row(name="Диффузор")]
    patched, warnings, accepted, rejected = apply_row_patches(
        rows,
        [
            {
                "row_index": 1,
                "page_ref": "1",
                "field": "name",
                "old_value": "wrong",
                "new_value": "Вентилятор",
                "confidence": "high",
            },
            {
                "row_index": 1,
                "page_ref": "1",
                "field": "page_ref",
                "old_value": "1",
                "new_value": "2",
                "confidence": "high",
            },
            {
                "row_index": 1,
                "page_ref": "1",
                "field": "name",
                "old_value": "Диффузор",
                "new_value": "Вентилятор",
                "confidence": "medium",
            },
        ],
        {"1"},
    )

    assert patched == rows
    assert not accepted
    assert len(rejected) == 3
    assert "third-layer rejected 3 unsafe patch(es)" in warnings
