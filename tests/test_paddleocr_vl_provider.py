from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

from spec_pipeline.ocr.providers.paddleocr_vl import (
    PaddleOcrVlConfig,
    PaddleOcrVlProvider,
    annotate_crossed_out_html_rows,
    create_table_bands,
    detect_crossed_out_html_rows,
    parse_paddleocr_vl_result,
)
from spec_pipeline.ocr.render import RenderedPage


def test_parse_paddleocr_vl_result_returns_region_tokens_and_text() -> None:
    result = {
        "blocks": [
            {"text": "Позиция", "bbox": [10, 20, 80, 35], "confidence": 0.98},
            {"text": "1 Диффузор", "bbox": [10, 50, 180, 65], "confidence": 0.91},
        ]
    }

    page = parse_paddleocr_vl_result(
        result,
        page_index=0,
        page_ref="1",
        width=1000,
        height=800,
        provider="paddleocr-vl",
        image_path="page_001.png",
    )

    assert page.provider == "paddleocr-vl"
    assert page.image_path == "page_001.png"
    assert page.text.splitlines() == [
        "[10.0,20.0,80.0,35.0] Позиция",
        "[10.0,50.0,180.0,65.0] 1 Диффузор",
    ]
    assert len(page.tokens) == 2
    assert page.tokens[0].confidence == 0.98


def test_parse_paddleocr_vl_result_preserves_reported_reading_order() -> None:
    page = parse_paddleocr_vl_result(
        {
            "blocks": [
                {"text": "Вторая визуально выше", "bbox": [10, 90, 80, 100]},
                {"text": "Первая по Paddle reading order", "bbox": [10, 10, 80, 20]},
            ]
        },
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert [token.text for token in page.tokens] == [
        "Вторая визуально выше",
        "Первая по Paddle reading order",
    ]


def test_parse_paddleocr_vl_result_scales_normalized_bbox() -> None:
    page = parse_paddleocr_vl_result(
        {"blocks": [{"text": "Диффузор", "bbox": [0.1, 0.2, 0.3, 0.4]}]},
        page_index=0,
        page_ref="1",
        width=1000,
        height=800,
        provider="paddleocr-vl",
    )

    assert page.tokens[0].x0 == 100
    assert page.tokens[0].y0 == 160
    assert page.tokens[0].x1 == 300
    assert page.tokens[0].y1 == 320


def test_parse_paddleocr_vl_result_converts_polygon_to_rect() -> None:
    page = parse_paddleocr_vl_result(
        {
            "blocks": [
                {
                    "text": "Наименование",
                    "polygon": [[10, 20], [80, 20], [80, 35], [10, 35]],
                }
            ]
        },
        page_index=0,
        page_ref="1",
        width=1000,
        height=800,
        provider="paddleocr-vl",
    )

    assert page.tokens[0].x0 == 10
    assert page.tokens[0].y0 == 20
    assert page.tokens[0].x1 == 80
    assert page.tokens[0].y1 == 35


def test_parse_paddleocr_vl_result_accepts_array_like_bbox_and_scalars() -> None:
    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeArray:
        def tolist(self):
            return [FakeScalar(1), FakeScalar(2), FakeScalar(3), FakeScalar(4)]

    page = parse_paddleocr_vl_result(
        {"blocks": [{"text": "Клапан", "block_bbox": FakeArray()}]},
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert page.tokens[0].x0 == 1
    assert page.tokens[0].y0 == 2
    assert page.tokens[0].x1 == 3
    assert page.tokens[0].y1 == 4


def test_parse_paddleocr_vl_result_skips_empty_text() -> None:
    page = parse_paddleocr_vl_result(
        {"blocks": [{"text": "   ", "bbox": [0, 0, 10, 10]}]},
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert page.text == ""
    assert page.tokens == []


def test_parse_paddleocr_vl_result_reads_nested_res_payload() -> None:
    page = parse_paddleocr_vl_result(
        {"res": {"parsing_res_list": [{"block_content": "Клапан", "block_bbox": [1, 2, 3, 4]}]}},
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert page.tokens[0].text == "Клапан"
    assert page.tokens[0].x0 == 1


def test_parse_paddleocr_vl_result_prefers_json_payload_on_result_objects() -> None:
    class FakeResult(dict):
        json = {
            "res": {"parsing_res_list": [{"block_content": "Клапан", "block_bbox": [1, 2, 3, 4]}]}
        }

    page = parse_paddleocr_vl_result(
        FakeResult(),
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert page.tokens[0].text == "Клапан"
    assert page.tokens[0].x0 == 1


def test_parse_paddleocr_vl_result_pairs_rec_texts_and_boxes() -> None:
    page = parse_paddleocr_vl_result(
        {
            "res": {
                "overall_ocr_res": {
                    "rec_texts": ["Позиция", "Диффузор"],
                    "rec_boxes": [[1, 2, 3, 4], [10, 20, 30, 40]],
                    "rec_scores": [0.99, 0.88],
                }
            }
        },
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert [token.text for token in page.tokens] == ["Позиция", "Диффузор"]
    assert page.tokens[1].confidence == 0.88


def test_parse_paddleocr_vl_result_pairs_array_like_boxes_without_bool_eval() -> None:
    class FakeArray:
        def __init__(self, values) -> None:
            self.values = values

        def __bool__(self) -> bool:
            raise AssertionError("array-like boxes must not be truth-tested")

        def tolist(self):
            return self.values

    page = parse_paddleocr_vl_result(
        {
            "res": {
                "overall_ocr_res": {
                    "rec_texts": FakeArray(["Позиция", "Диффузор"]),
                    "rec_boxes": FakeArray([[1, 2, 3, 4], [10, 20, 30, 40]]),
                    "rec_scores": FakeArray([0.99, 0.88]),
                }
            }
        },
        page_index=0,
        page_ref="1",
        width=100,
        height=100,
        provider="paddleocr-vl",
    )

    assert [token.text for token in page.tokens] == ["Позиция", "Диффузор"]
    assert page.tokens[1].x0 == 10
    assert page.tokens[1].confidence == 0.88


def test_paddleocr_vl_provider_renders_selected_pages_and_sets_image_path(tmp_path: Path) -> None:
    class FakeRenderer:
        def __init__(self) -> None:
            self.rendered: list[int] = []

        def page_count(self, pdf_path: Path) -> int:
            return 3

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            self.rendered.append(page_number)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake", encoding="utf-8")
            return RenderedPage(
                page_number=page_number,
                image_path=output_path,
                width=1000,
                height=800,
            )

    class FakeEngine:
        def predict(self, image_path: str):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            return {"blocks": [{"text": "Диффузор", "bbox": [10, 20, 80, 35]}]}

    renderer = FakeRenderer()
    provider = PaddleOcrVlProvider(renderer=renderer, engine=FakeEngine())

    pages = provider.extract_pages(
        tmp_path / "input.pdf",
        pages=[2],
        output_dir=tmp_path / "pages",
    )

    assert renderer.rendered == [2]
    assert [page.page_ref for page in pages] == ["2"]
    assert pages[0].provider == "paddleocr-vl"
    assert pages[0].image_path == str(tmp_path / "pages" / "page_002.png")
    assert pages[0].tokens[0].text == "Диффузор"


def test_paddleocr_vl_provider_materializes_predict_generators(tmp_path: Path) -> None:
    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 1

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake", encoding="utf-8")
            return RenderedPage(
                page_number=page_number, image_path=output_path, width=100, height=100
            )

    class FakeEngine:
        def predict(self, image_path: str):
            yield {"blocks": [{"text": "Клапан", "bbox": [1, 2, 3, 4]}]}

    provider = PaddleOcrVlProvider(renderer=FakeRenderer(), engine=FakeEngine())

    pages = provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")

    assert pages[0].tokens[0].text == "Клапан"


def test_paddleocr_vl_provider_passes_generation_overrides_to_predict(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 1

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake", encoding="utf-8")
            return RenderedPage(
                page_number=page_number, image_path=output_path, width=100, height=100
            )

    class FakeEngine:
        def predict(self, image_path: str, **kwargs):
            captured.update(kwargs)
            return {"blocks": [{"text": "Клапан", "bbox": [1, 2, 3, 4]}]}

    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(
            max_pixels=8_000_000,
            max_new_tokens=8192,
            repetition_penalty=1.08,
            temperature=0.0,
            top_p=0.95,
        ),
        renderer=FakeRenderer(),
        engine=FakeEngine(),
    )

    provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")

    assert captured == {
        "max_pixels": 8_000_000,
        "max_new_tokens": 8192,
        "repetition_penalty": 1.08,
        "temperature": 0.0,
        "top_p": 0.95,
    }


def test_paddleocr_vl_provider_times_out_stuck_predict(tmp_path: Path) -> None:
    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 1

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake", encoding="utf-8")
            return RenderedPage(
                page_number=page_number, image_path=output_path, width=100, height=100
            )

    class FakeEngine:
        def predict(self, image_path: str, **kwargs):
            time.sleep(0.2)
            return {"blocks": [{"text": "Поздно", "bbox": [1, 2, 3, 4]}]}

    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(predict_timeout_seconds=0, predict_heartbeat_seconds=0),
        renderer=FakeRenderer(),
        engine=FakeEngine(),
    )

    try:
        provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")
    except TimeoutError as exc:
        assert "PaddleOCR-VL predict timed out" in str(exc)
    else:
        raise AssertionError("stuck PaddleOCR-VL predict must time out")


def test_create_table_bands_splits_wide_grid_image(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 140, 180, 240, 300, 360, 420, 480, 540, 600, 660):
        draw.line((40, y, 960, y), fill="black", width=3)
    for x in (40, 160, 500, 720, 840, 960):
        draw.line((x, 20, x, 660), fill="black", width=3)
    image.save(image_path)

    bands = create_table_bands(image_path, band_height=220, overlap=40)

    assert len(bands) >= 2
    assert all(band.image_path.exists() for band in bands)
    assert bands[0].y0 < bands[0].y1
    assert bands[1].y0 < bands[0].y1


def test_annotate_crossed_out_html_rows_marks_red_strike_rows(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    image_path = tmp_path / "band.png"
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 70, 120, 170, 220):
        draw.line((20, y, 580, y), fill="black", width=2)
    for x in (20, 120, 400, 500, 580):
        draw.line((x, 20, x, 220), fill="black", width=2)
    draw.line((40, 145, 560, 145), fill="red", width=3)
    image.save(image_path)
    html = "<table><tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr><tr><td>4</td></tr></table>"

    marked = annotate_crossed_out_html_rows(html, image_path)

    assert "CROSSED_OUT_HTML_ROWS" in marked
    assert 'data-status="crossed_out"' in marked
    assert detect_crossed_out_html_rows(image_path, html_row_count=4)


def test_paddleocr_vl_provider_splits_wide_pages_into_table_bands(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    captured: list[tuple[str, dict[str, object]]] = []

    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 1

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            image = Image.new("RGB", (1000, 700), "white")
            draw = ImageDraw.Draw(image)
            for y in (20, 140, 180, 240, 300, 360, 420, 480, 540, 600, 660):
                draw.line((40, y, 960, y), fill="black", width=3)
            for x in (40, 160, 500, 720, 840, 960):
                draw.line((x, 20, x, 660), fill="black", width=3)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return RenderedPage(
                page_number=page_number, image_path=output_path, width=1000, height=700
            )

    class FakeEngine:
        def predict(self, image_path: str, **kwargs):
            captured.append((Path(image_path).name, kwargs))
            return {
                "blocks": [
                    {
                        "text": f"<table><tr><td>{Path(image_path).name}</td></tr></table>",
                        "bbox": [0, 0, 100, 50],
                    }
                ]
            }

    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(
            split_wide_pages=True,
            split_band_height=220,
            split_band_overlap=40,
            max_pixels=1_003_520,
            max_new_tokens=2048,
        ),
        renderer=FakeRenderer(),
        engine=FakeEngine(),
    )

    pages = provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")

    assert len(captured) >= 2
    assert all(item[1]["use_layout_detection"] is False for item in captured)
    assert all(item[1]["prompt_label"] == "table" for item in captured)
    assert all(item[1]["use_queues"] is False for item in captured)
    assert "OCR_BAND 1" in pages[0].text
    assert pages[0].image_path == str(tmp_path / "pages" / "page_001.png")


def test_paddleocr_vl_provider_initializes_current_api_kwargs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    paddle_module = types.ModuleType("paddleocr")
    local_model_path = tmp_path / "paddle-model"
    local_model_path.mkdir()
    layout_model_path = local_model_path / "PP-DocLayoutV2"
    layout_model_path.mkdir()
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            assert "model_name" not in kwargs
            assert "vl_rec_model_name" not in kwargs
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            assert os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"
            assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path)
            captured.update(kwargs)

    setattr(paddle_module, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle_module)
    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(model_name=str(local_model_path), device="cpu")
    )

    engine = provider._get_engine()

    assert isinstance(engine, FakePaddleOCRVL)
    assert captured == {
        "pipeline_version": "v1",
        "device": "cpu",
        "layout_detection_model_dir": str(layout_model_path),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "vl_rec_model_dir": str(local_model_path),
    }
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "PADDLE_PDX_CACHE_HOME" not in os.environ


def test_paddleocr_vl_provider_initializes_mlx_vlm_server_backend(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    paddle_module = types.ModuleType("paddleocr")
    local_model_path = tmp_path / "paddle-model"
    (local_model_path / "PP-DocLayoutV2").mkdir(parents=True)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    setattr(paddle_module, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle_module)
    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(
            model_name=str(local_model_path),
            device="cpu",
            vl_rec_backend="mlx-vlm-server",
            vl_rec_server_url="http://localhost:8111/",
            vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
            max_new_tokens=512,
        )
    )

    provider._get_engine()

    assert captured["vl_rec_backend"] == "mlx-vlm-server"
    assert captured["vl_rec_server_url"] == "http://localhost:8111/"
    assert captured["vl_rec_api_model_name"] == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert "vl_rec_model_dir" not in captured


def test_paddleocr_vl_provider_requires_cuda_and_defaults_gpu_device(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    checked: dict[str, object] = {}
    paddle_module = types.ModuleType("paddleocr")
    local_model_path = tmp_path / "paddle-model"
    (local_model_path / "PP-DocLayoutV2").mkdir(parents=True)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    def fake_ensure_cuda(*, device: str):
        checked["device"] = device
        return SimpleNamespace(version="3.2.1", cuda_device_count=1, selected_device=device)

    setattr(paddle_module, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle_module)
    monkeypatch.setattr(
        "spec_pipeline.ocr.providers.paddleocr_vl.ensure_paddle_cuda_available",
        fake_ensure_cuda,
    )

    provider = PaddleOcrVlProvider(
        config=PaddleOcrVlConfig(model_name=str(local_model_path), require_cuda=True)
    )

    provider._get_engine()

    assert checked["device"] == "gpu:0"
    assert captured["device"] == "gpu:0"


def test_paddleocr_vl_provider_rejects_local_model_without_layout_dir(
    tmp_path: Path, monkeypatch
) -> None:
    paddle_module = types.ModuleType("paddleocr")
    local_model_path = tmp_path / "paddle-model"
    local_model_path.mkdir()

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("engine must not initialize without managed layout model")

    setattr(paddle_module, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle_module)
    provider = PaddleOcrVlProvider(config=PaddleOcrVlConfig(model_name=str(local_model_path)))

    try:
        provider._get_engine()
    except RuntimeError as exc:
        assert "PP-DocLayoutV2" in str(exc)
    else:
        raise AssertionError("local PaddleOCR-VL model without layout model must be rejected")


def test_paddleocr_vl_provider_rejects_non_local_model_name(monkeypatch) -> None:
    paddle_module = types.ModuleType("paddleocr")

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("engine must not initialize for non-local model names")

    setattr(paddle_module, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle_module)
    provider = PaddleOcrVlProvider(config=PaddleOcrVlConfig(model_name="custom-model"))

    try:
        provider._get_engine()
    except RuntimeError as exc:
        assert "local model directory" in str(exc)
    else:
        raise AssertionError("non-local PaddleOCR-VL model names must be rejected")
