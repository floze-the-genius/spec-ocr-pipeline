from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from spec_pipeline.ocr.providers.dots_mocr import (
    DotsMocrConfig,
    DotsMocrProvider,
    MlxVlmDotsMocrGenerator,
    TransformersDotsMocrGenerator,
    _dots_mocr_process_lock,
    _resolve_dots_model,
    build_dots_mocr_prompt,
    create_dots_mocr_generator,
    detect_table_content_bands,
    parse_dots_mocr_html_rows,
    preprocess_table_image,
    validate_dots_mocr_html,
)
from spec_pipeline.ocr.render import RenderedPage


def test_create_dots_mocr_generator_selects_requested_backend() -> None:
    assert isinstance(
        create_dots_mocr_generator(DotsMocrConfig(backend="mlx-vlm", trust_model_code=True)),
        MlxVlmDotsMocrGenerator,
    )


def test_dots_mocr_process_lock_is_cross_platform(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "dots-mocr.lock"

    with _dots_mocr_process_lock(lock_path):
        assert lock_path.exists()
    assert isinstance(
        create_dots_mocr_generator(DotsMocrConfig(backend="transformers", trust_model_code=True)),
        TransformersDotsMocrGenerator,
    )


def test_dots_mocr_generator_requires_explicit_model_code_trust() -> None:
    try:
        create_dots_mocr_generator(DotsMocrConfig(backend="transformers"))
    except RuntimeError as exc:
        assert "--dots-trust-model-code" in str(exc)
    else:
        raise AssertionError("dots.mocr model code must be opt-in")


def test_dots_mocr_remote_model_download_is_disabled(tmp_path: Path) -> None:
    try:
        _resolve_dots_model("example/model", tmp_path)
    except RuntimeError as exc:
        assert "Automatic dots.mocr model downloads are disabled" in str(exc)
    else:
        raise AssertionError("remote model identifiers must fail closed")


def test_dots_mocr_provider_crops_upscales_and_returns_html_page(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 3

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            image = Image.new("RGB", (200, 180), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 50, 160, 130), outline="black", width=2)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return RenderedPage(
                page_number=page_number,
                image_path=output_path,
                width=200,
                height=180,
            )

    class FakeGenerator:
        def generate(self, **kwargs):
            captured.update(kwargs)
            cells = "".join(f"<td>{index}</td>" for index in range(1, 10))
            return f"<html><body><table><tr>{cells}</tr></table></body></html>"

    provider = DotsMocrProvider(
        config=DotsMocrConfig(
            model_name="test-model",
            upscale=2.0,
            max_image_edge=2600,
            crop_padding=0,
            progress=False,
            lock_path=None,
        ),
        renderer=FakeRenderer(),
        generator=FakeGenerator(),
    )

    pages = provider.extract_pages(tmp_path / "input.pdf", pages=[2], output_dir=tmp_path / "pages")

    assert pages[0].provider == "dots-mocr"
    assert pages[0].page_ref == "2"
    assert pages[0].text.startswith("DOTS_MOCR_HTML:\n<html>")
    assert pages[0].image_path is not None
    assert Path(pages[0].image_path).name == "page_002_crop_2x.png"
    assert Image.open(Path(pages[0].image_path)).size == (242, 162)
    assert captured["model_name"] == "test-model"
    assert captured["model_cache_dir"] is None
    assert captured["image_path"] == Path(pages[0].image_path)
    assert "red quantity is visually split" in str(captured["prompt"])


def test_preprocess_table_image_caps_upscaled_long_edge(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 500), "white").save(image_path)

    output_path, width, height = preprocess_table_image(
        image_path,
        crop_to_content=False,
        crop_threshold=12,
        crop_padding=40,
        upscale=2.0,
        max_image_edge=600,
    )

    assert output_path.name == "page_2x.png"
    assert (width, height) == (600, 300)
    assert Image.open(output_path).size == (600, 300)


def test_preprocess_table_image_auto_upscales_red_markup(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 50, 35), fill=(220, 0, 0))
    image.save(image_path)

    output_path, width, height = preprocess_table_image(
        image_path,
        crop_to_content=False,
        crop_threshold=12,
        crop_padding=40,
        upscale=0.0,
        max_image_edge=1000,
    )

    assert output_path.name == "page_2x.png"
    assert (width, height) == (200, 160)


def test_dots_mocr_prompt_anchors_main_table_and_split_quantities() -> None:
    prompt = build_dots_mocr_prompt()

    assert "main Russian construction specification table" in prompt
    assert "Ignore side stamps" in prompt
    assert "exactly 9 visual columns" in prompt
    assert "without inventing decimal commas" in prompt


def test_validate_dots_mocr_html_rejects_non_9_column_rows() -> None:
    good_cells = "".join(f"<td>{index}</td>" for index in range(1, 10))
    assert parse_dots_mocr_html_rows(f"<table><tr>{good_cells}</tr></table>") == [
        [str(index) for index in range(1, 10)]
    ]

    try:
        validate_dots_mocr_html("<table><tr><td>1</td></tr></table>")
    except RuntimeError as exc:
        assert "expected 9 cells" in str(exc)
    else:
        raise AssertionError("malformed dots HTML should fail validation")


def test_validate_dots_mocr_html_drops_empty_and_title_block_rows() -> None:
    good_cells = "".join(f"<td>{index}</td>" for index in range(1, 10))
    text = (
        "<html><body><table>"
        f"<tr>{good_cells}</tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td>Изм.</td><td>Кол.</td><td>Лист</td></tr>"
        "</table></body></html>"
    )

    normalized = validate_dots_mocr_html(text)

    assert parse_dots_mocr_html_rows(normalized) == [[str(index) for index in range(1, 10)]]


def test_validate_dots_mocr_html_expands_colspan_and_pads_trailing_empty_cells() -> None:
    normalized = validate_dots_mocr_html(
        "<table>"
        '<tr><td colspan="2">Кровля</td><td></td><td></td><td></td><td></td>'
        "<td></td><td></td><td></td></tr>"
        "<tr><td>1</td><td>Дефлектор Ø800</td><td></td><td></td><td></td>"
        "<td>шт</td><td>12</td><td></td></tr>"
        "</table>"
    )

    rows = parse_dots_mocr_html_rows(normalized)
    assert rows[0] == ["Кровля", "", "", "", "", "", "", "", ""]
    assert rows[1] == ["1", "Дефлектор Ø800", "", "", "", "шт", "12", "", ""]


def test_detect_table_content_bands_finds_separated_table_sections() -> None:
    image = Image.new("RGB", (800, 700), "white")
    draw = ImageDraw.Draw(image)
    for top in (100, 410):
        for row in range(5):
            draw.rectangle((120, top + row * 24, 260, top + row * 24 + 8), fill="black")
            draw.rectangle((610, top + row * 24, 630, top + row * 24 + 8), fill="black")

    bands = detect_table_content_bands(
        image,
        min_gap_px=120,
        padding_px=20,
        max_bands=4,
    )

    assert len(bands) == 2
    assert bands[0][0] < 100 < bands[0][1]
    assert bands[1][0] < 410 < bands[1][1]


def test_dots_mocr_provider_uses_band_split_when_it_finds_more_rows(tmp_path: Path) -> None:
    calls: list[str] = []

    def html_row(*cells: str) -> str:
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

    upper = html_row("1", "Вентилятор", "", "", "", "шт", "2", "", "")
    lower = html_row("2", "Дефлектор", "", "", "", "шт", "12", "", "")

    class FakeRenderer:
        def page_count(self, pdf_path: Path) -> int:
            return 1

        def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path):
            image = Image.new("RGB", (800, 700), "white")
            draw = ImageDraw.Draw(image)
            for top in (100, 410):
                for row in range(5):
                    draw.rectangle((120, top + row * 24, 260, top + row * 24 + 8), fill="black")
                    draw.rectangle((610, top + row * 24, 630, top + row * 24 + 8), fill="black")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return RenderedPage(
                page_number=page_number,
                image_path=output_path,
                width=800,
                height=700,
            )

    class FakeGenerator:
        def generate(self, **kwargs):
            calls.append(Path(kwargs["image_path"]).name)
            name = Path(kwargs["image_path"]).name
            if "_band_002" in name:
                return f"<table>{lower}</table>"
            return f"<table>{upper}</table>"

    provider = DotsMocrProvider(
        config=DotsMocrConfig(
            crop_to_content=False,
            upscale=1.0,
            max_image_edge=0,
            lock_path=None,
            band_min_gap_px=120,
            band_padding_px=20,
        ),
        renderer=FakeRenderer(),
        generator=FakeGenerator(),
    )

    pages = provider.extract_pages(tmp_path / "input.pdf", output_dir=tmp_path / "pages")

    rows = parse_dots_mocr_html_rows(pages[0].text)
    assert any(row[1] == "Вентилятор" for row in rows)
    assert any(row[1] == "Дефлектор" for row in rows)
    assert any("_band_001" in call for call in calls)
    assert any("_band_002" in call for call in calls)
