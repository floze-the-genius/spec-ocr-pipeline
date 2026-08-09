from __future__ import annotations

import sys
import types
from pathlib import Path

from spec_pipeline.ocr.render import PdfiumPdfRenderer


def test_pdfium_renderer_renders_page_with_scale(tmp_path: Path, monkeypatch) -> None:
    calls = {}
    pdfium_module = types.ModuleType("pypdfium2")

    class FakeImage:
        size = (123, 456)

        def save(self, path: Path) -> None:
            calls["saved_path"] = path
            path.write_text("png", encoding="utf-8")

    class FakeBitmap:
        def to_pil(self):
            return FakeImage()

    class FakePage:
        def render(self, *, scale: float):
            calls["scale"] = scale
            return FakeBitmap()

    class FakeDocument:
        def __init__(self, path: str) -> None:
            calls["path"] = path

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            calls["index"] = index
            return FakePage()

    setattr(pdfium_module, "PdfDocument", FakeDocument)
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium_module)

    output_path = tmp_path / "pages" / "page_001.png"
    rendered = PdfiumPdfRenderer(dpi=144).render_page(
        tmp_path / "input.pdf", page_number=1, output_path=output_path
    )

    assert calls["scale"] == 2.0
    assert calls["index"] == 0
    assert calls["saved_path"] == output_path
    assert rendered.width == 123
    assert rendered.height == 456
    assert output_path.exists()
