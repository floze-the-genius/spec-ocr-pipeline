from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class RenderedPage:
    page_number: int
    image_path: Path
    width: int
    height: int


class PdfRenderer(Protocol):
    dpi: int

    def page_count(self, pdf_path: Path) -> int: ...

    def render_page(
        self, pdf_path: Path, *, page_number: int, output_path: Path
    ) -> RenderedPage: ...


def create_pdf_renderer(*, dpi: int = 200) -> PdfRenderer:
    try:
        QuartzPdfRenderer._import_quartz()
    except RuntimeError:
        return PdfiumPdfRenderer(dpi=dpi)
    return QuartzPdfRenderer(dpi=dpi)


class QuartzPdfRenderer:
    """Render PDF pages to PNG using macOS Quartz, avoiding Poppler/Ghostscript."""

    def __init__(self, *, dpi: int = 200) -> None:
        self.dpi = dpi

    def page_count(self, pdf_path: Path) -> int:
        Quartz, Foundation = self._import_quartz()
        doc = self._open_document(Quartz, Foundation, pdf_path)
        return int(Quartz.CGPDFDocumentGetNumberOfPages(doc))

    def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path) -> RenderedPage:
        Quartz, Foundation = self._import_quartz()
        doc = self._open_document(Quartz, Foundation, pdf_path)
        page = Quartz.CGPDFDocumentGetPage(doc, page_number)
        if page is None:
            raise ValueError(f"PDF page does not exist: page={page_number} path={pdf_path}")

        box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
        scale = self.dpi / 72.0
        width = int(box.size.width * scale)
        height = int(box.size.height * scale)

        context = Quartz.CGBitmapContextCreate(
            None,
            width,
            height,
            8,
            0,
            Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaPremultipliedLast,
        )
        Quartz.CGContextSetRGBFillColor(context, 1, 1, 1, 1)
        Quartz.CGContextFillRect(context, Quartz.CGRectMake(0, 0, width, height))

        target = Quartz.CGRectMake(0, 0, width, height)
        transform = Quartz.CGPDFPageGetDrawingTransform(
            page, Quartz.kCGPDFMediaBox, target, 0, True
        )
        Quartz.CGContextConcatCTM(context, transform)
        Quartz.CGContextDrawPDFPage(context, page)

        image = Quartz.CGBitmapContextCreateImage(context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        destination = Quartz.CGImageDestinationCreateWithURL(
            Foundation.NSURL.fileURLWithPath_(str(output_path.resolve())),
            "public.png",
            1,
            None,
        )
        Quartz.CGImageDestinationAddImage(destination, image, None)
        Quartz.CGImageDestinationFinalize(destination)
        return RenderedPage(
            page_number=page_number,
            image_path=output_path,
            width=width,
            height=height,
        )

    @staticmethod
    def _import_quartz():
        try:
            import Foundation  # type: ignore[import-not-found]
            import Quartz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Quartz rendering is available only on macOS with PyObjC") from exc
        return Quartz, Foundation

    @staticmethod
    def _open_document(Quartz, Foundation, pdf_path: Path):
        url = Foundation.NSURL.fileURLWithPath_(str(pdf_path.resolve()))
        doc = Quartz.CGPDFDocumentCreateWithURL(url)
        if doc is None:
            raise ValueError(f"Cannot open PDF: {pdf_path}")
        return doc


class PdfiumPdfRenderer:
    """Render PDF pages to PNG using pypdfium2 for non-macOS environments."""

    def __init__(self, *, dpi: int = 200) -> None:
        self.dpi = dpi

    def page_count(self, pdf_path: Path) -> int:
        pdfium = self._import_pdfium()
        document = pdfium.PdfDocument(str(pdf_path.expanduser()))
        return len(document)

    def render_page(self, pdf_path: Path, *, page_number: int, output_path: Path) -> RenderedPage:
        pdfium = self._import_pdfium()
        document = pdfium.PdfDocument(str(pdf_path.expanduser()))
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"PDF page does not exist: page={page_number} path={pdf_path}")
        page = document[page_number - 1]
        image = page.render(scale=self.dpi / 72.0).to_pil()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        width, height = image.size
        return RenderedPage(
            page_number=page_number,
            image_path=output_path,
            width=int(width),
            height=int(height),
        )

    @staticmethod
    def _import_pdfium():
        try:
            import pypdfium2 as pdfium  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PDF rendering requires macOS Quartz or pypdfium2. "
                "Install the paddleocr-vl extra for cross-platform rendering."
            ) from exc
        return pdfium
