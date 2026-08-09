"""Concrete OCR providers."""

from spec_pipeline.ocr.providers.dots_mocr import DotsMocrProvider
from spec_pipeline.ocr.providers.paddleocr_vl import PaddleOcrVlProvider
from spec_pipeline.ocr.providers.pdf_text import PdfTextOcrProvider
from spec_pipeline.ocr.providers.tesseract import TesseractOcrProvider

__all__ = [
    "DotsMocrProvider",
    "PaddleOcrVlProvider",
    "PdfTextOcrProvider",
    "TesseractOcrProvider",
]
