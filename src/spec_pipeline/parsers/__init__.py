"""Input parsing layer for PDF/OCR sources."""

from spec_pipeline.parsers.ai_ocr_universal import UniversalAiOcrParser
from spec_pipeline.parsers.pdfminer_text import PdfMinerSpec8Parser

__all__ = ["PdfMinerSpec8Parser", "UniversalAiOcrParser"]
