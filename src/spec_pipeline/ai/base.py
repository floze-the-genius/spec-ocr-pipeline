from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from spec_pipeline.ocr.base import OcrPage


@dataclass(slots=True)
class AiExtractionRequest:
    sample_id: str
    source_pdf: Path
    pages: list[OcrPage]
    schema_version: str = "specification-row-v1"


@dataclass(slots=True)
class AiExtractionResult:
    rows: list[dict[str, object]]
    warnings: list[str]
    raw_response: dict[str, object]


@dataclass(slots=True)
class AiPatchReviewResult:
    patches: list[dict[str, object]]
    warnings: list[str]
    raw_response: dict[str, object]


class AiExtractionClient(Protocol):
    client_name: str

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult: ...
