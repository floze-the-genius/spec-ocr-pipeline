from __future__ import annotations

from pathlib import Path

from spec_pipeline.ai.base import AiExtractionClient, AiExtractionRequest
from spec_pipeline.ai.schema import AI_SCHEMA_VERSION, ai_row_to_specification_row, validate_ai_rows
from spec_pipeline.models import SampleDefinition, SpecificationRow
from spec_pipeline.ocr.base import OcrPage, OcrProvider
from spec_pipeline.parsers.base import ParseDiagnostics


class UniversalAiOcrParser:
    backend_name = "ai-ocr-universal"

    def __init__(
        self,
        *,
        ocr_provider: OcrProvider,
        ai_client: AiExtractionClient,
        page_limit: int | None = None,
        ocr_output_dir: Path | None = None,
    ) -> None:
        self.ocr_provider = ocr_provider
        self.ai_client = ai_client
        self.page_limit = page_limit
        self.ocr_output_dir = ocr_output_dir
        self._last_pages: list[OcrPage] = []
        self._last_warnings: list[str] = []

    def parse_rows(self, sample: SampleDefinition) -> list[SpecificationRow]:
        pages = self.ocr_provider.extract_pages(
            sample.source_pdf,
            pages=list(range(1, self.page_limit + 1)) if self.page_limit else None,
            output_dir=self.ocr_output_dir,
        )
        request = AiExtractionRequest(
            sample_id=sample.sample_id,
            source_pdf=sample.source_pdf,
            pages=pages,
            schema_version=AI_SCHEMA_VERSION,
        )
        result = self.ai_client.extract_rows(request)
        validate_ai_rows(result.rows)
        self._last_pages = pages
        self._last_warnings = result.warnings
        return [ai_row_to_specification_row(sample.sample_id, row) for row in result.rows]

    def diagnose(self, sample: SampleDefinition, limit: int = 40) -> ParseDiagnostics:
        rows = self.parse_rows(sample)
        pages = self._last_pages[:limit]
        return ParseDiagnostics(
            backend=self.backend_name,
            sample_id=sample.sample_id,
            page_count=len(self._last_pages),
            parsed_rows=len(rows),
            pages=[
                {
                    "page_ref": page.page_ref,
                    "ocr_provider": page.provider,
                    "ai_client": self.ai_client.client_name,
                    "text_chars": len(page.text),
                    "tokens": len(page.tokens),
                    "image_path": page.image_path,
                    "warnings": self._last_warnings,
                }
                for page in pages
            ],
        )
