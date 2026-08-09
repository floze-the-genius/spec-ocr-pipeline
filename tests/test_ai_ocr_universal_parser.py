from __future__ import annotations

from pathlib import Path

from spec_pipeline.ai.base import AiExtractionRequest, AiExtractionResult
from spec_pipeline.models import SampleDefinition
from spec_pipeline.ocr.base import OcrPage, OcrToken
from spec_pipeline.parsers.ai_ocr_universal import UniversalAiOcrParser


class FakeOcrProvider:
    provider_name = "fake-ocr"

    def extract_pages(self, pdf_path: Path, *, pages=None, output_dir=None):
        return [
            OcrPage(
                page_index=0,
                page_ref="1",
                width=1000,
                height=800,
                text="Позиция Наименование Количество\n1 Диффузор 2",
                tokens=[
                    OcrToken(
                        text="Диффузор",
                        page_ref="1",
                        x0=10,
                        y0=20,
                        x1=80,
                        y1=35,
                        confidence=91.0,
                    )
                ],
                provider=self.provider_name,
                image_path="page_001.png",
            )
        ]


class FakeAiClient:
    client_name = "fake-ai"

    def extract_rows(self, request: AiExtractionRequest) -> AiExtractionResult:
        return AiExtractionResult(
            rows=[
                {
                    "page_ref": "1",
                    "section": "Вентиляция",
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
                }
            ],
            warnings=[],
            raw_response={},
        )


def _sample(tmp_path: Path) -> SampleDefinition:
    return SampleDefinition(
        sample_id="sample",
        title="Sample",
        source_pdf=tmp_path / "input.pdf",
        ground_truth_tsv=None,
        expected_xlsx=None,
        visual_xlsx=None,
    )


def test_universal_ai_ocr_parser_converts_ai_rows(tmp_path: Path) -> None:
    parser = UniversalAiOcrParser(ocr_provider=FakeOcrProvider(), ai_client=FakeAiClient())

    rows = parser.parse_rows(_sample(tmp_path))

    assert len(rows) == 1
    assert rows[0].sample_id == "sample"
    assert rows[0].section == "Вентиляция"
    assert rows[0].name == "Диффузор"
    assert str(rows[0].quantity) == "2"


def test_universal_ai_ocr_parser_diagnostics_include_provider_metadata(tmp_path: Path) -> None:
    parser = UniversalAiOcrParser(ocr_provider=FakeOcrProvider(), ai_client=FakeAiClient())

    diagnostics = parser.diagnose(_sample(tmp_path))

    assert diagnostics.backend == "ai-ocr-universal"
    assert diagnostics.page_count == 1
    assert diagnostics.parsed_rows == 1
    assert diagnostics.pages[0]["ocr_provider"] == "fake-ocr"
    assert diagnostics.pages[0]["ai_client"] == "fake-ai"
