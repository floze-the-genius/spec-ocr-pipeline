from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from spec_pipeline.models import SampleDefinition, SpecificationRow


@dataclass(slots=True)
class ParseDiagnostics:
    backend: str
    sample_id: str
    page_count: int
    parsed_rows: int
    pages: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sample_id": self.sample_id,
            "page_count": self.page_count,
            "parsed_rows": self.parsed_rows,
            "pages": self.pages,
        }


class ParserAdapter(Protocol):
    backend_name: str

    def parse_rows(self, sample: SampleDefinition) -> list[SpecificationRow]: ...

    def diagnose(self, sample: SampleDefinition, limit: int = 40) -> ParseDiagnostics: ...
