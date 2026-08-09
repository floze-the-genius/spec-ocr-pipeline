from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class OcrToken:
    text: str
    page_ref: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "page_ref": self.page_ref,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OcrToken":
        return cls(
            text=str(payload.get("text") or ""),
            page_ref=str(payload.get("page_ref") or ""),
            x0=_float_or_zero(payload.get("x0")),
            y0=_float_or_zero(payload.get("y0")),
            x1=_float_or_zero(payload.get("x1")),
            y1=_float_or_zero(payload.get("y1")),
            confidence=_float_or_none(payload.get("confidence")),
        )


@dataclass(slots=True)
class OcrPage:
    page_index: int
    page_ref: str
    width: int
    height: int
    text: str
    tokens: list[OcrToken]
    provider: str
    image_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "page_ref": self.page_ref,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "provider": self.provider,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OcrPage":
        tokens = payload.get("tokens")
        return cls(
            page_index=_int_or_zero(payload.get("page_index")),
            page_ref=str(payload.get("page_ref") or ""),
            width=_int_or_zero(payload.get("width")),
            height=_int_or_zero(payload.get("height")),
            text=str(payload.get("text") or ""),
            tokens=[OcrToken.from_dict(token) for token in tokens if isinstance(token, dict)]
            if isinstance(tokens, list)
            else [],
            provider=str(payload.get("provider") or "unknown"),
            image_path=str(payload["image_path"])
            if payload.get("image_path") is not None
            else None,
        )


class OcrProvider(Protocol):
    provider_name: str

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        pages: list[int] | None = None,
        output_dir: Path | None = None,
    ) -> list[OcrPage]: ...


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _float_or_zero(value: object) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else 0.0


def _int_or_zero(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
