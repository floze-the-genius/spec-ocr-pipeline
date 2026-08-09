from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, NamedTuple

from spec_pipeline.models import clean_text
from spec_pipeline.ocr.base import OcrPage, OcrToken
from spec_pipeline.ocr.paddle_runtime import ensure_paddle_cuda_available, wants_cuda_device
from spec_pipeline.ocr.render import create_pdf_renderer

DEFAULT_PADDLEOCR_VL_MODEL = "PaddlePaddle/PaddleOCR-VL"
HTML_ROW_RE = re.compile(r"<tr\b[^>]*>", re.IGNORECASE)


@dataclass(slots=True)
class PaddleOcrVlConfig:
    model_name: str = DEFAULT_PADDLEOCR_VL_MODEL
    dpi: int = 200
    device: str | None = None
    pipeline_version: str = "v1"
    vl_rec_backend: str | None = None
    vl_rec_server_url: str | None = None
    vl_rec_api_model_name: str | None = None
    vl_rec_api_key: str | None = None
    vl_rec_max_concurrency: int | None = None
    engine: str | None = None
    cpu_threads: int | None = None
    require_cuda: bool = False
    enable_hpi: bool | None = None
    use_tensorrt: bool | None = None
    precision: str | None = None
    use_layout_detection: bool | None = None
    use_queues: bool | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    split_wide_pages: bool = False
    split_band_height: int = 520
    split_band_overlap: int = 80
    predict_timeout_seconds: int | None = None
    predict_heartbeat_seconds: int = 60
    progress: bool = False


@dataclass(slots=True)
class PaddleOcrVlRegion:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float | None = None


class PaddleOcrVlBand(NamedTuple):
    image_path: Path
    x0: int
    y0: int
    x1: int
    y1: int


class PaddleOcrVlProvider:
    provider_name = "paddleocr-vl"

    def __init__(
        self,
        *,
        config: PaddleOcrVlConfig | None = None,
        renderer: Any | None = None,
        engine: Any | None = None,
    ) -> None:
        self.config = config or PaddleOcrVlConfig()
        self.renderer = renderer or create_pdf_renderer(dpi=self.config.dpi)
        self._engine = engine

    def extract_pages(
        self,
        pdf_path: Path,
        *,
        pages: list[int] | None = None,
        output_dir: Path | None = None,
    ) -> list[OcrPage]:
        resolved_output_dir = output_dir or Path("artifacts") / "ocr" / pdf_path.stem
        page_numbers = pages or list(range(1, self.renderer.page_count(pdf_path) + 1))
        extracted: list[OcrPage] = []
        for page_number in page_numbers:
            started = time.perf_counter()
            self._log_progress(f"page {page_number} started")
            image_path = resolved_output_dir / f"page_{page_number:03d}.png"
            render_started = time.perf_counter()
            self._log_progress(f"page {page_number} render started")
            rendered = self.renderer.render_page(
                pdf_path, page_number=page_number, output_path=image_path
            )
            self._log_progress(
                f"page {page_number} render finished in {time.perf_counter() - render_started:.1f}s"
            )
            if self.config.split_wide_pages and rendered.width > rendered.height:
                bands = create_table_bands(
                    rendered.image_path,
                    band_height=self.config.split_band_height,
                    overlap=self.config.split_band_overlap,
                )
            else:
                bands = []
            if bands:
                self._log_progress(f"page {page_number} split into {len(bands)} table band(s)")
                extracted.append(
                    self._extract_split_page(
                        page_number=page_number,
                        width=rendered.width,
                        height=rendered.height,
                        image_path=rendered.image_path,
                        bands=bands,
                    )
                )
            else:
                result = self._run_paddleocr_vl(rendered.image_path)
                extracted.append(
                    parse_paddleocr_vl_result(
                        result,
                        page_index=page_number - 1,
                        page_ref=str(page_number),
                        width=rendered.width,
                        height=rendered.height,
                        provider=self.provider_name,
                        image_path=str(rendered.image_path),
                    )
                )
            elapsed = time.perf_counter() - started
            self._log_progress(f"page {page_number} finished in {elapsed:.1f}s")
        return extracted

    def _extract_split_page(
        self,
        *,
        page_number: int,
        width: int,
        height: int,
        image_path: Path,
        bands: Sequence[PaddleOcrVlBand],
    ) -> OcrPage:
        tokens: list[OcrToken] = []
        text_blocks: list[str] = []
        for index, band in enumerate(bands, start=1):
            self._log_progress(f"page {page_number} band {index}/{len(bands)} predict started")
            result = self._run_paddleocr_vl(
                band.image_path,
                predict_kwargs={
                    "use_layout_detection": False,
                    "prompt_label": "table",
                    "use_queues": False,
                },
            )
            band_page = parse_paddleocr_vl_result(
                result,
                page_index=page_number - 1,
                page_ref=str(page_number),
                width=band.x1 - band.x0,
                height=band.y1 - band.y0,
                provider=self.provider_name,
                image_path=str(band.image_path),
            )
            band_text = annotate_crossed_out_html_rows(band_page.text, band.image_path)
            text_blocks.append(f"OCR_BAND {index} y={band.y0}-{band.y1}\n{band_text}")
            for token in band_page.tokens:
                tokens.append(
                    OcrToken(
                        text=token.text,
                        page_ref=token.page_ref,
                        x0=token.x0 + band.x0,
                        y0=token.y0 + band.y0,
                        x1=token.x1 + band.x0,
                        y1=token.y1 + band.y0,
                        confidence=token.confidence,
                    )
                )
        return OcrPage(
            page_index=page_number - 1,
            page_ref=str(page_number),
            width=width,
            height=height,
            text="\n\n".join(text_blocks),
            tokens=tokens,
            provider=self.provider_name,
            image_path=str(image_path),
        )

    def _run_paddleocr_vl(
        self, image_path: Path, *, predict_kwargs: dict[str, object] | None = None
    ) -> object:
        self._log_progress("engine load/check started")
        started = time.perf_counter()
        engine = self._get_engine()
        self._log_progress(f"engine ready in {time.perf_counter() - started:.1f}s")
        predict = getattr(engine, "predict", None)
        kwargs = self._predict_kwargs()
        if predict_kwargs is not None:
            kwargs.update(predict_kwargs)
        with _offline_model_env():
            if callable(predict):
                self._log_progress(f"predict started for {image_path.name}")
                predict_started = time.perf_counter()
                result = self._run_predict_with_watchdog(
                    lambda: _materialize_paddle_result(predict(str(image_path), **kwargs)),
                    image_path=image_path,
                    started=predict_started,
                )
                self._log_progress(
                    f"predict finished in {time.perf_counter() - predict_started:.1f}s"
                )
                return result
            if callable(engine):
                self._log_progress(f"callable engine started for {image_path.name}")
                predict_started = time.perf_counter()
                result = self._run_predict_with_watchdog(
                    lambda: _materialize_paddle_result(engine(str(image_path))),
                    image_path=image_path,
                    started=predict_started,
                )
                self._log_progress(
                    f"predict finished in {time.perf_counter() - predict_started:.1f}s"
                )
                return result
        raise RuntimeError("PaddleOCR-VL engine must provide predict(image_path) or be callable")

    def _run_predict_with_watchdog(
        self, predict_call: Any, *, image_path: Path, started: float
    ) -> object:
        timeout = self.config.predict_timeout_seconds
        heartbeat = max(0, self.config.predict_heartbeat_seconds)
        if timeout is None and heartbeat <= 0:
            return predict_call()

        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(("ok", predict_call()))
            except BaseException as exc:  # noqa: BLE001 - propagate worker failure to caller.
                result_queue.put(("error", exc))

        thread = threading.Thread(target=worker, name="paddleocr-vl-predict", daemon=True)
        thread.start()
        next_heartbeat = heartbeat if heartbeat > 0 else None
        while True:
            elapsed = time.perf_counter() - started
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    "PaddleOCR-VL predict timed out after "
                    f"{timeout}s for {image_path.name}. "
                    "This usually means the scanned image fragment is too heavy or the "
                    "Paddle inference backend is stuck. Try a smaller OCR band or use a "
                    "PDF with an embedded text/table layer."
                )
            wait_for = 1.0
            if next_heartbeat is not None:
                wait_for = max(0.1, min(wait_for, next_heartbeat - elapsed))
            if timeout is not None:
                wait_for = max(0.1, min(wait_for, timeout - elapsed))
            try:
                status, payload = result_queue.get(timeout=wait_for)
            except queue.Empty:
                elapsed = time.perf_counter() - started
                if next_heartbeat is not None and elapsed >= next_heartbeat:
                    self._log_progress(
                        f"predict still running for {image_path.name}: {elapsed:.0f}s elapsed"
                    )
                    next_heartbeat += heartbeat
                continue
            if status == "error":
                raise payload  # type: ignore[misc]
            return payload

    def _log_progress(self, message: str) -> None:
        if self.config.progress:
            print(f"paddleocr-vl: {message}", file=sys.stderr, flush=True)

    def _get_engine(self) -> object:
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCRVL  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL provider requires optional dependencies. "
                "Install PaddleOCR and the matching PaddlePaddle runtime for your platform. "
                "Project extra: pip install 'spec-ocr-pipeline[paddleocr-vl]'"
            ) from exc

        model_path = Path(self.config.model_name).expanduser()
        if not model_path.is_dir():
            raise RuntimeError(
                "PaddleOCR-VL provider requires a local model directory. "
                "Use 'spec-pipeline models download paddleocr-vl' or pass an existing path."
            )

        layout_model_dir_name = (
            "PP-DocLayoutV3" if self.config.pipeline_version == "v1.5" else "PP-DocLayoutV2"
        )
        layout_model_path = model_path / layout_model_dir_name
        if not layout_model_path.is_dir():
            raise RuntimeError(
                f"PaddleOCR-VL local model directory is missing {layout_model_dir_name}. "
                "Run 'spec-pipeline models repair paddleocr-vl' and retry."
            )

        selected_device = self.config.device
        if self.config.require_cuda and selected_device is None:
            selected_device = "gpu:0"
        if self.config.require_cuda or wants_cuda_device(selected_device):
            info = ensure_paddle_cuda_available(device=selected_device or "gpu:0")
            self._log_progress(
                "paddle runtime CUDA ready: "
                f"version={info.version}, device_count={info.cuda_device_count}, "
                f"selected={info.selected_device or selected_device}"
            )

        kwargs: dict[str, object] = {
            "pipeline_version": self.config.pipeline_version,
            "layout_detection_model_dir": str(layout_model_path),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        if self.config.vl_rec_backend is not None:
            kwargs["vl_rec_backend"] = self.config.vl_rec_backend
        if self.config.vl_rec_backend in {None, "native"}:
            kwargs["vl_rec_model_dir"] = str(model_path)
        else:
            if self.config.vl_rec_server_url is not None:
                kwargs["vl_rec_server_url"] = self.config.vl_rec_server_url
            if self.config.vl_rec_api_model_name is not None:
                kwargs["vl_rec_api_model_name"] = self.config.vl_rec_api_model_name
            if self.config.vl_rec_api_key is not None:
                kwargs["vl_rec_api_key"] = self.config.vl_rec_api_key
            if self.config.vl_rec_max_concurrency is not None:
                kwargs["vl_rec_max_concurrency"] = self.config.vl_rec_max_concurrency
        if selected_device is not None:
            kwargs["device"] = selected_device
        if self.config.engine is not None:
            kwargs["engine"] = self.config.engine
        if self.config.cpu_threads is not None:
            kwargs["cpu_threads"] = self.config.cpu_threads
        if self.config.enable_hpi is not None:
            kwargs["enable_hpi"] = self.config.enable_hpi
        if self.config.use_tensorrt is not None:
            kwargs["use_tensorrt"] = self.config.use_tensorrt
        if self.config.precision is not None:
            kwargs["precision"] = self.config.precision
        try:
            with _offline_model_env(cache_home=model_path.parent):
                paddle_ocr_vl: Any = PaddleOCRVL
                self._engine = paddle_ocr_vl(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Cannot initialize PaddleOCR-VL. Verify PaddleOCR/PaddlePaddle installation, "
                "device selection, and local model directory."
            ) from exc
        return self._engine

    def _predict_kwargs(self) -> dict[str, object]:
        values = {
            "use_layout_detection": self.config.use_layout_detection,
            "use_queues": self.config.use_queues,
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "max_new_tokens": self.config.max_new_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        return {key: value for key, value in values.items() if value is not None}


@contextmanager
def _offline_model_env(*, cache_home: Path | None = None):
    flag_names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PADDLEOCR_OFFLINE",
        "PADDLE_HUB_OFFLINE",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    )
    names = (*flag_names, "PADDLE_PDX_CACHE_HOME")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in flag_names:
            os.environ[name] = "1"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        if cache_home is not None:
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def create_table_bands(
    image_path: Path, *, band_height: int, overlap: int
) -> list[PaddleOcrVlBand]:
    if band_height <= 0:
        return []
    try:
        from PIL import Image
    except Exception:
        return []

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        content_bbox = _dark_content_bbox(gray)
        if content_bbox is None:
            return []
        x0, y0, x1, y1 = content_bbox
        page_width, _page_height = gray.size
        if x1 - x0 < page_width * 0.75:
            # Some renderers keep thin table grid lines close to white. For landscape
            # engineering tables, preserve the full visual table width instead of
            # cropping to a central text-only bbox.
            x0 = max(0, int(page_width * 0.04))
            x1 = min(page_width, int(page_width * 0.985))
        top, bottom = _table_vertical_range(gray, (x0, y0, x1, y1))
        if bottom - top <= band_height:
            return []
        safe_overlap = max(0, min(overlap, band_height - 1))
        bands: list[PaddleOcrVlBand] = []
        current_top = top
        index = 1
        while current_top < bottom:
            current_bottom = min(current_top + band_height, bottom)
            band_path = image_path.with_name(f"{image_path.stem}_band_{index:02d}.png")
            rgb.crop((x0, current_top, x1, current_bottom)).save(band_path)
            bands.append(PaddleOcrVlBand(band_path, x0, current_top, x1, current_bottom))
            if current_bottom >= bottom:
                break
            current_top = current_bottom - safe_overlap
            index += 1
    return bands


def annotate_crossed_out_html_rows(text: str, image_path: Path) -> str:
    html_row_count = len(HTML_ROW_RE.findall(text))
    if html_row_count == 0:
        return text
    crossed_rows = detect_crossed_out_html_rows(image_path, html_row_count=html_row_count)
    if not crossed_rows:
        return text
    marked = _mark_html_rows(text, crossed_rows)
    row_labels = ",".join(str(row_index) for row_index in crossed_rows)
    return f"CROSSED_OUT_HTML_ROWS: {row_labels}\n{marked}"


def detect_crossed_out_html_rows(image_path: Path, *, html_row_count: int) -> list[int]:
    if html_row_count <= 0:
        return []
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return []

    try:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
    except Exception:
        return []

    height, width = pixels.shape[:2]
    if width <= 0 or height <= 0:
        return []
    red = (
        (pixels[:, :, 0] > 120)
        & (pixels[:, :, 1] < 135)
        & (pixels[:, :, 2] < 135)
        & ((pixels[:, :, 0].astype(int) - pixels[:, :, 1].astype(int)) > 30)
        & ((pixels[:, :, 0].astype(int) - pixels[:, :, 2].astype(int)) > 30)
    )
    red_counts = red.sum(axis=1)
    red_rows = np.where(red_counts > max(8, int(width * 0.02)))[0].tolist()
    red_groups = _index_groups([int(row) for row in red_rows], max_gap=4)
    if not red_groups:
        return []

    black = (pixels[:, :, 0] < 100) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 100)
    black_counts = black.sum(axis=1)
    separator_rows = np.where(black_counts > max(20, int(width * 0.25)))[0].tolist()
    separator_groups = _index_groups([int(row) for row in separator_rows], max_gap=5)
    separators = [(start + end) // 2 for start, end in separator_groups]
    if len(separators) >= 2:
        top = separators[0]
        bottom = separators[-1]
    else:
        top = 0
        bottom = height - 1
    if bottom <= top:
        return []

    crossed: set[int] = set()
    for start, end in red_groups:
        center = (start + end) / 2
        ratio = (center - top) / (bottom - top)
        row_index = int(ratio * html_row_count) + 1
        row_index = max(1, min(html_row_count, row_index))
        crossed.add(row_index)
    return sorted(crossed)


def _mark_html_rows(text: str, crossed_rows: list[int]) -> str:
    crossed = set(crossed_rows)
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        tag = match.group(0)
        if counter not in crossed or "data-status" in tag:
            return tag
        return tag[:-1] + ' data-status="crossed_out">'

    return HTML_ROW_RE.sub(replace, text)


def _dark_content_bbox(gray_image: Any) -> tuple[int, int, int, int] | None:
    mask = gray_image.point(lambda pixel: 255 if pixel < 245 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def _table_vertical_range(gray_image: Any, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    try:
        import numpy as np
    except Exception:
        return (y0, y1)
    pixels = np.asarray(gray_image)
    dark = pixels < 180
    row_counts = dark[:, x0:x1].sum(axis=1)
    threshold = max(10, int((x1 - x0) * 0.35))
    groups = _index_groups(np.where(row_counts > threshold)[0].tolist(), max_gap=4)
    centers = [(start + end) // 2 for start, end in groups]
    if len(centers) < 5:
        return (y0, y1)
    # Keep table headers. PaddleOCR-VL needs the column/header context to avoid
    # dropping section labels on scanned specification tables.
    top = max(y0, centers[0] - 20)
    # Exclude drawing title blocks below the table when a bottom table line is visible.
    bottom = min(y1, centers[-3] + 20) if len(centers) >= 7 else y1
    if bottom - top < 100:
        return (y0, y1)
    return (top, bottom)


def _index_groups(values: list[int], *, max_gap: int) -> list[tuple[int, int]]:
    if not values:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current - previous > max_gap:
            groups.append((start, previous))
            start = current
        previous = current
    groups.append((start, previous))
    return groups


def parse_paddleocr_vl_result(
    result: object,
    *,
    page_index: int,
    page_ref: str,
    width: int,
    height: int,
    provider: str,
    image_path: str | None = None,
) -> OcrPage:
    regions = _extract_regions(result, width=width, height=height)
    tokens = [
        OcrToken(
            text=region.text,
            page_ref=page_ref,
            x0=region.x0,
            y0=region.y0,
            x1=region.x1,
            y1=region.y1,
            confidence=region.confidence,
        )
        for region in regions
    ]
    text = "\n".join(
        f"[{region.x0:.1f},{region.y0:.1f},{region.x1:.1f},{region.y1:.1f}] {region.text}"
        for region in regions
    )
    return OcrPage(
        page_index=page_index,
        page_ref=page_ref,
        width=width,
        height=height,
        text=text,
        tokens=tokens,
        provider=provider,
        image_path=image_path,
    )


def _materialize_paddle_result(result: object) -> object:
    if isinstance(result, (str, bytes, dict, list, tuple)):
        return result
    if isinstance(result, Iterable):
        return list(result)
    return result


def _extract_regions(payload: object, *, width: int, height: int) -> list[PaddleOcrVlRegion]:
    if payload is None:
        return []
    if isinstance(payload, (list, tuple)):
        regions: list[PaddleOcrVlRegion] = []
        for item in payload:
            regions.extend(_extract_regions(item, width=width, height=height))
        return regions

    mapping = _as_mapping(payload)
    if mapping is None:
        return []

    paired_regions = _extract_paired_ocr_regions(mapping, width=width, height=height)
    if paired_regions:
        return paired_regions

    child_regions: list[PaddleOcrVlRegion] = []
    for key in (
        "blocks",
        "layout",
        "results",
        "res",
        "pages",
        "items",
        "cells",
        "tables",
        "parsing_res_list",
        "layout_det_res",
        "overall_ocr_res",
        "ocr_res",
        "text_res",
    ):
        if key in mapping:
            child_regions.extend(_extract_regions(mapping[key], width=width, height=height))
    if child_regions:
        return child_regions

    text = _extract_text(mapping)
    if text is None:
        return []
    bbox = _bbox_from_payload(mapping, width=width, height=height)
    if bbox is None:
        bbox = (0.0, 0.0, float(width), float(height))
    x0, y0, x1, y1 = bbox
    return [
        PaddleOcrVlRegion(
            text=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            confidence=_confidence_from_payload(mapping),
        )
    ]


def _as_mapping(payload: object) -> dict[str, object] | None:
    json_payload = getattr(payload, "json", None)
    if isinstance(json_payload, dict):
        return json_payload
    if isinstance(payload, dict):
        return payload
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, dict) else None
    attr_payload = {
        key: getattr(payload, key)
        for key in (
            "text",
            "content",
            "block_content",
            "bbox",
            "box",
            "block_bbox",
            "coordinate",
            "polygon",
            "poly",
            "points",
            "polygon_points",
            "confidence",
            "score",
        )
        if hasattr(payload, key)
    }
    return attr_payload or None


def _extract_text(payload: dict[str, object]) -> str | None:
    for key in ("text", "content", "block_content", "ocr_text", "markdown", "html"):
        value = payload.get(key)
        if isinstance(value, str):
            return clean_text(value)
    return None


def _bbox_from_payload(
    payload: dict[str, object], *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "box", "block_bbox", "dt_bbox", "rec_box", "coordinate", "rect"):
        bbox = _rect_from_sequence(payload.get(key), width=width, height=height)
        if bbox is not None:
            return bbox
    for key in ("polygon", "poly", "points"):
        bbox = _rect_from_polygon(payload.get(key), width=width, height=height)
        if bbox is not None:
            return bbox
    return None


def _extract_paired_ocr_regions(
    payload: dict[str, object], *, width: int, height: int
) -> list[PaddleOcrVlRegion]:
    texts = _first_present(payload, "rec_texts", "texts")
    boxes = _first_present(payload, "rec_boxes", "dt_boxes", "boxes")
    polygons = _first_present(payload, "rec_polys", "dt_polys", "polys")
    scores = _first_present(payload, "rec_scores", "scores")
    text_values = _to_sequence(texts)
    if text_values is None:
        return []
    coordinate_values = _to_sequence(boxes) or _to_sequence(polygons)
    if coordinate_values is None:
        return []
    score_values = _to_sequence(scores)

    regions: list[PaddleOcrVlRegion] = []
    for index, text_value in enumerate(text_values):
        if index >= len(coordinate_values):
            break
        text = clean_text(text_value) if isinstance(text_value, str) else None
        if text is None:
            continue
        coordinate_value = coordinate_values[index]
        bbox = _rect_from_sequence(coordinate_value, width=width, height=height)
        if bbox is None:
            bbox = _rect_from_polygon(coordinate_value, width=width, height=height)
        if bbox is None:
            continue
        confidence = None
        if score_values is not None and index < len(score_values):
            confidence = _float_or_none(score_values[index])
        x0, y0, x1, y1 = bbox
        regions.append(PaddleOcrVlRegion(text, x0, y0, x1, y1, confidence))
    return regions


def _first_present(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _rect_from_sequence(
    value: object, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    sequence = _to_sequence(value)
    if sequence is None or len(sequence) != 4:
        return None
    parsed = [_float_or_none(item) for item in sequence]
    if any(item is None for item in parsed):
        return None
    x0, y0, x1, y1 = [float(item) for item in parsed if item is not None]
    return _scale_bbox_if_normalized((x0, y0, x1, y1), width=width, height=height)


def _rect_from_polygon(
    value: object, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    points: list[tuple[float, float]] = []
    sequence = _to_sequence(value)
    if sequence is None:
        return None
    for item in sequence:
        point = _to_sequence(item)
        if point is None or len(point) < 2:
            return None
        x = _float_or_none(point[0])
        y = _float_or_none(point[1])
        if x is None or y is None:
            return None
        points.append((x, y))
    if not points:
        return None
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return _scale_bbox_if_normalized(
        (min(x_values), min(y_values), max(x_values), max(y_values)),
        width=width,
        height=height,
    )


def _scale_bbox_if_normalized(
    bbox: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    if all(0.0 <= value <= 1.0 for value in bbox):
        return (x0 * width, y0 * height, x1 * width, y1 * height)
    return bbox


def _confidence_from_payload(payload: dict[str, object]) -> float | None:
    for key in ("confidence", "score", "conf", "prob"):
        confidence = _float_or_none(payload.get(key))
        if confidence is not None:
            return confidence
    return None


def _float_or_none(value: object) -> float | None:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_sequence(value: object) -> list[object] | None:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return None
