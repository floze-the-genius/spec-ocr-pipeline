from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

from spec_pipeline.ocr.base import OcrPage
from spec_pipeline.ocr.render import create_pdf_renderer

DEFAULT_DOTS_MOCR_MODEL = "mlx-community/dots.mocr-4bit"
DOTS_MOCR_IGNORABLE_ROW_RE = re.compile(
    r"(?:\bизм\.?\b|\bкол\.?(?:\s*уч)?\b|\bлист\b|№\s*док|подп\.?|дата|стадия|листов|"
    r"строительство|многоквартирные|спецификация оборудования|"
    r"жил(?:ой|ые|ых)?\s+дом|разработал|н\.?\s*контр|гип|исполн|испопн|ооо\s*[\"«])",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DotsMocrConfig:
    model_name: str = DEFAULT_DOTS_MOCR_MODEL
    backend: str = "mlx-vlm"
    model_cache_dir: Path | None = None
    device: str | None = None
    device_map: str | None = None
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    trust_model_code: bool = False
    dpi: int = 200
    crop_to_content: bool = True
    crop_threshold: int = 12
    crop_padding: int = 40
    upscale: float = 2.0
    max_image_edge: int = 2600
    max_tokens: int = 4096
    temperature: float = 0.0
    progress: bool = False
    lock_path: Path | None = Path("artifacts/locks/dots-mocr.lock")
    split_table_bands: bool = True
    band_min_gap_px: int = 140
    band_padding_px: int = 50
    max_bands: int = 4


class DotsMocrGenerator(Protocol):
    def generate(
        self,
        *,
        model_name: str,
        model_cache_dir: Path | None,
        image_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class DotsMocrProvider:
    provider_name = "dots-mocr"

    def __init__(
        self,
        *,
        config: DotsMocrConfig | None = None,
        renderer: Any | None = None,
        generator: DotsMocrGenerator | None = None,
    ) -> None:
        self.config = config or DotsMocrConfig()
        self.renderer = renderer or create_pdf_renderer(dpi=self.config.dpi)
        self.generator = generator or create_dots_mocr_generator(self.config)

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
            rendered_path = resolved_output_dir / f"page_{page_number:03d}.png"
            rendered = self.renderer.render_page(
                pdf_path, page_number=page_number, output_path=rendered_path
            )
            image_path, width, height = preprocess_table_image(
                rendered.image_path,
                crop_to_content=self.config.crop_to_content,
                crop_threshold=self.config.crop_threshold,
                crop_padding=self.config.crop_padding,
                upscale=self.config.upscale,
                max_image_edge=self.config.max_image_edge,
            )
            text = self._extract_html(image_path)
            extracted.append(
                OcrPage(
                    page_index=page_number - 1,
                    page_ref=str(page_number),
                    width=width,
                    height=height,
                    text=format_dots_mocr_text(text),
                    tokens=[],
                    provider=self.provider_name,
                    image_path=str(image_path),
                )
            )
            self._log_progress(
                f"page {page_number} finished in {time.perf_counter() - started:.1f}s"
            )
        return extracted

    def _extract_html(self, image_path: Path) -> str:
        prompt = build_dots_mocr_prompt()
        with _dots_mocr_process_lock(self.config.lock_path):
            full_text = validate_dots_mocr_html(
                self.generator.generate(
                    model_name=self.config.model_name,
                    model_cache_dir=self.config.model_cache_dir,
                    image_path=image_path,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            )
            if not self.config.split_table_bands:
                return full_text
            band_paths = create_table_band_images(
                image_path,
                min_gap_px=self.config.band_min_gap_px,
                padding_px=self.config.band_padding_px,
                max_bands=self.config.max_bands,
            )
            if len(band_paths) < 2:
                return full_text
            band_texts: list[str] = []
            for band_path in band_paths:
                band_texts.append(
                    validate_dots_mocr_html(
                        self.generator.generate(
                            model_name=self.config.model_name,
                            model_cache_dir=self.config.model_cache_dir,
                            image_path=band_path,
                            prompt=prompt,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                        )
                    )
                )
        merged = merge_dots_mocr_html_tables(full_text, band_texts)
        full_quantity_count = _dots_quantity_like_count(full_text)
        merged_quantity_count = _dots_quantity_like_count(merged)
        if merged_quantity_count > full_quantity_count:
            self._log_progress(
                "band split improved quantity coverage: "
                f"{full_quantity_count} -> {merged_quantity_count} quantity cells"
            )
            return merged
        return full_text

    def _log_progress(self, message: str) -> None:
        if self.config.progress:
            print(f"dots-mocr: {message}", file=sys.stderr, flush=True)


def create_dots_mocr_generator(config: DotsMocrConfig) -> DotsMocrGenerator:
    backend = config.backend.lower().strip()
    if not config.trust_model_code:
        raise RuntimeError(
            "dots.mocr model code execution is disabled by default. Review an immutable local "
            "model snapshot, pass its directory with --dots-model, and opt in with "
            "--dots-trust-model-code."
        )
    if backend == "mlx-vlm":
        return MlxVlmDotsMocrGenerator()
    if backend == "transformers":
        return TransformersDotsMocrGenerator(
            device=config.device,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            trust_model_code=config.trust_model_code,
        )
    raise RuntimeError(
        f"Unsupported dots-mocr backend '{config.backend}'. "
        "Use 'mlx-vlm' on Apple Silicon or 'transformers' for Windows/Linux checks."
    )


class MlxVlmDotsMocrGenerator:
    def __init__(self) -> None:
        self._loaded: tuple[str, Any, Any] | None = None

    def generate(
        self,
        *,
        model_name: str,
        model_cache_dir: Path | None,
        image_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise RuntimeError(
                "dots-mocr mlx-vlm backend requires mlx-vlm. Install the macos-local "
                "extra on Apple Silicon, or use --dots-backend transformers on "
                "Windows/Linux."
            ) from exc

        resolved_model = _resolve_dots_model(model_name, model_cache_dir)
        model_key = str(resolved_model)
        if self._loaded is None or self._loaded[0] != model_key:
            model, processor = load(model_key)
            self._loaded = (model_key, model, processor)
        _, model, processor = self._loaded
        formatted_prompt = apply_chat_template(
            processor,
            getattr(model, "config", {}),
            prompt,
            add_generation_prompt=True,
            num_images=1,
            enable_thinking=False,
        )
        result = generate(
            model,
            processor,
            formatted_prompt,
            image=[str(image_path)],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        return str(getattr(result, "text", result))


DefaultDotsMocrGenerator = MlxVlmDotsMocrGenerator


class TransformersDotsMocrGenerator:
    def __init__(
        self,
        *,
        device: str | None = None,
        device_map: str | None = None,
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
        trust_model_code: bool = False,
    ) -> None:
        self.device = device
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.trust_model_code = trust_model_code
        self._loaded: tuple[str, Any, Any, Any] | None = None

    def generate(
        self,
        *,
        model_name: str,
        model_cache_dir: Path | None,
        image_path: Path,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "dots-mocr transformers backend requires torch, transformers, and Pillow. "
                "On Windows run CHECK_DOTS_MOCR_WINDOWS.cmd to install/check them."
            ) from exc

        resolved_model = _resolve_dots_model(model_name, model_cache_dir)
        model_key = str(resolved_model)
        requested_device = _resolve_transformers_device(torch, self.device)
        load_key = (model_key, requested_device, self.device_map, self.torch_dtype)
        if self._loaded is None or self._loaded[0] != repr(load_key):
            processor = AutoProcessor.from_pretrained(
                model_key, trust_remote_code=self.trust_model_code
            )
            load_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_model_code}
            dtype = _resolve_torch_dtype(torch, self.torch_dtype, requested_device)
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            if self.device_map:
                load_kwargs["device_map"] = self.device_map
            elif requested_device == "cuda":
                load_kwargs["device_map"] = "auto"
            model = _load_transformers_model(AutoModelForCausalLM, model_key, load_kwargs)
            if "device_map" not in load_kwargs:
                model = model.to(requested_device)
            model.eval()
            self._loaded = (repr(load_key), model, processor, torch)

        _, model, processor, torch_module = self._loaded
        image = Image.open(image_path).convert("RGB")
        formatted_prompt = _format_transformers_prompt(processor, image, prompt)
        inputs = processor(
            text=[formatted_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        target_device = _first_model_device(model, requested_device)
        inputs = _move_tensors(inputs, target_device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with torch_module.inference_mode():
            generated_ids = model.generate(**inputs, **generate_kwargs)
        input_token_count = int(inputs["input_ids"].shape[-1])
        completion_ids = generated_ids[:, input_token_count:]
        tokenizer = getattr(processor, "tokenizer", processor)
        decoded = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0] if decoded else "")


def _resolve_transformers_device(torch: Any, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(torch: Any, requested: str, device: str) -> Any | None:
    normalized = requested.lower().strip()
    if normalized in {"", "auto"}:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device in {"cuda", "mps"}:
            return torch.float16
        return None
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise RuntimeError(
            f"Unsupported --dots-torch-dtype '{requested}'. "
            "Use auto, float16, bfloat16, or float32."
        )
    return dtype_map[normalized]


def _load_transformers_model(
    model_cls: Any,
    model_key: str,
    load_kwargs: dict[str, Any],
) -> Any:
    try:
        return model_cls.from_pretrained(model_key, **load_kwargs)
    except TypeError:
        retry_kwargs = dict(load_kwargs)
        retry_kwargs.pop("attn_implementation", None)
        return model_cls.from_pretrained(model_key, **retry_kwargs)


def _format_transformers_prompt(processor: Any, image: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return str(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return "<|user|><|img|><|imgpad|><|endofimg|>" + prompt + "<|endofuser|><|assistant|>"


def _first_model_device(model: Any, fallback: str) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def _move_tensors(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def build_dots_mocr_prompt() -> str:
    return (
        "Convert only the main Russian construction specification table into clean HTML table. "
        "Ignore side stamps, page headers, page footers, signatures, and bottom title blocks. "
        "Preserve the visual row order and exactly 9 visual columns when the source table has "
        "the standard specification columns. Extract only visible text, including red "
        "handwritten or changed numbers. Do not infer missing values, do not translate, and do "
        "not merge different visual rows. Empty cells must stay empty. Important: when a red "
        "quantity is visually split across multiple lines inside one cell, concatenate the "
        "pieces in reading order without inventing decimal commas. Do not emit fully empty grid "
        "rows, but continue after visual separators to the next data section on the same page. "
        "Return HTML only and stop after the last row of the main specification table."
    )


def format_dots_mocr_text(text: str) -> str:
    cleaned = _strip_mlx_vlm_wrappers(text).strip()
    return "\n".join(["DOTS_MOCR_HTML:", cleaned])


def validate_dots_mocr_html(text: str) -> str:
    cleaned = _strip_mlx_vlm_wrappers(text).strip()
    rows = parse_dots_mocr_html_rows(cleaned)
    kept_rows: list[list[str]] = []
    bad_widths: list[tuple[int, int]] = []
    title_block_started = False
    for index, row in enumerate(rows, start=1):
        if not any(cell.strip() for cell in row):
            continue
        if _is_ignorable_dots_row(row):
            title_block_started = True
            continue
        normalized_row = _normalize_dots_row_width(row)
        if normalized_row is None:
            if title_block_started:
                title_block_started = True
                continue
            bad_widths.append((index, len(row)))
            continue
        kept_rows.append(normalized_row)
    if bad_widths:
        sample = ", ".join(f"row {index}: {width} cells" for index, width in bad_widths[:5])
        raise RuntimeError(
            f"dots-mocr returned malformed table; expected 9 cells per row: {sample}"
        )
    if not kept_rows:
        raise RuntimeError("dots-mocr returned no HTML table rows")
    return _rows_to_dots_html(kept_rows)


def parse_dots_mocr_html_rows(text: str) -> list[list[str]]:
    parser = _DotsMocrHtmlTableParser()
    parser.feed(text)
    return parser.rows


def merge_dots_mocr_html_tables(full_text: str, band_texts: list[str]) -> str:
    full_rows = parse_dots_mocr_html_rows(full_text)
    merged_rows: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    loose_seen: set[tuple[str, str, str, str]] = set()
    for row in full_rows:
        if _is_dots_header_row(row):
            continue
        key = tuple(_normalize_cell_for_key(cell) for cell in row)
        seen.add(key)
        loose_seen.add(_dots_loose_row_key(row))
        merged_rows.append(row)
    for text in band_texts:
        for row in parse_dots_mocr_html_rows(text):
            if _is_dots_header_row(row):
                continue
            key = tuple(_normalize_cell_for_key(cell) for cell in row)
            loose_key = _dots_loose_row_key(row)
            if key in seen or loose_key in loose_seen:
                continue
            seen.add(key)
            loose_seen.add(loose_key)
            merged_rows.append(row)
    if not merged_rows:
        return full_text
    header_rows = [row for row in full_rows if _is_dots_header_row(row)]
    return _rows_to_dots_html(header_rows + merged_rows)


def create_table_band_images(
    image_path: Path,
    *,
    min_gap_px: int,
    padding_px: int,
    max_bands: int,
) -> list[Path]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("dots-mocr provider requires Pillow for band preprocessing") from exc

    image = Image.open(image_path).convert("RGB")
    bands = detect_table_content_bands(
        image,
        min_gap_px=min_gap_px,
        padding_px=padding_px,
        max_bands=max_bands,
    )
    if len(bands) < 2:
        return []
    paths: list[Path] = []
    for index, (top, bottom) in enumerate(bands, start=1):
        output_path = image_path.with_name(f"{image_path.stem}_band_{index:03d}{image_path.suffix}")
        image.crop((0, top, image.width, bottom)).save(output_path)
        paths.append(output_path)
    return paths


def detect_table_content_bands(
    image: Any,
    *,
    min_gap_px: int,
    padding_px: int,
    max_bands: int,
) -> list[tuple[int, int]]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    sample_step = max(1, width // 1200)
    lower_dark_count = max(25, int(width * 0.012))
    upper_dark_count = max(lower_dark_count + 1, int(width * 0.25))
    active_rows: list[int] = []
    pixels = grayscale.load()
    for y in range(height):
        dark_count = 0
        for x in range(0, width, sample_step):
            if pixels[x, y] < 180:
                dark_count += sample_step
        if lower_dark_count <= dark_count <= upper_dark_count:
            active_rows.append(y)
    raw_bands = _merge_active_rows(active_rows, max_gap_px=25)
    bands = _merge_nearby_bands(raw_bands, max_gap_px=55)
    min_height = max(90, int(height * 0.055))
    filtered: list[tuple[int, int]] = []
    for top, bottom in bands:
        band_height = bottom - top
        if band_height < min_height:
            continue
        if top > int(height * 0.86) and band_height < int(height * 0.12):
            continue
        filtered.append((max(0, top - padding_px), min(height, bottom + padding_px)))
    if len(filtered) > max_bands:
        return []
    return filtered


def preprocess_table_image(
    image_path: Path,
    *,
    crop_to_content: bool,
    crop_threshold: int,
    crop_padding: int,
    upscale: float,
    max_image_edge: int,
) -> tuple[Path, int, int]:
    try:
        from PIL import Image, ImageChops, ImageEnhance
    except ImportError as exc:
        raise RuntimeError("dots-mocr provider requires Pillow for image preprocessing") from exc

    image = Image.open(image_path).convert("RGB")
    if crop_to_content:
        background = Image.new("RGB", image.size, (255, 255, 255))
        diff = ImageChops.difference(image, background).convert("L")
        mask = diff.point(lambda value: 255 if value > crop_threshold else 0)
        bbox = mask.getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            box = (
                max(0, left - crop_padding),
                max(0, top - crop_padding),
                min(image.width, right + crop_padding),
                min(image.height, bottom + crop_padding),
            )
            image = image.crop(box)

    scale = _resolve_upscale(image, upscale)
    if scale != 1:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        image = ImageEnhance.Sharpness(image).enhance(1.4)

    if max_image_edge > 0:
        long_edge = max(image.size)
        if long_edge > max_image_edge:
            factor = max_image_edge / long_edge
            image = image.resize(
                (max(1, int(image.width * factor)), max(1, int(image.height * factor))),
                Image.Resampling.LANCZOS,
            )

    suffix_parts = []
    if crop_to_content:
        suffix_parts.append("crop")
    if scale != 1:
        suffix_parts.append(f"{scale:g}x")
    output_path = image_path.with_name(
        f"{image_path.stem}_{'_'.join(suffix_parts) or 'preprocessed'}{image_path.suffix}"
    )
    image.save(output_path)
    return output_path, int(image.width), int(image.height)


def _strip_mlx_vlm_wrappers(text: str) -> str:
    match = re.search(r"(<(?:html|table)\b.*)", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else text


def _normalize_dots_row_width(row: list[str]) -> list[str] | None:
    if len(row) == 9:
        return row
    if len(row) > 9 and all(not cell for cell in row[9:]):
        return row[:9]
    if 6 <= len(row) < 9:
        return row + [""] * (9 - len(row))
    return None


def _is_ignorable_dots_row(row: list[str]) -> bool:
    joined = " ".join(cell for cell in row if cell).strip()
    if not joined:
        return True
    if DOTS_MOCR_IGNORABLE_ROW_RE.search(joined):
        return True
    return bool(re.fullmatch(r"(?:\d{2}/\d{2}|\d{2}-[А-ЯA-Z]{1,3})", joined))


def _rows_to_dots_html(rows: list[list[str]]) -> str:
    rendered_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(cell)}</td>" for cell in row)
        rendered_rows.append(f"<tr>{cells}</tr>")
    return "<html><body><table>" + "".join(rendered_rows) + "</table></body></html>"


def _dots_data_row_count(text: str) -> int:
    return sum(1 for row in parse_dots_mocr_html_rows(text) if not _is_dots_header_row(row))


def _dots_quantity_like_count(text: str) -> int:
    count = 0
    for row in parse_dots_mocr_html_rows(text):
        if len(row) < 7 or _is_dots_header_row(row):
            continue
        quantity = row[6].strip()
        if re.search(r"\d", quantity):
            count += 1
    return count


def _is_dots_header_row(row: list[str]) -> bool:
    if len(row) != 9:
        return False
    first = _normalize_cell_for_key(row[0])
    if first in {"позиция", "поз"}:
        return True
    return [_normalize_cell_for_key(cell) for cell in row] == [str(index) for index in range(1, 10)]


def _normalize_cell_for_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _dots_loose_row_key(row: list[str]) -> tuple[str, str, str, str]:
    padded = row + [""] * max(0, 9 - len(row))
    return (
        _normalize_cell_for_key(padded[0]),
        _normalize_cell_for_key(padded[1])[:48],
        _normalize_cell_for_key(padded[5]),
        _normalize_cell_for_key(padded[6]),
    )


def _merge_active_rows(rows: list[int], *, max_gap_px: int) -> list[tuple[int, int]]:
    if not rows:
        return []
    bands: list[tuple[int, int]] = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row - previous <= max_gap_px:
            previous = row
            continue
        bands.append((start, previous + 1))
        start = previous = row
    bands.append((start, previous + 1))
    return bands


def _merge_nearby_bands(bands: list[tuple[int, int]], *, max_gap_px: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for top, bottom in bands:
        if merged and top - merged[-1][1] <= max_gap_px:
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))
    return merged


def _resolve_dots_model(model_name: str, model_cache_dir: Path | None) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    raise RuntimeError(
        "Automatic dots.mocr model downloads are disabled because the model may contain "
        "executable Python code. Download and review an immutable revision separately, then "
        "pass its local directory with --dots-model."
    )


@contextmanager
def _dots_mocr_process_lock(lock_path: Path | None) -> Iterator[None]:
    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import msvcrt

        with lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _DotsMocrHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "table":
            self._table_depth += 1
            return
        if self._table_depth <= 0:
            return
        if lowered == "tr":
            self._row = []
        elif lowered in {"td", "th"}:
            self._cell = []
            self._cell_colspan = _parse_colspan(attrs)
        elif lowered == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth <= 0:
            return
        lowered = tag.lower()
        if lowered in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            for _ in range(1, self._cell_colspan):
                self._row.append("")
            self._cell = None
            self._cell_colspan = 1
        elif lowered == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None
        elif lowered == "table":
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _parse_colspan(attrs: list[tuple[str, str | None]]) -> int:
    for name, value in attrs:
        if name.lower() != "colspan" or value is None:
            continue
        try:
            return max(1, min(9, int(value)))
        except ValueError:
            return 1
    return 1


def _resolve_upscale(image: Any, requested_upscale: float) -> float:
    if requested_upscale > 0:
        return max(0.1, requested_upscale)
    return 2.0 if _has_red_markup(image) else 1.0


def _has_red_markup(image: Any) -> bool:
    sample_step = max(1, max(image.size) // 800)
    red_pixels = 0
    for y in range(0, image.height, sample_step):
        for x in range(0, image.width, sample_step):
            r, g, b = image.getpixel((x, y))
            if r >= 150 and g <= 110 and b <= 110 and r - max(g, b) >= 45:
                red_pixels += 1
                if red_pixels >= 12:
                    return True
    return False
