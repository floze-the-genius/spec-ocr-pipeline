from __future__ import annotations

import re
import warnings
from copy import copy
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

from openpyxl import load_workbook
from openpyxl.cell import Cell
from openpyxl.styles import Border, Font, Side
from openpyxl.utils import column_index_from_string
from openpyxl.workbook.properties import CalcProperties
from openpyxl.worksheet.worksheet import Worksheet

from spec_pipeline.exporters.customer_fields import (
    customer_description,
    customer_note,
    normalize_customer_text,
    normalize_customer_unit,
)
from spec_pipeline.exporters.excel_safe import excel_safe_text, excel_safe_value
from spec_pipeline.io import load_rows_from_jsonl
from spec_pipeline.models import SpecificationRow

MAIN_SHEET_NAME = "Quote"
BODY_START_ROW = 11
HEADER_NUMBER_ROW = 10
DEFAULT_KP_PREFIX_LABELS: tuple[str, ...] = ()
SUMMARY_LABEL = "Итого стоимость оборудования и материалов:"
SUMMARY_LABELS = [
    "Итого стоимость оборудования и материалов:",
    "Итого стоимость оборудования:",
    "Итого стоимость материалов:",
    "Стоимость монтажных работ:",
    "Расходные и крепежные материалы:",
    "Транспортные и накладные расходы:",
    "ИТОГО:",
    "в т.ч. НДС 22%",
]
COMMERCIAL_HEADER_ROWS = 5
VISIBLE_COLUMN_COUNT = 9
HEADER_TEMPLATE_PATTERN = "*КП*спец*8*.xlsx"
ASCII_KP_TEMPLATE_NAME = "kp-template.xlsx"
ASCII_KP_HEADER_TEMPLATE_NAME = "kp-header-template.xlsx"
LINEAR_UNITS = {"п.м", "п.м.", "п. м", "пм"}
FORMULA_COUNT_UNITS = {"шт"}
CROSSED_OUT_STATUS = "crossed_out"
CROSSED_OUT_FONT_COLOR = "FF0000"
QUANTITY_NUMBER_FORMAT = "0.00"
DEFAULT_BODY_ROW_HEIGHT = 15.0
BODY_ROW_LINE_HEIGHT = 15.0
NAME_WRAP_CHARS_PER_LINE = 85
NOTE_WRAP_CHARS_PER_LINE = 16
MAX_AUTO_BODY_ROW_HEIGHT = 180.0
SUMMARY_BORDER_SIDE = Side(style="thin", color="000000")
SUMMARY_BORDER = Border(
    left=SUMMARY_BORDER_SIDE,
    right=SUMMARY_BORDER_SIDE,
    top=SUMMARY_BORDER_SIDE,
    bottom=SUMMARY_BORDER_SIDE,
)
ROUND_DIMENSION_RE = re.compile(r"(?:^|\s)(?:ф|ø|Ø|d=?)\s*(\d{2,4})(?:\D|$)", re.IGNORECASE)
RECTANGULAR_DIMENSION_RE = re.compile(r"(\d{2,4})\s*[xх×]\s*(\d{2,4})(?![\d,.])", re.IGNORECASE)
GOST_MARK_RE = re.compile(r"\s*ГОСТ\s*14918-?80\*?", re.IGNORECASE)
SURFACE_AREA_QUANT = Decimal("0.1")
PI = Decimal("3.14")
QuantityValue = Decimal | str
POSITIONAL_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+$")
TEXT_STARTS_POSITION_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+")


@dataclass(slots=True)
class KpOutputRow:
    kind: str
    label: str | None = None
    source: SpecificationRow | None = None
    pos: str | None = None
    display_name: str | None = None
    unit: str | None = None
    quantity: QuantityValue | None = None
    calc_quantity: Decimal | None = None
    status: str | None = None


def default_kp_template_path(project_root: Path) -> Path:
    return project_root / "examples" / ASCII_KP_TEMPLATE_NAME


def default_kp_header_template_path(project_root: Path) -> Path:
    return project_root / "examples" / ASCII_KP_HEADER_TEMPLATE_NAME


def export_jsonl_to_kp_xlsx(
    *,
    input_jsonl_path: str | Path,
    output_path: str | Path,
    template_path: str | Path,
    sheet_name: str | None = None,
) -> tuple[Path, int]:
    rows = load_rows_from_jsonl(input_jsonl_path)
    generated_item_count = sum(1 for row in build_kp_output_rows(rows) if row.kind == "item")
    written_path = write_rows_to_kp_xlsx(
        rows,
        output_path=output_path,
        template_path=template_path,
        sheet_name=sheet_name,
    )
    return written_path, generated_item_count


def write_rows_to_kp_xlsx(
    rows: list[SpecificationRow],
    *,
    output_path: str | Path,
    template_path: str | Path,
    sheet_name: str | None = None,
) -> Path:
    resolved_output = Path(output_path).expanduser()
    if not resolved_output.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {resolved_output.parent}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Cannot parse header or footer.*",
            category=UserWarning,
            module=r"openpyxl\.worksheet\.header_footer",
        )
        workbook = load_workbook(template_path)
    worksheet = workbook[sheet_name] if sheet_name else workbook[MAIN_SHEET_NAME]
    copy_commercial_header_from_template(worksheet, discover_header_template_path(template_path))
    summary_start = find_summary_start_row(worksheet)
    item_style = capture_row_style(worksheet, find_first_item_style_row(worksheet))
    group_style = capture_row_style(worksheet, find_first_group_style_row(worksheet))

    kp_rows = build_kp_output_rows(rows)
    body_row_count = len(kp_rows)
    body_end = BODY_START_ROW + body_row_count - 1 if body_row_count else BODY_START_ROW - 1
    spacer_row = body_end + 1 if body_row_count else None
    new_summary_start = (spacer_row + 1) if spacer_row is not None else BODY_START_ROW

    remove_template_merges_below_body(worksheet)
    worksheet.delete_rows(BODY_START_ROW, summary_start - BODY_START_ROW)
    if body_row_count:
        worksheet.insert_rows(BODY_START_ROW, body_row_count + 1)

    for offset, kp_row in enumerate(kp_rows):
        excel_row = BODY_START_ROW + offset
        style = group_style if kp_row.kind == "group" else item_style
        apply_row_style(worksheet, excel_row, style)
        if kp_row.kind == "group":
            write_grouping_row(worksheet, excel_row, kp_row.label)
        elif kp_row.source is not None:
            write_item_output_row(worksheet, excel_row, kp_row)
            if kp_row.status == CROSSED_OUT_STATUS:
                apply_crossed_out_font(worksheet, excel_row)

    if spacer_row is not None:
        apply_row_style(worksheet, spacer_row, group_style)
        clear_visible_row(worksheet, spacer_row)

    rewrite_summary_block(worksheet, new_summary_start, BODY_START_ROW, body_end)
    rebuild_summary_merges(worksheet, new_summary_start)
    apply_summary_block_style(worksheet, new_summary_start)
    last_summary_row = new_summary_start + len(SUMMARY_LABELS) - 1
    trim_to_visible_area(worksheet, last_summary_row, VISIBLE_COLUMN_COUNT)
    worksheet.print_area = f"A1:I{last_summary_row}"
    remove_unused_template_sheets(workbook, worksheet)
    if workbook.calculation is None:
        workbook.calculation = CalcProperties()
    workbook.calculation.calcMode = "auto"
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True

    workbook.save(resolved_output)
    return resolved_output


def discover_header_template_path(template_path: str | Path) -> Path | None:
    resolved_template = Path(template_path).expanduser()
    ascii_header = resolved_template.parent / ASCII_KP_HEADER_TEMPLATE_NAME
    if ascii_header.exists():
        return ascii_header
    project_default = default_kp_header_template_path(resolved_template.parent.parent)
    if project_default.exists():
        return project_default
    matches = sorted(resolved_template.parent.glob(HEADER_TEMPLATE_PATTERN))
    return matches[0] if matches else None


def copy_commercial_header_from_template(
    worksheet: Worksheet, header_template_path: Path | None
) -> None:
    if header_template_path is None or not header_template_path.exists():
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Cannot parse header or footer.*",
            category=UserWarning,
            module=r"openpyxl\.worksheet\.header_footer",
        )
        header_workbook = load_workbook(header_template_path, data_only=False)
    source = header_workbook[header_workbook.sheetnames[0]]
    remove_merges_in_header_rows(worksheet)
    for row in range(1, COMMERCIAL_HEADER_ROWS + 1):
        worksheet.row_dimensions[row].height = source.row_dimensions[row].height
        for column in range(1, VISIBLE_COLUMN_COUNT + 1):
            source_cell = source.cell(row, column)
            target_cell = cell_at(worksheet, row, column)
            target_cell.value = source_cell.value
            if isinstance(source_cell, Cell):
                apply_cell_style(target_cell, copy_cell_style(source_cell))
            sanitize_commercial_header_cell(target_cell)

    for merged_range in source.merged_cells.ranges:
        if merged_range.min_row > COMMERCIAL_HEADER_ROWS:
            continue
        if merged_range.max_row > COMMERCIAL_HEADER_ROWS:
            continue
        worksheet.merge_cells(
            start_row=merged_range.min_row,
            start_column=merged_range.min_col,
            end_row=merged_range.max_row,
            end_column=min(merged_range.max_col, VISIBLE_COLUMN_COUNT),
        )


def sanitize_commercial_header_cell(cell: Cell) -> None:
    value = cell.value
    if not isinstance(value, str):
        return
    if "РОГА" in value.upper() or "КОПЫТА" in value.upper():
        cell.value = "Коммерческое предложение по спецификации"
    elif "ОБЪЕКТ МЖД" in value.upper():
        cell.value = None


def remove_merges_in_header_rows(worksheet: Worksheet) -> None:
    for merged_range in list(worksheet.merged_cells.ranges):
        if merged_range.min_row <= COMMERCIAL_HEADER_ROWS:
            worksheet.unmerge_cells(str(merged_range))


def build_kp_output_rows(rows: list[SpecificationRow]) -> list[KpOutputRow]:
    output: list[KpOutputRow] = []
    seen_initial_labels: set[str] = set()
    for label in DEFAULT_KP_PREFIX_LABELS:
        output.append(KpOutputRow(kind="group", label=label))
        seen_initial_labels.add(label)

    current_section: str | None = None
    current_subsection: str | None = None
    pending_header: SpecificationRow | None = None
    pending_header_used = False
    last_item: KpOutputRow | None = None
    for index, row in enumerate(rows):
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        if row.is_grouping_row():
            current_section, current_subsection = append_grouping_from_row(
                output, row, current_section, current_subsection, seen_initial_labels
            )
            pending_header = None
            pending_header_used = False
            last_item = None
            continue

        if is_short_section_like_header(row):
            output.append(KpOutputRow(kind="group", label=normalize_customer_text(row.name)))
            pending_header = None
            pending_header_used = False
            last_item = None
            continue

        if row.section and row.section != current_section:
            output.append(KpOutputRow(kind="group", label=row.section))
            current_section = row.section
            current_subsection = None
            pending_header = None
            pending_header_used = False
            last_item = None
        if row.subsection and row.subsection != current_subsection:
            output.append(KpOutputRow(kind="group", label=row.subsection))
            current_subsection = row.subsection

        if is_standalone_text_header(row, current_section, current_subsection):
            output.append(KpOutputRow(kind="group", label=normalize_customer_text(row.name)))
            pending_header = None
            pending_header_used = False
            last_item = None
            continue

        if is_non_prefix_header(row):
            item = build_item_output_row(row, None, False)
            output.append(item)
            last_item = item
            pending_header = None
            pending_header_used = False
            continue

        if is_positioned_text_header(row):
            output.append(KpOutputRow(kind="group", label=positioned_header_label(row)))
            pending_header = None
            pending_header_used = False
            last_item = None
            continue

        if is_context_header(row):
            pending_header = row
            pending_header_used = False
            continue

        if is_prefix_header_before_item(row, next_row):
            pending_header = row
            pending_header_used = False
            continue

        if is_text_continuation(row) and last_item is not None:
            last_item.display_name = join_display_parts(
                last_item.display_name, row.name, row.type_mark
            )
            if row.status == CROSSED_OUT_STATUS:
                last_item.status = CROSSED_OUT_STATUS
            continue
        if is_text_continuation(row):
            continue

        item = build_item_output_row(row, pending_header, pending_header_used)
        if pending_header is not None and row.unit is not None:
            pending_header_used = True
        output.append(item)
        last_item = item
    return output


def append_grouping_from_row(
    output: list[KpOutputRow],
    row: SpecificationRow,
    current_section: str | None,
    current_subsection: str | None,
    seen_initial_labels: set[str],
) -> tuple[str | None, str | None]:
    if row.section and row.section != current_section and row.section not in seen_initial_labels:
        output.append(KpOutputRow(kind="group", label=row.section))
        current_section = row.section
        current_subsection = None
    if row.subsection and row.subsection != current_subsection:
        output.append(KpOutputRow(kind="group", label=row.subsection))
        current_subsection = row.subsection
    return current_section, current_subsection


def kp_display_name(row: SpecificationRow) -> str | None:
    return normalize_kp_name(customer_description(row))


def is_context_header(row: SpecificationRow) -> bool:
    return (
        row.name is not None
        and row.pos is not None
        and not is_decimal_position(row.pos)
        and row.unit is None
        and row.quantity is None
        and not is_non_prefix_header(row)
    )


def is_prefix_header_before_item(row: SpecificationRow, next_row: SpecificationRow | None) -> bool:
    return (
        row.name is not None
        and row.pos is None
        and row.unit is None
        and row.quantity is None
        and row.quantity_raw is None
        and next_row is not None
        and next_row.pos is None
        and next_row.name is not None
        and (next_row.unit is not None or next_row.quantity is not None or next_row.quantity_raw)
        and any([row.type_mark, row.code, row.manufacturer, row.note])
    )


def is_short_section_like_header(row: SpecificationRow) -> bool:
    if not is_text_continuation(row):
        return False
    if any([row.type_mark, row.code, row.manufacturer, row.note]):
        return False
    label = normalize_customer_text(row.name)
    if label is None or len(label) > 48:
        return False
    lowered = label.lower()
    return (
        label == "КИПиА" or lowered in {"радиаторы:", "радиаторы"} or lowered.startswith("система ")
    )


def is_non_prefix_header(row: SpecificationRow) -> bool:
    return (
        row.name is not None
        and row.pos is not None
        and row.unit is None
        and row.quantity is None
        and is_composition_header_text(row.name)
    )


def is_composition_header_text(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    return "в составе" in normalized


def is_text_continuation(row: SpecificationRow) -> bool:
    return (
        row.name is not None
        and row.pos is None
        and row.unit is None
        and row.quantity is None
        and not row.quantity_raw
        and not TEXT_STARTS_POSITION_RE.match(row.name)
    )


def is_positioned_text_header(row: SpecificationRow) -> bool:
    return (
        row.name is not None
        and row.pos is not None
        and is_decimal_position(row.pos)
        and row.unit is None
        and row.quantity is None
        and not row.quantity_raw
        and not any(
            [row.type_mark, row.code, row.manufacturer, row.mass_kg, row.mass_kg_raw, row.note]
        )
    )


def is_decimal_position(value: str | None) -> bool:
    return bool(value and POSITIONAL_HEADING_RE.fullmatch(value))


def positioned_header_label(row: SpecificationRow) -> str | None:
    return normalize_customer_text(join_display_parts(row.pos, row.name))


def is_standalone_text_header(
    row: SpecificationRow, current_section: str | None, current_subsection: str | None
) -> bool:
    if not is_text_continuation(row):
        return False
    if any([row.type_mark, row.code, row.manufacturer, row.note]):
        return False
    label = normalize_customer_text(row.name)
    if label is None:
        return False
    current_labels = {value for value in (current_section, current_subsection) if value}
    if label in current_labels:
        return False
    return bool(row.section or row.subsection)


def build_item_output_row(
    row: SpecificationRow,
    pending_header: SpecificationRow | None,
    pending_header_used: bool,
) -> KpOutputRow:
    display_name = kp_display_name(row)
    pos = row.pos
    if pending_header is not None and row.pos is None and row.unit is not None:
        display_name = join_display_parts(
            kp_display_name(pending_header), normalize_dimension_text(display_name)
        )
        pos = pending_header.pos if not pending_header_used else None

    unit, quantity, calc_quantity = commercial_unit_quantity_for_row(display_name, row)
    return KpOutputRow(
        kind="item",
        source=row,
        pos=pos,
        display_name=display_name,
        unit=unit,
        quantity=quantity,
        calc_quantity=calc_quantity,
        status=row.status,
    )


def join_display_parts(*parts: str | None) -> str | None:
    values = [part.strip() for part in parts if part and part.strip()]
    return normalize_kp_name(" ".join(values)) if values else None


def normalize_dimension_text(value: str | None) -> str | None:
    return normalize_customer_text(value)


def normalize_kp_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = normalize_dimension_text(value) or value
    text = GOST_MARK_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def commercial_unit_quantity(
    display_name: str | None, unit: str | None, quantity: Decimal | None
) -> tuple[str | None, Decimal | None]:
    normalized_unit = normalize_unit(unit)
    area = None
    if normalized_unit in FORMULA_COUNT_UNITS:
        area = typical_item_surface_area(display_name, normalized_unit, quantity)
    if area is not None:
        return "м2", area
    if (
        normalized_unit in LINEAR_UNITS
        and quantity is not None
        and is_linear_surface_area_item(display_name)
    ):
        area = linear_quantity_to_surface_area(display_name, quantity)
        if area is not None:
            return "м2", area
    return normalized_unit, quantity


def commercial_unit_quantity_for_row(
    display_name: str | None, row: SpecificationRow
) -> tuple[str | None, QuantityValue | None, Decimal | None]:
    normalized_unit = normalize_unit(row.unit)
    raw_quantity = row.quantity_raw

    if raw_quantity is not None:
        area_quantity = area_quantity_from_composite(raw_quantity)
        if area_quantity is not None and unit_has_area_component(normalized_unit):
            return "м2", area_quantity, area_quantity
        expression_quantity = expression_sum_quantity(raw_quantity)
        if expression_quantity is not None:
            return normalized_unit, expression_quantity, expression_quantity
        first_quantity = first_decimal_in_text(raw_quantity)
        if first_quantity is not None:
            return normalized_unit, first_quantity, first_quantity
        return normalized_unit, raw_quantity, None

    if unit_has_count_and_area_components(normalized_unit) and row.mass_kg is not None:
        return "м2", row.mass_kg, row.mass_kg

    unit, quantity = commercial_unit_quantity(display_name, row.unit, row.quantity)
    return unit, quantity, quantity if isinstance(quantity, Decimal) else None


def unit_has_area_component(unit: str | None) -> bool:
    if unit is None:
        return False
    normalized = unit.replace("²", "2").lower()
    return "м2" in normalized or "m2" in normalized


def unit_has_count_and_area_components(unit: str | None) -> bool:
    if unit is None:
        return False
    normalized = unit.replace("²", "2").replace(" ", "").lower()
    return ("шт" in normalized or "pcs" in normalized) and unit_has_area_component(normalized)


def area_quantity_from_composite(value: str) -> Decimal | None:
    parts = re.split(r"/", value, maxsplit=1)
    if len(parts) != 2:
        return None
    return first_decimal_in_text(parts[1])


def expression_sum_quantity(value: str) -> Decimal | None:
    if "+" not in value:
        return None
    numbers = [
        decimal_from_text(match.group(0)) for match in re.finditer(r"\d+(?:[,.]\d+)?", value)
    ]
    if len(numbers) < 2:
        return None
    return sum(numbers, Decimal(0))


def first_decimal_in_text(value: str) -> Decimal | None:
    match = re.search(r"\d+(?:[,.]\d+)?", value)
    return decimal_from_text(match.group(0)) if match else None


def is_linear_surface_area_item(name: str | None) -> bool:
    if name is None:
        return False
    text = formula_text(name)
    return "воздуховод" in text


def typical_item_surface_area(
    name: str | None, unit: str | None, quantity: Decimal | None
) -> Decimal | None:
    if name is None or quantity is None:
        return None
    text = formula_text(name)
    if not text:
        return None

    if "воздуховод" in text:
        return duct_surface_area(text, unit, quantity)
    if "отвод" in text and "сечен" in text:
        return elbow_surface_area(text, quantity)
    if "переход" in text and "утка" in text:
        return offset_transition_surface_area(text, quantity)
    if "переход" in text and ("сечен" in text or "кругл" in text or "прямоуголь" in text):
        return transition_surface_area(text, quantity)
    if "тройник" in text and "сечен" in text:
        return tee_surface_area(text, quantity)
    if "заглуш" in text and "сечен" in text:
        return cap_surface_area(text, quantity)
    if "крестов" in text and "сечен" in text:
        return cross_surface_area(text, quantity)
    if "врезк" in text and ("кругл" in text or "прямоуголь" in text):
        return tap_surface_area(text, quantity)
    return None


def duct_surface_area(text: str, unit: str | None, quantity: Decimal) -> Decimal | None:
    if unit in LINEAR_UNITS:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is not None:
            return rectangular_duct_area(rectangular[0], rectangular[1], quantity, Decimal(1))
        diameter = first_round_dimension(text)
        if diameter is not None:
            return round_duct_area(diameter, quantity, Decimal(1))
        return None

    length = labeled_meters(text, ("l", "длина"))
    if length is None:
        return None
    if "прямоуголь" in text:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is None:
            return None
        return rectangular_duct_area(rectangular[0], rectangular[1], length, quantity)
    diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
    if diameter is None:
        return None
    return round_duct_area(diameter, length, quantity)


def elbow_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    angle = angle_degrees(text) or Decimal(90)
    allowance = allowance_meters(text) or Decimal(0)
    radius = labeled_meters(text, ("r", "радиус"))
    if "прямоуголь" in text:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is None or radius is None:
            return None
        width, height = rectangular
        outer_arc = ((PI * ((radius + width) ** 2)) - (PI * (radius**2))) / (Decimal(360) / angle)
        top_bottom = outer_arc * Decimal(2)
        outer_side = (Decimal(2) * PI * (radius + width) * height) / (Decimal(360) / angle)
        inner_side = (Decimal(2) * PI * radius * height) / (Decimal(360) / angle)
        allowance_area = ((width + height) * Decimal(2) * allowance) * Decimal(2)
        return quantize_surface_area(
            (top_bottom + outer_side + inner_side + allowance_area) * quantity
        )

    diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
    if diameter is None:
        return None
    if radius is None:
        radius = diameter
    bend = (Decimal(4) * PI * PI * radius * (diameter / Decimal(2))) / (Decimal(360) / angle)
    allowance_area = PI * diameter * (allowance * Decimal(2))
    return quantize_surface_area((bend + allowance_area) * quantity)


def transition_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    length = labeled_meters(text, ("l", "длина"))
    allowance = allowance_meters(text) or Decimal(0)
    if length is None:
        return None

    if "прямоуголь" in text and "кругл" in text:
        rectangular = first_rectangular_dimensions(text)
        diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
        if rectangular is None or diameter is None:
            return None
        width, height = rectangular
        side_a = sqrt_decimal(length**2 + ((height - diameter) / Decimal(2)) ** 2)
        side_b = sqrt_decimal(length**2 + ((width - diameter) / Decimal(2)) ** 2)
        allowance_round = PI * diameter * allowance
        allowance_rect = ((width + height) / Decimal(2)) * allowance
        trapezoid_a = (
            ((width + (PI * diameter / Decimal(4))) / Decimal(2)) * side_a * Decimal("1.07")
        )
        trapezoid_b = (
            ((height + (PI * diameter / Decimal(4))) / Decimal(2)) * side_b * Decimal("1.07")
        )
        return quantize_surface_area(
            ((trapezoid_a + trapezoid_b) * Decimal(2) + allowance_round + allowance_rect)
            * Decimal("1.05")
            * quantity
        )

    if "прямоуголь" in text:
        dims = rectangular_dimension_list(text)
        if len(dims) < 2:
            return None
        width1, height1 = dims[0]
        width2, height2 = dims[1]
        side1 = sqrt_decimal(((height1 - height2) / Decimal(2)) ** 2 + length**2)
        side2 = sqrt_decimal(((width1 - width2) / Decimal(2)) ** 2 + length**2)
        trapezoid1 = (width1 + width2) * side2
        trapezoid2 = (height1 + height2) * side1
        allowance1 = (width1 + height1) * Decimal(2) * allowance * Decimal("1.03")
        allowance2 = (width2 + height2) * Decimal(2) * allowance * Decimal("1.03")
        return quantize_surface_area((trapezoid1 + trapezoid2 + allowance1 + allowance2) * quantity)

    dimensions = round_dimension_list(text)
    if len(dimensions) < 2:
        return None
    diameter1, diameter2 = dimensions[0], dimensions[1]
    side = PI * length * ((diameter1 / Decimal(2)) + (diameter2 / Decimal(2))) * Decimal("1.039")
    allowance1 = Decimal("3.1428") * diameter1 * allowance * Decimal("1.035")
    allowance2 = Decimal("3.1428") * diameter2 * allowance * Decimal("1.035")
    return quantize_surface_area((side + allowance1 + allowance2) * quantity)


def tee_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    if "прямоугольного сечения" in text and "круг" in text:
        rectangular = first_rectangular_dimensions(text)
        diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
        main_length = labeled_meters(text, ("l", "длина"))
        branch_length = labeled_meters(text, ("l2", "l1")) or labeled_meters(text, ("длина2",))
        if rectangular is None or diameter is None or main_length is None or branch_length is None:
            return None
        width, height = rectangular
        main = (width + height) * Decimal(2) * main_length
        branch = PI * diameter * branch_length
        cutout = PI * (diameter / Decimal(2)) ** 2
        return quantize_surface_area((main + branch - cutout) * quantity)

    if "круглого сечения" in text and "прямоуголь" in text:
        diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
        rectangular = first_rectangular_dimensions(text)
        main_length = labeled_meters(text, ("l", "длина"))
        branch_length = labeled_meters(text, ("l2", "l1")) or labeled_meters(text, ("длина2",))
        if diameter is None or rectangular is None or main_length is None or branch_length is None:
            return None
        width, height = rectangular
        branch = (width + height) * Decimal(2) * branch_length
        cylinder = PI * diameter * main_length
        cutout = width * height
        add = height**3 * Decimal("0.95")
        return quantize_surface_area((branch + cylinder - cutout + add) * quantity)

    if "прямоуголь" in text and "кругл" in text:
        return None
    if "прямоуголь" in text:
        dims = rectangular_dimension_list(text)
        main_length = labeled_meters(text, ("l", "длина"))
        branch_length = labeled_meters(text, ("l2", "l1")) or labeled_meters(text, ("длина2",))
        if len(dims) < 2 or main_length is None or branch_length is None:
            return None
        width1, height1 = dims[0]
        width2, height2 = dims[1]
        main = (width1 + height1) * Decimal(2) * main_length
        branch = (width2 + height2) * Decimal(2) * branch_length
        cutout = width2 * height2
        return quantize_surface_area((main + branch - cutout) * quantity)

    dimensions = round_dimension_list(text)
    main_length = labeled_meters(text, ("l", "длина"))
    branch_length = labeled_meters(text, ("l2", "l1")) or labeled_meters(text, ("длина2",))
    if len(dimensions) < 2 or main_length is None or branch_length is None:
        return None
    diameter1, diameter2 = dimensions[0], dimensions[1]
    return quantize_surface_area(
        (PI * diameter1 * main_length + PI * diameter2 * branch_length) * quantity
    )


def cap_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    allowance = allowance_meters(text) or Decimal(0)
    if "прямоуголь" in text:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is None:
            return None
        width, height = rectangular
        return quantize_surface_area(
            (width * height + ((width + height) * Decimal(2) * allowance)) * quantity
        )

    diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
    if diameter is None:
        return None
    return quantize_surface_area(
        (PI * (diameter / Decimal(2)) ** 2 + PI * diameter * allowance) * quantity
    )


def offset_transition_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    length = labeled_meters(text, ("l", "длина"))
    offset = labeled_meters(text, ("offset", "смещение")) or labeled_meters(text, ("l1",))
    allowance = allowance_meters(text) or Decimal(0)
    if length is None or offset is None:
        return None
    if "прямоуголь" in text:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is None:
            return None
        width, height = rectangular
        slanted = sqrt_decimal(length**2 + (offset * Decimal("0.82")) ** 2)
        base_a = Decimal(2) * width * length * Decimal("0.95")
        base_b = slanted * height * Decimal(2) * Decimal("0.95")
        allowance_area = (width + height) * Decimal(2) * allowance / Decimal(2)
        extra = (offset * Decimal("7.5")) / (length * Decimal("8.63")) if length else Decimal(0)
        return quantize_surface_area((base_a + base_b + allowance_area + extra) * quantity)

    diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
    if diameter is None:
        return None
    slanted = sqrt_decimal(length**2 + offset**2)
    return quantize_surface_area(
        (PI * diameter * slanted + PI * diameter * allowance * Decimal(2)) * quantity
    )


def cross_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    if "прямоуголь" in text:
        dims = rectangular_dimension_list(text)
        lengths = labeled_meter_list(text, ("l", "l2", "l3", "длина"))
        if len(dims) < 3 or len(lengths) < 3:
            return None
        width1, height1 = dims[0]
        width2, height2 = dims[1]
        width3, height3 = dims[2]
        main = (width1 + height1) * Decimal(2) * lengths[0]
        branch2 = (width2 + height2) * Decimal(2) * lengths[1]
        branch3 = (width3 + height3) * Decimal(2) * lengths[2]
        cutout = width2 * height2 + width3 * height3
        return quantize_surface_area((main + branch2 + branch3 - cutout) * quantity)

    dimensions = round_dimension_list(text)
    lengths = labeled_meter_list(text, ("l", "l2", "l3", "длина"))
    if len(dimensions) < 3 or len(lengths) < 3:
        return None
    diameter1, diameter2, diameter3 = dimensions[0], dimensions[1], dimensions[2]
    main = PI * diameter1 * lengths[0]
    branch2 = PI * diameter2 * lengths[1] * Decimal("0.9")
    branch3 = PI * diameter3 * lengths[2] * Decimal("0.9")
    cutout = (diameter2**3) * Decimal(4) / Decimal(3) + (diameter3**3) * Decimal(4) / Decimal(3)
    return quantize_surface_area((main + branch2 + branch3 - cutout) * Decimal("1.005") * quantity)


def tap_surface_area(text: str, quantity: Decimal) -> Decimal | None:
    length = labeled_meters(text, ("l", "длина"))
    if length is None:
        return None
    if "прямоуголь" in text:
        rectangular = first_rectangular_dimensions(text)
        if rectangular is None:
            return None
        width, height = rectangular
        return quantize_surface_area((width + length) * Decimal(2) * height * quantity)

    diameter = labeled_meters(text, ("d", "диаметр")) or first_round_dimension(text)
    if diameter is None:
        return None
    return quantize_surface_area(PI * diameter * length * quantity)


def linear_quantity_to_surface_area(name: str | None, length: Decimal) -> Decimal | None:
    if name is None:
        return None
    rectangular = RECTANGULAR_DIMENSION_RE.search(name)
    if rectangular:
        width = Decimal(rectangular.group(1)) / Decimal(1000)
        height = Decimal(rectangular.group(2)) / Decimal(1000)
        return rectangular_duct_area(width, height, length, Decimal(1))

    round_match = ROUND_DIMENSION_RE.search(name)
    if round_match:
        diameter = Decimal(round_match.group(1)) / Decimal(1000)
        return round_duct_area(diameter, length, Decimal(1))
    return None


def formula_text(value: str) -> str:
    text = normalize_customer_text(value) or value
    text = re.sub(r"(?<=\d)\s*[xх×]\s*(?=\d)", "x", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().lower()


def quantize_surface_area(value: Decimal) -> Decimal:
    return value.quantize(SURFACE_AREA_QUANT, rounding=ROUND_HALF_UP)


def round_duct_area(diameter: Decimal, length: Decimal, count: Decimal) -> Decimal:
    return quantize_surface_area(PI * diameter * length * count)


def rectangular_duct_area(
    width: Decimal, height: Decimal, length: Decimal, count: Decimal
) -> Decimal:
    return quantize_surface_area(Decimal(2) * (width + height) * length * count)


def sqrt_decimal(value: Decimal) -> Decimal:
    return value.sqrt()


def decimal_from_text(value: str) -> Decimal:
    return Decimal(value.replace(",", "."))


def mm_to_meters(value: str) -> Decimal:
    return decimal_from_text(value) / Decimal(1000)


def length_to_meters(value: str, unit: str | None = None) -> Decimal:
    number = decimal_from_text(value)
    normalized_unit = (unit or "").strip().lower()
    if normalized_unit in {"м", "m"}:
        return number
    if normalized_unit in {"см", "cm"}:
        return number / Decimal(100)
    return number / Decimal(1000)


def labeled_meters(text: str, labels: tuple[str, ...]) -> Decimal | None:
    for label in sorted(labels, key=len, reverse=True):
        pattern = re.compile(
            rf"(?:^|[\s,;(]){re.escape(label)}(?:\s*=\s*|\s+)(?:[Øø]\s*)?(\d{{1,4}}(?:[,.]\d+)?)(?:\s*(мм|mm|см|cm|м|m)(?![A-Za-zА-Яа-я]))?",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return length_to_meters(match.group(1), match.group(2))
    return None


def labeled_meter_list(text: str, labels: tuple[str, ...]) -> list[Decimal]:
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    if not label_pattern:
        return []
    pattern = re.compile(
        rf"(?:^|[\s,;(])(?:{label_pattern})(?:\s*=\s*|\s+)(?:[Øø]\s*)?(\d{{1,4}}(?:[,.]\d+)?)(?:\s*(мм|mm|см|cm|м|m)(?![A-Za-zА-Яа-я]))?",
        re.IGNORECASE,
    )
    return [length_to_meters(match.group(1), match.group(2)) for match in pattern.finditer(text)]


def allowance_meters(text: str) -> Decimal | None:
    return labeled_meters(text, ("припуск", "p", "р"))


def angle_degrees(text: str) -> Decimal | None:
    explicit = re.search(r"(?:^|[\s,;(])(?:a|а|угол)\s*=?\s*(\d{1,3}(?:[,.]\d+)?)", text)
    if explicit:
        return decimal_from_text(explicit.group(1))
    suffix = re.search(r"(\d{1,3}(?:[,.]\d+)?)\s*(?:°|гр\.?|град)", text)
    if suffix:
        return decimal_from_text(suffix.group(1))
    return None


def first_rectangular_dimensions(text: str) -> tuple[Decimal, Decimal] | None:
    dimensions = rectangular_dimension_list(text)
    return dimensions[0] if dimensions else None


def rectangular_dimension_list(text: str) -> list[tuple[Decimal, Decimal]]:
    return [(mm_to_meters(a), mm_to_meters(b)) for a, b in RECTANGULAR_DIMENSION_RE.findall(text)]


def first_round_dimension(text: str) -> Decimal | None:
    dimensions = round_dimension_list(text)
    return dimensions[0] if dimensions else None


def round_dimension_list(text: str) -> list[Decimal]:
    pattern = re.compile(
        r"(?:[Øø]\s*(\d{2,4})(?![\d,.])|(?:^|[\s,;(])(?:диаметр|d3|d2|d1|d)(?:\s*=\s*|\s+)[Øø]?\s*(\d{1,4}(?:[,.]\d+)?)(?:\s*(мм|mm|см|cm|м|m)(?![A-Za-zА-Яа-я]))?)",
        re.IGNORECASE,
    )
    dimensions: list[Decimal] = []
    for match in pattern.finditer(text):
        if match.group(1) is not None:
            dimensions.append(mm_to_meters(match.group(1)))
        elif match.group(2) is not None:
            dimensions.append(length_to_meters(match.group(2), match.group(3)))
    return dimensions


def find_summary_start_row(worksheet: Worksheet) -> int:
    for row in range(BODY_START_ROW, worksheet.max_row + 1):
        for column in range(1, 10):
            if worksheet.cell(row, column).value == SUMMARY_LABEL:
                return row
    raise ValueError(f"Cannot find КП summary label: {SUMMARY_LABEL}")


def find_first_item_style_row(worksheet: Worksheet) -> int:
    summary_start = find_summary_start_row(worksheet)
    for row in range(BODY_START_ROW, summary_start):
        if worksheet.cell(row, 2).value and worksheet.cell(row, 3).value:
            return row
    return BODY_START_ROW


def find_first_group_style_row(worksheet: Worksheet) -> int:
    summary_start = find_summary_start_row(worksheet)
    for row in range(BODY_START_ROW, summary_start):
        if worksheet.cell(row, 2).value and not worksheet.cell(row, 3).value:
            return row
    return BODY_START_ROW


def capture_row_style(worksheet: Worksheet, row_number: int) -> dict[str, object]:
    return {
        "height": worksheet.row_dimensions[row_number].height,
        "cells": [
            copy_cell_style(cell_at(worksheet, row_number, column)) for column in range(1, 10)
        ],
    }


def copy_cell_style(cell: Cell) -> dict[str, Any]:
    return {
        "style": copy(cell._style),
        "font": copy(cell.font),
        "fill": copy(cell.fill),
        "border": copy(cell.border),
        "alignment": copy(cell.alignment),
        "number_format": cell.number_format,
        "protection": copy(cell.protection),
    }


def apply_row_style(worksheet: Worksheet, row_number: int, style: dict[str, object]) -> None:
    worksheet.row_dimensions[row_number].height = style["height"]
    cell_styles = style["cells"]
    if not isinstance(cell_styles, list):
        return
    for column, cell_style in enumerate(cell_styles, start=1):
        if isinstance(cell_style, dict):
            apply_cell_style(cell_at(worksheet, row_number, column), cell_style)


def apply_cell_style(cell: Cell, style: dict[str, Any]) -> None:
    cell._style = copy(style["style"])
    cell.font = copy(style["font"])
    cell.fill = copy(style["fill"])
    cell.border = copy(style["border"])
    cell.alignment = copy(style["alignment"])
    cell.number_format = str(style["number_format"])
    cell.protection = copy(style["protection"])


def write_grouping_row(worksheet: Worksheet, row_number: int, label: str | None) -> None:
    clear_visible_row(worksheet, row_number)
    cell_at(worksheet, row_number, 2).value = excel_safe_text(label)
    enable_wrap_text(cell_at(worksheet, row_number, 2))
    adjust_body_row_height(worksheet, row_number, name=label or "", note="")


def write_item_output_row(worksheet: Worksheet, row_number: int, output_row: KpOutputRow) -> None:
    row = output_row.source
    if row is None:
        return
    clear_visible_row(worksheet, row_number)
    display_name = output_row.display_name or kp_display_name(row)
    unit = output_row.unit
    quantity = output_row.quantity
    calc_quantity = output_row.calc_quantity
    if unit is None and quantity is None:
        unit, quantity, calc_quantity = commercial_unit_quantity_for_row(display_name, row)
    cell_at(worksheet, row_number, 1).value = excel_safe_text(
        output_row.pos if output_row.pos is not None else row.pos
    )
    cell_at(worksheet, row_number, 2).value = excel_safe_text(display_name)
    cell_at(worksheet, row_number, 3).value = excel_safe_text(unit)
    quantity_cell = cell_at(worksheet, row_number, 4)
    quantity_cell.value = excel_safe_value(decimal_to_excel(quantity))
    if isinstance(quantity, Decimal):
        quantity_cell.number_format = QUANTITY_NUMBER_FORMAT
    cell_at(worksheet, row_number, 5).value = None
    cell_at(worksheet, row_number, 6).value = None
    formula_quantity = quantity if isinstance(quantity, Decimal) else calc_quantity
    if isinstance(formula_quantity, Decimal):
        cell_at(worksheet, row_number, 7).value = f"=E{row_number}*D{row_number}"
        cell_at(worksheet, row_number, 8).value = f"=F{row_number}*D{row_number}"
        if not isinstance(quantity, Decimal):
            literal = excel_decimal_literal(formula_quantity)
            cell_at(worksheet, row_number, 7).value = f"=E{row_number}*{literal}"
            cell_at(worksheet, row_number, 8).value = f"=F{row_number}*{literal}"
    else:
        cell_at(worksheet, row_number, 7).value = None
        cell_at(worksheet, row_number, 8).value = None
    note = customer_note(row)
    cell_at(worksheet, row_number, 9).value = excel_safe_text(note)
    enable_wrap_text(cell_at(worksheet, row_number, 2))
    enable_wrap_text(cell_at(worksheet, row_number, 9))
    adjust_body_row_height(worksheet, row_number, name=display_name, note=note or "")


def enable_wrap_text(cell: Cell) -> None:
    alignment = copy(cell.alignment)
    alignment.wrap_text = True
    cell.alignment = alignment


def adjust_body_row_height(worksheet: Worksheet, row_number: int, *, name: str, note: str) -> None:
    name_wrap_chars = column_wrap_chars(
        worksheet, "B", fallback_chars_per_line=NAME_WRAP_CHARS_PER_LINE
    )
    note_wrap_chars = column_wrap_chars(
        worksheet, "I", fallback_chars_per_line=NOTE_WRAP_CHARS_PER_LINE
    )
    line_count = max(
        wrapped_line_count(name, name_wrap_chars),
        wrapped_line_count(note, note_wrap_chars),
        1,
    )
    current_height = worksheet.row_dimensions[row_number].height or DEFAULT_BODY_ROW_HEIGHT
    target_height = min(
        MAX_AUTO_BODY_ROW_HEIGHT, max(DEFAULT_BODY_ROW_HEIGHT, line_count * BODY_ROW_LINE_HEIGHT)
    )
    worksheet.row_dimensions[row_number].height = max(float(current_height), target_height)


def column_wrap_chars(
    worksheet: Worksheet, column_letter: str, *, fallback_chars_per_line: int
) -> int:
    width = worksheet.column_dimensions[column_letter].width
    if width is None:
        return fallback_chars_per_line
    return max(fallback_chars_per_line, int(width))


def wrapped_line_count(text: str | None, chars_per_line: int) -> int:
    if not text:
        return 1
    total = 0
    for line in str(text).splitlines() or [""]:
        visible_chars = len(line)
        total += max(1, (visible_chars + chars_per_line - 1) // chars_per_line)
    return total


def apply_crossed_out_font(worksheet: Worksheet, row_number: int) -> None:
    for column in range(1, 10):
        cell = cell_at(worksheet, row_number, column)
        source = cell.font
        cell.font = Font(
            name=source.name,
            sz=source.sz,
            b=source.b,
            i=source.i,
            vertAlign=source.vertAlign,
            underline=source.underline,
            strike=source.strike,
            color=CROSSED_OUT_FONT_COLOR,
            family=source.family,
            scheme=source.scheme,
        )


def rewrite_summary_block(
    worksheet: Worksheet, summary_start: int, body_start: int, body_end: int
) -> None:
    for offset, label in enumerate(SUMMARY_LABELS):
        row = summary_start + offset
        clear_visible_row(worksheet, row)
        cell_at(worksheet, row, 3).value = label
    material_range = f"G{body_start}:G{max(body_end, body_start)}"
    work_range = f"H{body_start}:H{max(body_end, body_start)}"
    cell_at(worksheet, summary_start, 8).value = f"=SUM({material_range})"
    cell_at(worksheet, summary_start + 1, 8).value = 0
    cell_at(worksheet, summary_start + 2, 8).value = f"=H{summary_start}-H{summary_start + 1}"
    cell_at(worksheet, summary_start + 3, 8).value = f"=SUM({work_range})"
    cell_at(worksheet, summary_start + 4, 8).value = 100000
    cell_at(worksheet, summary_start + 5, 8).value = 1200000
    cell_at(
        worksheet, summary_start + 6, 8
    ).value = f"=H{summary_start}+H{summary_start + 3}+H{summary_start + 4}+H{summary_start + 5}"
    cell_at(worksheet, summary_start + 7, 8).value = f"=H{summary_start + 6}/120*22"


def clear_visible_row(worksheet: Worksheet, row_number: int) -> None:
    for column in range(1, 10):
        cell_at(worksheet, row_number, column).value = None


def remove_template_merges_below_body(worksheet: Worksheet) -> None:
    for merged_range in list(worksheet.merged_cells.ranges):
        if merged_range.min_row >= BODY_START_ROW:
            worksheet.unmerge_cells(str(merged_range))


def rebuild_summary_merges(worksheet: Worksheet, summary_start: int) -> None:
    for offset in range(len(SUMMARY_LABELS)):
        row = summary_start + offset
        worksheet.merge_cells(start_row=row, start_column=3, end_row=row, end_column=7)


def apply_summary_block_style(worksheet: Worksheet, summary_start: int) -> None:
    for offset in range(len(SUMMARY_LABELS)):
        row = summary_start + offset
        worksheet.row_dimensions[row].height = max(
            float(worksheet.row_dimensions[row].height or 0), 18.0
        )
        for column in range(1, 10):
            cell_at(worksheet, row, column).border = copy(SUMMARY_BORDER)


def trim_to_visible_area(worksheet: Worksheet, last_row: int, last_column: int) -> None:
    if worksheet.max_row > last_row:
        worksheet.delete_rows(last_row + 1, worksheet.max_row - last_row)
    if worksheet.max_column > last_column:
        worksheet.delete_cols(last_column + 1, worksheet.max_column - last_column)
    for column_letter in list(worksheet.column_dimensions):
        if column_index_from_string(column_letter) > last_column:
            del worksheet.column_dimensions[column_letter]


def remove_unused_template_sheets(workbook: Any, worksheet: Worksheet) -> None:
    for sheet in list(workbook.worksheets):
        if sheet is not worksheet:
            workbook.remove(sheet)


def normalize_unit(unit: str | None) -> str | None:
    repaired_unit = normalize_customer_unit(unit)
    if repaired_unit == "шт":
        return "шт"
    if repaired_unit == "м²":
        return "м2"
    return repaired_unit


def decimal_to_excel(value: QuantityValue | None) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def excel_decimal_literal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def cell_at(worksheet: Worksheet, row: int, column: int) -> Cell:
    return cast(Cell, worksheet.cell(row, column))
