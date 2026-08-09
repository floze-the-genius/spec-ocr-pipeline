from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from spec_pipeline.io import load_rows_from_jsonl
from spec_pipeline.models import SampleDefinition, SpecificationRow
from spec_pipeline.samples import load_ground_truth_rows

FIELD_NAMES = [
    "page_ref",
    "section",
    "subsection",
    "pos",
    "name",
    "type_mark",
    "code",
    "manufacturer",
    "unit",
    "quantity",
    "mass_kg",
    "note",
]

NUMERIC_FIELDS = ["quantity", "mass_kg"]


@dataclass(slots=True)
class CompareResult:
    sample_id: str
    ground_truth_rows: int
    predicted_rows: int
    row_delta: int
    matched_rows: int
    exact_match_rate: float
    field_match_rate: float
    quantity_mae: float | None
    mass_kg_mae: float | None
    missing_keys: list[str]
    extra_keys: list[str]
    mismatches: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "ground_truth_rows": self.ground_truth_rows,
            "predicted_rows": self.predicted_rows,
            "row_delta": self.row_delta,
            "matched_rows": self.matched_rows,
            "exact_match_rate": self.exact_match_rate,
            "field_match_rate": self.field_match_rate,
            "quantity_mae": self.quantity_mae,
            "mass_kg_mae": self.mass_kg_mae,
            "missing_keys": self.missing_keys,
            "extra_keys": self.extra_keys,
            "mismatches": self.mismatches,
        }


def _row_key(rows: list[SpecificationRow], index: int, key_counters: dict[str, int]) -> str:
    row = rows[index]
    page = row.page_ref or ""
    if row.is_grouping_row():
        section = row.section or ""
        subsection = row.subsection or ""
        base_key = f"{page}|group|{section}|{subsection}"
    elif row.pos is not None:
        base_key = f"{page}|item|{row.pos}"
    else:
        base_key = f"{page}|cont"

    key_counters[base_key] += 1
    return f"{base_key}|{key_counters[base_key]}"


def _index_rows(rows: list[SpecificationRow]) -> dict[str, SpecificationRow]:
    key_counters: dict[str, int] = defaultdict(int)
    indexed: dict[str, SpecificationRow] = {}

    for index in range(len(rows)):
        key = _row_key(rows, index, key_counters)
        indexed[key] = rows[index]

    return indexed


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_safe(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    return value


def compare_rows(
    *,
    sample_id: str,
    ground_truth_rows: list[SpecificationRow],
    predicted_rows: list[SpecificationRow],
) -> CompareResult:
    gt_index = _index_rows(ground_truth_rows)
    pred_index = _index_rows(predicted_rows)

    gt_keys = set(gt_index)
    pred_keys = set(pred_index)
    common_keys = sorted(gt_keys & pred_keys)

    exact_matches = 0
    compared_field_total = 0
    compared_field_matches = 0
    quantity_errors: list[float] = []
    mass_errors: list[float] = []
    mismatches: list[dict[str, object]] = []

    for key in common_keys:
        gt_row = gt_index[key]
        pred_row = pred_index[key]

        differing_fields: dict[str, dict[str, object]] = {}
        all_equal = True
        for field_name in FIELD_NAMES:
            gt_value = getattr(gt_row, field_name)
            pred_value = getattr(pred_row, field_name)

            if gt_value is not None:
                compared_field_total += 1
                if gt_value == pred_value:
                    compared_field_matches += 1

            if field_name in NUMERIC_FIELDS and gt_value is not None and pred_value is not None:
                gt_num = _decimal_to_float(gt_value)
                pred_num = _decimal_to_float(pred_value)
                if gt_num is not None and pred_num is not None:
                    error = abs(gt_num - pred_num)
                    if field_name == "quantity":
                        quantity_errors.append(error)
                    else:
                        mass_errors.append(error)

            if gt_value != pred_value:
                all_equal = False
                differing_fields[field_name] = {
                    "ground_truth": _json_safe(gt_value),
                    "predicted": _json_safe(pred_value),
                }

        if all_equal:
            exact_matches += 1
        else:
            mismatches.append({"key": key, "fields": differing_fields})

    return CompareResult(
        sample_id=sample_id,
        ground_truth_rows=len(ground_truth_rows),
        predicted_rows=len(predicted_rows),
        row_delta=len(predicted_rows) - len(ground_truth_rows),
        matched_rows=len(common_keys),
        exact_match_rate=(exact_matches / len(ground_truth_rows)) if ground_truth_rows else 1.0,
        field_match_rate=(compared_field_matches / compared_field_total)
        if compared_field_total
        else 1.0,
        quantity_mae=(sum(quantity_errors) / len(quantity_errors)) if quantity_errors else None,
        mass_kg_mae=(sum(mass_errors) / len(mass_errors)) if mass_errors else None,
        missing_keys=sorted(gt_keys - pred_keys),
        extra_keys=sorted(pred_keys - gt_keys),
        mismatches=mismatches,
    )


def compare_sample_against_jsonl(
    *,
    sample: SampleDefinition,
    predicted_jsonl_path: str | Path,
) -> CompareResult:
    ground_truth_rows = load_ground_truth_rows(sample)
    predicted_rows = load_rows_from_jsonl(predicted_jsonl_path)
    return compare_rows(
        sample_id=sample.sample_id,
        ground_truth_rows=ground_truth_rows,
        predicted_rows=predicted_rows,
    )


def write_compare_summary(result: CompareResult, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
