from __future__ import annotations

from spec_pipeline.ai.schema import AI_ROW_FIELDS, AI_SCHEMA_VERSION
from spec_pipeline.ocr.base import OcrPage

SYSTEM_PROMPT = """You extract Russian construction specification tables into strict JSON.
Use only visible OCR/text input and attached page images. Do not invent missing values. Preserve original text, units, quantities, marks, and punctuation. For structured table OCR, keep the visual column placement first and use semantics only to name the target fields. Return JSON only. Do not output reasoning, markdown, comments, or <think> tags."""


def build_extraction_prompt(*, sample_id: str, pages: list[OcrPage]) -> str:
    fields = ", ".join(AI_ROW_FIELDS)
    page_blocks = []
    for page in pages:
        page_blocks.append(
            "\n".join(
                [
                    f"PAGE {page.page_ref}",
                    f"IMAGE_SIZE {page.width}x{page.height}",
                    "OCR_TEXT:",
                    page.text,
                ]
            )
        )

    return f"""{SYSTEM_PROMPT}

Schema version: {AI_SCHEMA_VERSION}
Sample id: {sample_id}

Return exactly this JSON shape:
{{
  "rows": [
    {{
      "page_ref": "string or null",
      "section": "string or null",
      "subsection": "string or null",
      "pos": "string or null",
      "name": "string or null",
      "type_mark": "string or null",
      "code": "string or null",
      "manufacturer": "string or null",
      "unit": "string or null",
      "quantity": "string or null",
      "mass_kg": "string or null",
      "note": "string or null",
      "status": "crossed_out or null"
    }}
  ],
  "warnings": ["string"]
}}

Required row fields: {fields}

Rules:
- Do not output repeated page headers, title frames, stamps, or footer rows.
- For missing or unreadable fields, use null.
- `page_ref` must be exactly the page number after the PAGE marker, for example "1". Do not write "PAGE 1".
- Keep quantity and mass as strings exactly as read.
- All JSON values must be strings or null. Even numeric pos/quantity values must be quoted strings, for example "1" and "2.5".
- Continue section/subsection across pages until a new one appears.
- Merge visual line wraps into one logical row, including wraps that cross a page break.
- If OCR_TEXT contains PDF_TABLE_LAYER/PDF_TABLE_ROW lines, prefer them over noisier OCR HTML and return every data-like PDF_TABLE_ROW, not just the first row.
- OCR_TEXT may contain DOTS_MOCR_HTML from visual table OCR. Treat its HTML table as the row-order and column-placement anchor, ignore side stamps/title blocks, and map its cells into the target schema without moving text between columns.
- OCR_TEXT may contain overlapping OCR_BAND blocks from the same page; deduplicate repeated visual rows from band overlaps.
- If an OCR table row is marked with data-status="crossed_out" or CROSSED_OUT_HTML_ROWS, include it and set `status` to "crossed_out"; otherwise set `status` to null.
- If a row has no visible position but has its own unit/quantity/type/code/manufacturer, keep it as a separate blank-position child row.
- If a line has no position, no unit, no quantity, and only continues the previous item name, append it to the previous logical row instead of emitting a separate row. This applies across page breaks.
- If a visual table leaves a cell blank, return null for that field. Only copy a previous value when the PDF visibly uses a ditto/same-as-above mark.
- Ignore fully empty OCR HTML/grid rows; they are page layout space, not specification items.
- Do not leave `name` null for a row that has unit/quantity/type/code/manufacturer unless the item is truly unreadable.
- Put only the company/factory name in `manufacturer`.
- Put text into `note` only when it is visibly in the original `Примечание` column. Do not move phones, email, ТКП numbers, contacts, or other text from name/type/code/manufacturer columns into `note`.
- If OCR text is noisy but a page image is attached, use the image/table layout to correct obvious OCR damage.
- If OCR_TEXT has no usable table rows but page images are attached, extract the specification rows directly from the visible page images.
- Preserve original item meaning; do not translate or paraphrase technical names.
- Treat OCR_TEXT as the row-order and row-count anchor. Use page images to verify/correct fields, not to replace the whole table from memory.
- Do not replace a visible brand, manufacturer, model, or code with a different one unless the original OCR value is unreadable and the image clearly shows the replacement.
- If the image is unclear, keep the OCR text and add a warning instead of guessing.
- Never output fewer rows just because some fields are uncertain; return the row with null/uncertain fields instead.
- Header synonyms include: №/Позиция/Поз. -> pos; Наименование/Материал/Оборудование -> name; Тип/Марка/Обозначение/Модель/Артикул -> type_mark; Производитель/Завод/Изготовитель -> manufacturer; Ед. изм./Единица -> unit; Кол./Количество/Кол-во -> quantity; Примечание -> note. ТКП/contact/phone/email text belongs in `note` only when it is visibly inside the original `Примечание` column.
- If a quantity is split across visual lines inside one cell, concatenate the pieces in reading order. Do not insert decimal commas or decimal points unless they are visibly present in the original cell.

Input pages:
{chr(10).join(page_blocks)}
"""
