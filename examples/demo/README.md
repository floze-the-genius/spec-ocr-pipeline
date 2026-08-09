# Synthetic demo

Every file in this directory is generated from synthetic data and is safe to publish.

Run the deterministic text-layer path from the repository root:

```bash
spec-ocr process examples/demo/specification.pdf \
  --output-dir demo-output \
  --quote
```

This creates `ocr.json`, `rows.jsonl`, `result.xlsx`, `quote.xlsx`, and `run.json` in
`demo-output/`. For this layout, `process` detects deterministic `PDF_TABLE_LAYER` rows and does
not call an external AI service.

The equivalent lower-level extraction and mapping steps are:

```bash
spec-ocr extract-ocr examples/demo/specification.pdf \
  --provider pdf-text \
  --output demo-output/ocr.json

spec-ocr map-ocr demo \
  --input-ocr demo-output/ocr.json \
  --source-pdf examples/demo/specification.pdf \
  --output demo-output/rows.jsonl \
  --no-image-attachments

spec-ocr export-jsonl-xlsx demo-output/rows.jsonl --output demo-output/result.xlsx
```
