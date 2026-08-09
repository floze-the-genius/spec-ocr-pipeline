import argparse
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextLine


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDFMiner text-line layout.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== PDFMiner Layout Diagnostic for spec_8 ===\n")

    # Extract first few pages with layout analysis
    page_count = 0
    lines_per_page = []
    for page_layout in extract_pages(args.pdf_path, laparams=LAParams()):
        page_count += 1
        lines = []
        for element in page_layout:
            if isinstance(element, LTTextLine):
                text = element.get_text().strip()
                if text:
                    lines.append(
                        {
                            "text": text,
                            "x0": round(element.x0, 1),
                            "y0": round(element.y0, 1),
                            "x1": round(element.x1, 1),
                            "y1": round(element.y1, 1),
                            "width": round(element.width, 1),
                        }
                    )

        # Sort by y descending (top to bottom), then x ascending
        lines.sort(key=lambda line: (-line["y1"], line["x0"]))
        lines_per_page.append(lines)

        print(f"--- Page {page_count} --- ({len(lines)} text lines)")
        for line in lines[:30]:  # first 30 lines
            print(
                f"  y={line['y1']:>6.1f} x={line['x0']:>6.1f}-{line['x1']:>6.1f}"
                f" | {line['text'][:80]}"
            )
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")
        print()

        if page_count >= 3:
            break

    print(f"Total pages processed: {page_count}")
    print(f"Total text lines in first 3 pages: {sum(len(lines) for lines in lines_per_page)}")


if __name__ == "__main__":
    main()
