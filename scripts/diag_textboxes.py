import argparse
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDFMiner horizontal text boxes.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== PDFMiner LTTextBoxHorizontal Diagnostic for spec_8 ===\n")

    page_count = 0
    all_boxes = []
    for page_layout in extract_pages(args.pdf_path, laparams=LAParams()):
        page_count += 1
        page_boxes = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text().strip()
                if text and len(text) > 1:
                    page_boxes.append(
                        {
                            "page": page_count,
                            "text": text,
                            "x0": round(element.x0, 1),
                            "y0": round(element.y0, 1),
                            "x1": round(element.x1, 1),
                            "y1": round(element.y1, 1),
                            "width": round(element.width, 1),
                            "height": round(element.height, 1),
                            "lines": text.count("\n") + 1,
                        }
                    )

        # Sort by y descending (top to bottom), then x ascending
        page_boxes.sort(key=lambda box: (-box["y1"], box["x0"]))
        all_boxes.extend(page_boxes)

        print(f"--- Page {page_count} --- ({len(page_boxes)} text boxes)")
        for box in page_boxes[:40]:
            text = box["text"][:90].replace("\n", " ")
            print(
                f"  y={box['y1']:>6.1f} x={box['x0']:>6.1f}-{box['x1']:>6.1f}"
                f" h={box['height']:>4.1f} | {text}"
            )
        if len(page_boxes) > 40:
            print(f"  ... ({len(page_boxes) - 40} more boxes)")
        print()

        if page_count >= 4:
            break

    print(f"\nTotal pages: {page_count}")
    total_boxes = sum(
        len([box for box in all_boxes if box["page"] == page]) for page in range(1, page_count + 1)
    )
    print(f"Total text boxes in first {page_count} pages: {total_boxes}")


if __name__ == "__main__":
    main()
