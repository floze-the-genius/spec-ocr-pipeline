import argparse
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDFMiner object types.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== PDFMiner Object Type Diagnostic for spec_8 ===\n")

    page_count = 0
    object_types = {}
    for page_layout in extract_pages(args.pdf_path, laparams=LAParams()):
        page_count += 1
        page_types = {}
        for element in page_layout:
            element_type = type(element).__name__
            page_types[element_type] = page_types.get(element_type, 0) + 1
            object_types[element_type] = object_types.get(element_type, 0) + 1

            # Show first few objects of each type per page
            if page_types[element_type] <= 3:
                print(
                    f"Page {page_count} {element_type}: bbox=({element.x0:.1f},"
                    f"{element.y0:.1f},{element.x1:.1f},{element.y1:.1f})"
                )
                if hasattr(element, "get_text"):
                    text = element.get_text().strip()
                    if text:
                        print(f"  text: {text[:100]}")
        print()
        if page_count >= 3:
            break

    print("\n=== Object type counts (first 3 pages) ===")
    for element_type, count in sorted(object_types.items(), key=lambda item: -item[1]):
        print(f"  {element_type}: {count}")


if __name__ == "__main__":
    main()
