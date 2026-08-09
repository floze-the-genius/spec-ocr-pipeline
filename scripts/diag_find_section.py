import argparse
import re
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal


def main() -> None:
    parser = argparse.ArgumentParser(description="Find section names and sheet numbers in a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== Scanning all pages for 'Общеобменная' and sheet numbers ===\n")

    for page_layout in extract_pages(args.pdf_path, laparams=LAParams()):
        page_num = getattr(page_layout, "pageid", "?")
        boxes = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text().strip()
                if text:
                    boxes.append(text)

        # Check for section name or sheet number pattern
        found = []
        for text in boxes:
            if "Общеобменная" in text or "Спецификация" in text:
                found.append(text[:100])
            if re.match(r"^\d{1,3}\s*$", text) or re.match(r"^Лист\s+\d+", text):
                found.append(f"[SHEET?] {text}")

        if found:
            print(f"--- Page {page_num} ---")
            for item in found[:10]:
                print(f"  {item}")
            print()


if __name__ == "__main__":
    main()
