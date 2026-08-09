import argparse
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal

SECTION_KEYWORDS = [
    "Общеобменная",
    "Вентиляция квартир",
    "Системы ВГ1",
    "Системы ВЕ1",
    "Приток в техподполье",
    "Вентилятор",
    "Диффузор",
    "Клапан",
    "Воздуховод",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find section and item keywords in a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== Scanning all pages for section/item keywords ===\n")

    for page_layout in extract_pages(args.pdf_path, laparams=LAParams()):
        page_num = getattr(page_layout, "pageid", "?")
        matches = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text().strip()
                if any(keyword in text for keyword in SECTION_KEYWORDS):
                    matches.append(
                        {
                            "text": text[:80],
                            "y": round(element.y1, 1),
                            "x": round(element.x0, 1),
                        }
                    )

        if matches:
            print(f"--- Page {page_num} ---")
            matches.sort(key=lambda match: (-match["y"], match["x"]))
            for match in matches[:15]:
                print(f"  y={match['y']:>6.1f} x={match['x']:>6.1f} | {match['text']}")
            print()


if __name__ == "__main__":
    main()
