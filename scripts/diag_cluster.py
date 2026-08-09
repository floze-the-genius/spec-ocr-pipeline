import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDFMiner text-box clustering.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to inspect.")
    args = parser.parse_args()

    print("=== Y-cluster heuristic test for spec_8, Page 1 ===\n")

    page_layout = next(extract_pages(args.pdf_path, laparams=LAParams()))
    boxes = []
    for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):
            text = element.get_text().strip()
            if text and len(text) > 1 and element.y1 > 50:  # skip footer artifacts
                boxes.append(
                    {
                        "text": text[:60],
                        "y": round(element.y1, 1),
                        "x0": round(element.x0, 1),
                        "x1": round(element.x1, 1),
                    }
                )

    # Simple y-clustering with tolerance
    tolerance = 3.0
    boxes.sort(key=lambda b: -b["y"])
    clusters = []
    for box in boxes:
        placed = False
        for cluster in clusters:
            if abs(cluster["y_mean"] - box["y"]) <= tolerance:
                cluster["boxes"].append(box)
                cluster["y_mean"] = statistics.mean(b["y"] for b in cluster["boxes"])
                placed = True
                break
        if not placed:
            clusters.append({"y_mean": box["y"], "boxes": [box]})

    print(f"Tolerance={tolerance}, boxes={len(boxes)}, clusters={len(clusters)}")
    for cluster in clusters[:25]:
        cluster["boxes"].sort(key=lambda b: b["x0"])
        texts = [f"x={b['x0']:.0f}|{b['text'][:25]}" for b in cluster["boxes"]]
        print(f"  y={cluster['y_mean']:>6.1f} | {'  '.join(texts)}")

    # X-range analysis for columns
    print("\n=== X-range histogram (all boxes on page 1) ===")
    x_ranges = defaultdict(int)
    for box in boxes:
        # bucket x0 into 10-pt bins
        bucket = int(box["x0"] / 10) * 10
        x_ranges[bucket] += 1

    for bucket in sorted(x_ranges):
        print(f"  x={bucket:>4}-{bucket + 10:<4} : {x_ranges[bucket]} boxes")


if __name__ == "__main__":
    main()
