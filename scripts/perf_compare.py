"""Print a markdown before/after table of tree time for labels in a results file.

Example:
    python scripts/perf_compare.py baseline item1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "agent/feat/optimize/results.jsonl"


def key(record: dict) -> tuple:
    return (record.get("grid", ""), Path(record["data"]).name, record["rows"], record["max_depth"], record["parallel"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+", help="First label is the reference.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--metric", default="tree_ms", choices=["tree_ms", "bin_s"])
    args = parser.parse_args()

    by_label: dict[str, dict[tuple, dict]] = {label: {} for label in args.labels}
    for line in args.results.read_text().splitlines():
        record = json.loads(line)
        if record.get("label") in by_label and "kind" not in record:
            by_label[record["label"]][key(record)] = record

    reference = by_label[args.labels[0]]
    unit = "ms" if args.metric == "tree_ms" else "s"
    header = ["Grid", "Data", "Rows", "Depth", "Mode", *[f"{label} ({unit})" for label in args.labels]]
    header += [f"{label} vs {args.labels[0]}" for label in args.labels[1:]]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * 5 + ["---:"] * (len(header) - 5)) + " |")
    for k in sorted(reference, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        grid, data, rows, depth, parallel = k
        values = [by_label[label].get(k, {}).get(args.metric) for label in args.labels]
        cells = [grid, data, f"{rows // 1000}k", str(depth), "8 threads" if parallel else "serial"]
        cells += [f"{v:.{1 if args.metric == 'tree_ms' else 2}f}" if v is not None else "-" for v in values]
        for v in values[1:]:
            cells.append(f"{(v / values[0] - 1) * 100:+.1f}%" if v is not None and values[0] else "-")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
