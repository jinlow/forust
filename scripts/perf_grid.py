"""Run the fixed performance grid from `agent/feat/optimize/plan.md` with a label.

Example:
    python scripts/perf_grid.py --label baseline --grids G1 G2 G3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_sweep import git_commit, run

DATA = Path("/tmp/forust-perf")
RESULTS = Path(__file__).resolve().parents[1] / "agent/feat/optimize/results.jsonl"

# name: (dataset, rows, max_depth, iterations, include serial)
GRIDS = {
    "G1": [("w200", rows, 5, 60, True) for rows in (25_000, 100_000, 250_000)],
    "G2": [("w200", rows, 8, 30, True) for rows in (25_000, 100_000)],
    "G3": [("w500", rows, 5, 30, True) for rows in (25_000, 100_000)],
    "G4": [("w200-1m", 1_000_000, 5, 20, False)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--grids", nargs="+", default=["G1", "G2", "G3"], choices=sorted(GRIDS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    commit = git_commit()
    print(f"label={args.label} commit={commit}")
    print(f"{'grid':>4} {'data':>8} {'rows':>8} {'depth':>5} {'mode':>10} {'tree ms':>9} {'bin s':>7}")
    with open(args.out, "a") as out:
        for grid in args.grids:
            for dataset, rows, depth, iterations, with_serial in GRIDS[grid]:
                extra = ["--max-depth", str(depth)]
                modes = ([(1, False)] if with_serial else []) + [(args.threads, True)]
                for threads, parallel in modes:
                    record = run(DATA / dataset, rows, threads, parallel, iterations, extra, args.repeats)
                    record.update({"label": args.label, "commit": commit, "grid": grid, "extra": extra})
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    mode = f"{threads} thr" if parallel else "serial"
                    print(
                        f"{grid:>4} {dataset:>8} {rows:>8} {depth:>5} {mode:>10} "
                        f"{record['tree_ms']:>9.1f} {record['bin_s']:>7.2f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
