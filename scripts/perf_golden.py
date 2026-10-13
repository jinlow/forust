"""Save or check golden tree dumps, to prove an optimization leaves models unchanged.

Example:
    python scripts/perf_golden.py save    # once, on the baseline code
    python scripts/perf_golden.py check   # after each change
"""

from __future__ import annotations

import argparse
import filecmp
import os
import subprocess
import sys
from pathlib import Path

# Set FORUST_PERF_BINARY to use a build in another target directory.
BINARY = Path(
    os.environ.get(
        "FORUST_PERF_BINARY",
        Path(__file__).resolve().parents[1] / "target/release/examples/perf_phases",
    )
)
DATA = Path("/tmp/forust-perf")
GOLDEN = DATA / "golden"
CHECK = DATA / "golden-check"

# name: (dataset, rows, max_depth, iterations, parallel, missing_branch, extra perf_phases args)
CONFIGS = {
    "w200_100k_d5_serial": ("w200", 100_000, 5, 50, False, False, []),
    "w200_100k_d5_parallel": ("w200", 100_000, 5, 50, True, False, []),
    "w200_25k_d8_parallel": ("w200", 25_000, 8, 20, True, False, []),
    "w500_25k_d5_parallel": ("w500", 25_000, 5, 20, True, False, []),
    "w200_25k_d5_parallel_missing_branch": ("w200", 25_000, 5, 20, True, True, []),
    "w200_25k_d5_random30": ("w200", 25_000, 5, 20, True, False, ["--sample-method", "random", "--subsample", "0.3"]),
    "w200_25k_d5_goss": ("w200", 25_000, 5, 20, True, False, ["--sample-method", "goss"]),
    "w200_25k_d8_lossguide": ("w200", 25_000, 8, 20, True, False, ["--grow-policy", "LossGuide", "--max-leaves", "32"]),
    "w200_25k_d5_nbins255": ("w200", 25_000, 5, 20, True, False, ["--nbins", "255"]),
    "w200_25k_d5_nbins64_missing_branch": ("w200", 25_000, 5, 20, True, True, ["--nbins", "64"]),
    "w200_25k_d8_missing_branch": ("w200", 25_000, 8, 20, True, True, []),
    "w200_25k_d6_max_leaves": ("w200", 25_000, 6, 20, True, False, ["--max-leaves", "20"]),
    "w200_25k_d6_max_leaves_missing_branch": ("w200", 25_000, 6, 20, True, True, ["--max-leaves", "21"]),
    "w200_25k_d5_colsample": ("w200", 25_000, 5, 20, True, False, ["--colsample", "0.5"]),
    "w200_25k_d5_goss_serial": ("w200", 25_000, 5, 20, False, False, ["--sample-method", "goss"]),
}


def train(name: str, out_dir: Path) -> Path:
    dataset, rows, depth, iterations, parallel, missing_branch, extra = CONFIGS[name]
    path = out_dir / f"{name}.json"
    command = [
        str(BINARY), "--mode", "fit",
        "--data", str(DATA / dataset),
        "--rows", str(rows),
        "--max-depth", str(depth),
        "--iterations", str(iterations),
        "--parallel", str(parallel).lower(),
        "--missing-branch", str(missing_branch).lower(),
        "--save-trees", str(path),
        *extra,
    ]
    subprocess.run(command, env={**os.environ, "RAYON_NUM_THREADS": "8"}, check=True, capture_output=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["save", "check"])
    args = parser.parse_args()

    out_dir = GOLDEN if args.action == "save" else CHECK
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in CONFIGS:
        path = train(name, out_dir)
        if args.action == "save":
            print(f"saved   {name}")
            continue
        same = filecmp.cmp(path, GOLDEN / path.name, shallow=False)
        failed |= not same
        print(f"{'same   ' if same else 'CHANGED'} {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
