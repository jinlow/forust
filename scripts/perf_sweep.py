"""Run the `perf_phases` example over a grid and report steady-state cost per tree.

Example:
    python scripts/perf_sweep.py --data /tmp/forust-perf/w200 --rows 50000 100000 \
        --threads 8 --repeats 3 --label baseline --out agent/feat/optimize/results.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/examples/perf_phases"
WARMUP_ITERATIONS = 10


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!agent")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def run_once(data: Path, rows: int, threads: int, parallel: bool, iterations: int, extra: list[str]) -> dict:
    command = [
        str(BINARY),
        "--data", str(data),
        "--rows", str(rows),
        "--iterations", str(iterations),
        "--parallel", str(parallel).lower(),
        *extra,
    ]
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    result = json.loads(subprocess.run(command, env=env, check=True, capture_output=True, text=True).stdout)
    steady = result["tree_s"][WARMUP_ITERATIONS:]
    phases = result["phases_s"]
    per_iteration = {
        name: 1000 * value / iterations for name, value in phases.items() if name != "bin"
    }
    return {
        "data": str(data),
        "rows": rows,
        "cols": result["config"]["cols"],
        "max_depth": result["config"]["max_depth"],
        "nbins": result["config"]["nbins"],
        "threads": threads,
        "parallel": parallel,
        "iterations": iterations,
        "tree_ms": 1000 * mean(steady),
        "bin_s": phases["bin"],
        "per_iteration_ms": per_iteration,
        "eval_logloss": result["eval_logloss"],
    }


def run(data: Path, rows: int, threads: int, parallel: bool, iterations: int, extra: list[str], repeats: int) -> dict:
    samples = [run_once(data, rows, threads, parallel, iterations, extra) for _ in range(repeats)]
    record = dict(samples[0])
    record["tree_ms_samples"] = [s["tree_ms"] for s in samples]
    record["bin_s_samples"] = [s["bin_s"] for s in samples]
    record["tree_ms"] = median(record["tree_ms_samples"])
    record["bin_s"] = median(record["bin_s_samples"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=1, help="Runs per configuration; the median is reported.")
    parser.add_argument("--label", default="", help="Tag for these results, such as `baseline` or `item1`.")
    parser.add_argument("--no-serial", action="store_true", help="Skip the serial reference run.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("extra", nargs="*", help="Extra perf_phases arguments, after `--`.")
    args = parser.parse_args()

    commit = git_commit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'data':>8} {'rows':>8} {'cols':>5} {'mode':>12} {'tree ms':>9} {'speedup':>8} {'bin s':>7}")
    with open(args.out, "a") as out:
        for data, rows in itertools.product(args.data, args.rows):
            records = [] if args.no_serial else [run(data, rows, 1, False, args.iterations, args.extra, args.repeats)]
            records += [
                run(data, rows, threads, True, args.iterations, args.extra, args.repeats) for threads in args.threads
            ]
            serial_ms = records[0]["tree_ms"] if not args.no_serial else None
            for record in records:
                record.update({"label": args.label, "commit": commit, "extra": args.extra})
                out.write(json.dumps(record) + "\n")
                mode = "serial" if not record["parallel"] else f"{record['threads']} threads"
                speedup = f"{serial_ms / record['tree_ms']:>7.2f}x" if serial_ms else f"{'-':>8}"
                print(
                    f"{data.name:>8} {rows:>8} {record['cols']:>5} {mode:>12} "
                    f"{record['tree_ms']:>9.1f} {speedup} {record['bin_s']:>7.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
