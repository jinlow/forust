"""Benchmark GOSS against no sampling and random sampling, with XGBoost and LightGBM references.

`run` appends one JSON record per fit to `--out`; `report` turns those records into
Markdown tables and checks the pass/fail rule for default GOSS. "Total s" includes
building the library's dataset (binning for Forust and LightGBM); "Train s" excludes
it for the references and equals the total for Forust, whose `fit` bins internally.

- accuracy: mean eval logloss over seeds at most 1% above no sampling;
- speed: full fit at least 1.5x faster than no sampling (checked on the largest
  dataset at 8 threads with a fixed number of iterations).

Example:
    python scripts/perf_goss.py run --data /tmp/forust-perf/w200 --rows 100000 --threads 8 \
        --iterations 100 --out agent/feat/goss/results.jsonl --references
    python scripts/perf_goss.py run --data /tmp/forust-perf/w200 --rows 100000 --threads 8 \
        --iterations 1000 --early-stopping-rounds 20 --out agent/feat/goss/results.jsonl
    python scripts/perf_goss.py report --results agent/feat/goss/results.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from perf_sweep import git_commit  # noqa: E402

BINARY = ROOT / "target/release/examples/perf_phases"

SETTINGS = {
    "none": [],
    "random30": ["--sample-method", "random", "--subsample", "0.3"],
    "goss": ["--sample-method", "goss", "--top-rate", "0.2", "--other-rate", "0.1"],
    "goss-small": ["--sample-method", "goss", "--top-rate", "0.1", "--other-rate", "0.1"],
}
REFERENCES = {
    "xgboost": {"library": "xgboost"},
    "lightgbm": {"library": "lightgbm"},
    "lightgbm-goss": {"library": "lightgbm", "goss": True},
}
SAMPLED = {"random30", "goss", "goss-small", "lightgbm-goss"}
LOGLOSS_TOLERANCE = 0.01
SPEEDUP_TARGET = 1.5


def run_forust(binary: Path, data: Path, rows: int, threads: int, iterations: int, max_depth: int,
               early_stopping_rounds: int | None, seed: int, setting: str) -> dict:
    command = [
        str(binary), "--mode", "fit",
        "--data", str(data),
        "--rows", str(rows),
        "--iterations", str(iterations),
        "--max-depth", str(max_depth),
        "--seed", str(seed),
        *SETTINGS[setting],
    ]
    if threads == 1:
        command += ["--parallel", "false"]
    else:
        command += ["--num-threads", str(threads)]
    if early_stopping_rounds:
        command += ["--early-stopping-rounds", str(early_stopping_rounds)]
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    result = json.loads(subprocess.run(command, env=env, check=True, capture_output=True, text=True).stdout)
    return {"total_s": result["total_s"], "fit_s": result["total_s"], "eval_logloss": result["eval_logloss"],
            "trees": result["trees"] if not early_stopping_rounds else (result["best_iteration"] or 0) + 1}


def cmd_run(args: argparse.Namespace) -> None:
    commit = git_commit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'data':>8} {'rows':>8} {'thr':>3} {'setting':>14} {'seed':>4} {'total s':>8} {'trees':>5} {'logloss':>9}")
    for data, rows, threads in itertools.product(args.data, args.rows, args.threads):
        loaded = None
        names = list(args.settings) + (list(REFERENCES) if args.references else [])
        for name in names:
            seeds = args.seeds if name in SAMPLED else list(range(args.repeats))
            for seed in seeds:
                if name in SETTINGS:
                    result = run_forust(args.binary, data, rows, threads, args.iterations, args.max_depth,
                                        args.early_stopping_rounds, seed, name)
                    library = "forust"
                else:
                    import perf_reference
                    from make_perf_data import load

                    if loaded is None:
                        loaded = load(data, max_rows=rows, max_eval_rows=rows // 4)
                    result = perf_reference.run(
                        data=loaded, threads=threads, iterations=args.iterations, max_depth=args.max_depth,
                        early_stopping_rounds=args.early_stopping_rounds, seed=seed, **REFERENCES[name],
                    )
                    library = REFERENCES[name]["library"]
                record = {
                    "setting": name, "library": library, "data": str(data), "rows": rows, "threads": threads,
                    "iterations": args.iterations, "max_depth": args.max_depth,
                    "early_stopping_rounds": args.early_stopping_rounds, "seed": seed,
                    "label": args.label, "commit": commit, **result,
                }
                with open(args.out, "a") as out:
                    out.write(json.dumps(record) + "\n")
                print(f"{data.name:>8} {rows:>8} {threads:>3} {name:>14} {seed:>4} {result['total_s']:>8.2f} "
                      f"{result['trees']:>5} {result['eval_logloss']:>9.5f}", flush=True)


def summarize(records: list[dict]) -> dict:
    losses = [r["eval_logloss"] for r in records]
    return {
        "n": len(records),
        "total_s": median(r["total_s"] for r in records),
        "fit_s": median(r["fit_s"] for r in records),
        "trees": mean(r["trees"] for r in records),
        "logloss": mean(losses),
        "logloss_sd": stdev(losses) if len(losses) > 1 else 0.0,
    }


def cmd_report(args: argparse.Namespace) -> None:
    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if args.label is not None:
        records = [r for r in records if r.get("label") == args.label]
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (Path(r["data"]).name, r["rows"], r["threads"], r["max_depth"], r["iterations"],
               r.get("early_stopping_rounds"), r.get("label", ""))
        groups[key][r["setting"]].append(r)

    largest_rows = max((key[1] for key in groups), default=0)
    checks = []
    order = list(SETTINGS) + list(REFERENCES)
    for key in sorted(groups, key=lambda k: (k[6], k[5] is not None, k[0], k[1], k[2], k[3])):
        data, rows, threads, depth, iterations, esr, label = key
        settings = groups[key]
        mode = f"early stopping {esr}, max {iterations} iterations" if esr else f"{iterations} iterations"
        print(f"\n### {data}, {rows:,} rows, {threads} threads, depth {depth}, {mode}"
              + (f" ({label})" if label else "") + "\n")
        print("| Setting | Runs | Total s | Train s | Speedup | Trees | Eval logloss | vs none |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        base = summarize(settings["none"]) if "none" in settings else None
        lgb_base = summarize(settings["lightgbm"]) if "lightgbm" in settings else None
        for name in order:
            if name not in settings:
                continue
            s = summarize(settings[name])
            # Each reference is compared against its own library's full-data run.
            reference = lgb_base if name == "lightgbm-goss" else base
            if name in REFERENCES and name != "lightgbm-goss":
                reference = None
            speedup = f"{reference['total_s'] / s['total_s']:.2f}x" if reference else "-"
            delta = f"{100 * (s['logloss'] / reference['logloss'] - 1):+.2f}%" if reference else "-"
            sd = f" ± {s['logloss_sd']:.5f}" if s["n"] > 1 else ""
            print(f"| {name} | {s['n']} | {s['total_s']:.2f} | {s['fit_s']:.2f} | {speedup} | {s['trees']:.0f} "
                  f"| {s['logloss']:.5f}{sd} | {delta} |")
        if base and "goss" in settings:
            goss = summarize(settings["goss"])
            ratio = goss["logloss"] / base["logloss"] - 1
            checks.append((f"logloss: {data} {rows:,} rows, {threads} thr, depth {depth}, {mode}",
                           f"{100 * ratio:+.2f}%", ratio <= LOGLOSS_TOLERANCE))
            if not esr and threads == 8 and rows == largest_rows:
                speedup = base["total_s"] / goss["total_s"]
                checks.append((f"speed: {data} {rows:,} rows, {threads} thr, depth {depth}, {mode}",
                               f"{speedup:.2f}x", speedup >= SPEEDUP_TARGET))

    if checks:
        print(f"\n### Pass/fail (logloss within {100 * LOGLOSS_TOLERANCE:.0f}%, "
              f"speedup at least {SPEEDUP_TARGET}x)\n")
        print("| Check | Value | Result |")
        print("| --- | ---: | --- |")
        for name, value, ok in checks:
            print(f"| {name} | {value} | {'pass' if ok else 'FAIL'} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run fits and append records.")
    run.add_argument("--data", type=Path, nargs="+", required=True)
    run.add_argument("--rows", type=int, nargs="+", required=True)
    run.add_argument("--threads", type=int, nargs="+", default=[8])
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--max-depth", type=int, default=5)
    run.add_argument("--early-stopping-rounds", type=int)
    run.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    run.add_argument("--references", action="store_true", help="Also run XGBoost and LightGBM.")
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds for sampled settings.")
    run.add_argument("--repeats", type=int, default=3, help="Timing repeats for unsampled settings.")
    run.add_argument("--binary", type=Path, default=BINARY)
    run.add_argument("--label", default="")
    run.add_argument("--out", type=Path, required=True)

    report = sub.add_parser("report", help="Print Markdown tables and pass/fail checks.")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--label", help="Only include records with this label.")

    args = parser.parse_args()
    cmd_run(args) if args.command == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
