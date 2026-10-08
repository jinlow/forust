"""Measure the fraction of training-loop wall time spent in serial (main-thread) code.

Runs `perf_phases` at 8 threads under `perf`, skipping data loading and binning, and
counts main-thread samples. The main thread is not a Rayon worker and blocks without
sampling while parallel work runs, so its sample share of the window is the serial
fraction.

Example:
    python scripts/perf_serial_share.py --label baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_sweep import BINARY, git_commit

RESULTS = Path(__file__).resolve().parents[1] / "target/perf/results.jsonl"
PERF_DATA = Path("/tmp/forust-perf/serial_share.data")
SAMPLE = re.compile(r"^\s*(\d+)/(\d+)\s+([\d.]+):\s*(?:[0-9a-f]+\s*)?(\S*)")


def short_name(symbol: str) -> str:
    """Reduce a v0-mangled Rust symbol such as `...15HistogramMatrix3new` to `HistogramMatrix::new`."""
    parts, end = [], len(symbol)
    while len(parts) < 2:
        for match in re.finditer(r"\d+", symbol[:end]):
            ident = symbol[match.end():end]
            if ident and not ident[0].isdigit() and len(ident) == int(match.group()):
                parts.insert(0, ident)
                end = match.start()
                break
        else:
            break
    return "::".join(parts) if parts else symbol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", default="/tmp/forust-perf/w200")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--delay-ms", type=int, default=3500, help="Skip loading and binning.")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "perf", "record", "-q", "-F", "999", "--delay", str(args.delay_ms), "-o", str(PERF_DATA), "--",
        str(BINARY), "--data", args.data, "--rows", str(args.rows),
        "--max-depth", str(args.max_depth), "--iterations", str(args.iterations),
    ]
    subprocess.run(command, env={**os.environ, "RAYON_NUM_THREADS": "8"}, check=True, capture_output=True)
    script = subprocess.run(
        ["perf", "script", "-i", str(PERF_DATA), "-F", "pid,tid,time,ip,sym"],
        check=True, capture_output=True, text=True,
    ).stdout

    times, main_symbols, main_samples, total_samples = [], Counter(), 0, 0
    for line in script.splitlines():
        match = SAMPLE.match(line)
        if not match:
            continue
        pid, tid, time, symbol = match.groups()
        times.append(float(time))
        total_samples += 1
        if pid == tid:
            main_samples += 1
            main_symbols[short_name(symbol)] += 1

    window_s = max(times) - min(times)
    serial_fraction = main_samples / 999 / window_s
    top = [(name, round(count / main_samples, 3)) for name, count in main_symbols.most_common(8)]
    record = {
        "label": args.label,
        "commit": git_commit(),
        "kind": "serial_share",
        "data": args.data,
        "rows": args.rows,
        "max_depth": args.max_depth,
        "window_s": round(window_s, 2),
        "main_samples": main_samples,
        "total_samples": total_samples,
        "serial_fraction": round(serial_fraction, 3),
        "main_thread_top": top,
    }
    with open(args.out, "a") as out:
        out.write(json.dumps(record) + "\n")
    print(f"serial fraction of loop wall time: {serial_fraction:.1%} (window {window_s:.1f} s)")
    for name, share in top:
        print(f"  {share:6.1%}  {name[:100]}")


if __name__ == "__main__":
    main()
