r"""Measure native forust parallelism from Python.

Run this with the release-built package, for example from py-forust:

    python ..\scripts\check_python_parallel.py

CPU percentages are process-wide and are normalized to one logical CPU. For
example, 100% is one busy logical CPU and 800% is eight busy logical CPUs.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections.abc import Callable

import numpy as np
from forust import GradientBooster
from xgboost import XGBClassifier


def measure_fit(
    fit: Callable[[], object],
    interval: float,
) -> tuple[float, float, float, int]:
    samples: list[float] = []
    stop = threading.Event()

    def monitor() -> None:
        previous_wall = time.perf_counter()
        previous_cpu = time.process_time()
        while not stop.wait(interval):
            current_wall = time.perf_counter()
            current_cpu = time.process_time()
            wall_delta = current_wall - previous_wall
            if wall_delta > 0:
                samples.append(100.0 * (current_cpu - previous_cpu) / wall_delta)
            previous_wall = current_wall
            previous_cpu = current_cpu

    watcher = threading.Thread(target=monitor, daemon=True)
    start = time.perf_counter()
    watcher.start()
    fit()
    elapsed = time.perf_counter() - start
    stop.set()
    watcher.join()

    average_cpu = sum(samples) / len(samples) if samples else 0.0
    peak_cpu = max(samples, default=0.0)
    return elapsed, average_cpu, peak_cpu, len(samples)


def measure_forust(
    X: np.ndarray,
    y: np.ndarray,
    *,
    parallel: bool,
    num_threads: int | None,
    iterations: int,
    interval: float,
) -> tuple[float, float, float, int]:
    model = GradientBooster(
        objective_type="LogLoss",
        iterations=iterations,
        max_depth=6,
        nbins=64,
        parallel=parallel,
        num_threads=num_threads,
    )
    return measure_fit(lambda: model.fit(X, y), interval)


def measure_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_threads: int | None,
    iterations: int,
    interval: float,
) -> tuple[float, float, float, int]:
    model = XGBClassifier(
        n_estimators=iterations,
        max_depth=6,
        max_bin=64,
        tree_method="hist",
        n_jobs=num_threads,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        verbosity=0,
    )
    return measure_fit(lambda: model.fit(X, y), interval)


def warm_up_xgboost(X: np.ndarray, y: np.ndarray) -> None:
    model = XGBClassifier(
        n_estimators=1,
        max_depth=6,
        max_bin=64,
        tree_method="hist",
        n_jobs=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        verbosity=0,
    )
    model.fit(X[: min(len(X), 1_000)], y[: min(len(y), 1_000)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()

    logical_cpus = os.cpu_count() or 1
    rng = np.random.default_rng(0)
    X = rng.random((args.rows, args.cols))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float64)
    warm_up_xgboost(X, y)

    cases = [
        ("serial", False, None, None),
        ("default", True, None, None),
        ("rayon-1", True, 1, None),
        ("rayon-8", True, 8, None),
        (f"rayon-{logical_cpus}", True, logical_cpus, None),
        ("xgboost-default", None, None, None),
        ("xgboost-1", None, 1, 1),
        ("xgboost-8", None, 8, 8),
        (f"xgboost-{logical_cpus}", None, logical_cpus, logical_cpus),
    ]
    print(f"data={args.rows}x{args.cols} iterations={args.iterations}")
    print(f"logical_cpus={logical_cpus}")
    print("case             elapsed_s  avg_cpu_%  peak_cpu_%  samples")
    for name, parallel, num_threads, xgb_threads in cases:
        if parallel is None:
            elapsed, average_cpu, peak_cpu, sample_count = measure_xgboost(
                X,
                y,
                num_threads=xgb_threads,
                iterations=args.iterations,
                interval=args.interval,
            )
        else:
            elapsed, average_cpu, peak_cpu, sample_count = measure_forust(
                X,
                y,
                parallel=parallel,
                num_threads=num_threads,
                iterations=args.iterations,
                interval=args.interval,
            )
        print(
            f"{name:<16} {elapsed:>9.3f} {average_cpu:>10.1f} "
            f"{peak_cpu:>10.1f} {sample_count:>8}"
        )


if __name__ == "__main__":
    main()