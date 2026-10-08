"""Time XGBoost `hist` on a dataset from `make_perf_data.py`, matching Forust's settings.

Example:
    python scripts/perf_xgboost.py --data /tmp/forust-perf/w200 --rows 100000 --threads 1 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_perf_data import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--nbins", type=int, default=256)
    parser.add_argument("--grow-policy", choices=["depthwise", "lossguide"], default="depthwise")
    parser.add_argument("--max-leaves", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    print(f"{'rows':>8} {'threads':>7} {'dmatrix s':>9} {'fit s':>8} {'tree ms':>8}")
    for rows in args.rows:
        data = load(args.data, max_rows=rows, max_eval_rows=rows // 4)
        for threads in args.threads:
            start = perf_counter()
            train = xgb.DMatrix(data["X_train"], label=data["y_train"], nthread=threads)
            evals = xgb.DMatrix(data["X_eval"], label=data["y_eval"], nthread=threads)
            dmatrix_s = perf_counter() - start
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "max_bin": args.nbins,
                "max_depth": args.max_depth,
                "grow_policy": args.grow_policy,
                "max_leaves": args.max_leaves,
                "eta": 0.1,
                "lambda": 1.0,
                "gamma": 0.0,
                "min_child_weight": 1.0,
                "nthread": threads,
            }
            start = perf_counter()
            xgb.train(params, train, args.iterations, evals=[(evals, "eval")], verbose_eval=False)
            fit_s = perf_counter() - start
            tree_ms = 1000 * fit_s / args.iterations
            print(f"{rows:>8} {threads:>7} {dmatrix_s:>9.2f} {fit_s:>8.2f} {tree_ms:>8.1f}", flush=True)
            if args.out:
                with open(args.out, "a") as out:
                    out.write(json.dumps({
                        "library": "xgboost",
                        "data": str(args.data),
                        "rows": rows,
                        "cols": data["metadata"]["cols"],
                        "threads": threads,
                        "max_depth": args.max_depth,
                        "grow_policy": args.grow_policy,
                        "max_leaves": args.max_leaves,
                        "nbins": args.nbins,
                        "dmatrix_s": dmatrix_s,
                        "fit_s": fit_s,
                        "iteration_ms": tree_ms,
                    }) + "\n")


if __name__ == "__main__":
    main()
