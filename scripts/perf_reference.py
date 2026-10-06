"""Time XGBoost `hist` and LightGBM (optionally with GOSS) on `make_perf_data.py` data.

Settings match `perf_phases --mode fit`: learning rate 0.1, L2 1, gamma 0, minimum
leaf hessian 1, and one evaluation set scored every iteration. `--subsample` sets
XGBoost's `subsample` or LightGBM's `bagging_fraction` (with `bagging_freq=1`). Reports the time to
build the library's dataset, the training time, and eval logloss computed from raw
scores with the same formula as Forust's `log_loss`.

Example:
    python scripts/perf_reference.py --data /tmp/forust-perf/w200 --rows 100000 \
        --threads 8 --library lightgbm --goss --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_perf_data import load


def logloss_from_margin(y: np.ndarray, margin: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-margin))
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


_xgboost_warm = False


def run_xgboost(data: dict, threads: int, iterations: int, max_depth: int, nbins: int,
                learning_rate: float, early_stopping_rounds: int | None, seed: int,
                subsample: float) -> dict:
    import xgboost as xgb

    global _xgboost_warm
    if not _xgboost_warm:
        # The first DMatrix in a process takes seconds of one-time setup; keep it out of timings.
        xgb.DMatrix(np.zeros((2, 2)), label=np.zeros(2), nthread=threads)
        _xgboost_warm = True
    start = perf_counter()
    # xgboost 1.7 is very slow to build a DMatrix from Fortran-ordered arrays.
    train = xgb.DMatrix(np.ascontiguousarray(data["X_train"]), label=data["y_train"], nthread=threads)
    evals = xgb.DMatrix(np.ascontiguousarray(data["X_eval"]), label=data["y_eval"], nthread=threads)
    dataset_s = perf_counter() - start
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_bin": nbins,
        "max_depth": max_depth,
        "eta": learning_rate,
        "lambda": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "base_score": float(np.mean(data["y_train"])),
        "nthread": threads,
        "seed": seed,
        "subsample": subsample,
    }
    start = perf_counter()
    booster = xgb.train(
        params, train, iterations, evals=[(evals, "eval")], verbose_eval=False,
        early_stopping_rounds=early_stopping_rounds,
    )
    fit_s = perf_counter() - start
    if early_stopping_rounds:
        trees = booster.best_iteration + 1
        margin = booster.predict(evals, output_margin=True, iteration_range=(0, trees))
    else:
        trees = booster.num_boosted_rounds()
        margin = booster.predict(evals, output_margin=True)
    return {"dataset_s": dataset_s, "fit_s": fit_s, "trees": trees,
            "eval_logloss": logloss_from_margin(data["y_eval"], margin)}


def run_lightgbm(data: dict, threads: int, iterations: int, max_depth: int, nbins: int,
                 learning_rate: float, early_stopping_rounds: int | None, seed: int,
                 goss: bool, top_rate: float, other_rate: float, subsample: float) -> dict:
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": 2 ** max_depth,
        "max_bin": nbins,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 1.0,
        "min_data_in_leaf": 1,
        "num_threads": threads,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "seed": seed,
        "verbose": -1,
    }
    if goss and subsample < 1:
        raise ValueError("LightGBM can't combine GOSS with bagging")
    if goss:
        params.update({"data_sample_strategy": "goss", "top_rate": top_rate, "other_rate": other_rate})
    elif subsample < 1:
        params.update({"bagging_fraction": subsample, "bagging_freq": 1})
    start = perf_counter()
    train = lgb.Dataset(data["X_train"], label=data["y_train"], params=params).construct()
    evals = lgb.Dataset(data["X_eval"], label=data["y_eval"], reference=train).construct()
    dataset_s = perf_counter() - start
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds else []
    start = perf_counter()
    booster = lgb.train(
        {**params, "metric": "binary_logloss"}, train, iterations, valid_sets=[evals], callbacks=callbacks
    )
    fit_s = perf_counter() - start
    trees = booster.best_iteration if early_stopping_rounds and booster.best_iteration > 0 else booster.num_trees()
    margin = booster.predict(data["X_eval"], raw_score=True, num_iteration=trees, num_threads=threads)
    return {"dataset_s": dataset_s, "fit_s": fit_s, "trees": trees,
            "eval_logloss": logloss_from_margin(data["y_eval"], margin)}


def run(library: str, data: dict, threads: int, iterations: int, max_depth: int = 5, nbins: int = 256,
        learning_rate: float = 0.1, early_stopping_rounds: int | None = None, seed: int = 0,
        goss: bool = False, top_rate: float = 0.2, other_rate: float = 0.1, subsample: float = 1.0) -> dict:
    if library == "xgboost":
        result = run_xgboost(data, threads, iterations, max_depth, nbins, learning_rate,
                             early_stopping_rounds, seed, subsample)
    elif library == "lightgbm":
        result = run_lightgbm(data, threads, iterations, max_depth, nbins, learning_rate,
                              early_stopping_rounds, seed, goss, top_rate, other_rate, subsample)
    else:
        raise ValueError(f"unknown library: {library}")
    result["total_s"] = result["dataset_s"] + result["fit_s"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--library", choices=["xgboost", "lightgbm"], default="lightgbm")
    parser.add_argument("--goss", action="store_true", help="LightGBM only: use GOSS sampling.")
    parser.add_argument("--top-rate", type=float, default=0.2)
    parser.add_argument("--other-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="XGBoost subsample, or LightGBM bagging_fraction with bagging_freq=1.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--nbins", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.goss and args.library != "lightgbm":
        parser.error("--goss is only supported with --library lightgbm")

    print(f"{'rows':>8} {'threads':>7} {'seed':>4} {'data s':>7} {'fit s':>8} {'trees':>5} {'logloss':>9}")
    for rows in args.rows:
        data = load(args.data, max_rows=rows, max_eval_rows=rows // 4)
        for threads in args.threads:
            for seed in args.seeds:
                result = run(args.library, data, threads, args.iterations, args.max_depth, args.nbins,
                             args.learning_rate, args.early_stopping_rounds, seed, args.goss,
                             args.top_rate, args.other_rate, args.subsample)
                print(f"{rows:>8} {threads:>7} {seed:>4} {result['dataset_s']:>7.2f} {result['fit_s']:>8.2f} "
                      f"{result['trees']:>5} {result['eval_logloss']:>9.5f}", flush=True)
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    with open(args.out, "a") as out:
                        out.write(json.dumps({
                            "library": args.library, "goss": args.goss, "subsample": args.subsample,
                            "data": str(args.data),
                            "rows": rows, "cols": data["metadata"]["cols"], "threads": threads,
                            "seed": seed, "iterations": args.iterations, "max_depth": args.max_depth,
                            "nbins": args.nbins, "early_stopping_rounds": args.early_stopping_rounds,
                            **result,
                        }) + "\n")


if __name__ == "__main__":
    main()
