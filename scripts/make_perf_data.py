"""Generate synthetic binary-classification datasets for Forust performance analysis.

Data is written as raw little-endian float64 in column-major order, so the Rust and
Python harnesses can load it without parsing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

DISCRETE_LEVELS = (2, 3, 4, 5, 8, 10, 20, 50)


def assign_kinds(cols: int, rng: np.random.Generator) -> list[str]:
    kinds = (
        ["discrete"] * int(cols * 0.25)
        + ["zero_inflated"] * int(cols * 0.15)
        + ["near_constant"] * max(1, int(cols * 0.03))
    )
    kinds += ["continuous"] * (cols - len(kinds))
    return [str(kind) for kind in rng.permutation(kinds)]


def transform_column(
    col: np.ndarray, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    if kind == "discrete":
        levels = int(rng.choice(DISCRETE_LEVELS))
        # Random cut points give imbalanced level frequencies, like real categorical codes.
        cuts = np.quantile(col, np.sort(rng.uniform(0.0, 1.0, levels - 1)))
        return np.searchsorted(cuts, col).astype(np.float64), {"levels": levels}
    if kind == "zero_inflated":
        zero_rate = float(rng.uniform(0.3, 0.9))
        z = (col - col.mean()) / col.std()
        threshold = np.quantile(z, zero_rate)
        amounts = np.round(np.expm1(np.maximum(z - threshold, 0.0) * 1.5) * 100.0, 2)
        return amounts, {"zero_rate": zero_rate}
    if kind == "near_constant":
        rare_rate = float(rng.uniform(0.001, 0.01))
        threshold = np.quantile(col, 1.0 - rare_rate)
        return (col > threshold).astype(np.float64), {"rare_rate": rare_rate}
    return col, {}


def generate(
    out: Path,
    rows: int,
    eval_rows: int,
    cols: int,
    seed: int,
    class_sep: float,
    flip_y: float,
    positive_rate: float,
    missing_col_rate: float,
) -> dict:
    rng = np.random.default_rng(seed)
    params = {
        "n_informative": max(2, int(cols * 0.2)),
        "n_redundant": int(cols * 0.2),
        "n_clusters_per_class": 3,
        "class_sep": class_sep,
        "flip_y": flip_y,
        "positive_rate": positive_rate,
        "missing_col_rate": missing_col_rate,
    }
    X, y = make_classification(
        n_samples=rows + eval_rows,
        n_features=cols,
        n_informative=params["n_informative"],
        n_redundant=params["n_redundant"],
        n_repeated=0,
        n_clusters_per_class=params["n_clusters_per_class"],
        weights=[1.0 - positive_rate],
        flip_y=flip_y,
        class_sep=class_sep,
        shuffle=True,
        random_state=seed,
    )

    out.mkdir(parents=True, exist_ok=True)
    kinds = assign_kinds(cols, rng)
    missing_cols = set(
        rng.choice(cols, size=int(cols * missing_col_rate), replace=False).tolist()
    )
    columns = []
    with open(out / "X_train.f64", "wb") as train_file, open(
        out / "X_eval.f64", "wb"
    ) as eval_file:
        for j in range(cols):
            col, details = transform_column(X[:, j].copy(), kinds[j], rng)
            missing_rate = 0.0
            if j in missing_cols:
                missing_rate = float(rng.uniform(0.01, 0.30))
                col[rng.random(col.shape[0]) < missing_rate] = np.nan
            train_col = col[:rows]
            train_col.astype("<f8", copy=False).tofile(train_file)
            col[rows:].astype("<f8", copy=False).tofile(eval_file)
            observed = train_col[~np.isnan(train_col)]
            columns.append(
                {
                    "kind": kinds[j],
                    "missing_rate": missing_rate,
                    "n_unique": int(np.unique(observed).size),
                    **details,
                }
            )

    y = y.astype("<f8")
    y[:rows].tofile(out / "y_train.f64")
    y[rows:].tofile(out / "y_eval.f64")

    unique_counts = np.array([column["n_unique"] for column in columns])
    metadata = {
        "format": "little-endian float64, column-major",
        "rows": rows,
        "eval_rows": eval_rows,
        "cols": cols,
        "seed": seed,
        "params": params,
        "positive_rate": {
            "train": float(y[:rows].mean()),
            "eval": float(y[rows:].mean()),
        },
        "summary": {
            "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            "columns_with_missing": len(missing_cols),
            "columns_with_at_most_256_unique": int((unique_counts <= 256).sum()),
        },
        "columns": columns,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load(
    path: Path, max_rows: int | None = None, max_eval_rows: int | None = None
) -> dict:
    """Load a generated dataset into memory as Fortran-ordered arrays."""
    metadata = json.loads((path / "metadata.json").read_text())
    cols = metadata["cols"]

    def read(name: str, rows: int, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
        view = np.memmap(path / f"X_{name}.f64", dtype="<f8", mode="r", shape=(cols, rows)).T
        y = np.fromfile(path / f"y_{name}.f64", dtype="<f8")
        n = rows if limit is None else min(limit, rows)
        return np.array(view[:n], order="F"), y[:n].copy()

    X_train, y_train = read("train", metadata["rows"], max_rows)
    X_eval, y_eval = read("eval", metadata["eval_rows"], max_eval_rows)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--eval-rows", type=int, default=250_000)
    parser.add_argument("--cols", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-sep", type=float, default=0.3)
    parser.add_argument("--flip-y", type=float, default=0.05)
    parser.add_argument("--positive-rate", type=float, default=0.15)
    parser.add_argument("--missing-col-rate", type=float, default=0.3)
    args = parser.parse_args()

    metadata = generate(
        out=args.out,
        rows=args.rows,
        eval_rows=args.eval_rows,
        cols=args.cols,
        seed=args.seed,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        positive_rate=args.positive_rate,
        missing_col_rate=args.missing_col_rate,
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "columns"}, indent=2))


if __name__ == "__main__":
    main()
