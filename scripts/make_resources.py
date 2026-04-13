import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    df = sns.load_dataset("titanic")
    X = df.select_dtypes("number").drop(columns=["survived"]).astype(float)
    y = df["survived"].astype(float)

    pd.Series(X.fillna(0).to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_no_missing.csv",
        index=False,
        header=False,
    )

    pd.Series(X.to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_with_missing.csv",
        index=False,
        header=False,
    )

    y.to_csv(
        "resources/performance.csv",
        index=False,
        header=False,
    )

    X.fare.to_csv(
        "resources/performance-fare.csv",
        index=False,
        header=False,
    )

    dfb = df.sample(
        100_000,
        random_state=0,
        replace=True,
    ).reset_index(drop=True)

    Xb = dfb.select_dtypes("number").drop(columns=["survived"]).astype(float)
    yb = dfb["survived"].astype(float)

    pd.Series(Xb.fillna(0).to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_no_missing_100k_samp_seed0.csv",
        index=False,
        header=False,
    )

    yb.to_csv(
        "resources/performance_100k_samp_seed0.csv",
        index=False,
        header=False,
    )

    # # --- Large realistic benchmark data: 500k rows x 200 columns ---
    # rng = np.random.default_rng(42)
    # n_rows = 500_000
    # n_cols = 200

    # # Mix of feature distributions to be realistic:
    # # - 100 normal features
    # # - 50 uniform features
    # # - 50 exponential features
    # X_large = np.column_stack([
    #     rng.standard_normal((n_rows, 100)),
    #     rng.uniform(0, 10, (n_rows, 50)),
    #     rng.exponential(1.0, (n_rows, 50)),
    # ])

    # # Binary target from a logistic model using a subset of features
    # logits = X_large[:, :10].sum(axis=1) + rng.standard_normal(n_rows) * 0.5
    # y_large = (1 / (1 + np.exp(-logits)) > 0.5).astype(float)

    # # Column-major (Fortran order) flattened, matching the Matrix layout
    # pd.Series(X_large.ravel(order="F")).to_csv(
    #     "resources/large_bench_500k_200col.csv",
    #     index=False,
    #     header=False,
    # )

    # pd.Series(y_large).to_csv(
    #     "resources/large_bench_500k_200col_y.csv",
    #     index=False,
    #     header=False,
    # )
