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
