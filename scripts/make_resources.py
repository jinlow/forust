import argparse

import pandas as pd
import seaborn as sns


class Inputs(argparse.Namespace):
    make_benchmark_files: bool


def main(args: Inputs):
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
    if args.make_benchmark_files:
        dfb = df.sample(
            100_000,
            random_state=0,
            replace=True,
        ).reset_index(drop=True)

        Xb = dfb.select_dtypes("number").drop(columns=["survived"]).astype(float)
        Xb = pd.concat([Xb] * 10, axis=0)
        yb = dfb["survived"].astype(float)
        print("benmark files sizes: {Xb.shape}, {y.shape}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-benchmark-files", "-mbf", action="store_true")
    args = parser.parse_args(namespace=Inputs())
    main(args)
