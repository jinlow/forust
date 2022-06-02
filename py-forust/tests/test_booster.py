from typing import Tuple
import pandas as pd
import numpy as np
from forust import GradientBooster
from xgboost import XGBClassifier
import pytest


@pytest.fixture
def X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../resources/titanic.csv")
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"].astype("float64")
    return X, y


def test_booster_to_xgboosts(X_y):
    X, y = X_y
    X = X.fillna(0)
    xmod = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=0,
        objective="binary:logitraw",
        tree_method="hist",
    )
    xmod.fit(X, y)
    xmod_preds = xmod.predict(X, output_margin=True)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        objective_type="LogLoss",
    )
    X_vec = X.to_numpy().ravel(order="F")
    y_ = y.to_numpy()
    fmod.fit(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
        y=y_,
        sample_weight=np.ones(y.shape),
        parallel=True,
    )
    fmod_preds = fmod.predict(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
    )
    assert np.allclose(fmod_preds, xmod_preds, rtol=0.0001)


def test_booster_to_xgboosts_with_missing(X_y):
    X, y = X_y
    X = X
    xmod = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        objective="binary:logitraw",
        eval_metric="auc",
        tree_method="hist",
        max_bin=10000,
        # tree_method="hist",
    )
    xmod.fit(X, y)
    xmod_preds = xmod.predict(X, output_margin=True)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=False,
    )
    X_vec = X.to_numpy().ravel(order="F")
    y_ = y.to_numpy()
    fmod.fit(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
        y=y_,
        sample_weight=np.ones(y.shape),
        parallel=True,
    )
    fmod_preds = fmod.predict(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
    )
    assert np.allclose(fmod_preds, xmod_preds, rtol=0.001)


def test_booster_to_xgboosts_weighted(X_y):
    X, y = X_y
    X = X.fillna(0)
    w = X["fare"].to_numpy() + 1
    xmod = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        objective="binary:logitraw",
        tree_method="hist",
    )
    xmod.fit(X, y, sample_weight=w)
    xmod_preds = xmod.predict(X, output_margin=True)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        objective_type="LogLoss",
    )
    X_vec = X.to_numpy().ravel(order="F")
    y_ = y.to_numpy()
    fmod.fit(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
        y=y_,
        sample_weight=w,
        parallel=True,
    )
    fmod_preds = fmod.predict(
        flat_data=X_vec,
        rows=X.shape[0],
        cols=X.shape[1],
    )
    assert np.allclose(fmod_preds, xmod_preds, rtol=0.001)
