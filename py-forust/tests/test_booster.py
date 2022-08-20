from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from forust import GradientBooster


@pytest.fixture
def X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../resources/titanic.csv")
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"]
    return X, y


def test_booster_to_xgboosts(X_y):
    X, y = X_y
    X = X.fillna(0)
    xmod = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1.0,
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
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


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
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


def test_booster_to_xgboosts_with_missing_sl(X_y):
    X, y = X_y
    X = X
    X["survived"] = y
    y = X["fare"]
    X = X.drop(columns=["fare"])
    xmod = XGBRegressor(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        eval_metric="auc",
        tree_method="hist",
        max_bin=10000,
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
        objective_type="SquaredLoss",
        nbins=500,
        parallel=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


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
        gamma=0,
        objective="binary:logitraw",
        tree_method="hist",
        max_bins=1000,
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
    fmod.fit(X, y=y, sample_weight=w)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.0001)


def test_booster_saving(X_y, tmp_path):
    # squared loss
    f64_model_path = tmp_path / "modelf64_sl.json"
    X, y = X_y
    X = X
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    # LogLoss
    f64_model_path = tmp_path / "modelf64_ll.json"
    X, y = X_y
    X = X
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
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))


def test_booster_saving_with_montone_constraints(X_y, tmp_path):
    # squared loss
    f64_model_path = tmp_path / "modelf64_sl.json"
    X, y = X_y
    X = X
    mono_ = X.apply(lambda x: int(np.sign(x.corr(y)))).to_dict()
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=False,
        monotone_constraints=mono_,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    # LogLoss
    f64_model_path = tmp_path / "modelf64_ll.json"
    X, y = X_y
    X = X
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
        monotone_constraints=mono_,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))


def test_monotone_constraints(X_y):
    X, y = X_y
    X = X
    mono_ = X.apply(lambda x: int(np.sign(x.corr(y)))).to_dict()
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=False,
        monotone_constraints=mono_,
    )
    fmod.fit(X, y=y)
    for f, m in mono_.items():
        p_d = fmod.partial_dependence(X, feature=f)
        p_d = p_d[~np.isnan(p_d[:, 0])]
        if m < 0:
            assert np.all(p_d[0:-1, 1] >= p_d[1:, 1])
        else:
            assert np.all(p_d[0:-1, 1] <= p_d[1:, 1])
