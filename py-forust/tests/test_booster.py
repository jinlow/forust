from typing import Tuple
import pandas as pd
import numpy as np
from forust import GradientBooster
from xgboost import XGBClassifier
import pytest


@pytest.fixture
def X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../resources/titanic.csv")
    X = (
        df.select_dtypes("number")
        .drop(columns="survived")
        .reset_index(drop=True)
    )
    y = df["survived"]
    return X, y


@pytest.fixture
def data_dtype():
    return "float64"


def test_booster_to_xgboosts(X_y, data_dtype):
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
        dtype=data_dtype,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.0001)


def test_booster_to_xgboosts_with_missing(X_y, data_dtype):
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
        dtype=data_dtype,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.0001)


def test_booster_to_xgboosts_weighted(X_y, data_dtype):
    X, y = X_y
    X = X.fillna(0)
    w = X["fare"].to_numpy().astype(data_dtype) + 1
    xmod = XGBClassifier(
        n_estimators=50,
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
        iterations=50,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        objective_type="LogLoss",
        dtype=data_dtype,
    )
    fmod.fit(X, y=y, sample_weight=w)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.0001)


def test_booster_saving(X_y, tmp_path):
    f32_model_path = tmp_path / "modelf32.json"
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
        dtype="float32",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f32_model_path)
    fmod_loaded = GradientBooster.load_booster(f32_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    f32_model_path = tmp_path / "modelf32.json"
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
        dtype="float32",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f32_model_path)
    fmod_loaded = GradientBooster.load_booster(f32_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    f64_model_path = tmp_path / "modelf64.json"
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
        dtype="float64",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    f64_model_path = tmp_path / "modelf64.json"
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
        dtype="float64",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    fmod.save_booster(f64_model_path)
    fmod_loaded = GradientBooster.load_booster(f64_model_path)
    assert all(fmod_preds == fmod_loaded.predict(X))

    


#    f32_model_path = tmp_path / "modelf32.json"
