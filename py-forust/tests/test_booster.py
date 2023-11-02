from __future__ import annotations

import itertools
import json
import pickle
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

import forust
from forust import GradientBooster


def loggodds_to_odds(v):
    return 1 / (1 + np.exp(-v))


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
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


def test_sklearn_clone(X_y):
    X, y = X_y
    fmod = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=True,
    )
    fmod_cloned = clone(fmod)
    fmod_cloned.fit(X, y=y)

    fmod.fit(X, y=y)

    # After it's fit it can still be cloned.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmod_cloned_post_fit = clone(fmod)
    fmod_cloned_post_fit.fit(X, y=y)

    fmod_preds = fmod.predict(X)
    fmod_cloned_preds = fmod_cloned.predict(X)
    fmod_cloned_post_fit_preds = fmod_cloned_post_fit.predict(X)

    assert np.allclose(fmod_preds, fmod_cloned_preds)
    assert np.allclose(fmod_preds, fmod_cloned_post_fit_preds)


def test_multiple_fit_calls(X_y):
    X, y = X_y
    fmod = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=True,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)

    fmod.fit(X, y=y)
    fmod_fit_again_preds = fmod.predict(X)

    assert np.allclose(fmod_preds, fmod_fit_again_preds)


def test_different_data_passed(X_y):
    X, y = X_y
    fmod = GradientBooster(
        objective_type="LogLoss",
    )
    fmod.fit(X, y=y)
    fmod.predict(X)
    with pytest.raises(ValueError, match="Columns mismatch"):
        fmod.predict(X.iloc[:, 0:4])

    with pytest.raises(ValueError, match="Columns mismatch"):
        X_ = X.rename(columns=lambda x: x + "_wrong" if x == "age" else x)
        fmod.predict(X_)


def test_booster_from_numpy(X_y):
    X, y = X_y
    X = X.astype("float32").astype("float64")
    fmod1 = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=False,
    )
    fmod1.fit(X, y=y)
    fmod1_preds = fmod1.predict(X)

    fmod2 = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=False,
    )
    fmod2.fit(X, y=y)
    fmod2_preds = fmod2.predict(X.to_numpy())

    fmod3 = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1.0,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=False,
    )
    fmod3.fit(X.to_numpy().astype("float32"), y=y)
    fmod3_preds = fmod3.predict(X)

    assert np.allclose(fmod1_preds, fmod2_preds)
    assert np.allclose(fmod2_preds, fmod3_preds, atol=0.00001)


@pytest.mark.parametrize(
    "with_mono,reverse,missing",
    itertools.product([True, False], [True, False], [-9999, np.nan, 11, 9999]),
)
def test_booster_to_xgboosts_with_missing(
    X_y, with_mono: bool, reverse: bool, missing: float
):
    X, y = X_y
    if with_mono:
        mono_ = X.apply(lambda x: int(np.sign(x.corr(y)))).to_dict()
    else:
        mono_ = None
    if reverse:
        y = 1 - y
    if not np.isnan(missing):
        X = X.fillna(missing)
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
        monotone_constraints=mono_,
        missing=missing,
    )
    xmod.fit(X, y)
    xmod_preds = xmod.predict(X, output_margin=True)

    fmod = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=False,
        monotone_constraints=mono_,
        missing=missing,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


def test_booster_to_xgboosts_trees_to_dataframe(X_y):
    X, y = X_y
    X = X
    xmod = XGBClassifier(
        n_estimators=50,
        learning_rate=0.03,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        objective="binary:logitraw",
        eval_metric="auc",
        tree_method="hist",
    )
    xmod.fit(X, y)
    xmod_preds = xmod.predict(X, output_margin=True)

    fmod = GradientBooster(
        base_score=0.5,
        iterations=50,
        learning_rate=0.03,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        parallel=True,
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)
    ftrees = fmod.trees_to_dataframe()
    xtrees = xmod.get_booster().trees_to_dataframe()
    assert ftrees.shape[0] == xtrees.shape[0]
    assert (ftrees["Feature"] == "Leaf").sum() == (xtrees["Feature"] == "Leaf").sum()


def test_get_node_list(X_y):
    X, y = X_y
    X = X
    fmod = GradientBooster(
        base_score=0.5,
        iterations=50,
        learning_rate=0.03,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        parallel=True,
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    assert all(
        [
            isinstance(n.split_feature, int)
            for i, tree in enumerate(fmod.get_node_lists(map_features_names=False))
            for n in tree
        ]
    )
    assert all(
        [
            isinstance(n.split_feature, str)
            for i, tree in enumerate(fmod.get_node_lists(map_features_names=True))
            for n in tree
        ]
    )


@pytest.mark.parametrize("is_numpy", [True, False])
def test_importance(X_y: tuple[pd.DataFrame, pd.Series], is_numpy: bool):
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
    if is_numpy:
        X_ = X.to_numpy()
    else:
        X_ = X
    xmod.fit(X_, y)
    fmod = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=False,
    )
    fmod.fit(X_, y)
    x_imp = xmod.get_booster().get_score(importance_type="weight")
    f_imp = fmod.calculate_feature_importance(method="Weight", normalize=False)

    def make_key(i: str | int, is_xgboost: bool, is_numpy) -> str | int:
        if is_numpy:
            if is_xgboost:
                return f"f{i}"
            else:
                return i
        else:
            return i

    assert all(
        [
            f_imp[make_key(f, False, is_numpy)] == x_imp[make_key(f, True, is_numpy)]
            for f in f_imp.keys()
        ]
    )

    x_imp = xmod.get_booster().get_score(importance_type="gain")
    f_imp = fmod.calculate_feature_importance(method="Gain", normalize=False)
    assert all(
        [
            np.allclose(
                f_imp[make_key(f, False, is_numpy)], x_imp[make_key(f, True, is_numpy)]
            )
            for f in f_imp.keys()
        ]
    )

    x_imp = xmod.get_booster().get_score(importance_type="total_gain")
    f_imp = fmod.calculate_feature_importance(method="TotalGain", normalize=False)
    assert all(
        [
            np.allclose(
                f_imp[make_key(f, False, is_numpy)], x_imp[make_key(f, True, is_numpy)]
            )
            for f in f_imp.keys()
        ]
    )

    x_imp = xmod.get_booster().get_score(importance_type="cover")
    f_imp = fmod.calculate_feature_importance(method="Cover", normalize=False)
    assert all(
        [
            np.allclose(
                f_imp[make_key(f, False, is_numpy)], x_imp[make_key(f, True, is_numpy)]
            )
            for f in f_imp.keys()
        ]
    )

    x_imp = xmod.get_booster().get_score(importance_type="total_cover")
    f_imp = fmod.calculate_feature_importance(method="TotalCover", normalize=False)
    assert all(
        [
            np.allclose(
                f_imp[make_key(f, False, is_numpy)], x_imp[make_key(f, True, is_numpy)]
            )
            for f in f_imp.keys()
        ]
    )


test_args = itertools.product(
    (True, False), ["Weight", "Cover", "Gain", "TotalGain", "TotalCover"]
)


@pytest.mark.parametrize("is_numpy,importance_method", test_args)
def test_feature_importance_method_init(
    X_y: tuple[pd.DataFrame, pd.Series], is_numpy: bool, importance_method: str
) -> None:
    X, y = X_y
    fmod = GradientBooster(
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=False,
        feature_importance_method=importance_method,
    )
    if is_numpy:
        X_ = X.to_numpy()
    else:
        X_ = X
    fmod.fit(X_, y)

    f_imp = fmod.calculate_feature_importance(method=importance_method, normalize=True)

    # for ft, cf in zip(fmod.feature_names_in_, fmod.feature_importances_):
    #     print(f_imp.get(ft, 0.0), cf)
    #     print(f_imp.get(ft, 0.0) == cf)

    if not is_numpy:
        assert all(
            [
                f_imp.get(ft, 0.0) == cf
                for ft, cf in zip(fmod.feature_names_in_, fmod.feature_importances_)
            ]
        )
    else:
        assert all(
            [
                f_imp.get(ft, 0.0) == cf
                for ft, cf in zip(range(fmod.n_features_), fmod.feature_importances_)
            ]
        )


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
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


def test_booster_with_new_missing(X_y):
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
        parallel=True,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)

    Xm = X.copy().fillna(-9999)
    fmod2 = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        missing=-9999,
    )
    fmod2.fit(Xm, y=y)
    fmod_preds2 = fmod2.predict(Xm)
    assert np.allclose(fmod_preds, fmod_preds2, atol=0.00001)


def test_booster_with_seed(X_y):
    X, y = X_y
    X = X
    fmod1 = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        subsample=0.5,
        seed=0,
    )
    fmod1.fit(X, y=y)
    fmod1_preds = fmod1.predict(X)

    fmod2 = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        subsample=0.5,
        seed=0,
    )
    fmod2.fit(X, y=y)
    fmod2_preds = fmod2.predict(X)
    assert np.allclose(fmod2_preds, fmod1_preds, atol=0.0000001)

    fmod3 = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        subsample=0.5,
        seed=1,
    )
    fmod3.fit(X, y=y)
    fmod3_preds = fmod3.predict(X)
    assert not np.allclose(fmod3_preds, fmod2_preds, atol=0.0000001)

    fmod4 = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
    )
    fmod4.fit(X, y=y)
    fmod4_preds = fmod4.predict(X)
    assert not np.allclose(fmod4_preds, fmod2_preds, atol=0.00001)


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
        base_score=0.5,
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        objective_type="LogLoss",
        initialize_base_score=False,
    )
    fmod.fit(X, y=y, sample_weight=w)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.0001)


def pickle_booster(model: GradientBooster, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(model, file)


def unpickle_booster(path: str) -> GradientBooster:
    with open(path, "rb") as file:
        return pickle.load(file)


def save_booster(model: GradientBooster, path: str) -> None:
    model.save_booster(path)


def load_booster(path: str) -> GradientBooster:
    return GradientBooster.load_booster(path)


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
        parallel=True,
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

    for f, m in mono_.items():
        p_d = fmod.partial_dependence(X, feature=f, samples=None)
        p_d = p_d[~np.isnan(p_d[:, 0])]
        if m < 0:
            assert np.all(p_d[0:-1, 1] >= p_d[1:, 1])
        else:
            assert np.all(p_d[0:-1, 1] <= p_d[1:, 1])


def test_partial_dependence_errors(X_y):
    X, y = X_y
    fmod = GradientBooster()
    fmod.fit(X, y=y)
    with pytest.raises(ValueError, match="If `feature` is a string, then"):
        fmod.partial_dependence(X.to_numpy(), "pclass")

    fmod = GradientBooster()
    fmod.fit(X.to_numpy(), y=y)
    with pytest.warns(
        expected_warning=UserWarning, match="No feature names were provided at fit"
    ):
        res = fmod.partial_dependence(X, "pclass")

    # This is the same as if we used a dataframe at fit.
    fmod = GradientBooster()
    fmod.fit(X, y=y)
    assert np.allclose(fmod.partial_dependence(X, "pclass"), res)

    fmod = GradientBooster()
    fmod.fit(X, y=y)
    pclass_n = next(i for i, ft in enumerate(X.columns) if ft == "pclass")
    assert np.allclose(
        fmod.partial_dependence(X, "pclass"), fmod.partial_dependence(X, pclass_n)
    )
    assert np.allclose(
        fmod.partial_dependence(X, "pclass"),
        fmod.partial_dependence(X.to_numpy(), pclass_n),
    )

    with pytest.raises(
        ValueError, match="The parameter `feature` must be a string, or an int"
    ):
        fmod.partial_dependence(X, 1.0)


def test_partial_dependence_exclude_missing(X_y):
    X, y = X_y
    fmod = GradientBooster()
    fmod.fit(X, y)
    res1 = fmod.partial_dependence(X, "age", samples=None)
    res2 = fmod.partial_dependence(X, "age", samples=None, exclude_missing=False)
    assert (res1.shape[0] + 1) == res2.shape[0]
    assert np.allclose(res2[~np.isnan(res2[:, 0]), :], res1)


def test_booster_to_xgboosts_with_contributions_missing_branch_methods(X_y):
    X, y = X_y
    X = X
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        create_missing_branch=True,
        allow_missing_splits=True,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
        missing_node_treatment="AssignToParent",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    contribs_average = fmod.predict_contributions(X)
    fmod_preds[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), fmod_preds)

    contribs_weight = fmod.predict_contributions(X, method="Weight")
    assert (contribs_weight[:, :-1][X.isna()] == 0).all()
    assert np.allclose(contribs_weight.sum(1), fmod_preds)
    assert not np.allclose(contribs_weight, contribs_average)

    contribs_branch_difference = fmod.predict_contributions(
        X, method="BranchDifference"
    )
    assert (contribs_branch_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_branch_difference.sum(1), fmod_preds)
    assert not np.allclose(contribs_branch_difference, contribs_average)

    contribs_midpoint_difference = fmod.predict_contributions(
        X, method="MidpointDifference"
    )
    assert (contribs_midpoint_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_midpoint_difference.sum(1), fmod_preds)
    assert not np.allclose(contribs_midpoint_difference, contribs_average)

    contribs_mode_difference = fmod.predict_contributions(X, method="ModeDifference")
    assert (contribs_mode_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_mode_difference.sum(1), fmod_preds)
    assert not np.allclose(contribs_mode_difference, contribs_average)


def test_booster_to_xgboosts_with_contributions(X_y):
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
        parallel=True,
        base_score=0.5,
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    contribs_average = fmod.predict_contributions(X)
    fmod_preds[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), fmod_preds)

    contribs_weight = fmod.predict_contributions(X, method="Weight")
    assert np.allclose(contribs_weight.sum(1), fmod_preds)
    assert not np.allclose(contribs_weight, contribs_average)

    contribs_difference = fmod.predict_contributions(X, method="BranchDifference")
    assert not np.allclose(contribs_difference.sum(1), fmod_preds)
    assert not np.allclose(contribs_difference, contribs_average)

    contribs_proba = fmod.predict_contributions(X, method="ProbabilityChange")
    assert np.allclose(contribs_proba.sum(1), loggodds_to_odds(fmod_preds))
    assert not np.allclose(contribs_proba, contribs_average)

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
        base_score=0.5,
    )
    xmod.fit(X, y)
    import xgboost as xgb

    xmod_contribs = xmod.get_booster().predict(
        xgb.DMatrix(X), approx_contribs=True, pred_contribs=True
    )
    assert np.allclose(contribs_average, xmod_contribs, atol=0.000001)
    assert np.allclose(fmod_preds, xmod.predict(X, output_margin=True), atol=0.00001)


def test_booster_to_xgboosts_with_contributions_shapley(X_y):
    X, y = X_y
    X = X.round(0)
    fmod = GradientBooster(
        iterations=2,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=1_000,
        parallel=True,
        base_score=0.5,
        initialize_base_score=False,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    contribs_average = fmod.predict_contributions(X)
    fmod_preds[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), fmod_preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), fmod_preds)

    contribs_shapley = fmod.predict_contributions(X, method="Shapley")
    assert np.allclose(contribs_shapley.sum(1), fmod_preds)
    assert not np.allclose(contribs_shapley, contribs_average)

    xmod = XGBClassifier(
        n_estimators=2,
        learning_rate=0.3,
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        objective="binary:logitraw",
        eval_metric="auc",
        tree_method="hist",
        max_bin=20000,
        base_score=0.5,
    )
    xmod.fit(X, y)
    import xgboost as xgb

    xmod_contribs_shapley = xmod.get_booster().predict(
        xgb.DMatrix(X), approx_contribs=False, pred_contribs=True
    )
    assert np.allclose(contribs_shapley, xmod_contribs_shapley, atol=0.00001)
    assert np.allclose(fmod_preds, xmod.predict(X, output_margin=True), atol=0.00001)


def test_missing_branch_with_contributions(X_y):
    X, y = X_y
    X = X
    fmod_miss_leaf = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
        allow_missing_splits=False,
        create_missing_branch=True,
    )
    fmod_miss_leaf.fit(X, y)
    fmod_miss_leaf_preds = fmod_miss_leaf.predict(X)
    fmod_miss_leaf_conts = fmod_miss_leaf.predict_contributions(X)
    assert np.allclose(fmod_miss_leaf_conts.sum(1), fmod_miss_leaf_preds)

    fmod_miss_leaf_conts = fmod_miss_leaf.predict_contributions(X, method="weight")
    assert np.allclose(fmod_miss_leaf_conts.sum(1), fmod_miss_leaf_preds)

    fmod_miss_leaf_conts = fmod_miss_leaf.predict_contributions(
        X, method="probability-change"
    )
    assert np.allclose(
        fmod_miss_leaf_conts.sum(1), loggodds_to_odds(fmod_miss_leaf_preds)
    )

    fmod_miss_branch = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
        allow_missing_splits=True,
        create_missing_branch=True,
    )
    fmod_miss_branch.fit(X, y)
    fmod_miss_branch_preds = fmod_miss_branch.predict(X)
    fmod_miss_branch_conts = fmod_miss_branch.predict_contributions(X)
    assert np.allclose(fmod_miss_branch_conts.sum(1), fmod_miss_branch_preds)
    assert not np.allclose(fmod_miss_branch_preds, fmod_miss_leaf_preds)

    fmod_miss_branch_conts = fmod_miss_branch.predict_contributions(X, method="weight")
    assert np.allclose(fmod_miss_branch_conts.sum(1), fmod_miss_branch_preds)

    fmod_miss_branch_conts = fmod_miss_branch.predict_contributions(
        X, method="probability-change"
    )
    assert np.allclose(
        fmod_miss_branch_conts.sum(1), loggodds_to_odds(fmod_miss_branch_preds)
    )


def test_text_dump(X_y):
    fmod = GradientBooster()
    fmod.fit(*X_y)
    assert len(fmod.text_dump()) > 0


def test_early_stopping_with_dev(X_y):
    X, y = X_y

    val = y.index.to_series().isin(y.sample(frac=0.25, random_state=0))

    model = GradientBooster(log_iterations=1, early_stopping_rounds=4, iterations=100)
    model.fit(
        X.loc[~val, :], y.loc[~val], evaluation_data=[(X.loc[val, :], y.loc[val])]
    )

    # Did we actually stop?
    n_trees = json.loads(model.json_dump())["trees"]
    # Didn't improve for 4 rounds, and one more triggered it.
    assert len(n_trees) == model.get_best_iteration() + 5
    assert len(n_trees) == model.get_evaluation_history().shape[0]
    assert model.get_best_iteration() < 99


def test_evaluation_history_none(X_y):
    X, y = X_y

    val = y.index.to_series().isin(y.sample(frac=0.25, random_state=0))

    model = GradientBooster(iterations=100)
    model.fit(X.loc[~val, :], y.loc[~val])

    assert model.get_evaluation_history() is None


def test_early_stopping_with_dev_val(X_y):
    X, y = X_y

    val = y.index.to_series().isin(y.sample(frac=0.25, random_state=0))

    w = np.random.default_rng(0).uniform(0.5, 1.0, size=val.sum())

    model = GradientBooster(log_iterations=1, early_stopping_rounds=4, iterations=100)

    with pytest.warns(UserWarning, match="Multiple evaluation datasets"):
        model.fit(
            X.loc[~val, :],
            y.loc[~val],
            evaluation_data=[
                (X.loc[val, :], y.loc[val], w),
                (X.loc[~val, :], y.loc[~val]),
                # Last item is used for early stopping.
                (X.loc[val, :], y.loc[val]),
            ],
        )

    # the weight is being used.
    history = model.get_evaluation_history()
    assert history[2, 0] != history[2, 1]

    # Did we actually stop?
    n_trees = json.loads(model.json_dump())["trees"]
    # Didn't improve for 4 rounds, and one more triggered it.
    assert len(n_trees) == model.get_best_iteration() + 5
    assert len(n_trees) == model.get_evaluation_history().shape[0]
    assert model.get_best_iteration() < 99
    assert model.number_of_trees == model.get_best_iteration() + 5


def test_goss_sampling_method(X_y):
    X, y = X_y
    X = X
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        sample_method="goss",
        min_leaf_weight=1,
        gamma=1,
        top_rate=0.2,
        other_rate=0.3,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
    )
    fmod.fit(X, y=y)

    assert True


def test_booster_to_xgboosts_with_base_score_log_loss(X_y):
    from scipy.special import logit

    X, y = X_y
    xmod = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        objective="binary:logitraw",
        base_score=logit(y.mean()),
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        eval_metric="auc",
        tree_method="hist",
        max_bin=500,
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
        parallel=True,
        initialize_base_score=True,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.00001)


def test_booster_to_xgboosts_with_base_score_squared_loss(X_y):
    X, y = X_y
    X = X
    X["survived"] = y
    y = X["fare"]
    X = X.drop(columns=["fare"])
    xmod = XGBRegressor(
        n_estimators=100,
        learning_rate=0.3,
        objective="reg:squarederror",
        base_score=np.mean(y),
        max_depth=5,
        reg_lambda=1,
        min_child_weight=1,
        gamma=1,
        eval_metric="auc",
        tree_method="hist",
        max_bin=500,
    )
    xmod.fit(X, y)
    xmod_preds = xmod.predict(X)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=True,
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    assert np.allclose(fmod_preds, xmod_preds, atol=0.001)


def test_booster_terminate_missing_features(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=10,
        l2=1,
        min_leaf_weight=0,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=True,
        allow_missing_splits=True,
        create_missing_branch=True,
        terminate_missing_features=["pclass", "fare"],
    )
    fmod.fit(X, y=y)
    [pclass_idx] = [i for i, f in enumerate(X.columns) if f == "pclass"]
    [fare_idx] = [i for i, f in enumerate(X.columns) if f == "fare"]

    def check_feature_split(feature: int, tree: dict, n: int):
        node = tree[n]
        if node["is_leaf"]:
            return
        if node["split_feature"] == feature:
            if not tree[node["missing_node"]]["is_leaf"]:
                raise ValueError("Node split more!")
        check_feature_split(feature, tree, node["missing_node"])
        check_feature_split(feature, tree, node["left_child"])
        check_feature_split(feature, tree, node["right_child"])

    # Does pclass never get split out?
    for tree in json.loads(fmod.json_dump())["trees"]:
        check_feature_split(pclass_idx, tree["nodes"], 0)
        check_feature_split(fare_idx, tree["nodes"], 0)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=True,
        allow_missing_splits=True,
        create_missing_branch=True,
        # terminate_missing_features=["pclass"]
    )
    fmod.fit(X, y=y)

    # Does age never get split out?
    pclass_one_bombed = False
    for tree in json.loads(fmod.json_dump())["trees"]:
        try:
            check_feature_split(pclass_idx, tree["nodes"], 0)
        except ValueError:
            pclass_one_bombed = True
    assert pclass_one_bombed

    fare_one_bombed = False
    for tree in json.loads(fmod.json_dump())["trees"]:
        try:
            check_feature_split(fare_idx, tree["nodes"], 0)
        except ValueError:
            fare_one_bombed = True
    assert fare_one_bombed


def test_missing_treatment(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=10,
        l2=1,
        min_leaf_weight=1,
        create_missing_branch=True,
        allow_missing_splits=False,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
        missing_node_treatment="AverageLeafWeight",
    )
    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    contribus_weight = fmod.predict_contributions(X, method="weight")
    assert contribus_weight.shape[1] == X.shape[1] + 1
    assert np.allclose(contribus_weight.sum(1), fmod_preds)
    assert np.allclose(contribus_weight[:, :-1][X.isna()], 0, atol=0.0000001)
    assert (contribus_weight[:, :-1][X.isna()] == 0).all()
    assert (X.isna().sum().sum()) > 0

    contribus_average = fmod.predict_contributions(X, method="average")
    assert contribus_average.shape[1] == X.shape[1] + 1
    # Wont be exactly zero because of floating point error.
    assert np.allclose(contribus_average[:, :-1][X.isna()], 0, atol=0.0000001)
    assert np.allclose(contribus_weight.sum(1), fmod_preds)
    assert (X.isna().sum().sum()) > 0

    # Even the contributions should be approximately the same.
    assert np.allclose(contribus_average, contribus_weight, atol=0.0000001)


def test_missing_treatment_split_further(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=10,
        l2=1,
        min_leaf_weight=1,
        create_missing_branch=True,
        allow_missing_splits=True,
        gamma=1,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        base_score=0.5,
        missing_node_treatment="AverageLeafWeight",
        terminate_missing_features=["pclass", "fare"],
    )

    [pclass_idx] = [i for i, f in enumerate(X.columns) if f == "pclass"]
    [fare_idx] = [i for i, f in enumerate(X.columns) if f == "fare"]

    fmod.fit(X, y=y)
    fmod_preds = fmod.predict(X)
    contribus_weight = fmod.predict_contributions(X, method="weight")

    # For the features that we terminate missing, these features should have a contribution
    # of zero.
    assert (contribus_weight[:, pclass_idx][X["pclass"].isna()] == 0).all()
    assert (contribus_weight[:, fare_idx][X["fare"].isna()] == 0).all()

    # For the others it might not be zero.
    all_others = [i for i in range(X.shape[1]) if i not in [fare_idx, pclass_idx]]
    assert (contribus_weight[:, all_others][X.iloc[:, all_others].isna()] != 0).any()
    assert (contribus_weight[:, all_others][X.iloc[:, all_others].isna()] != 0).any()

    contribus_average = fmod.predict_contributions(X, method="average")
    assert np.allclose(contribus_weight.sum(1), fmod_preds)
    assert np.allclose(contribus_average, contribus_weight, atol=0.0000001)


def test_AverageNodeWeight_missing_node_treatment(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=10,
        l2=1,
        min_leaf_weight=0,
        objective_type="LogLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=True,
        allow_missing_splits=True,
        create_missing_branch=True,
        terminate_missing_features=["pclass", "fare"],
        missing_node_treatment="AverageNodeWeight",
    )
    fmod.fit(X, y=y)

    def check_missing_is_average(tree: dict, n: int, learning_rate: float = 0.3):
        node = tree[n]
        if node["is_leaf"]:
            return
        missing_weight = tree[node["missing_node"]]["weight_value"]
        left = tree[node["left_child"]]
        right = tree[node["right_child"]]

        weighted_weight = (
            (left["weight_value"] * left["hessian_sum"])
            + (right["weight_value"] * right["hessian_sum"])
        ) / (left["hessian_sum"] + right["hessian_sum"])

        assert np.isclose(missing_weight, weighted_weight)

        check_missing_is_average(tree, node["missing_node"])
        check_missing_is_average(tree, node["left_child"])
        check_missing_is_average(tree, node["right_child"])

    # Does pclass never get split out?
    for tree in json.loads(fmod.json_dump())["trees"]:
        check_missing_is_average(tree["nodes"], 0)

    fmod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=1,
        objective_type="SquaredLoss",
        nbins=500,
        parallel=True,
        initialize_base_score=True,
        allow_missing_splits=True,
        create_missing_branch=True,
        # terminate_missing_features=["pclass"]
    )
    fmod.fit(X, y=y)

    with pytest.raises(AssertionError):
        for tree in json.loads(fmod.json_dump())["trees"]:
            check_missing_is_average(tree["nodes"], 0)


def test_get_params(X_y):
    X, y = X_y
    r = 0.00001
    fmod = GradientBooster(learning_rate=r)
    assert fmod.get_params()["learning_rate"] == r
    fmod.fit(X, y)
    assert fmod.get_params()["learning_rate"] == r


def test_set_params(X_y):
    X, y = X_y
    r = 0.00001
    fmod = GradientBooster()
    assert fmod.get_params()["learning_rate"] != r
    assert fmod.set_params(learning_rate=r)
    assert fmod.get_params()["learning_rate"] == r
    fmod.fit(X, y)


def test_compat_gridsearch(X_y):
    X, y = X_y
    fmod = GradientBooster()
    parameters = {"learning_rate": [0.1, 0.03], "subsample": [1.0, 0.8]}
    clf = GridSearchCV(
        fmod,
        parameters,
        scoring=lambda est, X, y: roc_auc_score(y, est.predict(X)),
    )
    clf.fit(X, y)
    assert len(clf.cv_results_["mean_test_score"]) > 0


# All save and load methods
@pytest.mark.parametrize(
    "load_func,save_func",
    [(unpickle_booster, pickle_booster), (load_booster, save_booster)],
)
class TestSaveLoadFunctions:
    def test_early_stopping_rounds(
        self, X_y: tuple[pd.DataFrame, pd.Series], tmp_path, load_func, save_func
    ):
        X, y = X_y
        X = X
        fmod = GradientBooster(
            iterations=200,
            learning_rate=0.3,
            max_depth=5,
            l2=1,
            min_leaf_weight=1,
            gamma=1,
            objective_type="LogLoss",
            nbins=500,
            parallel=True,
            base_score=0.5,
            early_stopping_rounds=2,
            evaluation_metric="AUC",
        )
        fmod.fit(X, y, evaluation_data=[(X, y)])
        preds = fmod.predict(X)
        mod_path = tmp_path / "early_stopping_model.json"
        save_func(fmod, mod_path)
        loaded = load_func(mod_path)
        assert np.allclose(loaded.predict(X), preds)
        history = fmod.get_evaluation_history()
        assert history is not None
        assert np.isclose(roc_auc_score(y, preds), history.max())
        best_iteration = fmod.get_best_iteration()
        assert best_iteration == fmod.best_iteration
        assert best_iteration is not None
        assert best_iteration < history.shape[0]

        assert fmod.prediction_iteration == (fmod.best_iteration + 1)
        fmod.set_prediction_iteration(4)
        assert fmod.prediction_iteration == 4
        new_preds = fmod.predict(X)
        assert not np.allclose(new_preds, preds)
        save_func(fmod, mod_path)
        loaded = load_func(mod_path)
        assert np.allclose(loaded.predict(X), new_preds)

    @pytest.mark.parametrize(
        "initialize_base_score",
        [True, False],
    )
    def test_booster_metadata(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
        tmp_path,
        initialize_base_score,
        load_func,
        save_func,
    ):
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
            parallel=True,
            base_score=0.5,
            initialize_base_score=initialize_base_score,
        )
        fmod.fit(X, y=y)
        fmod_preds = fmod.predict(X)
        save_func(fmod, f64_model_path)
        fmod.insert_metadata("test-info", "some-info")
        assert fmod.get_metadata("test-info") == "some-info"
        save_func(fmod, f64_model_path)

        loaded = load_func(f64_model_path)
        assert loaded.get_metadata("test-info") == "some-info"

        with pytest.raises(KeyError):
            loaded.get_metadata("No-key")

        # Make sure the base score is adjusted
        assert fmod.base_score == loaded.base_score
        if initialize_base_score:
            assert loaded.base_score != 0.5

        loaded_dict = loaded.__dict__
        fmod_dict = fmod.__dict__
        assert sorted(loaded_dict.keys()) == sorted(fmod_dict.keys())
        for k, v in loaded_dict.items():
            c_v = fmod_dict[k]
            if isinstance(v, float):
                if np.isnan(v):
                    assert np.isnan(c_v)
                else:
                    assert np.allclose(v, c_v)
            elif isinstance(v, forust.CrateGradientBooster):
                assert isinstance(c_v, forust.CrateGradientBooster)
            else:
                assert v == c_v, k
        fmod_loaded_preds = loaded.predict(X)
        assert np.allclose(fmod_preds, fmod_loaded_preds)

    def test_booster_saving(
        self, X_y: tuple[pd.DataFrame, pd.Series], tmp_path, load_func, save_func
    ):
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
            parallel=True,
        )
        fmod.fit(X, y=y)
        fmod_preds = fmod.predict(X)
        save_func(fmod, f64_model_path)
        fmod_loaded = load_func(f64_model_path)
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
            parallel=True,
        )
        fmod.fit(X, y=y)
        fmod_preds = fmod.predict(X)
        save_func(fmod, f64_model_path)
        fmod_loaded = load_func(f64_model_path)
        assert fmod_loaded.feature_names_in_ == fmod.feature_names_in_
        assert fmod_loaded.feature_names_in_ == X.columns.to_list()
        assert all(fmod_preds == fmod_loaded.predict(X))

    def test_booster_saving_with_monotone_constraints(
        self, X_y: tuple[pd.DataFrame, pd.Series], tmp_path, load_func, save_func
    ):
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
            parallel=True,
            monotone_constraints=mono_,
        )
        fmod.fit(X, y=y)
        fmod_preds = fmod.predict(X)
        save_func(fmod, f64_model_path)
        fmod_loaded = load_func(f64_model_path)
        assert fmod_loaded.feature_names_in_ == fmod.feature_names_in_
        assert fmod_loaded.feature_names_in_ == X.columns.to_list()
        assert all(fmod_preds == fmod_loaded.predict(X))
        assert all(
            [
                fmod.monotone_constraints[ft] == fmod_loaded.monotone_constraints[ft]
                for ft in fmod_loaded.feature_names_in_
            ]
        )
        assert all(
            [
                fmod.monotone_constraints[ft] == fmod_loaded.monotone_constraints[ft]
                for ft in fmod.feature_names_in_
            ]
        )
        assert all(
            [
                fmod.monotone_constraints[ft] == fmod_loaded.monotone_constraints[ft]
                for ft in mono_.keys()
            ]
        )

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
            parallel=True,
            monotone_constraints=mono_,
        )
        fmod.fit(X, y=y)
        fmod_preds = fmod.predict(X)
        save_func(fmod, f64_model_path)
        fmod_loaded = load_func(f64_model_path)
        assert all(fmod_preds == fmod_loaded.predict(X))
