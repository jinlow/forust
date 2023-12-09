from __future__ import annotations

import dataclasses
import inspect
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol, Union, cast

import numpy as np
import pandas as pd

from forust.forust import GradientBooster as CrateGradientBooster  # type: ignore
from forust.serialize import BaseSerializer, ObjectSerializer

__all__ = ["GradientBooster"]

ArrayLike = Union[pd.Series, np.ndarray]
FrameLike = Union[pd.DataFrame, np.ndarray]

CONTRIBUTION_METHODS = {
    "weight": "Weight",
    "Weight": "Weight",
    "average": "Average",
    "Average": "Average",
    "branch-difference": "BranchDifference",
    "branchdifference": "BranchDifference",
    "BranchDifference": "BranchDifference",
    "midpoint-difference": "MidpointDifference",
    "midpointdifference": "MidpointDifference",
    "MidpointDifference": "MidpointDifference",
    "mode-difference": "ModeDifference",
    "modedifference": "ModeDifference",
    "ModeDifference": "ModeDifference",
    "ProbabilityChange": "ProbabilityChange",
    "probabilitychange": "ProbabilityChange",
    "probability-change": "ProbabilityChange",
}

SAMPLE_METHODS = {
    "goss": "Goss",
    "Goss": "Goss",
    "random": "Random",
    "Random": "Random",
}


@dataclass
class Node:
    """Dataclass representation of a node, this represents all of the fields present in a tree node."""

    num: int
    weight_value: float
    hessian_sum: float
    depth: int
    split_value: float
    split_feature: int | str
    split_gain: float
    missing_node: int
    left_child: int
    right_child: int
    is_leaf: bool

    @classmethod
    def _from_xgboost_node(
        cls, xgb_node: dict[str, Any], feature_map: dict[Any, int]
    ) -> Node:
        return Node(
            num=xgb_node["nodeid"],
            weight_value=xgb_node.get("leaf", 0.0),
            hessian_sum=xgb_node["cover"],
            depth=xgb_node.get("depth", 0),
            split_value=float(np.float32(xgb_node.get("split_condition", 0.0))),
            split_feature=feature_map.get(xgb_node.get("split", 0), 0),
            split_gain=xgb_node.get("gain", 0.0),
            missing_node=xgb_node.get("missing", 0),
            left_child=xgb_node.get("yes", 0),
            right_child=xgb_node.get("no", 0),
            is_leaf="leaf" in xgb_node,
        )


def _xgboost_tree_to_nodes(
    tree: dict[str, Any], feature_map: dict[Any, int]
) -> list[dict[str, Any]]:
    buffer = [tree]
    node_list = []
    while len(buffer) > 0:
        xgb_node = buffer.pop(0)
        node_list.append(
            dataclasses.asdict(
                Node._from_xgboost_node(xgb_node, feature_map=feature_map)
            )
        )
        if "leaf" not in xgb_node:
            buffer.extend(xgb_node["children"])
    # Ensure the nodeids all align with the nodes index
    for idx, node in enumerate(node_list):
        if idx != node["num"]:
            raise ValueError(
                f"Nodes are unaligned for node {node['num']} at index {idx}"
            )
    return node_list


def _from_xgboost_model(model: Any) -> GradientBooster:
    import xgboost

    if isinstance(model, xgboost.XGBModel):
        booster = model.get_booster()
    else:
        booster = cast(xgboost.Booster, model)
    # Get the model dump...
    model_dump = booster.get_dump(dump_format="json", with_stats=True)
    features = booster.feature_names
    if features is None:
        feature_map = {}
    else:
        feature_map = {v: i for i, v in enumerate(features)}

    # Get the nodes
    trees = []
    for tree in model_dump:
        nodes = _xgboost_tree_to_nodes(tree=json.loads(tree), feature_map=feature_map)
        trees.append({"nodes": nodes})

    # This is would be wrong, for models trained with "binary:logistic"
    # because the base score is modified prior to predictions.
    # We would need to modify prior to handing it to the forust
    # model.
    learner_config = json.loads(model.get_booster().save_config())["learner"]
    base_score = float(learner_config["learner_model_param"]["base_score"])
    if learner_config["objective"]["name"] == "binary:logistic":
        base_score = np.log(base_score / (1 - base_score))

    # Get initial dump
    model_json = json.loads(GradientBooster().json_dump())
    model_json["base_score"] = base_score
    model_json["trees"] = trees

    # Populate booster from json
    final_model = GradientBooster()
    final_model.booster = CrateGradientBooster.from_json(json.dumps(model_json))
    if features is not None:
        final_model.feature_names_in_ = features
        final_model.n_features_ = len(features)
    return final_model


class BoosterType(Protocol):
    monotone_constraints: dict[int, int]
    prediction_iteration: None | int
    best_iteration: None | int
    base_score: float
    terminate_missing_features: set[int]
    number_of_trees: int

    def fit(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        y: np.ndarray,
        sample_weight: np.ndarray,
        evaluation_data: None
        | list[tuple[FrameLike, ArrayLike, None | ArrayLike]] = None,
        parallel: bool = True,
    ):
        """Fit method"""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict method"""

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        """method"""

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        """pass"""

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> dict[int, float]:
        """pass"""

    def text_dump(self) -> list[str]:
        """pass"""

    @classmethod
    def load_booster(cls, path: str) -> BoosterType:
        """pass"""

    def save_booster(self, path: str):
        """pass"""

    @classmethod
    def from_json(cls, json_str: str) -> BoosterType:
        """pass"""

    def json_dump(self) -> str:
        """pass"""

    def get_params(self) -> dict[str, Any]:
        """pass"""

    def insert_metadata(self, key: str, value: str) -> None:
        """pass"""

    def get_metadata(self, key: str) -> str:
        """pass"""

    def get_evaluation_history(self) -> tuple[int, int, np.ndarray] | None:
        """pass"""


def _convert_input_frame(X: FrameLike) -> tuple[list[str], np.ndarray, int, int]:
    """Convert data to format needed by booster.

    Returns:
        tuple[list[str], np.ndarray, int, int, ]: Return column names, the flat data, number of rows, and the number of columns
    """
    if isinstance(X, pd.DataFrame):
        X_ = X.to_numpy()
        features_ = X.columns.to_list()
    else:
        # Assume it's a numpy array.
        X_ = X
        features_ = []
    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape
    return features_, flat_data, rows, cols


def _convert_input_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        x_ = x.to_numpy()
    else:
        x_ = x
    if not np.issubdtype(x_.dtype, "float64"):
        x_ = x_.astype(dtype="float64", copy=False)
    return x_


class GradientBooster:
    # Define the metadata parameters
    # that are present on all instances of this class
    # this is useful for parameters that should be
    # attempted to be loaded in and set
    # as attributes on the booster after it is loaded.
    meta_data_attributes: dict[str, BaseSerializer] = {
        "feature_names_in_": ObjectSerializer(),
        "n_features_": ObjectSerializer(),
        "feature_importance_method": ObjectSerializer(),
    }

    def __init__(
        self,
        *,
        objective_type: str = "LogLoss",
        iterations: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 5,
        max_leaves: int = sys.maxsize,
        l2: float = 1.0,
        gamma: float = 0.0,
        min_leaf_weight: float = 1.0,
        base_score: float = 0.5,
        nbins: int = 256,
        parallel: bool = True,
        allow_missing_splits: bool = True,
        monotone_constraints: Union[dict[Any, int], None] = None,
        subsample: float = 1.0,
        top_rate: float = 0.1,
        other_rate: float = 0.2,
        colsample_bytree: float = 1.0,
        seed: int = 0,
        missing: float = np.nan,
        create_missing_branch: bool = False,
        sample_method: str | None = None,
        grow_policy: str = "DepthWise",
        evaluation_metric: str | None = None,
        early_stopping_rounds: int | None = None,
        initialize_base_score: bool = True,
        terminate_missing_features: Iterable[Any] | None = None,
        missing_node_treatment: str = "None",
        log_iterations: int = 0,
        feature_importance_method: str = "Gain",
        force_children_to_bound_parent: bool = False,
    ):
        """Gradient Booster Class, used to generate gradient boosted decision tree ensembles.

        Args:
            objective_type (str, optional): The name of objective function used to optimize.
                Valid options include "LogLoss" to use logistic loss as the objective function
                (binary classification), or "SquaredLoss" to use Squared Error as the objective
                function (continuous regression). Defaults to "LogLoss".
            iterations (int, optional): Total number of trees to train in the ensemble.
                Defaults to 100.
            learning_rate (float, optional): Step size to use at each iteration. Each
                leaf weight is multiplied by this number. The smaller the value, the more
                conservative the weights will be. Defaults to 0.3.
            max_depth (int, optional): Maximum depth of an individual tree. Valid values are 0 to infinity. Defaults to 5.
            max_leaves (int, optional): Maximum number of leaves allowed on a tree. Valid values are 0 to infinity. This is the total number of final nodes. Defaults to sys.maxsize.
            l2 (float, optional): L2 regularization term applied to the weights of the tree. Valid values are 0 to infinity. Defaults to 1.0.
            gamma (float, optional): The minimum amount of loss required to further split a node.
                Valid values are 0 to infinity. Defaults to 0.0.
            min_leaf_weight (float, optional): Minimum sum of the hessian values of the loss function
                required to be in a node. Defaults to 1.0.
            base_score (float, optional): The initial prediction value of the model. If `initialize_base_score`
                is set to True the `base_score` will automatically be updated based on the objective
                function at fit time. Defaults to 0.5.
            nbins (int, optional): Number of bins to calculate to partition the data. Setting this to
                a smaller number, will result in faster training time, while potentially sacrificing
                accuracy. If there are more bins, than unique values in a column, all unique values
                will be used. Defaults to 256.
            parallel (bool, optional): Should multiple cores be used when training and predicting
                with this model? Defaults to `True`.
            allow_missing_splits (bool, optional): Allow for splits to be made such that all missing values go
                down one branch, and all non-missing values go down the other, if this results
                in the greatest reduction of loss. If this is false, splits will only be made on non
                missing values. If `create_missing_branch` is set to `True` having this parameter be
                set to `True` will result in the missing branch further split, if this parameter
                is `False` then in that case the missing branch will always be a terminal node.
                Defaults to `True`.
            monotone_constraints (dict[Any, int], optional): Constraints that are used to enforce a
                specific relationship between the training features and the target variable. A dictionary
                should be provided where the keys are the feature index value if the model will be fit on
                a numpy array, or a feature name if it will be fit on a pandas Dataframe. The values of
                the dictionary should be an integer value of -1, 1, or 0 to specify the relationship
                that should be estimated between the respective feature and the target variable.
                Use a value of -1 to enforce a negative relationship, 1 a positive relationship,
                and 0 will enforce no specific relationship at all. Features not included in the
                mapping will not have any constraint applied. If `None` is passed no constraints
                will be enforced on any variable.  Defaults to `None`.
            subsample (float, optional): Percent of records to randomly sample at each iteration when
                training a tree. Defaults to 1.0, meaning all data is used to training.
            top_rate (float, optional): Used only in goss. The retain ratio of large gradient data.
            other_rate (float, optional): Used only in goss. the retain ratio of small gradient data.
            colsample_bytree (float, optional): Specify the fraction of columns that should be sampled at each iteration, valid values are in the range `(0.0,1.0]`.
            seed (integer, optional): Integer value used to seed any randomness used in the
                algorithm. Defaults to 0.
            missing (float, optional): Value to consider missing, when training and predicting
                with the booster. Defaults to `np.nan`.
            create_missing_branch (bool, optional): An experimental parameter, that if `True`, will
                create a separate branch for missing, creating a ternary tree, the missing node will be given the same
                weight value as the parent node. If this parameter is `False`, missing will be sent
                down either the left or right branch, creating a binary tree. Defaults to `False`.
            sample_method (str | None, optional): Optional string value to use to determine the method to
                use to sample the data while training. If this is None, no sample method will be used.
                If the `subsample` parameter is less than 1 and no sample_method is provided this `sample_method`
                will be automatically set to "random". Valid options are "goss" and "random".
                Defaults to `None`.
            grow_policy (str, optional): Optional string value that controls the way new nodes are added to the tree. Choices are `DepthWise` to split at nodes closest to the root, or `LossGuide` to split at nodes with the highest loss change.
            evaluation_metric (str | None, optional): Optional string value used to define an evaluation metric
                that will be calculated at each iteration if a `evaluation_dataset` is provided at fit time.
                The metric can be one of "AUC", "LogLoss", "RootMeanSquaredLogError", or "RootMeanSquaredError".
                If no `evaluation_metric` is passed, but an `evaluation_dataset` is passed, then "LogLoss", will
                be used with the "LogLoss" objective function, and "RootMeanSquaredLogError" will be used with
                "SquaredLoss".
            early_stopping_rounds (int | None, optional): If this is specified, and an `evaluation_dataset` is passed
                during fit, then an improvement in the `evaluation_metric` must be seen after at least this many
                iterations of training, otherwise training will be cut short.
            initialize_base_score (bool, optional): If this is specified, the `base_score` will be calculated at fit time using the `sample_weight` and y data in accordance with the requested `objective_type`. This will result in the passed `base_score` value being overridden.
            terminate_missing_features (set[Any], optional): An optional iterable of features (either strings, or integer values specifying the feature indices if numpy arrays are used for fitting), for which the missing node will always be terminated, even if `allow_missing_splits` is set to true. This value is only valid if `create_missing_branch` is also True.
            missing_node_treatment (str, optional): Method for selecting the `weight` for the missing node, if `create_missing_branch` is set to `True`. Defaults to "None". Valid options are:

                - "None": Calculate missing node weight values without any constraints.
                - "AssignToParent": Assign the weight of the missing node to that of the parent.
                - "AverageLeafWeight": After training each tree, starting from the bottom of the tree, assign the missing node weight to the weighted average of the left and right child nodes. Next assign the parent to the weighted average of the children nodes. This is performed recursively up through the entire tree. This is performed as a post processing step on each tree after it is built, and prior to updating the predictions for which to train the next tree.
                - "AverageNodeWeight": Set the missing node to be equal to the weighted average weight of the left and the right nodes.
            log_iterations (bool, optional): Setting to a value (N) other than zero will result in information being logged about ever N iterations, info can be interacted with directly with the python [`logging`](https://docs.python.org/3/howto/logging.html) module. For an example of how to utilize the logging information see the example [here](/#logging-output).
            feature_importance_method (str, optional): The feature importance method type that will be used to calculate the `feature_importances_` attribute on the booster.
            force_children_to_bound_parent (bool, optional): Setting this parameter to `True` will restrict children nodes, so that they always contain the parent node inside of their range. Without setting this it's possible that both, the left and the right nodes could be greater, than or less than, the parent node. Defaults to `False`.

        Raises:
            TypeError: Raised if an invalid dtype is passed.

        Example:
            Once, the booster has been initialized, it can be fit on a provided dataset, and performance field. After fitting, the model can be used to predict on a dataset.
            In the case of this example, the predictions are the log odds of a given record being 1.

            ```python
            # Small example dataset
            from seaborn import load_dataset

            df = load_dataset("titanic")
            X = df.select_dtypes("number").drop(columns=["survived"])
            y = df["survived"]

            # Initialize a booster with defaults.
            from forust import GradientBooster
            model = GradientBooster(objective_type="LogLoss")
            model.fit(X, y)

            # Predict on data
            model.predict(X.head())
            # array([-1.94919663,  2.25863229,  0.32963671,  2.48732194, -3.00371813])

            # predict contributions
            model.predict_contributions(X.head())
            # array([[-0.63014213,  0.33880048, -0.16520798, -0.07798772, -0.85083578,
            #        -1.07720813],
            #       [ 1.05406709,  0.08825999,  0.21662544, -0.12083538,  0.35209258,
            #        -1.07720813],
            ```

        """
        sample_method_ = (
            "None"
            if sample_method is None
            else SAMPLE_METHODS.get(sample_method, "Random")
        )
        sample_method_ = (
            "Random"
            if (subsample < 1) and (sample_method_ == "None")
            else sample_method_
        )
        terminate_missing_features_ = (
            set() if terminate_missing_features is None else terminate_missing_features
        )

        if (base_score != 0.5) and initialize_base_score:
            warnings.warn(
                "It appears as if you are modifying the `base_score` value, but "
                + "`initialize_base_score` is set to True. The `base_score` will be"
                + " calculated at `fit` time. If this it not the desired behavior, set"
                + " `initialize_base_score` to False.",
            )

        booster = CrateGradientBooster(
            objective_type=objective_type,
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaves=max_leaves,
            l2=l2,
            gamma=gamma,
            min_leaf_weight=min_leaf_weight,
            base_score=base_score,
            nbins=nbins,
            parallel=parallel,
            allow_missing_splits=allow_missing_splits,
            monotone_constraints={},
            subsample=subsample,
            top_rate=top_rate,
            other_rate=other_rate,
            colsample_bytree=colsample_bytree,
            seed=seed,
            missing=missing,
            create_missing_branch=create_missing_branch,
            sample_method=sample_method_,
            grow_policy=grow_policy,
            evaluation_metric=evaluation_metric,
            early_stopping_rounds=early_stopping_rounds,
            initialize_base_score=initialize_base_score,
            terminate_missing_features=set(),
            missing_node_treatment=missing_node_treatment,
            log_iterations=log_iterations,
            force_children_to_bound_parent=force_children_to_bound_parent,
        )
        monotone_constraints_ = (
            {} if monotone_constraints is None else monotone_constraints
        )
        self.booster = cast(BoosterType, booster)
        self.objective_type = objective_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.l2 = l2
        self.gamma = gamma
        self.min_leaf_weight = min_leaf_weight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.base_score = base_score
        self.nbins = nbins
        self.parallel = parallel
        self.allow_missing_splits = allow_missing_splits
        self.monotone_constraints = monotone_constraints_
        self.subsample = subsample
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.missing = missing
        self.create_missing_branch = create_missing_branch
        self.sample_method = sample_method
        self.grow_policy = grow_policy
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.evaluation_metric = evaluation_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.initialize_base_score = initialize_base_score
        self.terminate_missing_features = terminate_missing_features_
        self.missing_node_treatment = missing_node_treatment
        self.log_iterations = log_iterations
        self.feature_importance_method = feature_importance_method
        self.force_children_to_bound_parent = force_children_to_bound_parent

        self._set_metadata_attributes(
            "feature_importance_method", feature_importance_method
        )

    def fit(
        self,
        X: FrameLike,
        y: ArrayLike,
        sample_weight: Union[ArrayLike, None] = None,
        evaluation_data: None
        | list[
            tuple[FrameLike, ArrayLike, ArrayLike] | tuple[FrameLike, ArrayLike]
        ] = None,
    ) -> GradientBooster:
        """Fit the gradient booster on a provided dataset.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            y (ArrayLike): Either a pandas Series, or a 1 dimensional numpy array. If "LogLoss"
                was the objective type specified, then this should only contain 1 or 0 values,
                where 1 is the positive class being predicted. If "SquaredLoss" is the
                objective type, then any continuous variable can be provided.
            sample_weight (Union[ArrayLike, None], optional): Instance weights to use when
                training the model. If None is passed, a weight of 1 will be used for every record.
                Defaults to None.
            evaluation_data (tuple[FrameLike, ArrayLike, ArrayLike] | tuple[FrameLike, ArrayLike], optional):
                An optional list of tuples, where each tuple should contain a dataset, and equal length
                target array, and optional an equal length sample weight array. If this is provided
                metric values will be calculated at each iteration of training. If `early_stopping_rounds` is
                supplied, the last entry of this list will be used to determine if performance
                has improved over the last set of iterations, for which if no improvement is not seen
                in `early_stopping_rounds` training will be cut short.
        """

        features_, flat_data, rows, cols = _convert_input_frame(X)
        self.n_features_ = cols
        self._set_metadata_attributes("n_features_", self.n_features_)
        if len(features_) > 0:
            self.feature_names_in_ = features_
            self._set_metadata_attributes("feature_names_in_", self.feature_names_in_)

        y_ = _convert_input_array(y)

        if sample_weight is None:
            sample_weight = np.ones(y_.shape, dtype="float64")
        sample_weight_ = _convert_input_array(sample_weight)

        # Convert the monotone constraints into the form needed
        # by the rust code.
        monotone_constraints_ = self._standardize_monotonicity_map(X)
        self.booster.monotone_constraints = monotone_constraints_
        self.booster.terminate_missing_features = (
            self._standardize_terminate_missing_features(X)
        )

        # Create evaluation data
        if evaluation_data is not None:
            evaluation_data_ = []
            for eval_ in evaluation_data:
                if len(eval_) == 3:
                    eval_X, eval_y, eval_w = eval_  # type: ignore
                    eval_w_ = _convert_input_array(eval_w)
                else:
                    eval_X, eval_y = eval_  # type: ignore
                    eval_w_ = np.ones(eval_X.shape[0], dtype="float64")

                features_, eval_flat_data, eval_rows, eval_cols = _convert_input_frame(
                    eval_X
                )
                self._validate_features(features_)
                evaluation_data_.append(
                    (
                        eval_flat_data,
                        eval_rows,
                        eval_cols,
                        _convert_input_array(eval_y),
                        eval_w_,
                    )
                )
            if len(evaluation_data_) > 1:
                warnings.warn(
                    "Multiple evaluation datasets passed, only the last one will be used to determine early stopping."
                )
        else:
            evaluation_data_ = None

        self.booster.fit(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            sample_weight=sample_weight_,
            evaluation_data=evaluation_data_,  # type: ignore
        )

        # Once it's been fit, reset the `base_score`
        # this will account for the fact that's it's adjusted after fit.
        self.base_score = self.booster.base_score
        return self

    def _validate_features(self, features: list[str]):
        if len(features) > 0 and hasattr(self, "feature_names_in_"):
            if features != self.feature_names_in_:
                raise ValueError(
                    "Columns mismatch between data passed, and data used at fit."
                )

    def predict(self, X: FrameLike, parallel: Union[bool, None] = None) -> np.ndarray:
        """Predict with the fitted booster on new data.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            parallel (Union[bool, None], optional): Optionally specify if the predict
                function should run in parallel on multiple threads. If `None` is
                passed, the `parallel` attribute of the booster will be used.
                Defaults to `None`.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        features_, flat_data, rows, cols = _convert_input_frame(X)
        self._validate_features(features_)
        parallel_ = self.parallel if parallel is None else parallel
        return self.booster.predict(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            parallel=parallel_,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        vals = self.calculate_feature_importance(
            method=self.feature_importance_method, normalize=True
        )
        if hasattr(self, "feature_names_in_"):
            vals = cast(Dict[str, float], vals)
            return np.array([vals.get(ft, 0.0) for ft in self.feature_names_in_])
        else:
            vals = cast(Dict[int, float], vals)
            return np.array([vals.get(ft, 0.0) for ft in range(self.n_features_)])

    def predict_contributions(
        self, X: FrameLike, method: str = "Average", parallel: Union[bool, None] = None
    ) -> np.ndarray:
        """Predict with the fitted booster on new data, returning the feature
        contribution matrix. The last column is the bias term.


        When predicting with the data, the maximum iteration that will be used when predicting can be set using the `set_prediction_iteration` method. If `early_stopping_rounds` has been set, this will default to the best iteration, otherwise all of the trees will be used.

        If early stopping was used, the evaluation history can be retrieved with the [get_evaluation_history][forust.GradientBooster.get_evaluation_history] method.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            method (str, optional): Method to calculate the contributions, available options are:

                - "Average": If this option is specified, the average internal node values are calculated, this is equivalent to the `approx_contribs` parameter in XGBoost.
                - "Shapley": Using this option will calculate contributions using the tree shap algorithm.
                - "Weight": This method will use the internal leaf weights, to calculate the contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
                - "BranchDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the other non-missing branch. This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "MidpointDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the mid-point between the right and left node weighted by the cover of each node. This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "ModeDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the node with the largest cover (the mode node). This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "ProbabilityChange": This method is only valid when the objective type is set to "LogLoss". This method will calculate contributions as the change in a records probability of being 1 moving from a parent node to a child node. The sum of the returned contributions matrix, will be equal to the probability a record will be 1. For example, given a model, `model.predict_contributions(X, method="ProbabilityChange") == 1 / (1 + np.exp(-model.predict(X)))`
            parallel (Union[bool, None], optional): Optionally specify if the predict
                function should run in parallel on multiple threads. If `None` is
                passed, the `parallel` attribute of the booster will be used.
                Defaults to `None`.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        features_, flat_data, rows, cols = _convert_input_frame(X)
        self._validate_features(features_)
        parallel_ = self.parallel if parallel is None else parallel

        contributions = self.booster.predict_contributions(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            method=CONTRIBUTION_METHODS.get(method, method),
            parallel=parallel_,
        )
        return np.reshape(contributions, (rows, cols + 1))

    def set_prediction_iteration(self, iteration: int):
        """Set the iteration that should be used when predicting. If `early_stopping_rounds`
        has been set, this will default to the best iteration, otherwise all of the trees
        will be used.

        Args:
            iteration (int): Iteration number to use, this will use all trees, up to this
                index. Setting this to 10, would result in trees 0 through 9 used for predictions.
        """
        self.booster.prediction_iteration = iteration

    def partial_dependence(
        self,
        X: FrameLike,
        feature: Union[str, int],
        samples: int | None = 100,
        exclude_missing: bool = True,
        percentile_bounds: tuple[float, float] = (0.2, 0.98),
    ) -> np.ndarray:
        """Calculate the partial dependence values of a feature. For each unique
        value of the feature, this gives the estimate of the predicted value for that
        feature, with the effects of all features averaged out. This information gives
        an estimate of how a given feature impacts the model.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
                This should be the same data passed into the models fit, or predict,
                with the columns in the same order.
            feature (Union[str, int]): The feature for which to calculate the partial
                dependence values. This can be the name of a column, if the provided
                X is a pandas DataFrame, or the index of the feature.
            samples (int | None, optional): Number of evenly spaced samples to select. If None
                is passed all unique values will be used. Defaults to 100.
            exclude_missing (bool, optional): Should missing excluded from the features? Defaults to True.
            percentile_bounds (tuple[float, float], optional): Upper and lower percentiles to start at
                when calculating the samples. Defaults to (0.2, 0.98) to cap the samples selected
                at the 5th and 95th percentiles respectively.

        Raises:
            ValueError: An error will be raised if the provided X parameter is not a
                pandas DataFrame, and a string is provided for the feature.

        Returns:
            np.ndarray: A 2 dimensional numpy array, where the first column is the
                sorted unique values of the feature, and then the second column
                is the partial dependence values for each feature value.

        Example:
            This information can be plotted to visualize how a feature is used in the model, like so.

            ```python
            from seaborn import lineplot
            import matplotlib.pyplot as plt

            pd_values = model.partial_dependence(X=X, feature="age", samples=None)

            fig = lineplot(x=pd_values[:,0], y=pd_values[:,1],)
            plt.title("Partial Dependence Plot")
            plt.xlabel("Age")
            plt.ylabel("Log Odds")
            ```
            <img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age.png">

            We can see how this is impacted if a model is created, where a specific constraint is applied to the feature using the `monotone_constraint` parameter.

            ```python
            model = GradientBooster(
                objective_type="LogLoss",
                monotone_constraints={"age": -1},
            )
            model.fit(X, y)

            pd_values = model.partial_dependence(X=X, feature="age")
            fig = lineplot(
                x=pd_values[:, 0],
                y=pd_values[:, 1],
            )
            plt.title("Partial Dependence Plot with Monotonicity")
            plt.xlabel("Age")
            plt.ylabel("Log Odds")
            ```
            <img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age_mono.png">
        """
        if isinstance(feature, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "If `feature` is a string, then the object passed as `X` must be a pandas DataFrame."
                )
            values = X.loc[:, feature].to_numpy()
            if hasattr(self, "feature_names_in_"):
                [feature_idx] = [
                    i for i, v in enumerate(self.feature_names_in_) if v == feature
                ]
            else:
                w_msg = (
                    "No feature names were provided at fit, but feature was a string, attempting to "
                    + "determine feature index from DataFrame Column, "
                    + "ensure columns are the same order as data passed when fit."
                )
                warnings.warn(w_msg)
                [feature_idx] = [i for i, v in enumerate(X.columns) if v == feature]
        elif isinstance(feature, int):
            feature_idx = feature
            if isinstance(X, pd.DataFrame):
                values = X.iloc[:, feature].to_numpy()
            else:
                values = X[:, feature]
        else:
            raise ValueError(
                f"The parameter `feature` must be a string, or an int, however an object of type {type(feature)} was passed."
            )
        min_p, max_p = percentile_bounds
        values = values[~(np.isnan(values) | (values == self.missing))]
        if samples is None:
            search_values = np.sort(np.unique(values))
        else:
            # Exclude missing from this calculation.
            search_values = np.quantile(values, np.linspace(min_p, max_p, num=samples))

        # Add missing back, if they wanted it...
        if not exclude_missing:
            search_values = np.append([self.missing], search_values)

        res = []
        for v in search_values:
            res.append(
                (v, self.booster.value_partial_dependence(feature=feature_idx, value=v))
            )
        return np.array(res)

    def calculate_feature_importance(
        self, method: str = "Gain", normalize: bool = True
    ) -> dict[int, float] | dict[str, float]:
        """Feature importance values can be calculated with the `calculate_feature_importance` method. This function will return a dictionary of the features and their importance values. It should be noted that if a feature was never used for splitting it will not be returned in importance dictionary.

        Args:
            method (str, optional): Variable importance method. Defaults to "Gain". Valid options are:

                - "Weight": The number of times a feature is used to split the data across all trees.
                - "Gain": The average split gain across all splits the feature is used in.
                - "Cover": The average coverage across all splits the feature is used in.
                - "TotalGain": The total gain across all splits the feature is used in.
                - "TotalCover": The total coverage across all splits the feature is used in.
            normalize (bool, optional): Should the importance be normalized to sum to 1? Defaults to `True`.

        Returns:
            dict[str, float]: Variable importance values, for features present in the model.

        Example:
            ```python
            model.calculate_feature_importance("Gain")
            # {
            #   'parch': 0.0713072270154953,
            #   'age': 0.11609109491109848,
            #   'sibsp': 0.1486879289150238,
            #   'fare': 0.14309120178222656,
            #   'pclass': 0.5208225250244141
            # }
            ```
        """
        importance_: dict[int, float] = self.booster.calculate_feature_importance(
            method=method,
            normalize=normalize,
        )
        if hasattr(self, "feature_names_in_"):
            feature_map: dict[int, str] = {
                i: f for i, f in enumerate(self.feature_names_in_)
            }
            return {feature_map[i]: v for i, v in importance_.items()}
        return importance_

    def text_dump(self) -> list[str]:
        """Return all of the trees of the model in text form.

        Returns:
            list[str]: A list of strings, where each string is a text representation
                of the tree.
        Example:
            ```python
            model.text_dump()[0]
            # 0:[0 < 3] yes=1,no=2,missing=2,gain=91.50833,cover=209.388307
            #       1:[4 < 13.7917] yes=3,no=4,missing=4,gain=28.185467,cover=94.00148
            #             3:[1 < 18] yes=7,no=8,missing=8,gain=1.4576768,cover=22.090348
            #                   7:[1 < 17] yes=15,no=16,missing=16,gain=0.691266,cover=0.705011
            #                         15:leaf=-0.15120,cover=0.23500
            #                         16:leaf=0.154097,cover=0.470007
            ```
        """
        return self.booster.text_dump()

    def json_dump(self) -> str:
        """Return the booster object as a string.

        Returns:
            str: The booster dumped as a json object in string form.
        """
        return self.booster.json_dump()

    @classmethod
    def load_booster(cls, path: str) -> GradientBooster:
        """Load a booster object that was saved with the `save_booster` method.

        Args:
            path (str): Path to the saved booster file.

        Returns:
            GradientBooster: An initialized booster object.
        """
        booster = CrateGradientBooster.load_booster(str(path))

        params = booster.get_params()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cls(**params)
        c.booster = booster
        for m in c.meta_data_attributes:
            try:
                m_ = c._get_metadata_attributes(m)
                setattr(c, m, m_)
                # If "feature_names_in_" is present, we know a
                # pandas dataframe was used for fitting, in this case
                # get back the original monotonicity map, with the
                # feature names as keys.
                if m == "feature_names_in_":
                    if c.monotone_constraints is not None:
                        c.monotone_constraints = {
                            ft: c.monotone_constraints[i]
                            for i, ft in enumerate(c.feature_names_in_)
                        }
            except KeyError:
                pass
        return c

    def save_booster(self, path: str):
        """Save a booster object, the underlying representation is a json file.

        Args:
            path (str): Path to save the booster object.
        """
        self.booster.save_booster(str(path))

    def _standardize_monotonicity_map(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> dict[int, Any]:
        if isinstance(X, np.ndarray):
            return self.monotone_constraints
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return {feature_map[f]: v for f, v in self.monotone_constraints.items()}

    def _standardize_terminate_missing_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> set[int]:
        if isinstance(X, np.ndarray):
            return set(self.terminate_missing_features)
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return set(feature_map[f] for f in self.terminate_missing_features)

    def insert_metadata(self, key: str, value: str):
        """Insert data into the models metadata, this will be saved on the booster object.

        Args:
            key (str): Key to give the inserted value in the metadata.
            value (str): String value to assign to the key.
        """  # noqa: E501
        self.booster.insert_metadata(key=key, value=value)

    def get_metadata(self, key: str) -> str:
        """Get the value associated with a given key, on the boosters metadata.

        Args:
            key (str): Key of item in metadata.

        Returns:
            str: Value associated with the provided key in the boosters metadata.
        """
        v = self.booster.get_metadata(key=key)
        return v

    def _set_metadata_attributes(self, key: str, value: Any) -> None:
        value_ = self.meta_data_attributes[key].serialize(value)
        self.insert_metadata(key=key, value=value_)

    def _get_metadata_attributes(self, key: str) -> Any:
        value = self.get_metadata(key)
        return self.meta_data_attributes[key].deserialize(value)

    def get_evaluation_history(self) -> np.ndarray | None:
        """Get the results of the `evaluation_metric` calculated
        on the `evaluation_dataset` passed to [`fit`][forust.GradientBooster.fit], at each iteration.
        If no `evaluation_dataset` was passed, this will return None.

        Returns:
            np.ndarray | None: A numpy array equal to the shape of the number
            of evaluation datasets passed, and the number of trees in the model.

        Example:
            ```python
            model = GradientBooster(objective_type="LogLoss")
            model.fit(X, y, evaluation_data=[(X, y)])

            model.get_evaluation_history()[0:3]

            # array([[588.9158873 ],
            #        [532.01055803],
            #        [496.76933646]])
            ```
        """
        res = self.booster.get_evaluation_history()
        if res is None:
            return None
        r, v, d = res
        return d.reshape((r, v))

    @property
    def best_iteration(self) -> int | None:
        """Get the best iteration if `early_stopping_rounds` was used when fitting.

        Returns:
            int | None: The best iteration, or None if `early_stopping_rounds` wasn't used.
        """
        return self.booster.best_iteration

    @property
    def prediction_iteration(self) -> int | None:
        """The prediction_iteration that will be used when predicting, up to this many trees will be used.

        Returns:
            int | None: Int if this is set, otherwise, None, in which case all trees will be used.
        """
        return self.booster.prediction_iteration

    @property
    def number_of_trees(self) -> int:
        """The number of trees in the model.

        Returns:
            int: The total number of trees in the model.
        """
        return self.booster.number_of_trees

    def get_best_iteration(self) -> int | None:
        """Get the best iteration if `early_stopping_rounds` was used when fitting.

        Returns:
            int | None: The best iteration, or None if `early_stopping_rounds` wasn't used.
        """
        return self.booster.best_iteration

    # Make picklable with getstate and setstate
    def __getstate__(self) -> dict[Any, Any]:
        booster_json = self.json_dump()
        # Delete booster
        # Doing it like this, so it doesn't delete it globally.
        res = {k: v for k, v in self.__dict__.items() if k != "booster"}
        res["__booster_json_file__"] = booster_json
        return res

    def __setstate__(self, d: dict[Any, Any]) -> None:
        # Load the booster object the pickled JSon string.
        booster_object = CrateGradientBooster.from_json(d["__booster_json_file__"])
        d["booster"] = booster_object
        del d["__booster_json_file__"]
        self.__dict__ = d

    # Functions for scikit-learn compatibility, will feel out adding these manually,
    # and then if that feels too unwieldy will add scikit-learn as a dependency.
    def get_params(self, deep=True) -> dict[str, Any]:
        """Get all of the parameters for the booster.

        Args:
            deep (bool, optional): This argument does nothing, and is simply here for scikit-learn compatibility.. Defaults to True.

        Returns:
            dict[str, Any]: The parameters of the booster.
        """
        args = inspect.getfullargspec(GradientBooster).kwonlyargs
        return {param: getattr(self, param) for param in args}

    def set_params(self, **params: Any) -> GradientBooster:
        """Set the parameters of the booster, this has the same effect as reinstating the booster.

        Returns:
            GradientBooster: Booster with new parameters.
        """
        old_params = self.get_params()
        old_params.update(params)
        GradientBooster.__init__(self, **old_params)
        return self

    def get_node_lists(self, map_features_names: bool = True) -> list[list[Node]]:
        """Return the tree structures representation as a list of python objects.

        Args:
            map_features_names (bool, optional): Should the feature names tried to be mapped to a string, if a pandas dataframe was used. Defaults to True.

        Returns:
            list[list[Node]]: A list of lists where each sub list is a tree, with all of it's respective nodes.

        Example:
            This can be run directly to get the tree structure as python objects.

            ```python
            fmod = GradientBooster(max_depth=2)
            fmod.fit(X, y=y)

            fmod.get_node_lists()[0]

            # [Node(num=0, weight_value...,
            # Node(num=1, weight_value...,
            # Node(num=2, weight_value...,
            # Node(num=3, weight_value...,
            # Node(num=4, weight_value...,
            # Node(num=5, weight_value...,
            # Node(num=6, weight_value...,]
            ```
        """
        model = json.loads(self.json_dump())["trees"]
        feature_map: dict[int, str] | dict[int, int]
        leaf_split_feature: str | int
        if map_features_names and hasattr(self, "feature_names_in_"):
            feature_map = {i: ft for i, ft in enumerate(self.feature_names_in_)}
            leaf_split_feature = ""
        else:
            feature_map = {i: i for i in range(self.n_features_)}
            leaf_split_feature = -1

        trees = []
        for t in model:
            tree = []
            for n in t["nodes"]:
                if not n["is_leaf"]:
                    n["split_feature"] = feature_map[n["split_feature"]]
                else:
                    n["split_feature"] = leaf_split_feature
                tree.append(Node(**n))
            trees.append(tree)
        return trees

    def trees_to_dataframe(self) -> pd.DataFrame:
        """Return the tree structure as a pandas DataFrame object.

        Returns:
            pd.DataFrame: Trees in a pandas dataframe.

        Example:
            This can be used directly to print out the tree structure as a pandas dataframe. The Leaf values will have the "Gain" column replaced with the weight value.

            ```python
            model.trees_to_dataframe().head()
            ```

            |    |   Tree |   Node | ID   | Feature   |   Split | Yes   | No   | Missing   |    Gain |    Cover |
            |---:|-------:|-------:|:-----|:----------|--------:|:------|:-----|:----------|--------:|---------:|
            |  0 |      0 |      0 | 0-0  | pclass    |  3      | 0-1   | 0-2  | 0-2       | 91.5083 | 209.388  |
            |  1 |      0 |      1 | 0-1  | fare      | 13.7917 | 0-3   | 0-4  | 0-4       | 28.1855 |  94.0015 |
        """

        def node_to_row(
            n: Node,
            tree_n: int,
        ) -> dict[str, Any]:
            def _id(i: int) -> str:
                return f"{tree_n}-{i}"

            return dict(
                Tree=tree_n,
                Node=n.num,
                ID=_id(n.num),
                Feature="Leaf" if n.is_leaf else str(n.split_feature),
                Split=None if n.is_leaf else n.split_value,
                Yes=None if n.is_leaf else _id(n.left_child),
                No=None if n.is_leaf else _id(n.right_child),
                Missing=None if n.is_leaf else _id(n.missing_node),
                Gain=n.weight_value if n.is_leaf else n.split_gain,
                Cover=n.hessian_sum,
            )

        # Flatten list of lists using list comprehension
        vals = [
            node_to_row(n, i)
            for i, tree in enumerate(self.get_node_lists())
            for n in tree
        ]
        return pd.DataFrame.from_records(vals)
