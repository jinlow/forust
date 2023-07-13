from __future__ import annotations

import sys
import warnings
from ast import literal_eval
from typing import Any, Iterable, Protocol, Union, cast

import numpy as np
import pandas as pd

from .forust import GradientBooster as CrateGradientBooster  # type: ignore

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


class BoosterType(Protocol):
    monotone_constraints: dict[int, int]
    prediction_iteration: None | int
    best_iteration: None | int
    terminate_missing_features: set[int]

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
        ...

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        ...

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        ...

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        ...

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> dict[int, float]:
        ...

    def text_dump(self) -> list[str]:
        ...

    @classmethod
    def load_booster(cls, path: str) -> BoosterType:
        ...

    def save_booster(self, path: str):
        ...

    @classmethod
    def from_json(cls, json_str: str) -> BoosterType:
        ...

    def json_dump(self) -> str:
        ...

    def get_params(self) -> dict[str, Any]:
        ...

    def insert_metadata(self, key: str, value: str) -> None:
        ...

    def get_metadata(self, key: str) -> str:
        ...

    def get_evaluation_history(self) -> tuple[int, int, np.ndarray]:
        ...


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
    meta_data_attributes = ["feature_names_in_", "n_features_"]

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
        base_score: float | None = None,
        nbins: int = 256,
        parallel: bool = True,
        allow_missing_splits: bool = True,
        monotone_constraints: Union[dict[Any, int], None] = None,
        subsample: float = 1.0,
        top_rate: float = 0.1,
        other_rate: float = 0.2,
        seed: int = 0,
        missing: float = np.nan,
        create_missing_branch: bool = False,
        sample_method: str | None = None,
        grow_policy: str = "DepthWise",
        evaluation_metric: str | None = None,
        early_stopping_rounds: int | None = None,
        initialize_base_score: bool = False,
        terminate_missing_features: Iterable[Any] | None = None,
        missing_node_treatment: str = "AssignToParent",
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
            base_score (float, optional): The initial prediction value of the model. If set to None the parameter `initialize_base_score` will automatically be set to True, in which case the base score will be chosen based on the objective function at fit time. Defaults to None.
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
            other_rate (float, optional):Used only in goss. the retain ratio of small gradient data.
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
            initialize_base_score (bool, optional): If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
            terminate_missing_features (set[Any], optional): An optional iterable of features (either strings, or integer values specifying the feature indices if numpy arrays are used for fitting), for which the missing node will always be terminated, even if `allow_missing_splits` is set to true. This value is only valid if `create_missing_branch` is also True.
            missing_node_treatment (str, optional): Method for selecting the `weight` for the missing node, if `create_missing_branch` is set to `True`. Defaults to "AssignToParent". Valid options are:

                - "None": Calculate missing node weight values without any constraints.
                - "AssignToParent": Assign the weight of the missing node to that of the parent.
                - "AverageLeafWeight": After training each tree, starting from the bottom of the tree, assign the missing node weight to the weighted average of the left and right child nodes. Next assign the parent to the weighted average of the children nodes. This is performed recursively up through the entire tree. This is performed as a post processing step on each tree after it is built, and prior to updating the predictions for which to train the next tree.
                - "AverageNodeWeight": Set the missing node to be equal to the weighted average weight of the left and the right nodes.

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
            set()
            if terminate_missing_features is None
            else set(terminate_missing_features)
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
        self.base_score = base_score
        self.nbins = nbins
        self.parallel = parallel
        self.allow_missing_splits = allow_missing_splits
        self.monotone_constraints = monotone_constraints_
        self.subsample = subsample
        self.seed = seed
        self.missing = missing
        self.create_missing_branch = create_missing_branch
        self.sample_method = sample_method
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.evaluation_metric = evaluation_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.initialize_base_score = initialize_base_score
        self.terminate_missing_features = terminate_missing_features_
        self.missing_node_treatment = missing_node_treatment

    def fit(
        self,
        X: FrameLike,
        y: ArrayLike,
        sample_weight: Union[ArrayLike, None] = None,
        evaluation_data: None
        | list[
            tuple[FrameLike, ArrayLike, ArrayLike] | tuple[FrameLike, ArrayLike]
        ] = None,
    ):
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
                supplied, the first entry of this list will be used to determine if performance
                has improved over the last set of iterations, for which if no improvement is not seen
                in `early_stopping_rounds` training will be cut short.
        """

        features_, flat_data, rows, cols = _convert_input_frame(X)
        self.n_features_ = cols
        self.insert_metadata("n_features_", self.n_features_)
        if len(features_) > 0:
            self.feature_names_in_ = features_
            self.insert_metadata("feature_names_in_", self.feature_names_in_)

        y_ = _convert_input_array(y)

        if sample_weight is None:
            sample_weight = np.ones(y_.shape, dtype="float64")
        sample_weight_ = (
            sample_weight.to_numpy()
            if isinstance(sample_weight, pd.Series)
            else sample_weight
        )

        if not np.issubdtype(sample_weight_.dtype, "float64"):
            sample_weight_ = sample_weight_.astype("float64", copy=False)

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
                    eval_X, eval_y, eval_w = eval_
                    eval_w_ = _convert_input_array(eval_w)
                else:
                    eval_X, eval_y = eval_
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
        else:
            evaluation_data_ = None

        self.booster.fit(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            sample_weight=sample_weight_,
            evaluation_data=evaluation_data_,
        )

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
                index.
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
        is_dataframe = isinstance(X, pd.DataFrame)
        if isinstance(feature, str):
            if not is_dataframe:
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
            if is_dataframe:
                values = X.iloc[:, feature].unique()
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
        """Feature importance values can be calculated with the `calculate_feature_importance` method. This function will return a dictionary of the features and their importances. It should be noted that if a feature was never used for splitting it will not be returned in importance dictionary.

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
        c = cls(**params)
        c.booster = booster
        for m in c.meta_data_attributes:
            try:
                m_ = c.get_metadata(m)
                setattr(c, m, m_)
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
            return self.terminate_missing_features
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return set(feature_map[f] for f in self.terminate_missing_features)

    def insert_metadata(self, key: str, value: Any):
        """Insert data into the models metadata, this will be saved on the booster object.

        Args:
            key (str): Key to give the inserted value in the metadata.
            value (str): Value to assign the the key.
        """  # noqa: E501
        if isinstance(value, str):
            value_ = f"'{value}'"
        else:
            value_ = str(value)
        self.booster.insert_metadata(key=key, value=value_)

    def get_metadata(self, key: Any) -> Any:
        """Get the value associated with a given key, on the boosters metadata.

        Args:
            key (str): Key of item in metadata.

        Returns:
            str: Value associated with the provided key in the boosters metadata.
        """
        v = self.booster.get_metadata(key=key)
        return literal_eval(node_or_string=v)

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
        r, v, d = self.booster.get_evaluation_history()
        return d.reshape((r, v))

    def get_best_iteration(self) -> int | None:
        """Get the best iteration if `early_stopping_rounds` was used when fitting.

        Returns:
            int | None: The best iteration, or None if `early_stopping_rounds` wasn't used.
        """
        return self.booster.best_iteration
