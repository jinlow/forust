from __future__ import annotations

import sys
import warnings
from ast import literal_eval
from typing import Any, Union, cast

import numpy as np
import pandas as pd

from .forust import GradientBooster as CrateGradientBooster  # type: ignore

ArrayLike = Union[pd.Series, np.ndarray]
FrameLike = Union[pd.DataFrame, np.ndarray]


class BoosterType:
    monotone_constraints: dict[int, int]

    def fit(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        y: np.ndarray,
        sample_weight: np.ndarray,
        parallel: bool = True,
    ):
        raise NotImplementedError()

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError()

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        raise NotImplementedError()

    def text_dump(self) -> list[str]:
        raise NotImplementedError()

    @classmethod
    def load_booster(cls, path: str) -> BoosterType:
        raise NotImplementedError()

    def save_booster(self, path: str):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json_str: str) -> BoosterType:
        raise NotImplementedError()

    def json_dump(self) -> str:
        raise NotImplementedError()

    def get_params(self) -> dict[str, Any]:
        raise NotImplementedError()

    def insert_metadata(self, key: str, value: str) -> None:
        raise NotImplementedError()

    def get_metadata(self, key: str) -> str:
        raise NotImplementedError()


class GradientBooster:
    # Define the metadata parameters
    # that are present on all instances of this class
    # this is useful for parameters that should be
    # attempted to be loaded in and set
    # as attributes on the booster after it is loaded.
    meta_data_attributes = ["feature_names_in_"]

    def __init__(
        self,
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
        seed: int = 0,
        missing: float = np.nan,
        create_missing_branch: bool = False,
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
            max_depth (int, optional): Maximum depth of an individual tree. Valid values
            are 0 to infinity. Defaults to 5.
            max_leaves (int, optional): Maximum number of leaves allowed on a tree. Valid values
                are 0 to infinity. This is the total number of final nodes. Defaults to sys.maxsize.
            l2 (float, optional): L2 regularization term applied to the weights of the tree. Valid values
                are 0 to infinity. Defaults to 1.0.
            gamma (float, optional): The minimum amount of loss required to further split a node.
                Valid values are 0 to infinity. Defaults to 0.0.
            min_leaf_weight (float, optional): Minimum sum of the hessian values of the loss function
                required to be in a node. Defaults to 1.0.
            base_score (float, optional): The initial prediction value of the model. Defaults to 0.5.
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
            seed (integer, optional): Integer value used to seed any randomness used in the
                algorithm. Defaults to 0.
            missing (float, optional): Value to consider missing, when training and predicting
                with the booster. Defaults to `np.nan`.
            create_missing_branch (bool, optional): An experimental parameter, that if `True`, will
                create a separate branch for missing, creating a ternary tree, the missing node will be given the same
                weight value as the parent node. If this parameter is `False`, missing will be sent
                down either the left or right branch, creating a binary tree. Defaults to `False`.

        Raises:
            TypeError: Raised if an invalid dtype is passed.
        """
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
            seed=seed,
            missing=missing,
            create_missing_branch=create_missing_branch,
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

    def _convert_input_frame(
        self, X: FrameLike
    ) -> tuple[list[str], int, int, np.ndarray]:
        """Convert data to format needed by booster.

        Returns:
            tuple[list[str], int, int, np.ndarray]: Return column names, number of rows, and number of columns, and flat data.
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
        return features_, rows, cols, flat_data

    def fit(
        self,
        X: FrameLike,
        y: ArrayLike,
        sample_weight: Union[ArrayLike, None] = None,
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
        """

        features_, rows, cols, flat_data = self._convert_input_frame(X)
        if len(features_) > 0:
            self.feature_names_in_ = features_
            self.insert_metadata("feature_names_in_", self.feature_names_in_)

        y_ = y.to_numpy() if isinstance(y, pd.Series) else y

        if not np.issubdtype(y_.dtype, "float64"):
            y_ = y_.astype(dtype="float64", copy=False)

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

        self.booster.fit(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            sample_weight=sample_weight_,
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
        features_, rows, cols, flat_data = self._convert_input_frame(X)
        parallel_ = self.parallel if parallel is None else parallel
        return self.booster.predict(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            parallel=parallel_,
        )

    def predict_contributions(
        self, X: FrameLike, method: str = "average", parallel: Union[bool, None] = None
    ) -> np.ndarray:
        """Predict with the fitted booster on new data, returning the feature
        contribution matrix. The last column is the bias term.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            method (str, optional): Method to calculate the contributions, if "average"
                is specified, the average internal node values are calculated, this is
                equivalent to the `approx_contribs` parameter in XGBoost. The other supported
                method is "weight", this will use the internal leaf weights, to calculate the
                contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
            parallel (Union[bool, None], optional): Optionally specify if the predict
                function should run in parallel on multiple threads. If `None` is
                passed, the `parallel` attribute of the booster will be used.
                Defaults to `None`.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        features_, rows, cols, flat_data = self._convert_input_frame(X)
        parallel_ = self.parallel if parallel is None else parallel

        contributions = self.booster.predict_contributions(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            method=method,
            parallel=parallel_,
        )
        return np.reshape(contributions, (rows, cols + 1))

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

    def text_dump(self) -> list[str]:
        """Return all of the trees of the model in text form.

        Returns:
            list[str]: A list of strings, where each string is a text representation
                of the tree.
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

    def insert_metadata(self, key: str, value: Any):
        """Insert data into the models metadata, this will be saved on the booster object.

        Args:
            key (str): Key to give the inserted value in the metadata.
            value (str): Value to assign the the key.
        """
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
        # We use json to serialize/deserialize so that we can
        v = self.booster.get_metadata(key=key)
        return literal_eval(v)
