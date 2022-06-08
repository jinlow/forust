from __future__ import annotations

import json
import sys
from random import sample
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from .forust import GradientBoosterF32, GradientBoosterF64  # type: ignore

ArrayLike = Union[pd.Series, np.ndarray]
FrameLike = Union[pd.DataFrame, np.ndarray]


class BoosterType:
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

    def text_dump(self) -> List[str]:
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

    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError()


class GradientBooster:
    def __init__(
        self,
        objective_type: str = "LogLoss",
        iterations: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 5,
        max_leaves: int = sys.maxsize,
        l2: float = 1.0,
        gamma: float = 0.0,
        min_leaf_weight: float = 0.0,
        base_score: float = 0.5,
        nbins: int = 256,
        parallel: bool = True,
        dtype: Union[np.dtype, str] = "float64",
    ):
        """Gradient Booster Class, used to generate gradient boosted decision tree ensembles.

        Args:
            objective_type (str, optional): The name of objective function used to optimize.
                Valid options include "LogLoss" to use logistic loss as the objective function,
                or "SquaredLoss" to use Squared Error as the objective function.
                Defaults to "LogLoss".
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
                required to be in a node. Defaults to 0.0.
            base_score (float, optional): The initial prediction value of the model. Defaults to 0.5.
            nbins (int, optional): Number of bins to calculate to partition the data. Setting this to
                a smaller number, will result in faster training time, while potentially sacrificing
                accuracy. If there are more bins, than unique values in a column, all unique values
                will be used. Defaults to 256.
            parallel (bool, optional): Should multiple cores be used when training and predicting
                with this model? Defaults to True.
            dtype (Union[np.dtype, str], optional): Datatype used for the model. Valid options
                are a numpy 32 bit float, or numpy 64 bit float. Using 32 bit float could be faster
                in some instances, however this may lead to less precise results. Defaults to "float64".

        Raises:
            TypeError: Raised if an invalid dtype is passed.
        """
        if np.issubdtype(dtype, np.float32):
            booster = GradientBoosterF32(
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
            )
        elif np.issubdtype(dtype, np.float64):
            booster = GradientBoosterF64(
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
            )
        else:
            raise TypeError(
                f"Only a 32 bit or 64 bit floating point booster can be created, however {dtype} was passed."
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
        self.dtype = dtype

    def fit(
        self,
        X: FrameLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """Fit the gradient booster on a provided dataset.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            y (ArrayLike): Either a pandas Series, or a 1 dimensional numpy array.
            sample_weight (Optional[ArrayLike], optional): Instance weights to use when
                training the model. If None is passed, a weight of 1 will be used for every record.
                Defaults to None.
        """
        X_ = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if not np.issubdtype(X_.dtype, self.dtype):
            X_ = X_.astype(dtype=self.dtype, copy=False)

        y_ = y.to_numpy() if isinstance(y, pd.Series) else y
        if not np.issubdtype(y_.dtype, self.dtype):
            y_ = y_.astype(dtype=self.dtype, copy=False)

        if sample_weight is None:
            sample_weight = np.ones(y_.shape, dtype=self.dtype)
        sample_weight_ = (
            sample_weight.to_numpy()
            if isinstance(sample_weight, pd.Series)
            else sample_weight
        )
        if not np.issubdtype(sample_weight_.dtype, self.dtype):
            sample_weight_ = sample_weight_.astype(self.dtype, copy=False)

        flat_data = X_.ravel(order="F")
        rows, cols = X_.shape
        self.booster.fit(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            sample_weight=sample_weight_,
            parallel=self.parallel,
        )

    def predict(self, X: FrameLike) -> np.ndarray:
        """Predict with the fitted booster on new data.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        X_ = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if not np.issubdtype(X_.dtype, self.dtype):
            X_ = X_.astype(dtype=self.dtype, copy=False)

        flat_data = X_.ravel(order="F")
        rows, cols = X_.shape
        return self.booster.predict(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            parallel=self.parallel,
        )

    def text_dump(self) -> List[str]:
        """Return all of the trees of the model in text form.

        Returns:
            List[str]: A list of strings, where each string is a text representation
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
        with open(path, "r") as file:
            model_json = json.load(file)
        if model_json["dtype"] == "f32":
            booster = GradientBoosterF32.load_booster(str(path))
        else:
            booster = GradientBoosterF64.load_booster(str(path))
            
        params = booster.get_params()
        if params["dtype"] == "f64":
            params["dtype"] = "float64"
        else:
            params["dtype"] = "float32"
        c = cls(**params)
        c.booster = booster
        return c

    def save_booster(self, path: str):
        """Save a booster object, the underlying representation is a json file.

        Args:
            path (str): Path to save the booster object.
        """
        self.booster.save_booster(str(path))
