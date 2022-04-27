from typing import Optional, Tuple
import numpy as np

# https://arxiv.org/pdf/1603.02754.pdf
# https://github.com/Ekeany/XGBoost-From-Scratch/blob/master/XGBoost.py
# https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb


class LogLoss:
    """LogLoss, expects y_hat to be be a probability.
    Thus if it's on the logit scale, it will need to be converted
    to a probability using this function:
    y_hat = 1 / (1 + np.exp(-y_hat))
    """

    @staticmethod
    def loss(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return -1 * (y * np.log(y_hat) + (1 - y) * (np.log(1 - y_hat)))

    @staticmethod
    def grad(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return y_hat - y

    @staticmethod
    def hess(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return y_hat * (1 - y_hat)


class TreeNode:
    """Node of the tree, this determines the split, or if this
    is a terminal node value.
    """

    def __init__(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        lambda_val: float,
        gamma_val: float,
        min_child_weight: float,
    ):
        self.X = X
        self.grad = grad
        self.hess = hess
        self.lambda_val = lambda_val
        self.gamma_val = gamma_val
        self.min_child_weight = min_child_weight
        self.weight_ = self.weight()

    def weight(self) -> float:
        return -1 * (self.grad.sum() / (self.hess.sum() + self.lambda_val))

    def split_gain(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        gl = self.grad[left_mask].sum()
        gr = self.grad[right_mask].sum()
        hl = self.hess[left_mask].sum()
        hr = self.hess[right_mask].sum()
        l = (gl**2) / (hl + self.lambda_val)
        r = (gr**2) / (hr + self.lambda_val)
        lr = ((gl + gr) ** 2) / (hl + hr + self.lambda_val)
        return 0.5 * (l + r - lr) - self.gamma_val

    def find_best_split(self) -> Optional[Tuple[int, float, float]]:
        best_feature = -1
        best_split = None
        best_gain = -np.inf
        for i in range(len(self.X)):
            res = self.find_feature_best_split(self.X[:,i])
            if res is None:
                continue
            split, gain = res
            if gain > best_gain:
                best_feature = i
                best_split = split
                best_gain = gain
        if best_split is None:
            return None
        return best_feature, best_split, best_gain

    def find_feature_best_split(
        self, x: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Return the best split value"""
        best_split = None
        max_gain = -np.inf
        # Skip the minimum, because we are 
        # looking at less than.
        for v in np.unique(x)[1:]:
            left_mask = x < v
            right_mask = ~left_mask

            # Check the min child weight condition
            if (self.hess[left_mask].sum() < self.min_child_weight) or (
                self.hess[right_mask].sum() < self.min_child_weight
            ):
                continue
            split_gain = self.split_gain(left_mask, right_mask)
            if split_gain > max_gain:
                best_split = v
                max_gain = split_gain
        if best_split is None:
            return None
        return best_split, max_gain


def gain(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    lambda_val: float,
    gamma_val: float,
) -> float:
    gl = grad[left_mask].sum()
    gr = grad[right_mask].sum()
    hl = hess[left_mask].sum()
    hr = hess[right_mask].sum()
    l = (gl**2) / (hl + lambda_val)
    r = (gr**2) / (hr + lambda_val)
    lr = ((gl + gr) ** 2) / (hl + hr + lambda_val)
    return 0.5 * (l + r - lr) - gamma_val


def missing_gain(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    missing_mask: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    lambda_val: float,
    gamma_val: float,
) -> Tuple[float, float]:
    gl = grad[left_mask].sum()
    gr = grad[right_mask].sum()
    gm = grad[missing_mask].sum()
    hl = hess[left_mask].sum()
    hr = hess[right_mask].sum()
    hm = hess[missing_mask].sum()
    l = (gl**2) / (hl + lambda_val)
    lm = ((gl + gm) ** 2) / (hl + hm + lambda_val)
    r = (gr**2) / (hr + lambda_val)
    rm = ((gr + gm) ** 2) / (hr + hm + lambda_val)
    lrm = ((gl + gr + gm) ** 2) / (hl + hr + hm + lambda_val)
    gain_left = 0.5 * (lm + r - lrm) - gamma_val
    gain_right = 0.5 * (l + rm - lrm) - gamma_val
    return (gain_left, gain_right)


def weight(
    grad: np.ndarray,
    hess: np.ndarray,
    lambda_val: float,
) -> float:
    return -1 * (grad.sum() / (hess.sum() + lambda_val))
