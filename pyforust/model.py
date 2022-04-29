from __future__ import annotations
from optparse import Option
from typing import Any, Optional, Tuple, List
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


class Tree:
    """
    Define a tree structure that houses a vector
    of nodes.
    """

    def __init__(
        self,
        l2: float = 1,
        gamma: float = 0,
        max_leaves: int = 20,
        max_depth: int = 15,
        min_child_weight: float = 1,
        learning_rate: float = 0.3,
        objective: LogLoss = LogLoss(),
    ):
        self.l2 = l2
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.objective = objective
        self.nodes: List[TreeNode] = []

    def fit(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> Tree:
        grad_sum = grad.sum()
        hess_sum = hess.sum()
        root_gain = self.gain(grad_sum=grad_sum, hess_sum=hess_sum)
        root_weight = self.weight(grad_sum=grad_sum, hess_sum=hess_sum)
        root_node = TreeNode(
            node_idxs=np.arange(X.shape[0]),
            depth=0,
            weight_value=root_weight,
            gain_value=root_gain,
            cover_value=hess_sum,
        )
        self.nodes.append(root_node)
        growable = [0]
        while len(growable) > 0:
            # If we have hit max leaves break
            if len(self.nodes) >= self.max_leaves:
                break
            n_idx = growable.pop()
            n = self.nodes[n_idx]
            depth = n.depth + 1
            # if we have hit max depth, skip this node
            # but keep going, because there be other valid
            # shallower nodes.
            if depth > self.max_depth:
                continue

            # Try to find a valid split for this node.
            split_info = self.best_split(
                node=n,
                X=X,
                grad=grad,
                hess=hess,
            )
            # If this is None, this means there
            # are no more valid nodes.
            if split_info is None:
                continue
            left_node = TreeNode(
                node_idxs=split_info.left_idxs,
                weight_value=split_info.left_weight,
                gain_value=split_info.left_gain,
                cover_value=split_info.left_cover,
                depth=depth,
            )
            right_node = TreeNode(
                node_idxs=split_info.right_idxs,
                weight_value=split_info.right_weight,
                gain_value=split_info.right_gain,
                cover_value=split_info.right_cover,
                depth=depth,
            )
            left_idx = len(self.nodes)
            right_idx = left_idx + 1
            self.nodes.append(left_node)
            self.nodes.append(right_node)
            n.update_children(
                left_child=left_idx, right_child=right_idx, split_info=split_info
            )
            growable.append(left_idx)
            growable.append(right_idx)

        return self

    def best_split(
        self,
        node: TreeNode,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
    ) -> Optional[SplitInfo]:
        """
        Find the best split for this node out of all the features.
        """
        X_ = X[node.node_idxs, :]
        grad_ = grad[node.node_idxs]
        hess_ = hess[node.node_idxs]

        # Split info
        best_gain = -np.inf
        best_split_info = None

        for f in range(X_.shape[1]):
            split_info = self.best_feature_split(
                node=node,
                X=X_,
                feature=f,
                grad=grad_,
                hess=hess_,
            )

            if split_info is None:
                return None

            if split_info.split_gain > best_gain:
                best_gain = split_info.split_gain
                best_split_info = split_info
        return best_split_info

    def best_feature_split(
        self,
        node: TreeNode,
        X: np.ndarray,
        feature: int,
        grad: np.ndarray,
        hess: np.ndarray,
    ) -> Optional[SplitInfo]:
        """
        Find the best split for a given feature, if it is
        possible to create a split with this feature.
        """
        max_gain = -np.inf
        split_info = None

        # Skip the first value, because nothing is smaller
        # than the first value.
        x = X[:, feature]
        split_vals = np.unique(x)
        for v in split_vals[1:]:
            mask = x < v
            lidxs, ridxs = node.node_idxs[mask], node.node_idxs[~mask]
            lgs, lhs = grad[mask].sum(), hess[mask].sum()
            rgs, rhs = grad[~mask].sum(), hess[~mask].sum()
            # Don't even consider this if the min_child_weight
            # parameter is violated.
            if np.min([lhs, rhs]) < self.min_child_weight:
                continue
            l_gain = self.gain(lgs, lhs)
            r_gain = self.gain(rgs, rhs)
            split_gain = (l_gain + r_gain - node.gain_value) - self.gamma
            if split_gain > max_gain:
                max_gain = split_gain
                split_info = SplitInfo(
                    split_gain=split_gain,
                    split_feature=feature,
                    split_value=v,
                    left_gain=l_gain,
                    left_cover=lhs,
                    left_weight=self.weight(grad_sum=lgs, hess_sum=lhs),
                    left_idxs=lidxs,
                    right_gain=r_gain,
                    right_cover=rhs,
                    right_weight=self.weight(grad_sum=rgs, hess_sum=rhs),
                    right_idxs=ridxs,
                )
        return split_info

    def cover(
        self,
        hess: np.ndarray,
    ) -> float:
        return hess.sum()

    def gain(
        self,
        grad_sum: float,
        hess_sum: float,
    ) -> float:
        return (grad_sum**2) / (hess_sum + self.l2)

    def weight(self, grad_sum: float, hess_sum: float) -> float:
        return -1 * (grad_sum / (hess_sum + self.l2)) * self.learning_rate


from dataclasses import dataclass


@dataclass
class SplitInfo:
    split_gain: float
    split_feature: int
    split_value: Any

    left_gain: float
    left_cover: float
    left_weight: float
    left_idxs: np.ndarray

    right_gain: float
    right_cover: float
    right_weight: float
    right_idxs: np.ndarray


class TreeNode:
    """Node of the tree, this determines the split, or if this
    is a terminal node value.
    """

    def __init__(
        self,
        node_idxs: np.ndarray,
        weight_value: float,
        gain_value: float,
        cover_value: float,
        depth: int,
    ):
        self.node_idxs = node_idxs

        self.weight_value = weight_value
        self.gain_value = gain_value
        self.cover_value = cover_value
        self.depth = depth

        self.split_value_: Optional[float] = None
        self.split_feature_: Optional[float] = None
        self.split_gain_: Optional[float] = None
        self.left_child_: Optional[int] = None
        self.right_child_: Optional[int] = None

    def __repr__(self):
        return (
            "TreeNode{\n"
            + f"\tweight_value: {self.weight_value}\n"
            + f"\tgain_value: {self.gain_value}\n"
            + f"\tcover_value: {self.cover_value}\n"
            + f"\tdepth: {self.depth}\n"
            + f"\tsplit_value_: {self.split_value_}\n"
            + f"\tsplit_feature_: {self.split_feature_}\n"
            + f"\tsplit_gain_: {self.split_gain_}\n"
            + f"\tleft_child_: {self.left_child_}\n"
            + f"\tright_child_: {self.right_child_}\n"
            + "        }"
        )

    def update_children(
        self,
        left_child: int,
        right_child: int,
        split_info: SplitInfo,
    ):
        """
        Update the children, and split information for the node.
        """
        self.left_child_ = left_child
        self.right_child_ = right_child
        self.split_feature_ = split_info.split_feature
        self.split_gain_ = (
            split_info.left_gain + split_info.right_gain - self.gain_value
        )
        self.split_value_ = split_info.split_value

    # def optimal_split(
    #     self,
    #     X: np.ndarray,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    #     l2: float,
    #     gamma: float,
    #     min_child_weight: float,
    # ) -> Optional[SplitInfo]:
    #     pass

    # def fit(
    #     self,
    #     X: np.ndarray,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    # ) -> TreeNode:
    #     # Fit the node.
    #     self.weight_ = self.weight(grad=grad, hess=hess)
    #     self.children_ = self.find_best_split(X=X, grad=grad, hess=hess)
    #     return self

    # def weight(self, grad: np.ndarray, hess: np.ndarray) -> float:
    #     return -1 * (grad.sum() / (hess.sum() + self.l2))

    # def calc_gain_given_weight():
    #     pass

    # def gain(
    #     self,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    # ) -> float:
    #     return (grad.sum() ** 2) / (hess.sum() + self.l2)

    # def split_gain(
    #     self,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    #     left_mask: np.ndarray,
    #     right_mask: np.ndarray,
    # ) -> float:
    #     gl = grad[left_mask].sum()
    #     gr = grad[right_mask].sum()
    #     hl = hess[left_mask].sum()
    #     hr = hess[right_mask].sum()
    #     l = (gl**2) / (hl + self.l2)
    #     r = (gr**2) / (hr + self.l2)
    #     lr = ((gl + gr) ** 2) / (hl + hr + self.l2)
    #     # They don't multiply by 1/2 in the
    #     # actual code for some reason.
    #     return (l + r - lr) - self.gamma

    # def find_best_split(
    #     self,
    #     X: np.ndarray,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    # ) -> Optional[Tuple[int, float, float]]:
    #     best_feature = -1
    #     best_split = None
    #     best_gain = -np.inf
    #     for i in range(X.shape[1]):
    #         res = self.find_feature_best_split(
    #             x=X[:, i],
    #             grad=grad,
    #             hess=hess,
    #         )
    #         if res is None:
    #             continue
    #         split, gain = res
    #         if gain > best_gain:
    #             best_feature = i
    #             best_split = split
    #             best_gain = gain
    #     if best_split is None:
    #         return None
    #     return best_feature, best_split, best_gain

    # def find_feature_best_split(
    #     self,
    #     x: np.ndarray,
    #     grad: np.ndarray,
    #     hess: np.ndarray,
    # ) -> Optional[Tuple[float, float]]:
    #     """Return the best split value"""
    #     best_split = None
    #     max_gain = -np.inf
    #     # Skip the minimum, because we are
    #     # looking at less than.
    #     for v in np.unique(x)[1:]:
    #         left_mask = x < v
    #         right_mask = ~left_mask

    #         # Check the min child weight condition
    #         if (hess[left_mask].sum() < self.min_child_weight) or (
    #             hess[right_mask].sum() < self.min_child_weight
    #         ):
    #             continue
    #         split_gain = self.split_gain(
    #             left_mask=left_mask,
    #             right_mask=right_mask,
    #             grad=grad,
    #             hess=hess,
    #         )
    #         if split_gain > max_gain:
    #             best_split = v
    #             max_gain = split_gain
    #     if best_split is None:
    #         return None
    #     return best_split, max_gain


def gain(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    l2: float,
    gamma: float,
) -> float:
    gl = grad[left_mask].sum()
    gr = grad[right_mask].sum()
    hl = hess[left_mask].sum()
    hr = hess[right_mask].sum()
    l = (gl**2) / (hl + l2)
    r = (gr**2) / (hr + l2)
    lr = ((gl + gr) ** 2) / (hl + hr + l2)
    return 0.5 * (l + r - lr) - gamma


def missing_gain(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    missing_mask: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    l2: float,
    gamma: float,
) -> Tuple[float, float]:
    gl = grad[left_mask].sum()
    gr = grad[right_mask].sum()
    gm = grad[missing_mask].sum()
    hl = hess[left_mask].sum()
    hr = hess[right_mask].sum()
    hm = hess[missing_mask].sum()
    l = (gl**2) / (hl + l2)
    lm = ((gl + gm) ** 2) / (hl + hm + l2)
    r = (gr**2) / (hr + l2)
    rm = ((gr + gm) ** 2) / (hr + hm + l2)
    lrm = ((gl + gr + gm) ** 2) / (hl + hr + hm + l2)
    gain_left = 0.5 * (lm + r - lrm) - gamma
    gain_right = 0.5 * (l + rm - lrm) - gamma
    return (gain_left, gain_right)


def weight(
    grad: np.ndarray,
    hess: np.ndarray,
    l2: float,
) -> float:
    return -1 * (grad.sum() / (hess.sum() + l2))
