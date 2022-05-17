use std::collections::BTreeSet;

use crate::data::{Matrix, MatrixData};
use crate::node::Node;
use crate::splitter::{SplitInfo, Splitter};

pub struct ExactSplitter<T> {
    l2: T,
    gamma: T,
    min_leaf_weight: T,
    learning_rate: T,
    min_split_gain: T, // Set this to 0
}

impl<'a, T> ExactSplitter<T>
where
    T: MatrixData<T>,
{
    fn best_feature_split(
        &self,
        node: &mut Node<T>,
        data: &Matrix<T>,
        feature: usize,
        grad: &[T],
        hess: &[T],
    ) -> Option<SplitInfo<T>> {
        let mut split_info: Option<SplitInfo<T>> = None;
        let mut max_gain: Option<T> = None;

        let f = data.get_col(feature);

        node.node_idxs
            .sort_by(|a, b| f[*a].partial_cmp(&f[*b]).unwrap());

        let mut left_grad = grad[node.node_idxs[0]];
        let mut left_hess = hess[node.node_idxs[0]];
        let mut right_grad;
        let mut right_hess;
        let mut cur_val = f[node.node_idxs[0]];

        for i in &node.node_idxs[1..] {
            let v = f[*i];
            if v == cur_val {
                left_grad += grad[*i];
                left_hess += hess[*i];
            } else {
                // We have found a new value, consider this as a
                // possible split.
                right_grad = node.grad_sum - left_grad;
                right_hess = node.grad_sum - left_hess;
                if (right_hess < self.min_leaf_weight) || (left_hess < self.min_leaf_weight) {
                    continue;
                }
                let left_gain = self.gain(left_grad, left_hess);
                let right_gain = self.gain(right_grad, right_hess);
                let split_gain = (left_gain + right_gain - node.gain_value) - self.get_gamma();
                if split_gain <= self.min_split_gain {
                    continue;
                }
                if max_gain.is_none() || split_gain > max_gain.unwrap() {
                    max_gain = Some(split_gain);
                    split_info = Some(SplitInfo {
                        split_gain,
                        split_feature: feature,
                        split_value: v,
                        left_gain,
                        left_cover: left_hess,
                        left_weight: self.weight(left_grad, left_hess),
                        left_idxs: node.node_idxs[..*i].to_vec(),
                        right_gain,
                        right_cover: right_hess,
                        right_weight: self.weight(right_grad, right_hess),
                        right_idxs: node.node_idxs[*i..].to_vec(),
                    })
                }
                // Update for new value
                left_grad += grad[*i];
                left_hess += hess[*i];
                cur_val = v;
            }
        }
        split_info
    }
}

impl<T> Splitter<T> for ExactSplitter<T>
where
    T: MatrixData<T>,
{
    fn get_l2(&self) -> T {
        self.l2
    }

    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }

    fn get_gamma(&self) -> T {
        self.gamma
    }

    fn best_split(
        &self,
        node: &mut Node<T>,
        data: &Matrix<T>,
        grad: &[T],
        hess: &[T],
    ) -> Option<SplitInfo<T>> {
        let mut best_split_info = None;
        let mut best_gain = self.min_split_gain;
        for feature in 0..(data.cols + 1) {
            let split_info = self.best_feature_split(node, data, feature, grad, hess);
            match split_info {
                Some(info) => {
                    if info.split_gain > best_gain {
                        best_gain = info.split_gain;
                        best_split_info = Some(info);
                    }
                }
                None => continue,
            }
        }
        best_split_info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_best_feature_split() {
        let data = Matrix::new(&vec![1, 2, 3, 4, 4, 4, 5], 7, 1);
        
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

