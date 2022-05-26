use crate::data::{Matrix, MatrixData};
use crate::node::SplittableNode;
use crate::splitter::{SplitInfo, Splitter};
// use rayon::prelude::*;

pub struct ExactSplitter<T> {
    pub l2: T,
    pub gamma: T,
    pub min_leaf_weight: T,
    pub learning_rate: T,
}

impl<'a, T> ExactSplitter<T>
where
    T: MatrixData<T>,
{
    fn best_feature_split(
        &self,
        node: &mut SplittableNode<T>,
        data: &Matrix<T>,
        feature: usize,
        grad: &[T],
        hess: &[T],
        index: &mut [usize],
    ) -> Option<SplitInfo<T>> {
        let mut split_info: Option<SplitInfo<T>> = None;
        let mut max_gain: Option<T> = None;

        let f = data.get_col(feature);

        let node_idxs = &mut index[node.start_idx..node.stop_idx];
        node_idxs.sort_unstable_by(|a, b| f[*a].partial_cmp(&f[*b]).unwrap());

        let mut left_grad = grad[node_idxs[0]];
        let mut left_hess = hess[node_idxs[0]];
        let mut right_grad;
        let mut right_hess;
        let mut cur_val = f[node_idxs[0]];

        for (idx_, i) in node_idxs[1..].iter().enumerate() {
            let v = f[*i];
            if v == cur_val {
                left_grad += grad[*i];
                left_hess += hess[*i];
            } else {
                cur_val = v;
                // We have found a new value, consider this as a
                // possible split.
                right_grad = node.grad_sum - left_grad;
                right_hess = node.hess_sum - left_hess;

                if (right_hess < self.min_leaf_weight) || (left_hess < self.min_leaf_weight) {
                    // Update for new value
                    left_grad += grad[*i];
                    left_hess += hess[*i];
                    continue;
                }
                let left_gain = self.gain(left_grad, left_hess);
                let right_gain = self.gain(right_grad, right_hess);
                let split_gain = (left_gain + right_gain - node.gain_value) - self.get_gamma();
                if split_gain <= T::ZERO {
                    // Update for new value
                    left_grad += grad[*i];
                    left_hess += hess[*i];
                    continue;
                }
                if max_gain.is_none() || split_gain > max_gain.unwrap() {
                    max_gain = Some(split_gain);
                    split_info = Some(SplitInfo {
                        split_gain,
                        split_feature: feature,
                        split_value: v,
                        left_grad,
                        left_gain,
                        left_cover: left_hess,
                        left_weight: self.weight(left_grad, left_hess),
                        left_start_idx: node.start_idx,
                        left_stop_idx: idx_ + 1 + node.start_idx,
                        right_grad,
                        right_gain,
                        right_cover: right_hess,
                        right_weight: self.weight(right_grad, right_hess),
                        right_start_idx: idx_ + 1 + node.start_idx,
                        right_stop_idx: node.stop_idx,
                    });
                }
                // Update for new value
                left_grad += grad[*i];
                left_hess += hess[*i];
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
        node: &mut SplittableNode<T>,
        data: &Matrix<T>,
        grad: &[T],
        hess: &[T],
        index: &mut [usize],
    ) -> Option<SplitInfo<T>> {
        let mut best_split_info = None;
        let mut best_gain = T::ZERO;
        for feature in 0..(data.cols) {
            let split_info = self.best_feature_split(node, data, feature, grad, hess, index);
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
        // If there is best info, resort the index, so that
        // the start and stop are correct.
        if let Some(info) = &best_split_info {
            let f = data.get_col(info.split_feature);
            let node_idxs = &mut index[node.start_idx..node.stop_idx];
            node_idxs.sort_by(|a, b| f[*a].partial_cmp(&f[*b]).unwrap());
        }

        best_split_info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::SplittableNode;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use std::fs;
    #[test]
    fn test_best_feature_split() {
        let d = vec![4., 2., 3., 4., 5., 1., 4.];
        let data = Matrix::new(&d, 7, 1);
        let y = vec![0., 0., 0., 1., 1., 0., 1.];
        let yhat = vec![0.; 7];
        let w = vec![1.; y.len()];
        let grad = LogLoss::calc_grad(&y, &yhat, &w);
        let hess = LogLoss::calc_hess(&y, &yhat, &w);
        let es = ExactSplitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            0.0,
            0.14,
            grad.iter().sum::<f64>(),
            hess.iter().sum::<f64>(),
            0,
            0,
            grad.len(),
        );
        let mut index = data.index.to_owned();
        let s = es
            .best_feature_split(&mut n, &data, 0, &grad, &hess, &mut index)
            .unwrap();
        println!("{:?}", s);
        assert_eq!(s.split_value, 4.0);
        assert_eq!(s.left_cover, 0.75);
        assert_eq!(s.right_cover, 1.0);
        assert_eq!(s.left_gain, 3.0);
        assert_eq!(s.right_gain, 1.0);
        assert_eq!(s.split_gain, 3.86);
    }

    #[test]
    fn test_best_split() {
        let d: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 0., 4., 2., 3., 4., 5., 1., 4.];
        let data = Matrix::new(&d, 7, 2);
        let y = vec![0., 0., 0., 1., 1., 0., 1.];
        let yhat = vec![0.; 7];
        let w = vec![1.; y.len()];
        let grad = LogLoss::calc_grad(&y, &yhat, &w);
        let hess = LogLoss::calc_hess(&y, &yhat, &w);
        let es = ExactSplitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            0.0,
            0.14,
            grad.iter().sum::<f64>(),
            hess.iter().sum::<f64>(),
            0,
            0,
            grad.len(),
        );
        let mut index = data.index.to_owned();
        let index = index.as_mut();
        let s = es.best_split(&mut n, &data, &grad, &hess, index).unwrap();
        println!("{:?}", s);
        assert_eq!(s.split_feature, 1);
        assert_eq!(s.split_value, 4.);
        assert_eq!(s.left_cover, 0.75);
        assert_eq!(s.right_cover, 1.);
        assert_eq!(s.left_gain, 3.);
        assert_eq!(s.right_gain, 1.);
        assert_eq!(s.split_gain, 3.86);
    }

    #[test]
    fn test_data_split() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);

        let es = ExactSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
        };
        let grad_sum = g.iter().copied().sum();
        let hess_sum = h.iter().copied().sum();
        let root_gain = es.gain(grad_sum, hess_sum);
        let root_weight = es.weight(grad_sum, hess_sum);
        let data = Matrix::new(&data_vec, 891, 5);
        let mut n = SplittableNode::new(
            0,
            // (0..(data.rows - 1)).collect(),
            root_weight,
            root_gain,
            g.iter().copied().sum::<f64>(),
            h.iter().copied().sum::<f64>(),
            0,
            0,
            g.len(),
        );
        let mut index = data.index.to_owned();
        let index = index.as_mut();
        let s = es.best_split(&mut n, &data, &g, &h, index).unwrap();
        n.update_children(1, 2, &s);
        //println!("{}", n);
        assert_eq!(0, 0);
    }
}
