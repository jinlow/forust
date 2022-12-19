use crate::constraints::{Constraint, ConstraintMap};
use crate::data::FloatData;
use crate::histogram::HistogramMatrix;
use crate::missinghandler::MissingInfo;
use crate::node::SplittableNode;
use crate::utils::{constrained_weight, cull_gain, gain, gain_given_weight, weight};

#[derive(Debug)]
pub struct SplitInfo {
    pub split_gain: f32,
    pub split_feature: usize,
    pub split_value: f64,
    pub split_bin: u16,
    pub left_node: NodeInfo,
    pub right_node: NodeInfo,
    pub missing_node: MissingInfo,
}

#[derive(Debug)]
pub struct NodeInfo {
    pub grad: f32,
    pub gain: f32,
    pub cover: f32,
    pub weight: f32,
    pub bounds: (f32, f32),
}

pub struct Splitter {
    pub l2: f32,
    pub gamma: f32,
    pub min_leaf_weight: f32,
    pub learning_rate: f32,
    pub allow_missing_splits: bool,
    pub impute_missing: bool,
    pub constraints_map: ConstraintMap,
}

impl Splitter {
    pub fn new(
        l2: f32,
        gamma: f32,
        min_leaf_weight: f32,
        learning_rate: f32,
        allow_missing_splits: bool,
        impute_missing: bool,
        constraints_map: ConstraintMap,
    ) -> Self {
        Splitter {
            l2,
            gamma,
            min_leaf_weight,
            learning_rate,
            allow_missing_splits,
            impute_missing,
            constraints_map,
        }
    }

    pub fn best_split(&self, node: &SplittableNode) -> Option<SplitInfo> {
        let mut best_split_info = None;
        let mut best_gain = f32::ZERO;
        let HistogramMatrix(histograms) = &node.histograms;
        for i in 0..histograms.cols {
            let split_info = self.best_feature_split(node, i);
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

    pub fn best_feature_split(&self, node: &SplittableNode, feature: usize) -> Option<SplitInfo> {
        let mut split_info: Option<SplitInfo> = None;
        let mut max_gain: Option<f32> = None;

        let HistogramMatrix(histograms) = &node.histograms;
        let histogram = histograms.get_col(feature);

        // We also know we will have a missing bin.
        let missing = &histogram[0];
        let mut cuml_grad = 0.0; // first_bin.grad_sum;
        let mut cuml_hess = 0.0; // first_bin.hess_sum;
        let mut i = 0;
        let mut first_idx = 1;
        let constraint = self.constraints_map.get(&feature);

        if !self.allow_missing_splits {
            // If we don't want a split to be only on missing, we need
            // to start at the first bin that is populated.
            for bin in &histogram[1..] {
                i += 1;
                first_idx += 1;
                if (bin.grad_sum == f32::ZERO) && (bin.hess_sum == f32::ZERO) {
                    continue;
                }
                cuml_grad += bin.grad_sum;
                cuml_hess += bin.hess_sum;
                break;
            }
        }

        let elements = histogram.len();
        assert!(elements == histogram.len());

        for bin in &histogram[first_idx..] {
            i += 1;
            // If this bin is empty, continue...
            // however we are only concerned about this
            // if we don't want a split to happen only
            // on missing values.
            if (bin.grad_sum == f32::ZERO)
                && (bin.hess_sum == f32::ZERO)
                && !self.allow_missing_splits
            {
                continue;
            }
            // By default missing values will go into the right node.
            let mut missing_right = true;
            let mut left_grad = cuml_grad;
            let mut left_hess = cuml_hess;
            let mut right_grad = node.grad_sum - cuml_grad - missing.grad_sum;
            let mut right_hess = node.hess_sum - cuml_hess - missing.hess_sum;

            let mut left_weight = constrained_weight(
                &self.l2,
                left_grad,
                left_hess,
                node.lower_bound,
                node.upper_bound,
                constraint,
            );
            let mut right_weight = constrained_weight(
                &self.l2,
                right_grad,
                right_hess,
                node.lower_bound,
                node.upper_bound,
                constraint,
            );

            let mut left_gain = gain_given_weight(&self.l2, left_grad, left_hess, left_weight);
            let mut right_gain = gain_given_weight(&self.l2, right_grad, right_hess, right_weight);

            // let mut left_gain = self.gain(left_grad, left_hess);
            // let mut right_gain = self.gain(right_grad, right_hess);

            if !self.allow_missing_splits {
                // Check the min_hessian constraint first, if we do not
                // want to allow missing only splits.
                if (right_hess < self.min_leaf_weight) || (left_hess < self.min_leaf_weight) {
                    // Update for new value
                    cuml_grad += bin.grad_sum;
                    cuml_hess += bin.hess_sum;
                    continue;
                }
            }

            // Check Missing direction
            // Don't even worry about it, if there are no missing values
            // in this bin.
            if !self.impute_missing {
                // Missing goes right
                let missing_right_gain = gain(
                    &self.l2,
                    right_grad + missing.grad_sum,
                    right_hess + missing.hess_sum,
                );
                right_grad += missing.grad_sum;
                right_hess += missing.hess_sum;
                right_gain = missing_right_gain;
                missing_right = true;
            }
            // Otherwise, are there even any missing to worry about?
            else if (missing.grad_sum != f32::ZERO) || (missing.hess_sum != f32::ZERO) {
                // TODO: Consider making this safer, by casting to f64, summing, and then
                // back to f32...

                // The weight if missing went left
                let missing_left_weight = constrained_weight(
                    &self.l2,
                    left_grad + missing.grad_sum,
                    left_hess + missing.hess_sum,
                    node.lower_bound,
                    node.upper_bound,
                    constraint,
                );
                // The gain if missing went left
                let missing_left_gain = gain_given_weight(
                    &self.l2,
                    left_grad + missing.grad_sum,
                    left_hess + missing.hess_sum,
                    missing_left_weight,
                );
                // Confirm this wouldn't break monotonicity.
                let missing_left_gain = cull_gain(
                    missing_left_gain,
                    missing_left_weight,
                    right_weight,
                    constraint,
                );

                // The gain if missing went right
                let missing_right_weight = weight(
                    &self.l2,
                    right_grad + missing.grad_sum,
                    right_hess + missing.hess_sum,
                );
                // The gain is missing went right
                let missing_right_gain = gain_given_weight(
                    &self.l2,
                    right_grad + missing.grad_sum,
                    right_hess + missing.hess_sum,
                    missing_right_weight,
                );
                // Confirm this wouldn't break monotonicity.
                let missing_left_gain = cull_gain(
                    missing_left_gain,
                    missing_left_weight,
                    right_weight,
                    constraint,
                );

                if (missing_right_gain - right_gain) < (missing_left_gain - left_gain) {
                    // Missing goes left
                    left_grad += missing.grad_sum;
                    left_hess += missing.hess_sum;
                    left_gain = missing_left_gain;
                    left_weight = missing_left_weight;
                    missing_right = false;
                } else {
                    // Missing goes right
                    right_grad += missing.grad_sum;
                    right_hess += missing.hess_sum;
                    right_gain = missing_right_gain;
                    right_weight = missing_right_weight;
                    missing_right = true;
                }
            }

            if (right_hess < self.min_leaf_weight) || (left_hess < self.min_leaf_weight) {
                // Update for new value
                cuml_grad += bin.grad_sum;
                cuml_hess += bin.hess_sum;
                continue;
            }

            let split_gain = (left_gain + right_gain - node.gain_value) - self.gamma;
            // Check monotonicity holds
            let split_gain = cull_gain(split_gain, left_weight, right_weight, constraint);

            if split_gain <= f32::ZERO {
                // Update for new value
                cuml_grad += bin.grad_sum;
                cuml_hess += bin.hess_sum;
                continue;
            }

            let mid = (left_weight + right_weight) / 2.0;
            let (left_bounds, right_bounds) = match constraint {
                None | Some(Constraint::Unconstrained) => (
                    (node.lower_bound, node.upper_bound),
                    (node.lower_bound, node.upper_bound),
                ),
                Some(Constraint::Negative) => ((mid, node.upper_bound), (node.lower_bound, mid)),
                Some(Constraint::Positive) => ((node.lower_bound, mid), (mid, node.upper_bound)),
            };

            // If split gain is NaN, one of the sides is empty, do not allow
            // this split.
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };
            if max_gain.is_none() || split_gain > max_gain.unwrap() {
                let missing_node = if missing_right {
                    MissingInfo::Right
                } else {
                    MissingInfo::Left
                };
                max_gain = Some(split_gain);
                split_info = Some(SplitInfo {
                    split_gain,
                    split_feature: feature,
                    split_value: bin.cut_value,
                    split_bin: i as u16,
                    left_node: NodeInfo {
                        grad: left_grad,
                        gain: left_gain,
                        cover: left_hess,
                        weight: left_weight * self.learning_rate,
                        bounds: left_bounds,
                    },
                    right_node: NodeInfo {
                        grad: right_grad,
                        gain: right_gain,
                        cover: right_hess,
                        weight: right_weight * self.learning_rate,
                        bounds: right_bounds,
                    },
                    missing_node,
                });
            }
            // Update for new value
            cuml_grad += bin.grad_sum;
            cuml_hess += bin.hess_sum;
        }
        split_info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::data::Matrix;
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

        let b = bin_matrix(&data, &w, 10).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = HistogramMatrix::new(&bdata, &b.cuts, &grad, &hess, &index, true, false);
        let splitter = Splitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
            allow_missing_splits: true,
            impute_missing: true,
            constraints_map: ConstraintMap::new(),
        };
        // println!("{:?}", hists);
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f32>(),
            hess.iter().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_feature_split(&mut n, 0).unwrap();
        println!("{:?}", s);
        assert_eq!(s.split_value, 4.0);
        assert_eq!(s.left_node.cover, 0.75);
        assert_eq!(s.right_node.cover, 1.0);
        assert_eq!(s.left_node.gain, 3.0);
        assert_eq!(s.right_node.gain, 1.0);
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

        let b = bin_matrix(&data, &w, 10).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = HistogramMatrix::new(&bdata, &b.cuts, &grad, &hess, &index, true, false);
        println!("{:?}", hists);
        let splitter = Splitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
            allow_missing_splits: true,
            impute_missing: true,
            constraints_map: ConstraintMap::new(),
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f32>(),
            hess.iter().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_split(&mut n).unwrap();
        println!("{:?}", s);
        assert_eq!(s.split_feature, 1);
        assert_eq!(s.split_value, 4.);
        assert_eq!(s.left_node.cover, 0.75);
        assert_eq!(s.right_node.cover, 1.);
        assert_eq!(s.left_node.gain, 3.);
        assert_eq!(s.right_node.gain, 1.);
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
        let grad = LogLoss::calc_grad(&y, &yhat, &w);
        let hess = LogLoss::calc_hess(&y, &yhat, &w);

        let splitter = Splitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            impute_missing: true,
            constraints_map: ConstraintMap::new(),
        };
        let grad_sum = grad.iter().copied().sum();
        let hess_sum = hess.iter().copied().sum();
        let root_gain = gain(&splitter.l2, grad_sum, hess_sum);
        let root_weight = weight(&splitter.l2, grad_sum, hess_sum);
        // let gain_given_weight = splitter.gain_given_weight(grad_sum, hess_sum, root_weight);
        // println!("gain: {}, weight: {}, gain from weight: {}", root_gain, root_weight, gain_given_weight);
        let data = Matrix::new(&data_vec, 891, 5);

        let b = bin_matrix(&data, &w, 10).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = HistogramMatrix::new(&bdata, &b.cuts, &grad, &hess, &index, true, false);

        let mut n = SplittableNode::new(
            0,
            // (0..(data.rows - 1)).collect(),
            hists,
            root_weight,
            root_gain,
            grad.iter().copied().sum::<f32>(),
            hess.iter().copied().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_split(&mut n).unwrap();
        println!("{:?}", s);
        n.update_children(1, 2, &s);
        assert_eq!(0, s.split_feature);
    }
}
