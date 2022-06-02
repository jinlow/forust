use crate::data::MatrixData;
use crate::histogram::Histograms;
use crate::node::SplittableNode;

#[derive(Debug)]
pub struct SplitInfo<T> {
    pub split_gain: T,
    pub split_feature: usize,
    pub split_value: T,
    pub split_bin: u16,
    pub missing_right: bool,
    pub left_grad: T,
    pub left_gain: T,
    pub left_cover: T,
    pub left_weight: T,
    pub right_grad: T,
    pub right_gain: T,
    pub right_cover: T,
    pub right_weight: T,
}

pub struct HistogramSplitter<T> {
    pub l2: T,
    pub gamma: T,
    pub min_leaf_weight: T,
    pub learning_rate: T,
}

impl<T> HistogramSplitter<T>
where
    T: MatrixData<T>,
{
    pub fn new(l2: T, gamma: T, min_leaf_weight: T, learning_rate: T) -> Self {
        HistogramSplitter {
            l2,
            gamma,
            min_leaf_weight,
            learning_rate,
        }
    }

    pub fn best_split(&self, node: &SplittableNode<T>) -> Option<SplitInfo<T>> {
        let mut best_split_info = None;
        let mut best_gain = T::ZERO;
        let Histograms(hists) = &node.histograms;
        for i in 0..hists.len() {
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

    pub fn best_feature_split(
        &self,
        node: &SplittableNode<T>,
        feature: usize,
    ) -> Option<SplitInfo<T>> {
        let mut split_info: Option<SplitInfo<T>> = None;
        let mut max_gain: Option<T> = None;

        let histogram = &node.histograms.0[feature];

        // We know that at least one value will be populated.
        let first_bin = histogram.get(&1).unwrap();
        // We also know we will have a missing bin.
        let missing = histogram.get(&0).unwrap();
        let mut cuml_grad = first_bin.grad_sum;
        let mut cuml_hess = first_bin.hess_sum;

        let elements = (histogram.len() as u16) + 1;

        // We start at the second element, this is because
        // there will be no element less than the first element,
        // as this would lead to a split occurring on only missing, or
        // not missing. Maybe we would want to allow this, but
        // I think it would be easier to just encode missing as a real
        // value if we really wanted to allow this. I have problems with
        // this in a production setting with other packages, such as
        // XGBoost
        for i in 2..elements {
            if let Some(bin) = histogram.get(&i) {
                // If this bin is empty, continue...
                if (bin.grad_sum == T::ZERO) && (bin.hess_sum == T::ZERO) {
                    continue;
                }
                // By default missing values will go into the left node.
                let mut missing_right = true;
                let mut left_grad = cuml_grad;
                let mut left_hess = cuml_hess;
                let mut right_grad = node.grad_sum - cuml_grad - missing.grad_sum;
                let mut right_hess = node.hess_sum - cuml_hess - missing.hess_sum;

                let mut left_gain = self.gain(left_grad, left_hess);
                let mut right_gain = self.gain(right_grad, right_hess);

                // Check Missing direction
                // Don't even worry about it, if there are no missing values
                // in this bin.
                if (missing.grad_sum != T::ZERO) && (missing.hess_sum != T::ZERO) {
                    // The gain if missing went left
                    let missing_left_gain =
                        self.gain(left_grad + missing.grad_sum, left_hess + missing.hess_sum);
                    // The gain is missing went right
                    let missing_right_gain =
                        self.gain(right_grad + missing.grad_sum, right_hess + missing.hess_sum);

                    if (missing_right_gain - right_gain) > (missing_left_gain - left_gain) {
                        // Missing goes right
                        right_grad += missing.grad_sum;
                        right_hess += missing.hess_sum;
                        right_gain = missing_right_gain;
                        missing_right = true;
                    } else {
                        // Missing goes left
                        left_grad += missing.grad_sum;
                        left_hess += missing.hess_sum;
                        left_gain = missing_left_gain;
                        missing_right = false;
                    }
                }

                // Should this be after we add in the missing hessians?
                // Right now this means that only real values will be considered
                // in this calculation I think...
                if (right_hess < self.min_leaf_weight) || (left_hess < self.min_leaf_weight) {
                    // Update for new value
                    cuml_grad += bin.grad_sum;
                    cuml_hess += bin.hess_sum;
                    continue;
                }

                let split_gain = (left_gain + right_gain - node.gain_value) - self.get_gamma();
                if split_gain <= T::ZERO {
                    // Update for new value
                    cuml_grad += bin.grad_sum;
                    cuml_hess += bin.hess_sum;
                    continue;
                }
                if max_gain.is_none() || split_gain > max_gain.unwrap() {
                    max_gain = Some(split_gain);
                    split_info = Some(SplitInfo {
                        split_gain,
                        split_feature: feature,
                        split_value: bin.cut_value,
                        split_bin: i,
                        missing_right,
                        left_grad,
                        left_gain,
                        left_cover: left_hess,
                        left_weight: self.weight(left_grad, left_hess),
                        right_grad,
                        right_gain,
                        right_cover: right_hess,
                        right_weight: self.weight(right_grad, right_hess),
                    });
                }
                // Update for new value
                cuml_grad += bin.grad_sum;
                cuml_hess += bin.hess_sum;
            }
        }
        split_info
    }

    pub fn gain(&self, grad_sum: T, hess_sum: T) -> T {
        (grad_sum * grad_sum) / (hess_sum + self.get_l2())
    }

    pub fn weight(&self, grad_sum: T, hess_sum: T) -> T {
        -((grad_sum / (hess_sum + self.get_l2())) * self.get_learning_rate())
    }

    pub fn get_l2(&self) -> T {
        self.l2
    }

    pub fn get_learning_rate(&self) -> T {
        self.learning_rate
    }

    pub fn get_gamma(&self) -> T {
        self.gamma
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
        let hists = Histograms::new(&bdata, &b.cuts, &grad, &hess, &index, true);
        let splitter = HistogramSplitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f64>(),
            hess.iter().sum::<f64>(),
            0,
            true,
            0,
            grad.len(),
        );
        let s = splitter.best_feature_split(&mut n, 0).unwrap();
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

        let b = bin_matrix(&data, &w, 10).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = Histograms::new(&bdata, &b.cuts, &grad, &hess, &index, true);

        let splitter = HistogramSplitter {
            l2: 0.0,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f64>(),
            hess.iter().sum::<f64>(),
            0,
            true,
            0,
            grad.len(),
        );
        let s = splitter.best_split(&mut n).unwrap();
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
        let grad = LogLoss::calc_grad(&y, &yhat, &w);
        let hess = LogLoss::calc_hess(&y, &yhat, &w);

        let splitter = HistogramSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
        };
        let grad_sum = grad.iter().copied().sum();
        let hess_sum = hess.iter().copied().sum();
        let root_gain = splitter.gain(grad_sum, hess_sum);
        let root_weight = splitter.weight(grad_sum, hess_sum);
        let data = Matrix::new(&data_vec, 891, 5);

        let b = bin_matrix(&data, &w, 10).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = Histograms::new(&bdata, &b.cuts, &grad, &hess, &index, true);

        let mut n = SplittableNode::new(
            0,
            // (0..(data.rows - 1)).collect(),
            hists,
            root_weight,
            root_gain,
            grad.iter().copied().sum::<f64>(),
            hess.iter().copied().sum::<f64>(),
            0,
            true,
            0,
            grad.len(),
        );
        let s = splitter.best_split(&mut n).unwrap();
        println!("{:?}", s);
        n.update_children(1, 2, &s);
        assert_eq!(0, s.split_feature);
    }
}
