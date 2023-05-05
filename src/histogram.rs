use crate::data::{FloatData, JaggedMatrix, Matrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Bin<T> {
    /// The sum of the gradient for this bin.
    pub gradient_sum: T,
    /// The sum of the hession values for this bin.
    pub hessian_sum: T,
    /// The value used to split at, this is for deciding
    /// the split value for non-binned values.
    /// This value will be missing for the missing bin.
    pub cut_value: f64,
}

impl Bin<f32> {
    pub fn new_f32(cut_value: f64) -> Self {
        Bin {
            gradient_sum: f32::ZERO,
            hessian_sum: f32::ZERO,
            cut_value,
        }
    }

    /// Calculate a new bin, using the subtraction trick on the parent bin,
    /// and the child bin.
    pub fn from_parent_child(root_bin: &Bin<f32>, child_bin: &Bin<f32>) -> Self {
        Bin {
            gradient_sum: root_bin.gradient_sum - child_bin.gradient_sum,
            hessian_sum: root_bin.hessian_sum - child_bin.hessian_sum,
            cut_value: root_bin.cut_value,
        }
    }

    /// Calculate a new bin, using the subtraction trick when the parent node
    /// has three directions, left, right, and missing.
    pub fn from_parent_two_children(
        root_bin: &Bin<f32>,
        first_child_bin: &Bin<f32>,
        second_child_bin: &Bin<f32>,
    ) -> Self {
        Bin {
            gradient_sum: root_bin.gradient_sum
                - (first_child_bin.gradient_sum + second_child_bin.gradient_sum),
            hessian_sum: root_bin.hessian_sum
                - (first_child_bin.hessian_sum + second_child_bin.hessian_sum),
            cut_value: root_bin.cut_value,
        }
    }
}

impl Bin<f64> {
    pub fn new_f64(cut_value: f64) -> Self {
        Bin {
            gradient_sum: f64::ZERO,
            hessian_sum: f64::ZERO,
            cut_value,
        }
    }

    pub fn as_f32_bin(&self) -> Bin<f32> {
        Bin {
            gradient_sum: self.gradient_sum as f32,
            hessian_sum: self.hessian_sum as f32,
            cut_value: self.cut_value,
        }
    }
}

/// Histograms implemented as as jagged matrix.
#[derive(Debug, Deserialize, Serialize)]
pub struct HistogramMatrix(pub JaggedMatrix<Bin<f32>>);

/// Create a histogram for a given feature, we use f64
/// values to accumulate, so that we don't lose precision,
/// but then still return f32 values for memory efficiency
/// and speed.
pub fn create_feature_histogram(
    feature: &[u16],
    cuts: &[f64],
    sorted_grad: &[f32],
    sorted_hess: &[f32],
    index: &[usize],
) -> Vec<Bin<f32>> {
    let mut histogram: Vec<Bin<f64>> = Vec::with_capacity(cuts.len());
    // The first value is missing, it seems to not matter that we are using
    // Missing here, rather than the booster "missing" definition, because
    // we just always assume the first bin of the histogram is missing.
    histogram.push(Bin::new_f64(f64::NAN));
    // The last cut value is simply the maximum possible value, so we don't need it.
    // This value is needed initially for binning, but we don't need to count it as
    // a histogram bin.
    histogram.extend(cuts[..(cuts.len() - 1)].iter().map(|c| Bin::new_f64(*c)));
    index
        .iter()
        .zip(sorted_grad)
        .zip(sorted_hess)
        .for_each(|((i, g), h)| {
            if let Some(v) = histogram.get_mut(feature[*i] as usize) {
                v.gradient_sum += f64::from(*g);
                v.hessian_sum += f64::from(*h);
            }
        });
    histogram.iter().map(|b| b.as_f32_bin()).collect()
}

impl HistogramMatrix {
    /// Create an empty histogram matrix.
    pub fn empty() -> Self {
        HistogramMatrix(JaggedMatrix {
            data: Vec::new(),
            ends: Vec::new(),
            cols: 0,
            n_records: 0,
        })
    }
    pub fn new(
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        parallel: bool,
        sort: bool,
    ) -> Self {
        let col_index: Vec<usize> = (0..data.cols).collect();
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        let (sorted_grad, sorted_hess) = if !sort {
            (grad.to_vec(), hess.to_vec())
        } else {
            let mut n_grad = Vec::new();
            let mut n_hess = Vec::new();
            for i in index {
                let i_ = *i;
                n_grad.push(grad[i_]);
                n_hess.push(hess[i_]);
            }
            (n_grad, n_hess)
        };

        let histograms = if parallel {
            col_index
                .par_iter()
                .flat_map(|col| {
                    create_feature_histogram(
                        data.get_col(*col),
                        cuts.get_col(*col),
                        &sorted_grad,
                        &sorted_hess,
                        index,
                    )
                })
                .collect::<Vec<Bin<f32>>>()
        } else {
            col_index
                .iter()
                .flat_map(|col| {
                    create_feature_histogram(
                        data.get_col(*col),
                        cuts.get_col(*col),
                        &sorted_grad,
                        &sorted_hess,
                        index,
                    )
                })
                .collect::<Vec<Bin<f32>>>()
        };
        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends: cuts.ends.to_owned(),
            cols: cuts.cols,
            n_records: cuts.n_records,
        })
    }

    /// Calculate the histogram matrix, for a child, given the parent histogram
    /// matrix, and the other child histogram matrix. This should be used
    /// when the node has only two possible splits, left and right.
    pub fn from_parent_child(
        root_histogram: &HistogramMatrix,
        child_histogram: &HistogramMatrix,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(child) = child_histogram;
        let histograms = root
            .data
            .iter()
            .zip(child.data.iter())
            .map(|(root_bin, child_bin)| Bin::from_parent_child(root_bin, child_bin))
            .collect();
        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends: child.ends.to_owned(),
            cols: child.cols,
            n_records: child.n_records,
        })
    }

    /// Calculate the histogram matrix for a child, given the parent histogram
    /// and two other child histograms. This should be used with the node has
    /// three possible split paths, right, left, and missing.
    pub fn from_parent_two_children(
        root_histogram: &HistogramMatrix,
        first_child_histogram: &HistogramMatrix,
        second_child_histogram: &HistogramMatrix,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(first_child) = first_child_histogram;
        let HistogramMatrix(second_child) = second_child_histogram;
        let histograms = root
            .data
            .iter()
            .zip(first_child.data.iter())
            .zip(second_child.data.iter())
            .map(|((root_bin, first_child_bin), second_child_bin)| {
                Bin::from_parent_two_children(root_bin, first_child_bin, second_child_bin)
            })
            .collect();
        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends: first_child.ends.to_owned(),
            cols: first_child.cols,
            n_records: first_child.n_records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use std::fs;
    #[test]
    fn test_single_histogram() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let sample_weight = vec![1.; data.rows];
        let b = bin_matrix(&data, &sample_weight, 10, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);
        let hist =
            create_feature_histogram(&bdata.get_col(1), &b.cuts.get_col(1), &g, &h, &bdata.index);
        // println!("{:?}", hist);
        let mut f = bdata.get_col(1).to_owned();
        println!("{:?}", hist);
        f.sort();
        f.dedup();
        assert_eq!(f.len() + 1, hist.len());
    }
}
