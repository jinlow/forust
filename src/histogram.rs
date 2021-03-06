use crate::data::{FloatData, JaggedMatrix, Matrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Bin<T> {
    /// The sum of the gradient for this bin.
    pub grad_sum: T,
    /// The sum of the hession values for this bin.
    pub hess_sum: T,
    /// The value used to split at, this is for deciding
    /// the split value for non-binned values.
    /// This value will be missing for the missing bin.
    pub cut_value: f64,
}

impl Bin<f32> {
    pub fn new_f32(cut_value: f64) -> Self {
        Bin {
            grad_sum: f32::ZERO,
            hess_sum: f32::ZERO,
            cut_value,
        }
    }

    pub fn from_parent_child(root_bin: &Bin<f32>, child_bin: &Bin<f32>) -> Self {
        Bin {
            grad_sum: root_bin.grad_sum - child_bin.grad_sum,
            hess_sum: root_bin.hess_sum - child_bin.hess_sum,
            cut_value: root_bin.cut_value,
        }
    }
}

impl Bin<f64> {
    pub fn new_f64(cut_value: f64) -> Self {
        Bin {
            grad_sum: f64::ZERO,
            hess_sum: f64::ZERO,
            cut_value,
        }
    }

    pub fn as_f32_bin(&self) -> Bin<f32> {
        Bin {
            grad_sum: self.grad_sum as f32,
            hess_sum: self.hess_sum as f32,
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
    histogram.push(Bin::new_f64(f64::NAN));
    histogram.extend(cuts[..(cuts.len() - 1)].iter().map(|c| Bin::new_f64(*c)));
    index
        .iter()
        .zip(sorted_grad)
        .zip(sorted_hess)
        .for_each(|((i, g), h)| {
            if let Some(v) = histogram.get_mut(feature[*i] as usize) {
                v.grad_sum += f64::from(*g);
                v.hess_sum += f64::from(*h);
            }
        });
    histogram.iter().map(|b| b.as_f32_bin()).collect()
}

impl HistogramMatrix {
    pub fn new(
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        parallel: bool,
        root_node: bool,
    ) -> Self {
        let col_index: Vec<usize> = (0..data.cols).collect();
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        let sorted_grad = if root_node {
            grad.to_vec()
        } else {
            index.iter().map(|i| grad[*i]).collect::<Vec<f32>>()
        };

        let sorted_hess = if root_node {
            hess.to_vec()
        } else {
            index.iter().map(|i| hess[*i]).collect::<Vec<f32>>()
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
        let b = bin_matrix(&data, &sample_weight, 10).unwrap();
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
