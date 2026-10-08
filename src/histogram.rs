use crate::data::{FloatData, JaggedMatrix, Matrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many rows, gathering gradients serially is cheaper than scheduling Rayon tasks.
const PARALLEL_GATHER_MIN_ROWS: usize = 16_384;
/// Bins per Rayon task when subtracting histograms in parallel.
const PARALLEL_SUBTRACT_MIN_BINS: usize = 4_096;

/// Struct to hold the information of a given bin. Bin `k > 0` of a feature's
/// histogram covers values below cut `k - 1` of that feature; bin 0 is missing.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct Bin<T> {
    /// The sum of the gradient for this bin.
    pub gradient_sum: T,
    /// The sum of the hession values for this bin.
    pub hessian_sum: T,
}

impl Bin<f32> {
    pub fn new_f32() -> Self {
        Bin {
            gradient_sum: f32::ZERO,
            hessian_sum: f32::ZERO,
        }
    }

    /// Calculate a new bin, using the subtraction trick on the parent bin,
    /// and the child bin.
    pub fn from_parent_child(root_bin: &Bin<f32>, child_bin: &Bin<f32>) -> Self {
        Bin {
            gradient_sum: root_bin.gradient_sum - child_bin.gradient_sum,
            hessian_sum: root_bin.hessian_sum - child_bin.hessian_sum,
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
        }
    }
}

impl Bin<f64> {
    pub fn new_f64() -> Self {
        Bin {
            gradient_sum: f64::ZERO,
            hessian_sum: f64::ZERO,
        }
    }

    pub fn as_f32_bin(&self) -> Bin<f32> {
        Bin {
            gradient_sum: self.gradient_sum as f32,
            hessian_sum: self.hessian_sum as f32,
        }
    }
}

/// Histograms implemented as as jagged matrix.
#[derive(Debug, Deserialize, Serialize)]
pub struct HistogramMatrix(pub JaggedMatrix<Bin<f32>>);

/// Fill the histogram for a given feature. Sums accumulate in f64 so we don't
/// lose precision, but are stored as f32 values for memory efficiency and speed.
/// `out` must have one bin per cut value: the missing bin, then one per cut
/// excluding the final maximum.
pub fn fill_feature_histogram(
    out: &mut [Bin<f32>],
    feature: &[u16],
    sorted_grad: &[f32],
    sorted_hess: &[f32],
    index: &[usize],
) {
    let mut sums = vec![(f64::ZERO, f64::ZERO); out.len()];
    index
        .iter()
        .zip(sorted_grad)
        .zip(sorted_hess)
        .for_each(|((i, g), h)| {
            if let Some(v) = sums.get_mut(feature[*i] as usize) {
                v.0 += f64::from(*g);
                v.1 += f64::from(*h);
            }
        });
    for (bin, (g, h)) in out.iter_mut().zip(sums) {
        *bin = Bin {
            gradient_sum: g as f32,
            hessian_sum: h as f32,
        };
    }
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        col_index: &[usize],
        parallel: bool,
        sort: bool,
    ) -> Self {
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        let gathered: (Vec<f32>, Vec<f32>);
        let (sorted_grad, sorted_hess): (&[f32], &[f32]) = if !sort {
            (grad, hess)
        } else {
            gathered = if parallel && index.len() >= PARALLEL_GATHER_MIN_ROWS {
                index.par_iter().map(|&i| (grad[i], hess[i])).unzip()
            } else {
                index.iter().map(|&i| (grad[i], hess[i])).unzip()
            };
            (&gathered.0, &gathered.1)
        };

        // If we have sampled down the columns, we need to recalculate the ends.
        // we can do this by iterating over the cut's, as this will be the size
        // of the histograms.
        let ends: Vec<usize> = if col_index.len() == data.cols {
            cuts.ends.to_owned()
        } else {
            col_index
                .iter()
                .scan(0_usize, |state, i| {
                    *state += cuts.get_col(*i).len();
                    Some(*state)
                })
                .collect()
        };
        let n_records = if col_index.len() == data.cols {
            cuts.n_records
        } else {
            ends.iter().sum()
        };

        let total_bins = ends.last().copied().unwrap_or(0);
        let mut histograms: Vec<Bin<f32>> = Vec::with_capacity(total_bins);
        if parallel {
            (0..total_bins)
                .into_par_iter()
                .map(|_| Bin::new_f32())
                .collect_into_vec(&mut histograms);
        } else {
            histograms.resize(total_bins, Bin::new_f32());
        }
        let mut column_bins: Vec<&mut [Bin<f32>]> = Vec::with_capacity(col_index.len());
        let mut rest = histograms.as_mut_slice();
        for col in col_index {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(cuts.get_col(*col).len());
            column_bins.push(head);
            rest = tail;
        }
        let fill = |(out, col): (&mut [Bin<f32>], &usize)| {
            fill_feature_histogram(out, data.get_col(*col), sorted_grad, sorted_hess, index)
        };
        if parallel {
            column_bins.into_par_iter().zip(col_index).for_each(fill);
        } else {
            column_bins.into_iter().zip(col_index).for_each(fill);
        }

        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends,
            cols: col_index.len(),
            n_records,
        })
    }

    /// Calculate the histogram matrix, for a child, given the parent histogram
    /// matrix, and the other child histogram matrix. This should be used
    /// when the node has only two possible splits, left and right.
    pub fn from_parent_child(
        root_histogram: &HistogramMatrix,
        child_histogram: &HistogramMatrix,
        parallel: bool,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(child) = child_histogram;
        let histograms = if parallel {
            root.data
                .par_iter()
                .zip(child.data.par_iter())
                .with_min_len(PARALLEL_SUBTRACT_MIN_BINS)
                .map(|(root_bin, child_bin)| Bin::from_parent_child(root_bin, child_bin))
                .collect()
        } else {
            root.data
                .iter()
                .zip(child.data.iter())
                .map(|(root_bin, child_bin)| Bin::from_parent_child(root_bin, child_bin))
                .collect()
        };
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
        parallel: bool,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(first_child) = first_child_histogram;
        let HistogramMatrix(second_child) = second_child_histogram;
        let histograms = if parallel {
            root.data
                .par_iter()
                .zip(first_child.data.par_iter())
                .zip(second_child.data.par_iter())
                .with_min_len(PARALLEL_SUBTRACT_MIN_BINS)
                .map(|((root_bin, first_child_bin), second_child_bin)| {
                    Bin::from_parent_two_children(root_bin, first_child_bin, second_child_bin)
                })
                .collect()
        } else {
            root.data
                .iter()
                .zip(first_child.data.iter())
                .zip(second_child.data.iter())
                .map(|((root_bin, first_child_bin), second_child_bin)| {
                    Bin::from_parent_two_children(root_bin, first_child_bin, second_child_bin)
                })
                .collect()
        };
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
        let b = bin_matrix(&data, &sample_weight, 10, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        let cuts = b.cuts.get_col(1);
        let mut hist = vec![Bin::new_f32(); cuts.len()];
        fill_feature_histogram(&mut hist, &bdata.get_col(1), &g, &h, &bdata.index);
        // println!("{:?}", hist);
        let mut f = bdata.get_col(1).to_owned();
        println!("{:?}", hist);
        f.sort();
        f.dedup();
        assert_eq!(f.len() + 1, hist.len());
    }
}
