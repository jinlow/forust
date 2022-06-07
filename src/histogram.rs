use std::collections::HashMap;

use crate::data::{Matrix, MatrixData};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize)]
pub struct Bin<T> {
    /// The sum of the gradient for this bin.
    pub grad_sum: T,
    /// The sum of the hession values for this bin.
    pub hess_sum: T,
    /// The value used to split at, this is for deciding
    /// the split value for non-binned values.
    /// This value will be missing for the missing bin.
    pub cut_value: T,
}

impl<T> Bin<T>
where
    T: MatrixData<T>,
{
    pub fn new(cut_value: T) -> Self {
        Bin {
            grad_sum: T::ZERO,
            hess_sum: T::ZERO,
            cut_value,
        }
    }

    pub fn from_parent_child(root_bin: &Bin<T>, child_bin: &Bin<T>) -> Self {
        Bin {
            grad_sum: root_bin.grad_sum - child_bin.grad_sum,
            hess_sum: root_bin.hess_sum - child_bin.hess_sum,
            cut_value: root_bin.cut_value,
        }
    }
}

pub type Hist<T> = HashMap<u16, Bin<T>>;

#[derive(Debug, Deserialize, Serialize)]
pub struct Histograms<T>(pub Vec<Hist<T>>);

pub fn create_feature_histogram<T: MatrixData<T>>(
    feature: &[u16],
    cuts: &[T],
    grad: &[T],
    hess: &[T],
    index: &[usize],
) -> Hist<T> {
    let mut histogram: Hist<T> = cuts[..(cuts.len() - 1)]
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                // This should never panic, because the maximum number
                // of cuts possible is u16::MAX
                u16::try_from(i).expect("To many bins for a u16.") + 1,
                Bin::new(*c),
            )
        })
        .collect();
    // Add the missing bin
    histogram.insert(0, Bin::new(T::NAN));
    // Now add all of the stats
    index.iter().for_each(|i| {
        if let Some(v) = histogram.get_mut(&feature[*i]) {
            v.grad_sum += grad[*i];
            v.hess_sum += hess[*i];
        }
    });
    histogram
}

pub fn create_feature_histogram_from_parent_child<T: MatrixData<T>>(
    root_histogram: &Hist<T>,
    child_histogram: &Hist<T>,
) -> Hist<T> {
    let mut histogram: Hist<T> = HashMap::new();
    root_histogram.keys().for_each(|k| {
        let root_bin = root_histogram.get(k).unwrap();
        let child_bin = child_histogram.get(k).unwrap();
        histogram.insert(*k, Bin::from_parent_child(root_bin, child_bin));
    });
    histogram
}

impl<T> Histograms<T>
where
    T: MatrixData<T>,
{
    pub fn new(
        data: &Matrix<u16>,
        cuts: &[Vec<T>],
        grad: &[T],
        hess: &[T],
        index: &[usize],
        parallel: bool,
    ) -> Self {
        let col_index: Vec<usize> = (0..data.cols).collect();
        if parallel {
            Histograms(
                col_index
                    .par_iter()
                    .map(|i| {
                        create_feature_histogram(data.get_col(*i), &cuts[*i], grad, hess, index)
                    })
                    .collect(),
            )
        } else {
            Histograms(
                col_index
                    .iter()
                    .map(|i| {
                        create_feature_histogram(data.get_col(*i), &cuts[*i], grad, hess, index)
                    })
                    .collect(),
            )
        }
    }

    pub fn from_parent_child(
        root_histograms: &Histograms<T>,
        child_histograms: &Histograms<T>,
    ) -> Self {
        let Histograms(root_hists) = root_histograms;
        let Histograms(child_hists) = child_histograms;

        let histograms = root_hists
            .iter()
            .zip(child_hists)
            .map(|(rh, ch)| create_feature_histogram_from_parent_child(rh, ch))
            .collect();
        Histograms(histograms)
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
        let hist = create_feature_histogram(&bdata.get_col(1), &b.cuts[1], &g, &h, &bdata.index);
        // println!("{:?}", hist);
        let mut f = bdata.get_col(1).to_owned();
        f.sort();
        f.dedup();
        assert_eq!(f.len() + 1, hist.len());
    }
}
