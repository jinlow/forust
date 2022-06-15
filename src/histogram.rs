use crate::data::{JaggedMatrix, Matrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Bin {
    /// The sum of the gradient for this bin.
    pub grad_sum: f32,
    /// The sum of the hession values for this bin.
    pub hess_sum: f32,
    /// The value used to split at, this is for deciding
    /// the split value for non-binned values.
    /// This value will be missing for the missing bin.
    pub cut_value: f64,
}

impl Bin {
    pub fn new(cut_value: f64) -> Self {
        Bin {
            grad_sum: 0.,
            hess_sum: 0.,
            cut_value,
        }
    }

    pub fn from_parent_child(root_bin: &Bin, child_bin: &Bin) -> Self {
        Bin {
            grad_sum: root_bin.grad_sum - child_bin.grad_sum,
            hess_sum: root_bin.hess_sum - child_bin.hess_sum,
            cut_value: root_bin.cut_value,
        }
    }
}

/// Histograms implemented as as jagged matrix.
#[derive(Debug, Deserialize, Serialize)]
pub struct HistogramMatrix(pub JaggedMatrix<Bin>);

pub fn create_feature_histogram(
    feature: &[u16],
    cuts: &[f64],
    sorted_grad: &[f32],
    sorted_hess: &[f32],
    index: &[usize],
) -> Vec<Bin> {
    let mut histogram: Vec<Bin> = Vec::with_capacity(cuts.len());
    histogram.push(Bin::new(f64::NAN));
    histogram.extend(cuts[..(cuts.len() - 1)].iter().map(|c| Bin::new(*c)));
    let mut n_recs = 0;
    // println!("glen: {}, hlen: {}, ilen: {}", sorted_grad.len(), sorted_hess.len(), &index.len());
    index
        .iter()
        .zip(sorted_grad)
        .zip(sorted_hess)
        .for_each(|((i, g), h)| {
            
            if let Some(v) = histogram.get_mut(usize::from(feature[*i])) {
                v.grad_sum += *g;
                v.hess_sum += *h;
            }
            n_recs += 1;
        });
        println!("Recs in hist: {}, hess_sum: {}, true hess sum: {}", n_recs, histogram.iter().fold(0., |n, b| n + b.hess_sum), sorted_hess.iter().sum::<f32>());
        println!("Recs in hist: {}", n_recs);
    histogram
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
                .collect::<Vec<Bin>>()
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
                .collect::<Vec<Bin>>()
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
