use crate::data::{Matrix, MatrixData};
use crate::utils::{first_greater_than, percentiles_nunique};

// We want to be able to bin our dataset into discrete buckets.
// First we will calculate percentils and the number of unique values
// for each feature.
// Then we will bucket them into bins from 0 to N + 1 where N is the number
// of unique bin values created from the percentiles, and the very last
// bin is missing values.
// For now, we will just use usize, although, it would be good to see if
// we can use something smaller, u8 for instance.

struct BinnedData<T> {
    binned_data: Vec<usize>,
    cuts: Vec<Vec<T>>,
    nunique: Vec<usize>,
}

/// Convert a matrix of data, into a binned matrix.
fn bin_matrix_from_cuts<T: std::cmp::PartialOrd>(data: &Matrix<T>, cuts: &[Vec<T>]) -> Vec<usize> {
    // loop through the matrix, binning the data.
    // We will determine the column we are in, by
    // using the modulo operator, on the record value.
    data.data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let col = i % data.cols;
            first_greater_than(&cuts[col], v)
        })
        .collect()
}

pub fn bin_matrix<T: MatrixData<T>>(data: &Matrix<T>, sample_weights: &[T], nbins: usize) {
    // -> BinnedData<T> {
    let mut percentiles = Vec::new();
    let nbins_ = T::from_usize(nbins);
    for i in 0..nbins {
        let v = T::from_usize(i) / nbins_;
        percentiles.push(v);
    }
    // First we need to generate the bins for each of the columns.
    // We will loop through all of the columns, and generate the cuts.
    let mut cuts = Vec::new();
    let mut nunique = Vec::new();
    for i in 0..data.cols {
        let no_miss: Vec<T> = data
            .get_col(i)
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();
        let (col_cuts, col_nunique) = percentiles_nunique(&no_miss, sample_weights, &percentiles);
        cuts.push(col_cuts);
        nunique.push(col_nunique);
    }
}
