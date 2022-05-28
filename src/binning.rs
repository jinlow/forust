use crate::data::{Matrix, MatrixData};
use crate::utils::{first_greater_than, percentiles};

// We want to be able to bin our dataset into discrete buckets.
// First we will calculate percentils and the number of unique values
// for each feature.
// Then we will bucket them into bins from 0 to N + 1 where N is the number
// of unique bin values created from the percentiles, and the very last
// bin is missing values.
// For now, we will just use usize, although, it would be good to see if
// we can use something smaller, u8 for instance.

pub struct BinnedData<T> {
    pub binned_data: Vec<usize>,
    pub cuts: Vec<Vec<T>>,
    pub nunique: Vec<usize>,
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
            let col = i / data.rows;
            first_greater_than(&cuts[col], v)
        })
        .collect()
}

pub fn bin_matrix<T: MatrixData<T>>(
    data: &Matrix<T>,
    sample_weight: &[T],
    nbins: usize,
) -> BinnedData<T> {
    let mut pcts = Vec::new();
    let nbins_ = T::from_usize(nbins - 1);
    for i in 0..nbins {
        let v = T::from_usize(i) / nbins_;
        pcts.push(v);
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
        let mut col_cuts = percentiles(&no_miss, sample_weight, &pcts);
        col_cuts.insert(0, T::MIN);
        col_cuts.dedup();
        // There will be one less bins, then there are cuts.
        // The first value will be for missing.
        nunique.push(col_cuts.len().clone());
        cuts.push(col_cuts);
    }

    let binned_data = bin_matrix_from_cuts(data, &cuts);

    BinnedData {
        binned_data,
        cuts,
        nunique,
    }
}

mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn test_tree_fit() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let sample_weight = vec![1.; data.rows];
        let b = bin_matrix(&data, &sample_weight, 101);
        // println!("{:?}", b.binned_data);
    }
}
