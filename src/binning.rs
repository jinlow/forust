use crate::data::{FloatData, JaggedMatrix, Matrix};
use crate::errors::ForustError;
use crate::utils::{fast_sum, is_missing, map_bin, percentiles_of_sorted};
use rayon::prelude::*;

/// The distinct values of an ascending sequence, or `None` if there are more than `max`.
fn distinct_up_to<T: FloatData<T>>(sorted: impl Iterator<Item = T>, max: usize) -> Option<Vec<T>> {
    let mut unique: Vec<T> = Vec::new();
    for x in sorted {
        if unique.last() != Some(&x) {
            if unique.len() == max {
                return None;
            }
            unique.push(x);
        }
    }
    Some(unique)
}

/// If there are fewer unique values than their are
/// percentiles, just return the unique values of the
/// vectors.
///
/// * `v` - A numeric slice to calculate percentiles for.
/// * `sample_weight` - Instance weights for each row in the data.
fn percentiles_or_value<T>(v: &[T], sample_weight: &[T], pcts: &[T]) -> Vec<T>
where
    T: FloatData<T>,
{
    let max_unique = pcts.len() + 1;
    let total_weight = fast_sum(sample_weight);
    // With equal weights, the order of tied values can't change the percentile walk,
    // so sorting the values directly gives the same cuts as sorting an index.
    if sample_weight.windows(2).all(|w| w[0] == w[1]) {
        let mut sorted = v.to_owned();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        distinct_up_to(sorted.iter().copied(), max_unique).unwrap_or_else(|| {
            percentiles_of_sorted(
                sorted.len(),
                |k| sorted[k],
                |_| sample_weight[0],
                total_weight,
                pcts,
            )
        })
    } else {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
        distinct_up_to(idx.iter().map(|i| v[*i]), max_unique).unwrap_or_else(|| {
            percentiles_of_sorted(
                idx.len(),
                |k| v[idx[k]],
                |k| sample_weight[idx[k]],
                total_weight,
                pcts,
            )
        })
    }
}

// We want to be able to bin our dataset into discrete buckets.
// First we will calculate percentiles and the number of unique values
// for each feature.
// Then we will bucket them into bins from 0 to N + 1 where N is the number
// of unique bin values created from the percentiles, and the very last
// bin is missing values.
// For now, we will just use usize, although, it would be good to see if
// we can use something smaller, u8 for instance.
// If we generated these cuts:
// [0.0, 7.8958, 14.4542, 31.0, 512.3292, inf]
// We would have a number with bins 0 (missing), 1 [MIN, 0.0), 2 (0.0, 7], 3 [], 4, 5
// a split that is [feature < 5] would translate to [feature < 31.0 ]
pub struct BinnedData<T> {
    pub binned_data: Vec<u16>,
    pub cuts: JaggedMatrix<T>,
    pub nunique: Vec<usize>,
}

/// The bin of missing values in `bin_for_prediction`.
pub const PREDICT_MISSING_BIN: u16 = u16::MAX;
/// The bin of NAN values in `bin_for_prediction`, when the missing value isn't NAN.
pub const PREDICT_NAN_BIN: u16 = u16::MAX - 1;

/// Can trees be predicted from bins made with these cuts? A value's bin can be as
/// large as its column's number of cuts, which must stay below the reserved bins.
pub fn cuts_support_bin_prediction(cuts: &JaggedMatrix<f64>) -> bool {
    (0..cuts.cols).all(|c| cuts.get_col(c).len() < usize::from(PREDICT_NAN_BIN))
}

/// Bin data with existing cuts, column-major like the data, for predicting trees from
/// bins. Each value gets the number of cuts at or below it, so a value is below a
/// cut exactly when its bin is below the cut's split bin, including values below the
/// smallest cut (bin 0). Missing values get `PREDICT_MISSING_BIN`, and NAN values get
/// `PREDICT_NAN_BIN` when `missing` isn't NAN.
pub fn bin_for_prediction(
    data: &Matrix<f64>,
    cuts: &JaggedMatrix<f64>,
    missing: &f64,
    parallel: bool,
) -> Vec<u16> {
    let mut binned = vec![0; data.rows * data.cols];
    if data.rows == 0 {
        return binned;
    }
    let bin_column = |(col, out): (usize, &mut [u16])| {
        let col_cuts = cuts.get_col(col);
        for (b, v) in out.iter_mut().zip(data.get_col(col)) {
            *b = if v.is_nan() && !missing.is_nan() {
                PREDICT_NAN_BIN
            } else if is_missing(v, missing) {
                PREDICT_MISSING_BIN
            } else {
                col_cuts.partition_point(|c| c <= v) as u16
            };
        }
    };
    if parallel {
        binned
            .par_chunks_mut(data.rows)
            .enumerate()
            .for_each(bin_column);
    } else {
        binned
            .chunks_mut(data.rows)
            .enumerate()
            .for_each(bin_column);
    }
    binned
}

/// Convert a matrix of data, into a binned matrix.
///
/// * `data` - Numeric data to be binned.
/// * `cuts` - A slice of Vectors, where the vectors are the corresponding
///     cut values for each of the columns.
fn bin_matrix_from_cuts<T: FloatData<T>>(
    data: &Matrix<T>,
    cuts: &JaggedMatrix<T>,
    missing: &T,
    parallel: bool,
) -> Vec<u16> {
    let mut binned = vec![0; data.data.len()];
    if data.rows == 0 {
        return binned;
    }
    let bin_column = |(col, out): (usize, &mut [u16])| {
        for (b, v) in out.iter_mut().zip(data.get_col(col)) {
            // This will always be smaller than u16::MAX so we
            // are good to just unwrap here.
            *b = map_bin(cuts.get_col(col), v, missing).unwrap();
        }
    };
    if parallel {
        binned
            .par_chunks_mut(data.rows)
            .enumerate()
            .for_each(bin_column);
    } else {
        binned
            .chunks_mut(data.rows)
            .enumerate()
            .for_each(bin_column);
    }
    binned
}

/// Bin a numeric matrix.
///
/// * `data` - A numeric matrix, of data to be binned.
/// * `sample_weight` - Instance weights for each row of the data.
/// * `nbins` - The number of bins each column should be binned into.
/// * `missing` - Float value to consider as missing.
/// * `parallel` - Bin the columns in parallel.
pub fn bin_matrix(
    data: &Matrix<f64>,
    sample_weight: &[f64],
    nbins: u16,
    missing: f64,
    parallel: bool,
) -> Result<BinnedData<f64>, ForustError> {
    let mut pcts = Vec::new();
    let nbins_ = f64::from_u16(nbins);
    for i in 0..nbins {
        let v = f64::from_u16(i) / nbins_;
        pcts.push(v);
    }

    // First we need to generate the bins for each of the columns.
    let column_cuts = |i: usize| {
        let (no_miss, w): (Vec<f64>, Vec<f64>) = data
            .get_col(i)
            .iter()
            .zip(sample_weight.iter())
            // It is unrecoverable if they have provided missing values in
            // the data other than the specificized missing.
            .filter(|(v, _)| !is_missing(v, &missing))
            .unzip();
        assert_eq!(no_miss.len(), w.len());
        let mut col_cuts = percentiles_or_value(&no_miss, &w, &pcts);
        col_cuts.push(f64::MAX);
        col_cuts.dedup();
        col_cuts
    };
    let all_cuts: Vec<Vec<f64>> = if parallel {
        (0..data.cols).into_par_iter().map(column_cuts).collect()
    } else {
        (0..data.cols).map(column_cuts).collect()
    };

    let mut cuts = JaggedMatrix::new();
    let mut nunique = Vec::new();
    for col_cuts in all_cuts {
        // if col_cuts.len() < 2 {
        //     return Err(ForustError::NoVariance(i));
        // }
        // There will be one less bins, then there are cuts.
        // The first value will be for missing.
        nunique.push(col_cuts.len());
        let l = col_cuts.len();
        cuts.data.extend(col_cuts);
        let e = match cuts.ends.last() {
            Some(v) => v + l,
            None => l,
        };
        cuts.ends.push(e);
        cuts.cols = cuts.ends.len();
        cuts.n_records = cuts.ends.iter().sum();
    }

    let binned_data = bin_matrix_from_cuts(data, &cuts, &missing, parallel);

    Ok(BinnedData {
        binned_data,
        cuts,
        nunique,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuts_support_bin_prediction() {
        let cuts_of_len = |len: usize| {
            let mut cuts = JaggedMatrix::new();
            cuts.data = (0..len + 3).map(|v| v as f64).collect();
            cuts.ends = vec![3, len + 3];
            cuts.cols = 2;
            cuts.n_records = len + 3;
            cuts
        };
        assert!(cuts_support_bin_prediction(&cuts_of_len(257)));
        assert!(cuts_support_bin_prediction(&cuts_of_len(65533)));
        assert!(!cuts_support_bin_prediction(&cuts_of_len(65534)));
        assert!(!cuts_support_bin_prediction(&cuts_of_len(65536)));
    }
    use crate::utils::percentiles;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::fs;

    #[test]
    fn test_percentiles_or_value_matches_original_algorithm() {
        let mut rng = StdRng::seed_from_u64(0);
        let pcts: Vec<f64> = (0..64).map(|i| i as f64 / 64.0).collect();
        // (rows, distinct levels or 0 for continuous, weighted)
        for (n, levels, weighted) in [
            (2000, 10, false),
            (2000, 10, true),
            (5000, 1000, false),
            (5000, 1000, true),
            (5000, 0, false),
            (5000, 0, true),
            (1, 0, false),
        ] {
            let v: Vec<f64> = (0..n)
                .map(|_| {
                    let x: f64 = rng.gen();
                    if levels == 0 {
                        x
                    } else {
                        (x * levels as f64).floor()
                    }
                })
                .collect();
            let w: Vec<f64> = (0..n)
                .map(|_| {
                    if weighted {
                        rng.gen_range(0.5..2.0)
                    } else {
                        1.0
                    }
                })
                .collect();
            let mut expected = v.clone();
            expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            expected.dedup();
            if expected.len() > pcts.len() + 1 {
                expected = percentiles(&v, &w, &pcts);
            }
            assert_eq!(percentiles_or_value(&v, &w, &pcts), expected);
        }
    }

    #[test]
    fn test_bin_matrix_parallel_matches_serial() {
        let (rows, cols) = (3000, 12);
        let mut rng = StdRng::seed_from_u64(1);
        let data_vec: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let x: f64 = rng.gen();
                if i % 7 == 0 {
                    f64::NAN
                } else if (i / rows) % 3 == 0 {
                    (x * 5.0).floor()
                } else {
                    x
                }
            })
            .collect();
        let data = Matrix::new(&data_vec, rows, cols);
        let weighted: Vec<f64> = (0..rows).map(|_| rng.gen_range(0.5..2.0)).collect();
        for w in [vec![1.; rows], weighted] {
            let serial = bin_matrix(&data, &w, 64, f64::NAN, false).unwrap();
            let parallel = bin_matrix(&data, &w, 64, f64::NAN, true).unwrap();
            assert_eq!(serial.binned_data, parallel.binned_data);
            assert_eq!(serial.cuts.data, parallel.cuts.data);
            assert_eq!(serial.cuts.ends, parallel.cuts.ends);
            assert_eq!(serial.nunique, parallel.nunique);
        }
    }

    #[test]
    fn test_bin_data() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let sample_weight = vec![1.; data.rows];
        let b = bin_matrix(&data, &sample_weight, 50, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        for column in 0..data.cols {
            let mut b_compare = 1;
            for cuts in b.cuts.get_col(column).windows(2) {
                let c1 = cuts[0];
                let c2 = cuts[1];
                let mut n_v = 0;
                let mut n_b = 0;
                for (bin, value) in bdata.get_col(column).iter().zip(data.get_col(column)) {
                    if *bin == b_compare {
                        n_b += 1;
                    }
                    if (c1 <= *value) && (*value < c2) {
                        n_v += 1;
                    }
                }
                assert_eq!(n_v, n_b);
                b_compare += 1;
            }
        }
    }
}
