use crate::data::MatrixData;
use std::cmp::Ordering;
use std::collections::VecDeque;

/// Naive weighted percentiles calculation.
///
/// Currently this function does not support missing values.
///   
/// * `v` - A Vector of which to find percentiles for.
/// * `sample_weight` - Sample weights for the instances of the vector.
/// * `percentiles` - Percentiles to look for in the data. This should be
///     values from 0 to 1, and in sorted order.
pub fn percentiles<T>(v: &[T], sample_weight: &[T], percentiles: &[T]) -> Vec<T>
where
    T: MatrixData<T>,
{
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());

    // Setup percentiles
    let mut pcts = VecDeque::from_iter(percentiles.iter());
    let mut current_pct = *pcts.pop_front().expect("No percentiles were provided");

    // Prepare a vector to put the percentiles in...
    let mut p = Vec::new();
    let mut cuml_pct = T::ZERO;
    let mut current_value = v[idx[0]];
    let total_values = sample_weight.iter().copied().sum();

    for i in idx.iter() {
        if current_value != v[*i] {
            current_value = v[*i];
        }
        cuml_pct += sample_weight[*i] / total_values;
        if (current_pct == T::ZERO) || (cuml_pct >= current_pct) {
            // We loop here, because the same number might be a valid
            // value to make the percentile several times.
            while cuml_pct >= current_pct {
                p.push(current_value);
                match pcts.pop_front() {
                    Some(p_) => current_pct = *p_,
                    None => return p,
                }
            }
        } else if current_pct == T::ONE {
            if let Some(i_) = idx.last() {
                p.push(v[*i_]);
                break;
            }
        }
    }
    p
}

// Return the index of the first value in a slice that
// is less another number. This will return the first index for
// missing values.
/// Return the index of the first value in a sorted
/// vector that is greater than a provided value.
///
/// * `x` - The sorted slice of values.
/// * `v` - The value used to calculate the first
///   value larger than it.
pub fn map_bin<T: std::cmp::PartialOrd>(x: &[T], v: &T) -> Option<u16> {
    let mut low = 0;
    let mut high = x.len();
    while low != high {
        let mid = (low + high) / 2;
        // This will always be false for NaNs.
        // This it will force us to the bottom,
        // and thus Zero.
        if x[mid] <= *v {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    u16::try_from(low).ok()
}

/// Provided a list of index values, pivot those values
/// around a specific split value so all of the values less
/// than the split value are on one side, and then all of the
/// values greater than or equal to the split value are above.
///
/// * `index` - The index values to sort.
/// * `feature` - The feature vector to use to sort the index by.
/// * `split_value` - the split value to use to pivot on.
/// * `missing_right` - Should missing values go to the left, or
///    to the right of the split value.
pub fn pivot_on_split(
    index: &mut [usize],
    feature: &[u16],
    split_value: u16,
    missing_right: bool,
) -> usize {
    // I think we can do this in O(n) time...
    let mut low = 0;
    let mut high = index.len() - 1;
    let mad_idx = high;
    while low < high {
        // Go until we find a low value that needs to
        // be swapped, this will be the first value
        // that our split value is less or equal to.
        while low < mad_idx {
            let l = feature[index[low]];
            match missing_compare(&split_value, l, missing_right) {
                Ordering::Less | Ordering::Equal => break,
                Ordering::Greater => low += 1,
            }
        }
        while high > 0 {
            let h = feature[index[high]];
            // Go until we find a high value that needs to be
            // swapped, this will be the first value that our
            // split_value is greater than.
            match missing_compare(&split_value, h, missing_right) {
                Ordering::Less | Ordering::Equal => high -= 1,
                Ordering::Greater => break,
            }
        }
        if low < high {
            index.swap(high, low);
        }
    }
    low
}

/// Function to compare a value to our split value.
/// Our split value will _never_ be missing (0), thus we
/// don't have to worry about that.
pub fn missing_compare(split_value: &u16, cmp_value: u16, missing_right: bool) -> Ordering {
    if cmp_value == 0 {
        if missing_right {
            // If missing is right, then our split_value
            // will always be considered less than missing.
            Ordering::Less
        } else {
            // Otherwise less to send it left by considering
            // our split value being always greater than missing
            Ordering::Greater
        }
    } else {
        split_value.cmp(&cmp_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_percentiles() {
        let v = vec![4., 5., 6., 1., 2., 3., 7., 8., 9., 10.];
        let w = vec![1.; v.len()];
        let p = vec![0.3, 0.5, 0.75, 1.0];
        let p = percentiles(&v, &w, &p);
        assert_eq!(p, vec![3.0, 5.0, 8.0, 10.0]);
    }

    #[test]
    fn test_percentiles_weighted() {
        let v = vec![10., 8., 9., 1., 2., 3., 6., 7., 4., 5.];
        let w = vec![1., 1., 1., 1., 1., 2., 1., 1., 5., 1.];
        let p = vec![0.3, 0.5, 0.75, 1.0];
        let p = percentiles(&v, &w, &p);
        assert_eq!(p, vec![4.0, 4.0, 7.0, 10.0]);
    }

    #[test]
    fn test_map_bin_or_equal() {
        let v = vec![f64::MIN, 1., 4., 8., 9.];
        assert_eq!(1, map_bin(&v, &0.).unwrap());
        assert_eq!(2, map_bin(&v, &1.).unwrap());
        // Less than the bin value of 2, means the value is less
        // than 4...
        assert_eq!(2, map_bin(&v, &2.).unwrap());
        assert_eq!(3, map_bin(&v, &4.).unwrap());
        assert_eq!(5, map_bin(&v, &9.).unwrap());
        assert_eq!(5, map_bin(&v, &10.).unwrap());
        assert_eq!(2, map_bin(&v, &1.).unwrap());
        assert_eq!(0, map_bin(&v, &f64::NAN).unwrap());
    }

    #[test]
    fn test_missing_compare() {
        assert_eq!(missing_compare(&10, 0, true), Ordering::Less);
        assert_eq!(missing_compare(&10, 0, false), Ordering::Greater);
        assert_eq!(missing_compare(&10, 11, true), Ordering::Less);
        assert_eq!(missing_compare(&10, 1, true), Ordering::Greater);
    }

    #[test]
    fn test_pivot() {
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(&mut idx, &f, 10, true);
        for i in 0..split_i {
            assert!((f[idx[i]] < 10) && f[idx[i]] != 0);
        }
        for i in idx[split_i..].iter() {
            assert!((f[*i] >= 10) || (f[*i] == 0));
        }
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(&mut idx, &f, 10, false);
        for i in 0..split_i {
            assert!((f[idx[i]] < 10) || (f[idx[i]] == 0));
        }
        for i in idx[split_i..].iter() {
            assert!((f[*i] >= 10) || (f[*i] != 0));
        }
    }
}
