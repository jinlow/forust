use crate::data::MatrixData;
use std::collections::VecDeque;

/// Niave percentiles calculation.
/// Currently this function does not support missing values.
///
/// Params:
/// v - A Vector of which to find percentiles for.
/// sample_weight - Sample weights for the instances of the vector.
/// percentiles - Percentiles to look for in the data. This should be
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
pub fn first_greater_than<T: std::cmp::PartialOrd>(x: &[T], v: &T) -> usize {
    let mut low = 0;
    let mut high = x.len();
    while low != high {
        let mid = (low + high) / 2;
        // This will always be false for NaNs.
        // This it will force us to the bottom,
        // and thus Zero.
        if x[mid] <  *v {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
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
    fn test_first_greater_than_or_equal() {
        let v = vec![f64::MIN, 1., 4., 8., 9.];
        assert_eq!(1, first_greater_than(&v, &0.));
        assert_eq!(1, first_greater_than(&v, &1.));
        // Less than the bin value of 2, means the value is less
        // than 4...
        assert_eq!(2, first_greater_than(&v, &2.));
        assert_eq!(2, first_greater_than(&v, &4.));
        assert_eq!(4, first_greater_than(&v, &9.));
        assert_eq!(5, first_greater_than(&v, &10.));
        assert_eq!(1, first_greater_than(&v, &1.));
        assert_eq!(0, first_greater_than(&v, &f64::NAN));
    }
}
