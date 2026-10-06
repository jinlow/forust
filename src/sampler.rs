use crate::data::Matrix;
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Copy the sampled rows into a contiguous buffer when at most this share of
/// rows is sampled, as LightGBM does for GOSS and bagging.
pub const SUBSET_MAX_FRACTION: f64 = 0.5;
/// Below this many sampled rows, gathering gradients serially is cheaper.
const PARALLEL_GATHER_MIN_ROWS: usize = 16_384;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleMethod {
    None,
    Random,
    Goss,
}

// A sampler can be used to subset the data prior to fitting a new tree.
pub trait Sampler {
    /// Sample the data, returning a tuple, where the first item is the samples
    /// chosen for training, and the second are the samples excluded.
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>);
}

pub struct RandomSampler {
    subsample: f32,
}

impl RandomSampler {
    #[allow(dead_code)]
    pub fn new(subsample: f32) -> Self {
        RandomSampler { subsample }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        _grad: &mut [f32],
        _hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>) {
        let subsample = self.subsample;
        let mut chosen = Vec::new();
        let mut excluded = Vec::new();
        for i in index {
            if rng.gen_range(0.0..1.0) < subsample {
                chosen.push(*i);
            } else {
                excluded.push(*i)
            }
        }
        (chosen, excluded)
    }
}

/// Gradient-based One-Side Sampling (GOSS), following LightGBM.
///
/// Keeps the `top_rate` share of rows with the largest `|gradient * hessian|`,
/// then samples exactly `other_rate * n` of the remaining rows, scaling their
/// gradients and hessians by `(n - top_k) / other_k` so the split gains stay
/// close to unbiased.
/// See <https://lightgbm.readthedocs.io/en/latest/Parameters.html#top_rate>.
pub struct GossSampler {
    top_rate: f64,
    other_rate: f64,
}

impl Default for GossSampler {
    fn default() -> Self {
        GossSampler {
            top_rate: 0.2,
            other_rate: 0.1,
        }
    }
}

impl GossSampler {
    pub fn new(top_rate: f64, other_rate: f64) -> Self {
        GossSampler {
            top_rate,
            other_rate,
        }
    }

    /// Number of initial iterations trained on all rows before GOSS starts.
    /// As in LightGBM, early gradients carry little information about which
    /// rows matter, so the first `1 / learning_rate` trees use every row.
    pub fn warmup_iterations(learning_rate: f32) -> usize {
        (1.0f32 / learning_rate) as usize
    }
}

impl Sampler for GossSampler {
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>) {
        let n = index.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let top_k = ((n as f64 * self.top_rate) as usize).clamp(1, n);
        let other_k = (n as f64 * self.other_rate) as usize;

        let scores: Vec<f32> = index.iter().map(|&i| (grad[i] * hess[i]).abs()).collect();
        // The top_k-th largest score; selection is O(n), unlike a full sort.
        let threshold = {
            let mut buffer = scores.clone();
            *buffer
                .select_nth_unstable_by(top_k - 1, |a, b| b.total_cmp(a))
                .1
        };
        let multiply = (n - top_k) as f32 / other_k as f32;

        let mut chosen = Vec::with_capacity(top_k + other_k);
        let mut excluded = Vec::with_capacity(n.saturating_sub(top_k + other_k));
        let mut big_count: i64 = 0;
        for (position, (&i, &score)) in index.iter().zip(&scores).enumerate() {
            if score >= threshold {
                chosen.push(i);
                big_count += 1;
                continue;
            }
            // Sampling with probability rest_need / rest_all draws exactly other_k rows.
            let sampled = chosen.len() as i64 - big_count;
            let rest_need = other_k as i64 - sampled;
            let rest_all = (n - position) as i64 - (top_k as i64 - big_count);
            let probability = rest_need as f64 / rest_all as f64;
            if rng.gen::<f64>() < probability {
                grad[i] *= multiply;
                hess[i] *= multiply;
                chosen.push(i);
            } else {
                excluded.push(i);
            }
        }
        (chosen, excluded)
    }
}

/// The sampled rows of the binned data, gradients and hessians, copied into
/// contiguous buffers so trees can be built with sequential memory access.
///
/// Trees fit on the subset (with row index `0..rows`) are identical to trees fit
/// on the full data with the sampled index, because trees only store bin-based
/// split values. Buffers are reused across iterations.
#[derive(Default)]
pub struct RowSubset {
    pub binned: Vec<u16>,
    pub grad: Vec<f32>,
    pub hess: Vec<f32>,
    pub rows: usize,
    cols: usize,
}

impl RowSubset {
    /// Should `sampled` of `total` rows be copied into a subset?
    pub fn should_use(sampled: usize, total: usize) -> bool {
        sampled > 0 && (sampled as f64) <= SUBSET_MAX_FRACTION * (total as f64)
    }

    /// Copy `rows` of the columns in `col_index` from `data`, and their gradients and
    /// hessians. Columns not in `col_index` are left unfilled and must not be read.
    pub fn fill(
        &mut self,
        data: &Matrix<u16>,
        rows: &[usize],
        col_index: &[usize],
        grad: &[f32],
        hess: &[f32],
        parallel: bool,
    ) {
        let m = rows.len();
        self.rows = m;
        self.cols = data.cols;
        if m == 0 {
            self.binned.clear();
            self.grad.clear();
            self.hess.clear();
            return;
        }
        let mut used = vec![false; data.cols];
        col_index.iter().for_each(|&c| used[c] = true);
        self.binned.resize(data.cols * m, 0);
        let fill_col = |(c, out): (usize, &mut [u16])| {
            if used[c] {
                let col = data.get_col(c);
                out.iter_mut().zip(rows).for_each(|(o, &r)| *o = col[r]);
            }
        };
        if parallel {
            self.binned.par_chunks_mut(m).enumerate().for_each(fill_col);
        } else {
            self.binned.chunks_mut(m).enumerate().for_each(fill_col);
        }
        if parallel && m >= PARALLEL_GATHER_MIN_ROWS {
            rows.par_iter()
                .map(|&r| grad[r])
                .collect_into_vec(&mut self.grad);
            rows.par_iter()
                .map(|&r| hess[r])
                .collect_into_vec(&mut self.hess);
        } else {
            self.grad.clear();
            self.grad.extend(rows.iter().map(|&r| grad[r]));
            self.hess.clear();
            self.hess.extend(rows.iter().map(|&r| hess[r]));
        }
    }

    /// The copied binned data, and the row index `0..rows` to fit a tree with.
    pub fn matrix(&self) -> (Matrix<'_, u16>, Vec<usize>) {
        let mut matrix = Matrix::new(&self.binned, self.rows, self.cols);
        let index = std::mem::take(&mut matrix.index);
        (matrix, index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // Distinct |g * h| scores: row i scores i + 1 (rows are shuffled by a stride).
    fn gradients(n: usize) -> (Vec<f32>, Vec<f32>) {
        let grad = (0..n).map(|i| ((i * 37) % n + 1) as f32).collect();
        let hess = vec![1.0; n];
        (grad, hess)
    }

    #[test]
    fn test_goss_sample_counts_and_weights() {
        let n = 1000;
        let (mut grad, mut hess) = gradients(n);
        let (orig_grad, orig_hess) = (grad.clone(), hess.clone());
        let index: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(0);
        let (chosen, excluded) =
            GossSampler::new(0.2, 0.1).sample(&mut rng, &index, &mut grad, &mut hess);

        assert_eq!(chosen.len(), 200 + 100);
        assert_eq!(excluded.len(), n - chosen.len());
        assert!(chosen.windows(2).all(|w| w[0] < w[1]));

        let multiply = (n - 200) as f32 / 100.0;
        let threshold = (n - 200 + 1) as f32;
        let mut top = 0;
        for &i in &chosen {
            if orig_grad[i] >= threshold {
                top += 1;
                assert_eq!(grad[i], orig_grad[i]);
                assert_eq!(hess[i], orig_hess[i]);
            } else {
                assert_eq!(grad[i], orig_grad[i] * multiply);
                assert_eq!(hess[i], orig_hess[i] * multiply);
            }
        }
        assert_eq!(top, 200);
        for &i in &excluded {
            assert!(orig_grad[i] < threshold);
            assert_eq!(grad[i], orig_grad[i]);
            assert_eq!(hess[i], orig_hess[i]);
        }
    }

    #[test]
    fn test_goss_ranks_by_gradient_times_hessian() {
        // Row 0 has the largest gradient but a tiny hessian; row 1 has the largest product.
        let mut grad = vec![10.0, 2.0, 1.0, 1.0];
        let mut hess = vec![0.01, 1.0, 0.5, 0.5];
        let index = vec![0, 1, 2, 3];
        let mut rng = StdRng::seed_from_u64(0);
        let (chosen, _) =
            GossSampler::new(0.25, 0.25).sample(&mut rng, &index, &mut grad, &mut hess);
        assert!(chosen.contains(&1));
        assert_eq!(grad[1], 2.0);
    }

    #[test]
    fn test_goss_returns_values_from_index() {
        let n = 100;
        let (mut grad, mut hess) = gradients(n);
        let index: Vec<usize> = (50..n).collect();
        let mut rng = StdRng::seed_from_u64(1);
        let (chosen, excluded) =
            GossSampler::new(0.2, 0.2).sample(&mut rng, &index, &mut grad, &mut hess);
        assert_eq!(chosen.len(), 10 + 10);
        assert!(chosen.iter().chain(&excluded).all(|i| (50..n).contains(i)));
        assert_eq!(chosen.len() + excluded.len(), index.len());
    }

    #[test]
    fn test_goss_is_deterministic_for_a_seed() {
        let n = 500;
        let index: Vec<usize> = (0..n).collect();
        let run = |seed| {
            let (mut grad, mut hess) = gradients(n);
            let mut rng = StdRng::seed_from_u64(seed);
            GossSampler::default()
                .sample(&mut rng, &index, &mut grad, &mut hess)
                .0
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    #[test]
    fn test_row_subset() {
        assert!(RowSubset::should_use(50, 100));
        assert!(!RowSubset::should_use(51, 100));
        assert!(!RowSubset::should_use(0, 100));

        // 4 rows x 3 columns, column major.
        let values: Vec<u16> = vec![0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let data = Matrix::new(&values, 4, 3);
        let grad = vec![0.0, 0.1, 0.2, 0.3];
        let hess = vec![1.0, 1.1, 1.2, 1.3];
        let mut subset = RowSubset::default();
        for parallel in [false, true] {
            subset.fill(&data, &[1, 3], &[0, 2], &grad, &hess, parallel);
            let (matrix, index) = subset.matrix();
            assert_eq!(index, vec![0, 1]);
            assert_eq!(matrix.get_col(0), &[1, 3]);
            assert_eq!(matrix.get_col(2), &[21, 23]);
            assert_eq!(subset.grad, vec![0.1, 0.3]);
            assert_eq!(subset.hess, vec![1.1, 1.3]);
        }
    }

    #[test]
    fn test_goss_warmup_iterations() {
        assert_eq!(GossSampler::warmup_iterations(0.1), 10);
        assert_eq!(GossSampler::warmup_iterations(0.3), 3);
        assert_eq!(GossSampler::warmup_iterations(1.0), 1);
    }
}
