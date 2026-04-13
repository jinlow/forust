//! End-to-end smoke test for QuantileLoss: fit + predict on synthetic data
//! with heteroscedastic noise. Confirms alpha=0.70 predictions > alpha=0.30
//! predictions on average (by construction of the pinball loss).

use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};

fn synthetic_dataset(n: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut data = Vec::with_capacity(n * cols);
    let mut y = Vec::with_capacity(n);
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;
    let mut next_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };
    let mut rows_row_major = Vec::with_capacity(n * cols);
    for i in 0..n {
        let signal = (i as f64) / (n as f64);
        for j in 0..cols {
            rows_row_major.push(signal + (j as f64) * 0.01);
        }
        let noise = next_f64() - 0.5;
        y.push(signal * 10.0 + 2.0 * noise);
    }
    // Row-major → column-major (Matrix expects column-major).
    data.resize(n * cols, 0.0);
    for row in 0..n {
        for col in 0..cols {
            data[col * n + row] = rows_row_major[row * cols + col];
        }
    }
    (data, y)
}

fn train_quantile(
    data_vec: &[f64],
    y: &[f64],
    n: usize,
    cols: usize,
    alpha: f64,
) -> GradientBooster {
    let data = Matrix::new(data_vec, n, cols);
    let w = vec![1.0_f64; n];
    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::QuantileLoss { alpha })
        .set_iterations(60)
        .set_max_depth(4)
        .set_learning_rate(0.08);
    booster.initialize_base_score = true;
    booster.fit(&data, y, &w, None).unwrap();
    booster
}

#[test]
fn quantile_alpha_70_predicts_above_alpha_30_on_average() {
    let n = 400;
    let cols = 4;
    let (data_vec, y) = synthetic_dataset(n, cols);
    let data = Matrix::new(&data_vec, n, cols);

    let q30 = train_quantile(&data_vec, &y, n, cols, 0.30);
    let q70 = train_quantile(&data_vec, &y, n, cols, 0.70);

    let p30 = q30.predict(&data, true);
    let p70 = q70.predict(&data, true);

    let mean_30: f64 = p30.iter().sum::<f64>() / p30.len() as f64;
    let mean_70: f64 = p70.iter().sum::<f64>() / p70.len() as f64;
    assert!(
        mean_70 > mean_30,
        "expected q70 mean ({mean_70}) > q30 mean ({mean_30})"
    );

    // Empirical coverage: fraction of y above the prediction.
    let above = |preds: &[f64]| -> f64 {
        let hits = y
            .iter()
            .zip(preds.iter())
            .filter(|(y_, p_)| **y_ > **p_)
            .count();
        hits as f64 / y.len() as f64
    };
    let cov30 = above(&p30);
    let cov70 = above(&p70);
    // alpha=0.30 → ~70% of y should exceed the prediction; alpha=0.70 → ~30%.
    assert!(cov30 > cov70, "expected cov30 ({cov30}) > cov70 ({cov70})");
    assert!(cov30 > 0.55 && cov30 < 0.85, "cov30 = {cov30}");
    assert!(cov70 > 0.15 && cov70 < 0.45, "cov70 = {cov70}");
}
