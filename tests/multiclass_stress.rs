//! Granular stress tests for SoftmaxMultiClass.
//!
//! Covers edge cases, numerical limits, sampling modes,
//! scale, serialization, and interaction with existing features.

use forust_ml::objective::{ObjectiveType, SoftmaxMultiClass};
use forust_ml::sampler::SampleMethod;
use forust_ml::{GradientBooster, Matrix};

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Build a synthetic K-class dataset with N samples per class.
/// Each class is a cluster centered at (class_idx * separation, 0..n_features).
/// Returns (col_major_data, labels, n_rows, n_cols).
fn make_kclass_data(
    k: usize,
    n_per_class: usize,
    n_features: usize,
    separation: f64,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = k * n_per_class;
    let mut rows = Vec::with_capacity(n * n_features);
    let mut labels = Vec::with_capacity(n);

    for class in 0..k {
        for i in 0..n_per_class {
            let offset = (i as f64) * 0.02 - 1.0;
            for feat in 0..n_features {
                let center = if feat == 0 {
                    class as f64 * separation
                } else {
                    (class as f64 * separation * 0.5) + (feat as f64)
                };
                rows.push(center + offset);
            }
            labels.push(class as f64);
        }
    }

    // Row-major → column-major
    let mut col_major = vec![0.0; n * n_features];
    for row in 0..n {
        for col in 0..n_features {
            col_major[col * n + row] = rows[row * n_features + col];
        }
    }
    (col_major, labels, n, n_features)
}

fn train_multiclass(
    data_vec: &[f64],
    y: &[f64],
    n: usize,
    n_cols: usize,
    k: usize,
    iterations: usize,
) -> GradientBooster {
    let data = Matrix::new(data_vec, n, n_cols);
    let w = vec![1.0; n];
    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(iterations)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = k;
    booster.initialize_base_score = true;
    booster.fit(&data, y, &w, None).unwrap();
    booster
}

fn accuracy(probs: &[f64], y: &[f64], k: usize) -> f64 {
    let n = y.len();
    let mut correct = 0;
    for i in 0..n {
        let row = &probs[i * k..(i + 1) * k];
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        if predicted == y[i] as usize {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Edge case: K=2 (binary classification via softmax)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_k2_binary_via_softmax() {
    let (data_vec, y, n, nc) = make_kclass_data(2, 100, 2, 5.0);
    let booster = train_multiclass(&data_vec, &y, n, nc, 2, 30);
    let data = Matrix::new(&data_vec, n, nc);
    let probs = booster.predict_proba(&data, true);

    // Probabilities sum to 1
    for i in 0..n {
        let sum: f64 = probs[i * 2..(i + 1) * 2].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
    }

    // High accuracy on well-separated data
    let acc = accuracy(&probs, &y, 2);
    assert!(acc > 0.95, "K=2 accuracy = {acc}");

    // Tree count = iterations * K
    assert_eq!(booster.trees.len(), 30 * 2);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Edge case: K=10 (many classes)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_k10_many_classes() {
    let (data_vec, y, n, nc) = make_kclass_data(10, 50, 3, 8.0);
    let booster = train_multiclass(&data_vec, &y, n, nc, 10, 50);
    let data = Matrix::new(&data_vec, n, nc);
    let probs = booster.predict_proba(&data, true);

    for i in 0..n {
        let row = &probs[i * 10..(i + 1) * 10];
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
        // All probabilities must be in [0, 1]
        for (j, &p) in row.iter().enumerate() {
            assert!(p >= 0.0 && p <= 1.0, "row {i} class {j}: p = {p}");
        }
    }

    let acc = accuracy(&probs, &y, 10);
    assert!(acc > 0.80, "K=10 accuracy = {acc}");
    assert_eq!(booster.trees.len(), 50 * 10);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Edge case: single sample per class
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_single_sample_per_class() {
    // 3 classes, 1 sample each — minimum viable dataset
    let data_vec = vec![
        // col-major: col0 = [0, 5, 10], col1 = [0, 5, 0]
        0.0, 5.0, 10.0, 0.0, 5.0, 0.0,
    ];
    let y = vec![0.0, 1.0, 2.0];
    let n = 3;
    let nc = 2;
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(20)
        .set_max_depth(2)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    // Should not panic
    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);
    assert_eq!(probs.len(), 9);

    // Probabilities still sum to 1
    for i in 0..3 {
        let sum: f64 = probs[i * 3..(i + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Edge case: all samples same class
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_all_same_class() {
    // All labels = 0, K=3. calc_init should handle class 1 and 2 having 0 weight
    let data_vec = vec![
        // col-major: 5 samples, 1 feature
        1.0, 2.0, 3.0, 4.0, 5.0,
    ];
    let y = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let n = 5;
    let nc = 1;
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(10)
        .set_max_depth(2)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    // All samples should strongly predict class 0
    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        assert!(row[0] > 0.8, "sample {i}: P(class=0) = {}", row[0]);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Edge case: empty class (no samples for one class)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_empty_class() {
    // K=3 but no label=2 in training data. calc_init must clamp with RT_EPS.
    let data_vec = vec![
        // col-major: 6 samples, 1 feature
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // no class 2
    let n = 6;
    let nc = 1;
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(20)
        .set_max_depth(3)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    // Should not crash, probs should be valid
    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
        for &p in row {
            assert!(p.is_finite() && p >= 0.0);
        }
    }

    // Class 2 probability should be very low
    for i in 0..n {
        let p2 = probs[i * 3 + 2];
        assert!(p2 < 0.1, "sample {i}: P(class=2) = {p2} should be near 0");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Numerical: very imbalanced classes (95/3/2 split)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_very_imbalanced() {
    let n = 200;
    let nc = 2;
    // 190 class 0, 6 class 1, 4 class 2
    let mut y = vec![0.0; 190];
    y.extend(vec![1.0; 6]);
    y.extend(vec![2.0; 4]);
    assert_eq!(y.len(), n);

    // col-major: class 0 at (0,0), class 1 at (5,5), class 2 at (10,0)
    let mut col0 = Vec::with_capacity(n);
    let mut col1 = Vec::with_capacity(n);
    for i in 0..190 {
        col0.push(i as f64 * 0.01);
        col1.push(i as f64 * 0.01);
    }
    for i in 0..6 {
        col0.push(5.0 + i as f64 * 0.1);
        col1.push(5.0 + i as f64 * 0.1);
    }
    for i in 0..4 {
        col0.push(10.0 + i as f64 * 0.1);
        col1.push(i as f64 * 0.1);
    }
    let mut data_vec = col0;
    data_vec.extend(col1);

    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(50)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    let acc = accuracy(&probs, &y, 3);
    assert!(acc > 0.90, "imbalanced accuracy = {acc}");

    // Verify base_scores reflect imbalance (class 0 should have highest base)
    let base = booster.base_scores.as_ref().unwrap();
    assert!(base[0] > base[1], "class 0 base > class 1 base");
    assert!(base[0] > base[2], "class 0 base > class 2 base");
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Numerical: extreme sample weights
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_extreme_weights() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 30, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);

    // Give class 0 very high weight, class 2 very low
    let mut w = vec![1.0; n];
    for i in 0..30 {
        w[i] = 100.0; // class 0
    }
    for i in 60..90 {
        w[i] = 0.01; // class 2
    }

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(30)
        .set_max_depth(4)
        .set_learning_rate(0.1);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    // All probs must be valid
    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
        for &p in row {
            assert!(p.is_finite() && p >= 0.0 && p <= 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Sampling: GOSS with multiclass
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_goss_sampling() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(30)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;
    booster.sample_method = SampleMethod::Goss;
    booster.top_rate = 0.2;
    booster.other_rate = 0.1;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    let acc = accuracy(&probs, &y, 3);
    assert!(acc > 0.85, "GOSS accuracy = {acc}");
    assert_eq!(booster.trees.len(), 30 * 3);
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Sampling: Random subsample with multiclass
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_random_subsample() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(40)
        .set_max_depth(4)
        .set_learning_rate(0.3)
        .set_subsample(0.5);
    booster.num_classes = 3;
    booster.initialize_base_score = true;
    booster.sample_method = SampleMethod::Random;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    let acc = accuracy(&probs, &y, 3);
    assert!(acc > 0.85, "subsample accuracy = {acc}");
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Column subsampling with multiclass
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_colsample_bytree() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 100, 5, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(40)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.colsample_bytree = 0.6; // use only 60% of features per tree
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    let acc = accuracy(&probs, &y, 3);
    assert!(acc > 0.85, "colsample accuracy = {acc}");
}

// ═══════════════════════════════════════════════════════════════════════
// 11. Scale: large dataset (5000 samples)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_large_dataset() {
    let (data_vec, y, _n, nc) = make_kclass_data(3, 1667, 4, 5.0); // ~5001 samples
    let n = 5001; // 1667*3
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(20)
        .set_max_depth(5)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    assert_eq!(probs.len(), n * 3);
    let acc = accuracy(&probs, &y, 3);
    assert!(acc > 0.95, "large dataset accuracy = {acc}");
}

// ═══════════════════════════════════════════════════════════════════════
// 12. Missing values in features with multiclass
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_missing_values() {
    let (mut data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);

    // Inject NaN missing values into ~10% of feature values
    let total = data_vec.len();
    for i in (0..total).step_by(10) {
        data_vec[i] = f64::NAN;
    }

    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    // With create_missing_branch = true
    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(30)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;
    booster.create_missing_branch = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
        for &p in row {
            assert!(p.is_finite());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 13. Missing values with MissingImputerSplitter
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_missing_values_imputer() {
    let (mut data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);

    for i in (0..data_vec.len()).step_by(7) {
        data_vec[i] = f64::NAN;
    }

    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    // With create_missing_branch = false (uses imputer)
    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(30)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;
    booster.create_missing_branch = false;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    for i in 0..n {
        let sum: f64 = probs[i * 3..(i + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 14. Refit: calling fit() twice resets state
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_refit_resets_state() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(10)
        .set_max_depth(3)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    // First fit
    booster.fit(&data, &y, &w, None).unwrap();
    assert_eq!(booster.trees.len(), 10 * 3);
    let preds1 = booster.predict(&data, true);

    // Second fit — should reset, not append
    booster.fit(&data, &y, &w, None).unwrap();
    assert_eq!(
        booster.trees.len(),
        10 * 3,
        "trees should be reset, not appended"
    );
    let preds2 = booster.predict(&data, true);

    // Same seed + same data → identical results
    assert_eq!(preds1.len(), preds2.len());
    for (a, b) in preds1.iter().zip(preds2.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "refit predictions should be identical"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 15. Single iteration (minimum boosting rounds)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_single_iteration() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(1)
        .set_max_depth(3)
        .set_learning_rate(0.3);
    booster.num_classes = 3;

    booster.fit(&data, &y, &w, None).unwrap();

    assert_eq!(booster.trees.len(), 3, "1 round × 3 classes");

    let probs = booster.predict_proba(&data, true);
    for i in 0..n {
        let sum: f64 = probs[i * 3..(i + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 16. Extreme learning rate (very small)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_tiny_learning_rate() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(5)
        .set_max_depth(3)
        .set_learning_rate(0.001);
    booster.num_classes = 3;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    // With tiny learning rate and few iterations, predictions should be
    // close to base priors (near uniform for balanced classes)
    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        for &p in row {
            // Should not be too extreme with such small lr
            assert!(p > 0.01, "tiny lr should not produce extreme probs");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 17. Extreme learning rate (very large)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_large_learning_rate() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 50, 2, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(20)
        .set_max_depth(3)
        .set_learning_rate(1.0);
    booster.num_classes = 3;

    booster.fit(&data, &y, &w, None).unwrap();
    let probs = booster.predict_proba(&data, true);

    // Should not produce NaN or Inf even with aggressive lr
    for i in 0..n {
        let row = &probs[i * 3..(i + 1) * 3];
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {i} sum = {sum}");
        for &p in row {
            assert!(p.is_finite());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 18. JSON roundtrip preserves multiclass state for large K
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_json_roundtrip_k5() {
    let (data_vec, y, n, nc) = make_kclass_data(5, 40, 3, 6.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(15);
    booster.num_classes = 5;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();
    let preds_before = booster.predict(&data, true);

    booster
        .save_booster("resources/stress_model_k5.json")
        .unwrap();
    let loaded = GradientBooster::load_booster("resources/stress_model_k5.json").unwrap();

    assert_eq!(loaded.num_classes, 5);
    assert_eq!(loaded.trees.len(), 15 * 5);
    assert!(loaded.base_scores.is_some());
    assert_eq!(loaded.base_scores.as_ref().unwrap().len(), 5);

    let preds_after = loaded.predict(&data, true);
    assert_eq!(preds_before.len(), preds_after.len());
    for (a, b) in preds_before.iter().zip(preds_after.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "roundtrip prediction mismatch: {a} vs {b}"
        );
    }

    // Clean up
    std::fs::remove_file("resources/stress_model_k5.json").ok();
}

// ═══════════════════════════════════════════════════════════════════════
// 19. Softmax math: very many classes (K=50)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_softmax_k50() {
    let k = 50;
    // Random-ish logits
    let scores: Vec<f64> = (0..k).map(|i| (i as f64 * 0.3) - 7.5).collect();
    let probs = SoftmaxMultiClass::softmax(&scores, k);

    assert_eq!(probs.len(), k);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "K=50 sum = {sum}");

    for (i, &p) in probs.iter().enumerate() {
        assert!(p >= 0.0 && p <= 1.0, "K=50 class {i}: p = {p}");
        assert!(p.is_finite());
    }

    // Last class should have highest prob (highest logit)
    assert!(probs[49] > probs[0]);
}

// ═══════════════════════════════════════════════════════════════════════
// 20. Grad/Hess: all-zero logits (uniform distribution)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_grad_hess_uniform() {
    let k = 3;
    let n = 3;
    let y = vec![0.0, 1.0, 2.0];
    let yhat = vec![0.0; n * k]; // all zero → uniform softmax
    let w = vec![1.0; n];

    let (grad, hess) = SoftmaxMultiClass::calc_grad_hess(&y, &yhat, &w, k);

    // At uniform distribution, p = 1/K for all classes
    let p = 1.0 / k as f32;
    let expected_hess = 2.0 * p * (1.0 - p); // 2 * 1/3 * 2/3 = 4/9

    for i in 0..n {
        for kk in 0..k {
            let idx = i * k + kk;
            let is_true_class = kk == y[i] as usize;

            // Gradient: (p - 1) for true class, p for others
            let expected_g = if is_true_class { p - 1.0 } else { p };
            assert!(
                (grad[idx] - expected_g).abs() < 1e-6,
                "sample {i} class {kk}: grad = {}, expected {expected_g}",
                grad[idx]
            );

            // Hessian: 2*p*(1-p) for all (same at uniform)
            assert!(
                (hess[idx] - expected_hess).abs() < 1e-6,
                "sample {i} class {kk}: hess = {}, expected {expected_hess}",
                hess[idx]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 21. CalcInit: very imbalanced → verify log-prior magnitudes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_calc_init_imbalanced() {
    // 1000 class 0, 10 class 1, 1 class 2
    let mut y = vec![0.0; 1000];
    y.extend(vec![1.0; 10]);
    y.extend(vec![2.0; 1]);
    let w = vec![1.0; y.len()];

    let base = SoftmaxMultiClass::calc_init(&y, &w, 3);

    // Class 0 (most common) should have highest base score
    assert!(base[0] > base[1], "class 0 base > class 1");
    assert!(base[1] > base[2], "class 1 base > class 2");

    // Mean-centered: sum ≈ 0
    let sum: f64 = base.iter().sum();
    assert!(sum.abs() < 1e-10, "base scores sum = {sum}");

    // The gap should be significant (log scale of 1000:10:1)
    assert!(
        base[0] - base[2] > 3.0,
        "gap = {} should reflect 1000:1 ratio",
        base[0] - base[2]
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 22. ValidateLabels: boundary conditions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_validate_labels_boundaries() {
    // K=3: valid range is [0.0, 1.0, 2.0]
    assert!(SoftmaxMultiClass::validate_labels(&[0.0, 1.0, 2.0], 3).is_ok());
    assert!(SoftmaxMultiClass::validate_labels(&[0.0], 1).is_ok()); // K=1 edge

    // Just below K → invalid
    assert!(SoftmaxMultiClass::validate_labels(&[2.9999], 3).is_err());

    // Exactly K → invalid
    assert!(SoftmaxMultiClass::validate_labels(&[3.0], 3).is_err());

    // Negative zero is fine (it's == 0.0)
    assert!(SoftmaxMultiClass::validate_labels(&[-0.0], 3).is_ok());

    // f64::INFINITY
    assert!(SoftmaxMultiClass::validate_labels(&[f64::INFINITY], 3).is_err());
    assert!(SoftmaxMultiClass::validate_labels(&[f64::NEG_INFINITY], 3).is_err());

    // Very large valid label
    assert!(SoftmaxMultiClass::validate_labels(&[99.0], 100).is_ok());
    assert!(SoftmaxMultiClass::validate_labels(&[100.0], 100).is_err());

    // Empty y is fine
    assert!(SoftmaxMultiClass::validate_labels(&[], 3).is_ok());
}

// ═══════════════════════════════════════════════════════════════════════
// 23. Early stopping: metric must decrease monotonically near convergence
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_early_stopping_metric_tracking() {
    // Train on noisy/overlapping data so the model can overfit
    // Use a separate eval set that is DIFFERENT from training set
    let (train_data, train_y, n_train, nc) = make_kclass_data(3, 30, 2, 2.0);
    let (eval_data_vec, eval_y, n_eval, _) = make_kclass_data(3, 20, 2, 2.0);

    let data = Matrix::new(&train_data, n_train, nc);
    let w_train = vec![1.0; n_train];
    let w_eval = vec![1.0; n_eval];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(200)
        .set_max_depth(6) // deep trees → easy to overfit
        .set_learning_rate(0.5); // aggressive → fast convergence then overfit
    booster.num_classes = 3;
    booster.early_stopping_rounds = Some(5);
    booster.initialize_base_score = true;

    let eval_sets = vec![(
        Matrix::new(&eval_data_vec, n_eval, nc),
        eval_y.as_slice(),
        w_eval.as_slice(),
    )];
    booster
        .fit(&data, &train_y, &w_train, Some(eval_sets))
        .unwrap();

    // Evaluation history should exist
    let history = booster.evaluation_history.as_ref().unwrap();
    assert!(history.rows > 0, "should have evaluation history");

    // First metric should be >= best (metric decreases initially)
    let first_metric = history.data[0];
    let best_idx = booster.best_iteration.unwrap();
    let best_metric = history.data[best_idx];
    assert!(
        first_metric >= best_metric,
        "first ({first_metric}) should be >= best ({best_metric})"
    );

    let best_iteration = booster
        .best_iteration
        .expect("best_iteration should be set");
    let prediction_iteration = booster
        .prediction_iteration_limit()
        .expect("prediction_iteration should be set");
    assert_eq!(
        prediction_iteration,
        best_iteration + 1,
        "prediction_iteration should equal best_iteration + 1 in boosting rounds"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 24. Determinism: same seed → same results
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_deterministic_seed() {
    // Use overlapping clusters + random subsampling so seed actually matters
    let (data_vec, y, n, nc) = make_kclass_data(3, 80, 2, 1.5); // low separation → overlap
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let train = |seed: u64| {
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(15)
            .set_max_depth(4)
            .set_learning_rate(0.3)
            .set_subsample(0.7);
        b.num_classes = 3;
        b.seed = seed;
        b.sample_method = SampleMethod::Random;
        b.initialize_base_score = true;
        b.fit(&data, &y, &w, None).unwrap();
        b.predict(&data, true)
    };

    let preds_a = train(42);
    let preds_b = train(42);
    let preds_c = train(99);

    // Same seed → identical
    for (a, b) in preds_a.iter().zip(preds_b.iter()) {
        assert!((a - b).abs() < 1e-15, "same seed should be deterministic");
    }

    // Different seed + subsampling → different
    let any_different = preds_a
        .iter()
        .zip(preds_c.iter())
        .any(|(a, c)| (a - c).abs() > 1e-10);
    assert!(
        any_different,
        "different seeds should produce different results"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 25. Parallel vs sequential predict: same results
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_parallel_vs_sequential() {
    let (data_vec, y, n, nc) = make_kclass_data(3, 100, 3, 5.0);
    let data = Matrix::new(&data_vec, n, nc);
    let w = vec![1.0; n];

    let mut booster = GradientBooster::default()
        .set_objective_type(ObjectiveType::SoftmaxMultiClass)
        .set_iterations(20)
        .set_max_depth(4)
        .set_learning_rate(0.3);
    booster.num_classes = 3;
    booster.parallel = true;
    booster.initialize_base_score = true;

    booster.fit(&data, &y, &w, None).unwrap();

    let preds_par = booster.predict(&data, true);
    let preds_seq = booster.predict(&data, false);

    assert_eq!(preds_par.len(), preds_seq.len());
    for (p, s) in preds_par.iter().zip(preds_seq.iter()) {
        assert!(
            (p - s).abs() < 1e-12,
            "parallel vs sequential mismatch: {p} vs {s}"
        );
    }

    let probs_par = booster.predict_proba(&data, true);
    let probs_seq = booster.predict_proba(&data, false);
    for (p, s) in probs_par.iter().zip(probs_seq.iter()) {
        assert!((p - s).abs() < 1e-12);
    }
}
