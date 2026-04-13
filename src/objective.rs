use crate::errors::ForustError;
use crate::{data::FloatData, metric::Metric};
use serde::{Deserialize, Serialize};

type ObjFn = fn(&[f64], &[f64], &[f64]) -> (Vec<f32>, Vec<f32>);
const LOG_LOSS_EPS: f64 = 1e-16;
const XGBOOST_RT_EPS: f64 = 1e-6;

#[inline]
fn logistic_probability(margin: f64) -> f64 {
    f64::ONE / (f64::ONE + (-margin).exp())
}

#[inline]
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.max(LOG_LOSS_EPS).ln()
    }
}

#[inline]
fn binary_log_loss(label: f64, margin: f64) -> f64 {
    let prob = logistic_probability(margin).clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS);
    -(xlogy(label, prob) + xlogy(1.0 - label, 1.0 - prob))
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum ObjectiveType {
    LogLoss,
    SquaredLoss,
    SoftmaxMultiClass,
    /// Quantile regression (pinball loss) with configurable alpha ∈ (0, 1).
    /// alpha = 0.5 is median regression; alpha = 0.70 gives a 70th-percentile
    /// regressor (used by xgboost-v3 for MFE/MAE quantile heads).
    QuantileLoss {
        alpha: f64,
    },
}

pub fn gradient_hessian_callables(objective_type: &ObjectiveType) -> ObjFn {
    match objective_type {
        ObjectiveType::LogLoss => LogLoss::calc_grad_hess,
        ObjectiveType::SquaredLoss => SquaredLoss::calc_grad_hess,
        ObjectiveType::SoftmaxMultiClass => {
            panic!("SoftmaxMultiClass uses direct calls, not function-pointer dispatch")
        }
        ObjectiveType::QuantileLoss { .. } => {
            panic!("QuantileLoss uses direct calls, not function-pointer dispatch")
        }
    }
}

pub fn calc_init_callables(objective_type: &ObjectiveType) -> fn(&[f64], &[f64]) -> f64 {
    match objective_type {
        ObjectiveType::LogLoss => LogLoss::calc_init,
        ObjectiveType::SquaredLoss => SquaredLoss::calc_init,
        ObjectiveType::SoftmaxMultiClass => {
            panic!("SoftmaxMultiClass uses direct calls, not function-pointer dispatch")
        }
        ObjectiveType::QuantileLoss { .. } => {
            panic!("QuantileLoss uses direct calls, not function-pointer dispatch")
        }
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>);
    fn calc_init(y: &[f64], sample_weight: &[f64]) -> f64;
    fn default_metric() -> Metric;
}

#[derive(Default)]
pub struct LogLoss {}

impl LogLoss {
    /// Validate logistic labels using XGBoost's `[0, 1]` convention.
    pub fn validate_labels(y: &[f64]) -> Result<(), ForustError> {
        for (i, &yi) in y.iter().enumerate() {
            if !yi.is_finite() || !(0.0..=1.0).contains(&yi) {
                return Err(ForustError::InvalidInput(format!(
                    "LogLoss: label[{}] = {} is not a valid value in [0, 1]",
                    i, yi
                )));
            }
        }
        Ok(())
    }
}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| (binary_log_loss(*y_, *yhat_) * *w_) as f32)
            .collect()
    }

    fn calc_init(y: &[f64], sample_weight: &[f64]) -> f64 {
        let mut ytot: f64 = 0.;
        let mut ntot: f64 = 0.;
        for i in 0..y.len() {
            ytot += sample_weight[i] * y[i];
            ntot += sample_weight[i];
        }
        if !ntot.is_finite() || ntot <= 0.0 {
            return 0.0;
        }
        let p = (ytot / ntot).clamp(XGBOOST_RT_EPS, 1.0 - XGBOOST_RT_EPS);
        f64::ln(p / (1.0 - p))
    }

    #[inline]
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>) {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let yhat_ = logistic_probability(*yhat_);
                (
                    ((yhat_ - *y_) * *w_) as f32,
                    ((yhat_ * (f64::ONE - yhat_)).max(LOG_LOSS_EPS) * *w_) as f32,
                )
            })
            .unzip()
    }

    fn default_metric() -> Metric {
        Metric::LogLoss
    }
}

#[derive(Default)]
pub struct SquaredLoss {}

impl ObjectiveFunction for SquaredLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let s = *y_ - *yhat_;
                (s * s * *w_) as f32
            })
            .collect()
    }

    fn calc_init(y: &[f64], sample_weight: &[f64]) -> f64 {
        let mut ytot: f64 = 0.;
        let mut ntot: f64 = 0.;
        for i in 0..y.len() {
            ytot += sample_weight[i] * y[i];
            ntot += sample_weight[i];
        }
        if !ntot.is_finite() || ntot <= 0.0 {
            return 0.0;
        }

        ytot / ntot
    }

    #[inline]
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>) {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| (((yhat_ - *y_) * *w_) as f32, *w_ as f32))
            .unzip()
    }
    fn default_metric() -> Metric {
        Metric::RootMeanSquaredError
    }
}

/// Multi-class softmax objective.
/// Ported from XGBoost C++ `multiclass_obj.cu`.
pub struct SoftmaxMultiClass;

impl SoftmaxMultiClass {
    /// Numerically stable softmax over N×K row-major scores.
    /// Matches xgboost/src/common/math.h `common::Softmax`.
    pub fn softmax(scores: &[f64], num_classes: usize) -> Vec<f64> {
        let n = scores.len() / num_classes;
        let mut probs = Vec::with_capacity(scores.len());
        for i in 0..n {
            let row = &scores[i * num_classes..(i + 1) * num_classes];
            let wmax = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&s| (s - wmax).exp()).collect();
            let wsum: f64 = exps.iter().sum();
            for &e in &exps {
                probs.push(e / wsum);
            }
        }
        probs
    }

    /// Gradient and hessian for softmax cross-entropy.
    /// Matches xgboost/src/objective/multiclass_obj.cu `GetGradient`.
    ///
    /// y: integer class labels as f64 (0.0, ..., K-1), length N.
    /// yhat: raw logits, N×K row-major, length N×K.
    /// w: sample weights, length N.
    /// Returns (grad, hess) each length N×K as f32.
    pub fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        w: &[f64],
        num_classes: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        const K_EPS: f32 = 1e-16;
        let n = y.len();
        let mut grad = vec![0.0f32; n * num_classes];
        let mut hess = vec![0.0f32; n * num_classes];

        for i in 0..n {
            let row = &yhat[i * num_classes..(i + 1) * num_classes];
            let wmax = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let wsum: f64 = row.iter().map(|&s| (s - wmax).exp()).sum();

            let label = y[i] as usize;
            let wt = w[i] as f32;

            for k in 0..num_classes {
                let p = ((row[k] - wmax).exp() / wsum) as f32;
                let idx = i * num_classes + k;
                // Hessian first (uses original p, not adjusted)
                hess[idx] = f32::max(2.0 * p * (1.0 - p) * wt, K_EPS);
                // Gradient (adjust p for true class)
                let p_adj = if k == label { p - 1.0 } else { p };
                grad[idx] = p_adj * wt;
            }
        }
        (grad, hess)
    }

    /// Base score: log of class priors, centered by mean.
    /// Matches xgboost InitEstimation: ln(prob_k) - mean(ln(prob)).
    pub fn calc_init(y: &[f64], w: &[f64], num_classes: usize) -> Vec<f64> {
        let mut class_weights = vec![0.0f64; num_classes];
        let mut total = 0.0f64;
        for (&yi, &wi) in y.iter().zip(w.iter()) {
            let k = yi as usize;
            if k < num_classes {
                class_weights[k] += wi;
            }
            total += wi;
        }
        if !total.is_finite() || total <= 0.0 {
            return vec![0.0; num_classes];
        }
        let logs: Vec<f64> = class_weights
            .iter()
            .map(|cw| (cw / total).max(XGBOOST_RT_EPS).ln())
            .collect();
        let mean_log: f64 = logs.iter().sum::<f64>() / num_classes as f64;
        logs.iter().map(|l| l - mean_log).collect()
    }

    /// Validate that all labels are integers in [0, num_classes).
    /// Matches xgboost ValidateLabel.
    pub fn validate_labels(y: &[f64], num_classes: usize) -> Result<(), ForustError> {
        for (i, &yi) in y.iter().enumerate() {
            if yi < 0.0 || yi >= num_classes as f64 || yi.floor() != yi || yi.is_nan() {
                return Err(ForustError::InvalidInput(format!(
                    "SoftmaxMultiClass: label[{}] = {} is not a valid integer in [0, {})",
                    i, yi, num_classes
                )));
            }
        }
        Ok(())
    }

    pub fn default_metric() -> Metric {
        Metric::MultiClassLogLoss
    }
}

/// Quantile regression with pinball loss.
///
/// The pinball loss for a target y, prediction ŷ, and quantile α is
/// `ρ_α(r) = max(α·r, (α−1)·r)` where `r = y − ŷ`.
///
/// XGBoost emits gradients with respect to prediction `ŷ`, using
/// `d = ŷ - y`:
/// `g_ŷ = 1 - α` when `d >= 0` (over-prediction) and `g_ŷ = -α`
/// when `d < 0` (under-prediction).
///
/// Hessian for pinball loss is technically zero everywhere the function
/// is differentiable, which breaks Newton-style boosting. Following
/// XGBoost and LightGBM, we use a constant pseudo-hessian of 1.0 per
/// sample (scaled by weight). This produces first-order gradient steps
/// similar to absolute-error (MAE) boosting.
pub struct QuantileLoss;

impl QuantileLoss {
    /// Pinball loss `ρ_α(y − ŷ)` per sample, weighted.
    pub fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64], alpha: f64) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let r = *y_ - *yhat_;
                let loss = if r >= 0.0 {
                    alpha * r
                } else {
                    (alpha - 1.0) * r
                };
                (loss * *w_) as f32
            })
            .collect()
    }

    /// Quantile (pinball) loss gradient emitted in the XGBoost-compatible form
    /// used by this crate.
    ///
    /// When `y > ŷ` (under-prediction), `grad = -α < 0` → pushes ŷ up.
    /// When `y ≤ ŷ` (over-prediction), `grad = 1 - α > 0` → pushes ŷ down.
    pub fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: &[f64],
        alpha: f64,
    ) -> (Vec<f32>, Vec<f32>) {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let grad = if *yhat_ >= *y_ { 1.0 - alpha } else { -alpha } * *w_;
                // Constant pseudo-hessian avoids division-by-zero in Newton updates.
                let hess = *w_;
                (grad as f32, hess as f32)
            })
            .unzip()
    }

    /// Base score: weighted α-quantile of y. Falls back to the unweighted
    /// quantile if all weights are equal or zero.
    pub fn calc_init(y: &[f64], sample_weight: &[f64], alpha: f64) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let mut pairs: Vec<(f64, f64)> =
            y.iter().zip(sample_weight).map(|(&v, &w)| (v, w)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let total_w: f64 = pairs.iter().map(|(_, w)| *w).sum();
        if total_w <= 0.0 {
            return pairs[pairs.len() / 2].0;
        }
        let target = alpha * total_w;
        let mut cumulative = 0.0f64;
        for (v, w) in &pairs {
            cumulative += *w;
            if cumulative >= target {
                return *v;
            }
        }
        pairs.last().unwrap().0
    }

    pub fn default_metric() -> Metric {
        Metric::QuantileLoss
    }

    pub fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], alpha: f64) -> f64 {
        let total_weight: f64 = sample_weight.iter().sum();
        let total_loss = y
            .iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let r = *y_ - *yhat_;
                let loss = if r >= 0.0 {
                    alpha * r
                } else {
                    (alpha - 1.0) * r
                };
                loss * *w_
            })
            .sum::<f64>();
        if total_weight == 0.0 {
            total_loss
        } else {
            total_loss / total_weight
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_logloss_loss() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let l1 = LogLoss::calc_loss(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = LogLoss::calc_loss(&y, &yhat2, &w);
        assert!(l1.iter().sum::<f32>() < l2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let (g1, _) = LogLoss::calc_grad_hess(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let (g2, _) = LogLoss::calc_grad_hess(&y, &yhat2, &w);
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_hess() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let (_, h1) = LogLoss::calc_grad_hess(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let (_, h2) = LogLoss::calc_grad_hess(&y, &yhat2, &w);
        assert!(h1.iter().sum::<f32>() < h2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_hess_is_floored_for_extreme_margins() {
        let y = vec![1.0];
        let yhat = vec![1000.0];
        let w = vec![1.0];
        let (_, hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        assert!(hess[0].is_finite());
        assert!(hess[0] >= LOG_LOSS_EPS as f32);
    }

    #[test]
    fn test_logloss_loss_is_finite_for_extreme_margins() {
        let y = vec![0.0, 1.0];
        let yhat = vec![-1000.0, 1000.0];
        let w = vec![1.0, 1.0];
        let loss = LogLoss::calc_loss(&y, &yhat, &w);
        assert!(loss.iter().all(|v| v.is_finite()));
        assert!(loss.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn test_logloss_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let l1 = LogLoss::calc_init(&y, &w);
        assert!(l1 == 0.);

        let y = vec![1.0; 6];
        let l2 = LogLoss::calc_init(&y, &w);
        assert!(l2.is_finite());
        assert!((l2 - f64::ln((1.0 - XGBOOST_RT_EPS) / XGBOOST_RT_EPS)).abs() < 1e-10);

        let y = vec![0.0; 6];
        let l3 = LogLoss::calc_init(&y, &w);
        assert!(l3.is_finite());
        assert!((l3 - f64::ln(XGBOOST_RT_EPS / (1.0 - XGBOOST_RT_EPS))).abs() < 1e-10);

        let y = vec![0., 0., 0., 0., 1., 1.];
        let l4 = LogLoss::calc_init(&y, &w);
        assert!((l4 - f64::ln(2. / 4.)).abs() < 1e-12);
    }

    #[test]
    fn test_logloss_init_zero_sum_weights_is_neutral() {
        let y = vec![0.0, 1.0];
        let w = vec![0.0, 0.0];
        assert_eq!(LogLoss::calc_init(&y, &w), 0.0);
    }

    #[test]
    fn test_logloss_validate_labels_rejects_bad() {
        assert!(LogLoss::validate_labels(&[-1.0]).is_err());
        assert!(LogLoss::validate_labels(&[2.0]).is_err());
        assert!(LogLoss::validate_labels(&[f64::NAN]).is_err());
        assert!(LogLoss::validate_labels(&[f64::INFINITY]).is_err());
    }

    #[test]
    fn test_logloss_validate_labels_accepts_xgboost_style_soft_labels() {
        assert!(LogLoss::validate_labels(&[0.0, 0.25, 0.5, 1.0]).is_ok());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        // Random inputs
        let scores = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let probs = SoftmaxMultiClass::softmax(&scores, 3);
        for row in probs.chunks(3) {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "sum = {}", sum);
        }
        // All equal → uniform
        let scores = vec![5.0, 5.0, 5.0];
        let probs = SoftmaxMultiClass::softmax(&scores, 3);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large equal values → should not overflow
        let scores = vec![1000.0, 1000.0, 1000.0];
        let probs = SoftmaxMultiClass::softmax(&scores, 3);
        for &p in &probs {
            assert!(!p.is_nan(), "softmax produced NaN");
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
        // Extreme range → should not overflow
        let scores = vec![-1000.0, 0.0, 1000.0];
        let probs = SoftmaxMultiClass::softmax(&scores, 3);
        assert!((probs[2] - 1.0).abs() < 1e-10, "max class should be ~1.0");
        assert!(probs[0] < 1e-100, "min class should be ~0.0");
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_hess_3class() {
        // y=0, logits=[1.0, 0.0, 0.0], w=1.0
        let y = vec![0.0];
        let yhat = vec![1.0, 0.0, 0.0];
        let w = vec![1.0];
        let (grad, hess) = SoftmaxMultiClass::calc_grad_hess(&y, &yhat, &w, 3);

        // Compute expected probabilities
        let probs = SoftmaxMultiClass::softmax(&yhat, 3);
        let p0 = probs[0] as f32;
        let p1 = probs[1] as f32;
        let p2 = probs[2] as f32;

        // grad[0] = (p0 - 1) * w, grad[1] = p1 * w, grad[2] = p2 * w
        assert!((grad[0] - (p0 - 1.0)).abs() < 1e-6);
        assert!((grad[1] - p1).abs() < 1e-6);
        assert!((grad[2] - p2).abs() < 1e-6);

        // All hessians should be positive
        for &h in &hess {
            assert!(h > 0.0);
        }
    }

    #[test]
    fn test_hess_2x_factor() {
        let y = vec![1.0];
        let yhat = vec![0.5, 0.5, 0.5];
        let w = vec![2.0];
        let (_, hess) = SoftmaxMultiClass::calc_grad_hess(&y, &yhat, &w, 3);
        let probs = SoftmaxMultiClass::softmax(&yhat, 3);
        for k in 0..3 {
            let p = probs[k] as f32;
            let expected = 2.0 * p * (1.0 - p) * 2.0; // 2x factor * weight
            assert!(
                (hess[k] - expected).abs() < 1e-6,
                "hess[{}] = {}, expected {}",
                k,
                hess[k],
                expected
            );
        }
    }

    #[test]
    fn test_calc_init_centered() {
        // Balanced 3-class → base scores should sum to ~0
        let y = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let w = vec![1.0; 6];
        let base = SoftmaxMultiClass::calc_init(&y, &w, 3);
        assert_eq!(base.len(), 3);
        let sum: f64 = base.iter().sum();
        assert!(sum.abs() < 1e-10, "base scores sum = {}", sum);
        // All equal priors → all base scores should be 0
        for &b in &base {
            assert!(b.abs() < 1e-10);
        }
    }

    #[test]
    fn test_calc_init_zero_sum_weights_is_neutral() {
        let y = vec![0.0, 1.0, 2.0];
        let w = vec![0.0, 0.0, 0.0];
        assert_eq!(SoftmaxMultiClass::calc_init(&y, &w, 3), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_validate_labels_rejects_bad() {
        assert!(SoftmaxMultiClass::validate_labels(&[-1.0], 3).is_err());
        assert!(SoftmaxMultiClass::validate_labels(&[3.0], 3).is_err());
        assert!(SoftmaxMultiClass::validate_labels(&[1.5], 3).is_err());
        assert!(SoftmaxMultiClass::validate_labels(&[f64::NAN], 3).is_err());
    }

    #[test]
    fn test_validate_labels_accepts_good() {
        assert!(SoftmaxMultiClass::validate_labels(&[0.0, 1.0, 2.0], 3).is_ok());
        assert!(SoftmaxMultiClass::validate_labels(&[0.0, 0.0, 1.0, 1.0], 2).is_ok());
    }

    #[test]
    fn test_mse_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let l1 = SquaredLoss::calc_init(&y, &w);
        assert!(l1 == 0.5);

        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let l2 = SquaredLoss::calc_init(&y, &w);
        assert!(l2 == 1.);

        let y = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let l3 = SquaredLoss::calc_init(&y, &w);
        assert!(l3 == -1.);

        let y = vec![-1.0, -1.0, -1.0, 1., 1., 1.];
        let l4 = SquaredLoss::calc_init(&y, &w);
        assert!(l4 == 0.);
    }

    #[test]
    fn test_squared_loss_default_metric_is_rmse() {
        assert!(matches!(
            SquaredLoss::default_metric(),
            Metric::RootMeanSquaredError
        ));
    }

    #[test]
    fn test_squaredloss_init_zero_sum_weights_is_zero() {
        let y = vec![1.0, 2.0];
        let w = vec![0.0, 0.0];
        assert_eq!(SquaredLoss::calc_init(&y, &w), 0.0);
    }

    #[test]
    fn test_quantile_default_metric_is_quantile_loss() {
        assert!(matches!(
            QuantileLoss::default_metric(),
            Metric::QuantileLoss
        ));
    }

    #[test]
    fn test_quantile_init_equal_weights() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0; 5];
        // α = 0.5 → weighted median = 3.0
        let q50 = QuantileLoss::calc_init(&y, &w, 0.5);
        assert_eq!(q50, 3.0);
        // α = 0.70 → 70th percentile. cum = 0.2,0.4,0.6,0.8,1.0. target=0.7·5=3.5 → 4.0
        let q70 = QuantileLoss::calc_init(&y, &w, 0.70);
        assert_eq!(q70, 4.0);
        // α = 0.0 → smallest
        let q0 = QuantileLoss::calc_init(&y, &w, 0.0);
        assert_eq!(q0, 1.0);
        // α = 1.0 → largest
        let q1 = QuantileLoss::calc_init(&y, &w, 1.0);
        assert_eq!(q1, 5.0);
    }

    #[test]
    fn test_quantile_grad_sign_for_alpha_70() {
        let alpha = 0.70;
        let y = vec![1.0, 1.0];
        let yhat = vec![0.0, 2.0]; // underprediction, overprediction
        let w = vec![1.0, 1.0];
        let (grad, hess) = QuantileLoss::calc_grad_hess(&y, &yhat, &w, alpha);
        // y=1 > ŷ=0 → underprediction → grad = -0.70
        // y=1 < ŷ=2 → overprediction → grad = 1 - 0.70 = 0.30
        assert!((grad[0] + 0.70).abs() < 1e-6);
        assert!((grad[1] - 0.30).abs() < 1e-6);
        // Hessian = weight
        assert!((hess[0] - 1.0).abs() < 1e-6);
        assert!((hess[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_grad_sign_for_alpha_30() {
        let alpha = 0.30;
        let y = vec![1.0, 1.0];
        let yhat = vec![0.0, 2.0];
        let w = vec![1.0, 1.0];
        let (grad, _) = QuantileLoss::calc_grad_hess(&y, &yhat, &w, alpha);
        // y > ŷ → underprediction → grad = -0.30
        // y < ŷ → overprediction → grad = 0.70
        assert!((grad[0] + 0.30).abs() < 1e-6);
        assert!((grad[1] - 0.70).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_loss_pinball() {
        let y = vec![1.0, 1.0];
        let yhat = vec![0.0, 2.0];
        let w = vec![1.0, 1.0];
        let alpha = 0.70;
        let losses = QuantileLoss::calc_loss(&y, &yhat, &w, alpha);
        // r = y − ŷ
        // r=+1 → α·1 = 0.70
        // r=−1 → (α−1)·−1 = 0.30
        assert!((losses[0] - 0.70).abs() < 1e-6);
        assert!((losses[1] - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_metric_zero_sum_weights_returns_zero() {
        let y = vec![1.0, 2.0];
        let yhat = vec![1.5, 2.5];
        let w = vec![0.0, 0.0];
        assert_eq!(QuantileLoss::calculate_metric(&y, &yhat, &w, 0.5), 0.0);
    }

    #[test]
    fn test_quantile_init_weighted_quantile() {
        let y = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 5.0, 1.0]; // heavy weight on 2.0
                                     // total_w = 7, α=0.5 target = 3.5
                                     // cum after sort: 1 (1.0), 6 (2.0), 7 (3.0) → 2.0
        let q50 = QuantileLoss::calc_init(&y, &w, 0.5);
        assert_eq!(q50, 2.0);
    }
}
