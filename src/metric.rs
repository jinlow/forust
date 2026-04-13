use crate::data::FloatData;
use crate::errors::ForustError;
use crate::objective::SoftmaxMultiClass;
use crate::utils::items_to_strings;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub type MetricFn = fn(&[f64], &[f64], &[f64]) -> f64;
const LOG_LOSS_EPS: f64 = 1e-16;

#[inline]
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.max(LOG_LOSS_EPS).ln()
    }
}

#[inline]
fn weighted_mean(total: f64, weight_sum: f64) -> f64 {
    if weight_sum == 0.0 {
        total
    } else {
        total / weight_sum
    }
}

#[inline]
fn weighted_root_mean(total: f64, weight_sum: f64) -> f64 {
    weighted_mean(total, weight_sum).sqrt()
}

/// Compare to metric values, determining if b is better.
/// If one of them is NaN favor the non NaN value.
/// If both are NaN, consider the first value to be better.
pub fn is_comparison_better(value: f64, comparison: f64, maximize: bool) -> bool {
    match (value.is_nan(), comparison.is_nan()) {
        // Both nan, comparison is not better,
        // Or comparison is nan, also not better
        (true, true) | (false, true) => false,
        // comparison is not Nan, it's better
        (true, false) => true,
        // Perform numerical comparison.
        (false, false) => {
            // If we are maximizing is the comparison
            // greater, than the current value
            if maximize {
                value < comparison
            // If we are minimizing is the comparison
            // less than the current value.
            } else {
                value > comparison
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum Metric {
    AUC,
    LogLoss,
    QuantileLoss,
    RootMeanSquaredLogError,
    RootMeanSquaredError,
    MultiClassLogLoss,
}

impl FromStr for Metric {
    type Err = ForustError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "AUC" => Ok(Metric::AUC),
            "LogLoss" => Ok(Metric::LogLoss),
            "QuantileLoss" => Ok(Metric::QuantileLoss),
            "RootMeanSquaredLogError" => Ok(Metric::RootMeanSquaredLogError),
            "RootMeanSquaredError" => Ok(Metric::RootMeanSquaredError),
            "MultiClassLogLoss" => Ok(Metric::MultiClassLogLoss),

            _ => Err(ForustError::ParseString(
                s.to_string(),
                "Metric".to_string(),
                items_to_strings(vec![
                    "AUC",
                    "LogLoss",
                    "QuantileLoss",
                    "RootMeanSquaredLogError",
                    "RootMeanSquaredError",
                    "MultiClassLogLoss",
                ]),
            )),
        }
    }
}

pub fn metric_callables(metric_type: &Metric) -> (MetricFn, bool) {
    match metric_type {
        Metric::AUC => (AUCMetric::calculate_metric, AUCMetric::maximize()),
        Metric::LogLoss => (LogLossMetric::calculate_metric, LogLossMetric::maximize()),
        Metric::QuantileLoss => {
            panic!("QuantileLoss uses objective-aware dispatch because the metric depends on alpha")
        }
        Metric::RootMeanSquaredLogError => (
            RootMeanSquaredLogErrorMetric::calculate_metric,
            RootMeanSquaredLogErrorMetric::maximize(),
        ),
        Metric::RootMeanSquaredError => (
            RootMeanSquaredErrorMetric::calculate_metric,
            RootMeanSquaredErrorMetric::maximize(),
        ),
        Metric::MultiClassLogLoss => panic!(
            "MultiClassLogLoss uses multiclass_log_loss() directly, not function-pointer dispatch"
        ),
    }
}

pub trait EvaluationMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64;
    fn maximize() -> bool;
}

pub struct LogLossMetric {}
impl EvaluationMetric for LogLossMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
        log_loss(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub struct AUCMetric {}
impl EvaluationMetric for AUCMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
        roc_auc_score(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        true
    }
}

pub struct RootMeanSquaredLogErrorMetric {}
impl EvaluationMetric for RootMeanSquaredLogErrorMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
        root_mean_squared_log_error(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub struct RootMeanSquaredErrorMetric {}
impl EvaluationMetric for RootMeanSquaredErrorMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
        root_mean_squared_error(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub fn log_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            let yhat_ =
                (f64::ONE / (f64::ONE + (-*yhat_).exp())).clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS);
            -(xlogy(*y_, yhat_) + xlogy(f64::ONE - *y_, f64::ONE - yhat_)) * *w_
        })
        .sum::<f64>();
    weighted_mean(res, w_sum)
}

pub fn root_mean_squared_log_error(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            (y_.ln_1p() - yhat_.ln_1p()).powi(2) * *w_
        })
        .sum::<f64>();
    weighted_root_mean(res, w_sum)
}

pub fn root_mean_squared_error(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            (y_ - yhat_).powi(2) * *w_
        })
        .sum::<f64>();
    weighted_root_mean(res, w_sum)
}

/// Multi-class log loss (cross-entropy).
/// y: labels (N), yhat: raw logits (N×K), w: weights (N).
pub fn multiclass_log_loss(y: &[f64], yhat: &[f64], w: &[f64], num_classes: usize) -> f64 {
    let probs = SoftmaxMultiClass::softmax(yhat, num_classes);
    let n = y.len();
    let mut loss = 0.0;
    let mut w_sum = 0.0;
    for i in 0..n {
        let k = y[i] as usize;
        let p = probs[i * num_classes + k].max(1e-15);
        loss -= p.ln() * w[i];
        w_sum += w[i];
    }
    weighted_mean(loss, w_sum)
}

fn trapezoid_area(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    (x0 - x1).abs() * (y0 + y1) * 0.5
}

pub fn roc_auc_score(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut indices = (0..y.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| yhat[b].total_cmp(&yhat[a]));
    let mut auc: f64 = 0.0;

    let mut label = y[indices[0]];
    let mut w = sample_weight[indices[0]];
    let mut fp = (1.0 - label) * w;
    let mut tp: f64 = label * w;
    let mut tp_prev: f64 = 0.0;
    let mut fp_prev: f64 = 0.0;

    for i in 1..indices.len() {
        if yhat[indices[i]] != yhat[indices[i - 1]] {
            auc += trapezoid_area(fp_prev, fp, tp_prev, tp);
            tp_prev = tp;
            fp_prev = fp;
        }
        label = y[indices[i]];
        w = sample_weight[indices[i]];
        fp += (1.0 - label) * w;
        tp += label * w;
    }

    auc += trapezoid_area(fp_prev, fp, tp_prev, tp);
    if fp <= 0.0 || tp <= 0.0 {
        auc = 0.0;
        fp = 0.0;
        tp = 0.0;
    }

    auc / (tp * fp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::precision_round;
    #[test]
    fn test_root_mean_squared_log_error() {
        let y = vec![1., 3., 4., 5., 2., 4., 6.];
        let yhat = vec![3., 2., 3., 4., 4., 4., 4.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = root_mean_squared_log_error(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 4), 0.3549);
    }
    #[test]
    fn test_root_mean_squared_error() {
        let y = vec![1., 3., 4., 5., 2., 4., 6.];
        let yhat = vec![3., 2., 3., 4., 4., 4., 4.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = root_mean_squared_error(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 6), 1.452966);
    }

    #[test]
    fn test_log_loss() {
        let y = vec![1., 0., 1., 0., 0., 0., 0.];
        let yhat = vec![0.5, 0.01, -0., 1.05, 0., -4., 0.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = log_loss(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 5), 0.59235);
    }

    #[test]
    fn test_log_loss_is_finite_for_extreme_margins() {
        let y = vec![0.0, 1.0];
        let yhat = vec![-1000.0, 1000.0];
        let sample_weight = vec![1.0, 1.0];
        let res = log_loss(&y, &yhat, &sample_weight);
        assert!(res.is_finite());
        assert!(res >= 0.0);
    }

    #[test]
    fn test_log_loss_zero_weight_sum_returns_zero() {
        let y = vec![0.0, 1.0];
        let yhat = vec![0.0, 0.0];
        let sample_weight = vec![0.0, 0.0];
        assert_eq!(log_loss(&y, &yhat, &sample_weight), 0.0);
    }

    #[test]
    fn test_auc_real_data() {
        let y = vec![1., 0., 1., 0., 0., 0., 0.];
        let yhat = vec![0.5, 0.01, -0., 1.05, 0., -4., 0.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 5), 0.67857);
    }

    #[test]
    fn test_multiclass_log_loss() {
        // Perfect predictions → low loss
        let y = vec![0.0, 1.0, 2.0];
        let good_logits = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let w = vec![1.0; 3];
        let good_loss = multiclass_log_loss(&y, &good_logits, &w, 3);

        // Random predictions → higher loss
        let bad_logits = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bad_loss = multiclass_log_loss(&y, &bad_logits, &w, 3);

        assert!(good_loss > 0.0, "loss should be positive");
        assert!(
            good_loss < bad_loss,
            "better predictions should have lower loss"
        );
        // Uniform distribution → log(3) ≈ 1.0986
        assert!((bad_loss - 3.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_log_loss_zero_weight_sum_returns_zero() {
        let y = vec![0.0, 1.0];
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let w = vec![0.0, 0.0];
        assert_eq!(multiclass_log_loss(&y, &logits, &w, 2), 0.0);
    }

    #[test]
    fn test_auc_generc() {
        let sample_weight: Vec<f64> = vec![1.; 2];

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![0., 1.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.);

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 1.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.0);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![0.5, 0.5];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![0., 0.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());

        let y: Vec<f64> = vec![1., 1.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());
    }
}
