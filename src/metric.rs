use crate::data::FloatData;
use serde::{Deserialize, Serialize};

type MetricFn = fn(&[f64], &[f64], &[f64]) -> f64;

#[derive(Debug, Deserialize, Serialize)]
pub enum Metric {
    LogLoss,
    RootMeanSquaredLogError,
    RootMeanSquaredError,
}

pub fn metric_callables(metric_type: &Metric) -> MetricFn {
    match metric_type {
        Metric::LogLoss => log_loss,
        Metric::RootMeanSquaredLogError => root_mean_squared_log_error,
        Metric::RootMeanSquaredError => root_mean_squared_error,
    }
}

pub fn log_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    y.iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
            -(*y_ * yhat_.ln() + (f64::ONE - *y_) * ((f64::ONE - yhat_).ln())) * *w_
        })
        .sum::<f64>()
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
    (res / w_sum).sqrt()
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
    (res / w_sum).sqrt()
}

fn trapezoid_area(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    (x0 - x1).abs() * (y0 + y1) * 0.5
}

pub fn auc(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
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
        assert_eq!(precision_round(res, 5), 5.33118);
    }

    #[test]
    fn test_auc_real_data() {
        let y = vec![1., 0., 1., 0., 0., 0., 0.];
        let yhat = vec![0.5, 0.01, -0., 1.05, 0., -4., 0.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = auc(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 5), 0.67857);
    }

    #[test]
    fn test_auc_generc() {
        let sample_weight: Vec<f64> = vec![1.; 2];

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![0., 1.];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.);

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 1.];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.0);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![0.5, 0.5];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![0., 0.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());

        let y: Vec<f64> = vec![1., 1.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = auc(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());
    }
}
