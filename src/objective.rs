use crate::data::FloatData;
use serde::{Deserialize, Serialize};

type ObjFn = fn(&[f64], &[f64], &[f64]) -> Vec<f32>;

#[derive(Debug, Deserialize, Serialize)]
pub enum ObjectiveType {
    LogLoss,
    SquaredLoss,
}

pub fn gradient_hessian_callables(objective_type: &ObjectiveType) -> (ObjFn, ObjFn) {
    match objective_type {
        ObjectiveType::LogLoss => (LogLoss::calc_grad, LogLoss::calc_hess),
        ObjectiveType::SquaredLoss => (SquaredLoss::calc_grad, SquaredLoss::calc_hess),
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_grad(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
}

#[derive(Default)]
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                (-(*y_ * yhat_.ln() + (f64::ONE - *y_) * ((f64::ONE - yhat_).ln())) * *w_) as f32
            })
            .collect()
    }

    #[inline]
    fn calc_grad(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                ((yhat_ - *y_) * *w_) as f32
            })
            .collect()
    }
    #[inline]
    fn calc_hess(_: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        yhat.iter()
            .zip(sample_weight)
            .map(|(yhat_, w_)| {
                let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                (yhat_ * (f64::ONE - yhat_) * *w_) as f32
            })
            .collect()
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

    #[inline]
    fn calc_grad(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| ((*yhat_ - *y_) * *w_) as f32)
            .collect()
    }

    #[inline]
    fn calc_hess(_: &[f64], _: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        sample_weight.iter().map(|v| *v as f32).collect()
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
        let g1 = LogLoss::calc_grad(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let g2 = LogLoss::calc_grad(&y, &yhat2, &w);
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_hess() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let h1 = LogLoss::calc_hess(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let h2 = LogLoss::calc_hess(&y, &yhat2, &w);
        assert!(h1.iter().sum::<f32>() < h2.iter().sum::<f32>());
    }
}
