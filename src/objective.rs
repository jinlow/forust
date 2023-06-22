use crate::{data::FloatData, metric::Metric};
use serde::{Deserialize, Serialize};

type ObjFn = fn(&[f64], &[f64], &[f64]) -> (Vec<f32>, Vec<f32>);

#[derive(Debug, Deserialize, Serialize)]
pub enum ObjectiveType {
    LogLoss,
    SquaredLoss,
}

pub fn gradient_hessian_callables(objective_type: &ObjectiveType) -> ObjFn {
    match objective_type {
        ObjectiveType::LogLoss => LogLoss::calc_grad_hess,
        ObjectiveType::SquaredLoss => SquaredLoss::calc_grad_hess,
    }
}

pub fn calc_init_callables(objective_type: &ObjectiveType) -> fn(&[f64], &[f64]) -> f64 {
    match objective_type {
        ObjectiveType::LogLoss => LogLoss::calc_init,
        ObjectiveType::SquaredLoss => SquaredLoss::calc_init,
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>);
    fn calc_grad(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_init(y: &[f64], sample_weight: &[f64]) -> f64;
    fn default_metric() -> Metric;
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

    fn calc_init(y: &[f64], sample_weight: &[f64]) -> f64 {
        let mut ytot: f64 = 0.;
        let mut ntot: f64 = 0.;
        for i in 0..y.len() {
            ytot += sample_weight[i] * y[i];
            ntot += sample_weight[i];
        }
        f64::ln(ytot / (ntot - ytot))
    }

    #[inline]
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>) {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| {
                let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                (
                    ((yhat_ - *y_) * *w_) as f32,
                    (yhat_ * (f64::ONE - yhat_) * *w_) as f32,
                )
            })
            .unzip()
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

        ytot / ntot
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
    #[inline]
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> (Vec<f32>, Vec<f32>) {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| (((yhat_ - *y_) * *w_) as f32, *w_ as f32))
            .unzip()
    }
    fn default_metric() -> Metric {
        Metric::RootMeanSquaredLogError
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

    #[test]
    fn test_logloss_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let l1 = LogLoss::calc_init(&y, &w);
        assert!(l1 == 0.);

        let y = vec![1.0; 6];
        let l2 = LogLoss::calc_init(&y, &w);
        assert!(l2 == f64::INFINITY);

        let y = vec![0.0; 6];
        let l3 = LogLoss::calc_init(&y, &w);
        assert!(l3 == f64::NEG_INFINITY);

        let y = vec![0., 0., 0., 0., 1., 1.];
        let l4 = LogLoss::calc_init(&y, &w);
        assert!(l4 == f64::ln(2. / 4.));
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
}
