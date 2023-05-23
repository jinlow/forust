use std::str::FromStr;

use crate::{data::FloatData, errors::ForustError, metric::Metric, utils::items_to_strings};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub enum ObjectiveType {
    LogLoss,
    SquaredLoss,
}

impl FromStr for ObjectiveType {
    type Err = ForustError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LogLoss" => Ok(ObjectiveType::LogLoss),
            "SquaredLoss" => Ok(ObjectiveType::SquaredLoss),
            _ => Err(ForustError::ParseString(
                s.to_string(),
                "ObjectiveType".to_string(),
                items_to_strings(vec!["LogLoss", "SquaredLoss"]),
            )),
        }
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_grad(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn calc_hess(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32>;
    fn default_metric(&self) -> Metric;
}

#[derive(Default)]
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
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
    fn calc_grad(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
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
    fn calc_hess(&self, _: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        yhat.iter()
            .zip(sample_weight)
            .map(|(yhat_, w_)| {
                let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                (yhat_ * (f64::ONE - yhat_) * *w_) as f32
            })
            .collect()
    }
    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }
}

#[derive(Default)]
pub struct SquaredLoss {}

impl ObjectiveFunction for SquaredLoss {
    #[inline]
    fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
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
    fn calc_grad(&self, y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .zip(sample_weight)
            .map(|((y_, yhat_), w_)| ((*yhat_ - *y_) * *w_) as f32)
            .collect()
    }

    #[inline]
    fn calc_hess(&self, _: &[f64], _: &[f64], sample_weight: &[f64]) -> Vec<f32> {
        sample_weight.iter().map(|v| *v as f32).collect()
    }
    fn default_metric(&self) -> Metric {
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
        let l1 = LogLoss::default().calc_loss(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = LogLoss::default().calc_loss(&y, &yhat2, &w);
        assert!(l1.iter().sum::<f32>() < l2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let g1 = LogLoss::default().calc_grad(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let g2 = LogLoss::default().calc_grad(&y, &yhat2, &w);
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_hess() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = vec![1.; y.len()];
        let h1 = LogLoss::default().calc_hess(&y, &yhat1, &w);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let h2 = LogLoss::default().calc_hess(&y, &yhat2, &w);
        assert!(h1.iter().sum::<f32>() < h2.iter().sum::<f32>());
    }
}
