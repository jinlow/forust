use crate::data::MatrixData;

pub trait ObjectiveFunction<T>
where
    T: MatrixData<T>,
{
    fn calc_loss(y: &[T], yhat: &[T]) -> Vec<T>;
    fn calc_grad(y: &[T], yhat: &[T]) -> Vec<T>;
    fn calc_hess(y: &[T], yhat: &[T]) -> Vec<T>;
}

pub struct LogLoss {}

impl<T> ObjectiveFunction<T> for LogLoss
where
    T: MatrixData<T>,
{
    fn calc_loss(y: &[T], yhat: &[T]) -> Vec<T> {
        y.iter()
            .zip(yhat)
            .map(|(y_, yhat_)| {
                let yhat_ = T::one() / (T::one() + (-*yhat_).exp());
                -(*y_ * yhat_.ln() + (T::one() - *y_) * ((T::one() - yhat_).ln()))
            })
            .collect()
    }

    fn calc_grad(y: &[T], yhat: &[T]) -> Vec<T> {
        y.iter()
            .zip(yhat)
            .map(|(y_, yhat_)| {
                let yhat_ = T::one() / (T::one() + (-*yhat_).exp());
                yhat_ - *y_
            })
            .collect()
    }

    fn calc_hess(_: &[T], yhat: &[T]) -> Vec<T> {
        yhat.iter()
            .map(|yhat_| {
                let yhat_ = T::one() / (T::one() + (-*yhat_).exp());
                yhat_ * (T::one() - yhat_)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_logloss_loss() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let l1 = LogLoss::calc_loss(&y, &yhat1);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = LogLoss::calc_loss(&y, &yhat2);
        assert!(l1.iter().sum::<f64>() < l2.iter().sum::<f64>());
    }

    #[test]
    fn test_logloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let g1 = LogLoss::calc_grad(&y, &yhat1);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let g2 = LogLoss::calc_grad(&y, &yhat2);
        assert!(g1.iter().sum::<f64>() < g2.iter().sum::<f64>());
    }

    #[test]
    fn test_logloss_hess() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let h1 = LogLoss::calc_hess(&y, &yhat1);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let h2 = LogLoss::calc_hess(&y, &yhat2);
        assert!(h1.iter().sum::<f64>() < h2.iter().sum::<f64>());
    }
}
