use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForustError {
    #[error("Feature provided with no variance, when missing values are excluded.")]
    NoVariance,
    // #[error("NaN found in {0}.")]
    // ContainsNaN(String),
    // #[error("Unable to calculate prediction.")]
    // Prediction,
}
