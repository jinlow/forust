use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForustError {
    #[error("Feature provided with no variance, when missing values are excluded.")]
    NoVariance,
    #[error("Unable to write model to file: {0}")]
    UnableToWrite(String),
    #[error("Unable to read model from a file {0}")]
    UnableToRead(String),
    // #[error("NaN found in {0}.")]
    // ContainsNaN(String),
    // #[error("Unable to calculate prediction.")]
    // Prediction,
}
