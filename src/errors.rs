use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForustError {
    #[error("Feature number {0} has no variance, when missing values are excluded.")]
    NoVariance(usize),
    #[error("Unable to write model to file: {0}")]
    UnableToWrite(String),
    #[error("Unable to read model from a file {0}")]
    UnableToRead(String),
    #[error("The value {0} is set to missing, but a NaN value was found in the data.")]
    NANVAlueFound(f64),
    // #[error("NaN found in {0}.")]
    // ContainsNaN(String),
    // #[error("Unable to calculate prediction.")]
    // Prediction,
}
