use std::collections::HashMap;

use crate::data::{Matrix, MatrixData};

struct HistogramBuilder<T> {
    // HashMap of features, the index of the feature,
    // and the unique values of that feature.
    features: HashMap<usize, T>,
}
