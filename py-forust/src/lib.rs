use numpy::{PyReadonlyArray1};
use forust::data::Matrix;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn print_matrix(x: PyReadonlyArray1<f64>, rows: usize, cols: usize) -> PyResult<()> {
    let m = Matrix::new(x.as_slice()?, rows, cols);
    println!("{}", m);
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn forust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_matrix, m)?)?;
    Ok(())
}