use forust::binning::bin_matrix;
use forust::data::Matrix;
use forust::gradientbooster::GradientBooster as CrateGradientBooster;
use forust::objective::ObjectiveType;
use forust::utils::percentiles as crate_percentiles;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyType;

// This macro is used to define the base implementation of
// the booster.
macro_rules! generate_booster_pyclass {
    ($name:ident, $type:ty) => {
        #[pyclass(subclass)]
        struct $name {
            booster: CrateGradientBooster<$type>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(
                objective_type: &str,
                iterations: usize,
                learning_rate: $type,
                max_depth: usize,
                max_leaves: usize,
                l2: $type,
                gamma: $type,
                min_leaf_weight: $type,
                base_score: $type,
                nbins: u16,
                parallel: bool,
            ) -> PyResult<Self> {
                let objective_ = match objective_type {
                    "LogLoss" => Ok(ObjectiveType::LogLoss),
                    "SquaredLoss" => Ok(ObjectiveType::SquaredLoss),
                    _ => Err(PyValueError::new_err(format!("Not a valid objective type passed, expected one of 'LogLoss', 'SquaredLoss', but '{}' was provied.", objective_type))),
                }?;
                let booster = CrateGradientBooster::<$type>::new(
                    objective_,
                    iterations,
                    learning_rate,
                    max_depth,
                    max_leaves,
                    l2,
                    gamma,
                    min_leaf_weight,
                    base_score,
                    nbins,
                    parallel,
                );
                Ok($name { booster })
            }

            pub fn fit(
                &mut self,
                flat_data: PyReadonlyArray1<$type>,
                rows: usize,
                cols: usize,
                y: PyReadonlyArray1<$type>,
                sample_weight: PyReadonlyArray1<$type>,
                parallel: Option<bool>,
            ) -> PyResult<()> {
                let flat_data = flat_data.as_slice()?;
                let data = Matrix::new(flat_data, rows, cols);
                let y = y.as_slice()?;
                let sample_weight = sample_weight.as_slice()?;
                let parallel = match parallel {
                    None => true,
                    Some(v) => v,
                };
                self.booster
                    .fit(&data, &y, &sample_weight, parallel)
                    .unwrap();
                Ok(())
            }
            pub fn predict<'py>(
                &self,
                py: Python<'py>,
                flat_data: PyReadonlyArray1<$type>,
                rows: usize,
                cols: usize,
                parallel: Option<bool>,
            ) -> PyResult<&'py PyArray1<$type>> {
                let flat_data = flat_data.as_slice()?;
                let data = Matrix::new(flat_data, rows, cols);
                let parallel = match parallel {
                    None => true,
                    Some(v) => v,
                };
                Ok(self.booster.predict(&data, parallel).into_pyarray(py))
            }

            pub fn text_dump(&self) -> PyResult<Vec<String>> {
                let mut trees = Vec::new();
                for t in &self.booster.trees {
                    trees.push(format!("{}", t));
                }
                return Ok(trees);
            }

            pub fn save_booster(&self, path: &str) -> PyResult<()> {
                match self.booster.save_booster(path) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }
            }

            pub fn json_dump(&self) -> PyResult<String> {
                match self.booster.json_dump() {
                    Ok(m) => Ok(m),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }
            }

            #[classmethod]
            pub fn load_booster(_: &PyType, path: String) -> PyResult<Self> {
                let booster = match CrateGradientBooster::load_booster(path.as_str()) {
                    Ok(m) => Ok(m),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }?;
                Ok($name { booster })
            }

            #[classmethod]
            pub fn from_json(_: &PyType, json_str: &str) -> PyResult<Self> {
                let booster = match CrateGradientBooster::from_json(json_str) {
                    Ok(m) => Ok(m),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }?;
                Ok($name { booster })
            }

            pub fn get_params<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
                let objective_ = match self.booster.objective_type {
                    ObjectiveType::LogLoss => "LogLoss",
                    ObjectiveType::SquaredLoss => "SquaredLoss",
                };
                let key_vals: Vec<(&str, PyObject)> = vec![
                    ("objective_type", objective_.to_object(py)),
                    ("iterations", self.booster.iterations.to_object(py)),
                    ("learning_rate", self.booster.learning_rate.to_object(py)),
                    ("max_depth", self.booster.max_depth.to_object(py)),
                    ("max_leaves", self.booster.max_leaves.to_object(py)),
                    ("l2", self.booster.l2.to_object(py)),
                    ("gamma", self.booster.gamma.to_object(py)),
                    (
                        "min_leaf_weight",
                        self.booster.min_leaf_weight.to_object(py),
                    ),
                    ("base_score", self.booster.base_score.to_object(py)),
                    ("nbins", self.booster.nbins.to_object(py)),
                    ("parallel", self.booster.parallel.to_object(py)),
                    ("dtype", self.booster.dtype.to_object(py))
                ];
                let dict = key_vals.into_py_dict(py);
                Ok(dict.to_object(py))
            }
        }
    };
}

generate_booster_pyclass!(GradientBoosterF32, f32);
generate_booster_pyclass!(GradientBoosterF64, f64);

#[pyfunction]
fn rust_bin_matrix<'py>(
    py: Python<'py>,
    flat_data: PyReadonlyArray1<f32>,
    rows: usize,
    cols: usize,
    sample_weight: PyReadonlyArray1<f32>,
    nbins: u16,
) -> PyResult<(&'py PyArray1<u16>, Vec<Vec<f32>>, Vec<usize>)> {
    let flat_data = flat_data.as_slice()?;
    let sample_weight = sample_weight.as_slice()?;
    let data = Matrix::new(flat_data, rows, cols);
    let r = bin_matrix(&data, sample_weight, nbins).unwrap();
    Ok((r.binned_data.into_pyarray(py), r.cuts, r.nunique))
}

#[pyfunction]
fn print_matrix(x: PyReadonlyArray1<f32>, rows: usize, cols: usize) -> PyResult<()> {
    let m = Matrix::new(x.as_slice()?, rows, cols);
    println!("{}", m);
    Ok(())
}

#[pyfunction]
fn percentiles<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<f32>,
    sample_weight: PyReadonlyArray1<f32>,
    percentiles: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let v_ = v.as_slice()?;
    let sample_weight_ = sample_weight.as_slice()?;
    let percentiles_ = percentiles.as_slice()?;
    let p = crate_percentiles(v_, sample_weight_, percentiles_);
    Ok(p.into_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn forust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(rust_bin_matrix, m)?)?;
    m.add_class::<GradientBoosterF32>()?;
    m.add_class::<GradientBoosterF64>()?;
    Ok(())
}
