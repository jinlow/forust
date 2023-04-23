use forust_ml::constraints::{Constraint, ConstraintMap};
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster as CrateGradientBooster;
use forust_ml::objective::ObjectiveType;
use forust_ml::utils::percentiles as crate_percentiles;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyType;
use std::collections::HashMap;

fn int_map_to_constraint_map(int_map: HashMap<usize, i8>) -> PyResult<ConstraintMap> {
    let mut constraints: ConstraintMap = HashMap::new();
    for (f, c) in int_map.iter() {
        let c_ = match c {
                -1 => Ok(Constraint::Negative),
                1 => Ok(Constraint::Positive),
                0 => Ok(Constraint::Unconstrained),
                _ => Err(PyValueError::new_err(format!("Valid monotone constraints are -1, 1 or 0, but '{}' was provided for feature number {}.", c, f))),
            }?;
        constraints.insert(*f, c_);
    }
    Ok(constraints)
}

// This macro is used to define the base implementation of
// the booster.
#[pyclass(subclass)]
struct GradientBooster {
    booster: CrateGradientBooster,
}

#[pymethods]
impl GradientBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        objective_type: &str,
        iterations: usize,
        learning_rate: f32,
        max_depth: usize,
        max_leaves: usize,
        l2: f32,
        gamma: f32,
        min_leaf_weight: f32,
        base_score: f64,
        nbins: u16,
        parallel: bool,
        allow_missing_splits: bool,
        monotone_constraints: HashMap<usize, i8>,
    ) -> PyResult<Self> {
        let constraints = int_map_to_constraint_map(monotone_constraints)?;
        let objective_ = match objective_type {
            "LogLoss" => Ok(ObjectiveType::LogLoss),
            "SquaredLoss" => Ok(ObjectiveType::SquaredLoss),
            _ => Err(PyValueError::new_err(format!("Not a valid objective type passed, expected one of 'LogLoss', 'SquaredLoss', but '{}' was provided.", objective_type))),
        }?;
        let booster = CrateGradientBooster::new(
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
            allow_missing_splits,
            Some(constraints),
        );
        Ok(GradientBooster { booster })
    }

    #[setter]
    fn set_monotone_constraints(&mut self, value: HashMap<usize, i8>) -> PyResult<()> {
        let map = int_map_to_constraint_map(value)?;
        self.booster.monotone_constraints = Some(map);
        Ok(())
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        sample_weight: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight = sample_weight.as_slice()?;
        match self.booster.fit(&data, y, sample_weight) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(())
    }
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        Ok(self.booster.predict(&data, parallel).into_pyarray(py))
    }
    pub fn predict_contributions<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        Ok(self
            .booster
            .predict_contributions(&data, parallel)
            .into_pyarray(py))
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> PyResult<f64> {
        Ok(self.booster.value_partial_dependence(feature, value))
    }

    pub fn text_dump(&self) -> PyResult<Vec<String>> {
        let mut trees = Vec::new();
        for t in &self.booster.trees {
            trees.push(format!("{}", t));
        }
        Ok(trees)
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
        Ok(GradientBooster { booster })
    }

    #[classmethod]
    pub fn from_json(_: &PyType, json_str: &str) -> PyResult<Self> {
        let booster = match CrateGradientBooster::from_json(json_str) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(GradientBooster { booster })
    }

    pub fn get_params(&self, py: Python) -> PyResult<PyObject> {
        let objective_ = match self.booster.objective_type {
            ObjectiveType::LogLoss => "LogLoss",
            ObjectiveType::SquaredLoss => "SquaredLoss",
        };
        let constraints: HashMap<usize, i8> = self
            .booster
            .monotone_constraints
            .as_ref()
            .unwrap()
            .iter()
            .map(|(f, c)| {
                let c_ = match c {
                    Constraint::Negative => -1,
                    Constraint::Positive => 1,
                    Constraint::Unconstrained => 0,
                };
                (*f, c_)
            })
            .collect();
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
            (
                "allow_missing_splits",
                self.booster.allow_missing_splits.to_object(py),
            ),
            ("monotone_constraints", constraints.to_object(py)),
        ];
        let dict = key_vals.into_py_dict(py);
        Ok(dict.to_object(py))
    }
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
    v: PyReadonlyArray1<f64>,
    sample_weight: PyReadonlyArray1<f64>,
    percentiles: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
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
    m.add_class::<GradientBooster>()?;
    Ok(())
}
