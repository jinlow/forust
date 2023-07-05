use forust_ml::constraints::{Constraint, ConstraintMap};
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::{EvaluationData, MissingNodeTreatment};
use forust_ml::gradientbooster::{GradientBooster as CrateGradientBooster, GrowPolicy};
use forust_ml::metric::Metric;
use forust_ml::objective::ObjectiveType;
use forust_ml::sampler::SampleMethod;
use forust_ml::utils::percentiles as crate_percentiles;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyType;
use std::collections::{HashMap, HashSet};

type PyEvaluationData<'a> = (
    PyReadonlyArray1<'a, f64>,
    usize,
    usize,
    PyReadonlyArray1<'a, f64>,
    PyReadonlyArray1<'a, f64>,
);

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

fn to_value_error<T, E: std::fmt::Display>(value: Result<T, E>) -> Result<T, PyErr> {
    match value {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyclass(subclass)]
struct GradientBooster {
    booster: CrateGradientBooster,
}

#[pymethods]
impl GradientBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        objective_type,
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
        monotone_constraints,
        subsample,
        top_rate,
        other_rate,
        seed,
        missing,
        create_missing_branch,
        sample_method,
        grow_policy,
        evaluation_metric,
        early_stopping_rounds,
        initialize_base_score,
        terminate_missing_features,
        missing_node_treatment,
    ))]
    pub fn new(
        objective_type: &str,
        iterations: usize,
        learning_rate: f32,
        max_depth: usize,
        max_leaves: usize,
        l2: f32,
        gamma: f32,
        min_leaf_weight: f32,
        base_score: Option<f64>,
        nbins: u16,
        parallel: bool,
        allow_missing_splits: bool,
        monotone_constraints: HashMap<usize, i8>,
        subsample: f32,
        top_rate: f64,
        other_rate: f64,
        seed: u64,
        missing: f64,
        create_missing_branch: bool,
        sample_method: Option<&str>,
        grow_policy: &str,
        evaluation_metric: Option<&str>,
        early_stopping_rounds: Option<usize>,
        initialize_base_score: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: &str,
    ) -> PyResult<Self> {
        let constraints = int_map_to_constraint_map(monotone_constraints)?;
        let objective_ = to_value_error(serde_plain::from_str(objective_type))?;
        let sample_method_ = match sample_method {
            Some(s) => to_value_error(serde_plain::from_str(s))?,
            None => SampleMethod::None,
        };
        let grow_policy_ = to_value_error(serde_plain::from_str(grow_policy))?;
        let evaluation_metric_ = match evaluation_metric {
            Some(s) => Some(to_value_error(serde_plain::from_str(s))?),
            None => None,
        };
        let missing_node_treatment_ =
            to_value_error(serde_plain::from_str(missing_node_treatment))?;
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
            subsample,
            top_rate,
            other_rate,
            seed,
            missing,
            create_missing_branch,
            sample_method_,
            grow_policy_,
            evaluation_metric_,
            early_stopping_rounds,
            initialize_base_score,
            terminate_missing_features,
            missing_node_treatment_,
        );
        Ok(GradientBooster {
            booster: to_value_error(booster)?,
        })
    }

    #[setter]
    fn set_monotone_constraints(&mut self, value: HashMap<usize, i8>) -> PyResult<()> {
        let map = int_map_to_constraint_map(value)?;
        self.booster.monotone_constraints = Some(map);
        Ok(())
    }

    #[setter]
    fn set_terminate_missing_features(&mut self, value: HashSet<usize>) -> PyResult<()> {
        self.booster.terminate_missing_features = value;
        Ok(())
    }

    #[setter]
    fn set_prediction_iteration(&mut self, value: Option<usize>) -> PyResult<()> {
        self.booster.prediction_iteration = value;
        Ok(())
    }

    #[getter]
    fn best_iteration(&self) -> PyResult<Option<usize>> {
        Ok(self.booster.best_iteration)
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        sample_weight: PyReadonlyArray1<f64>,
        evaluation_data: Option<Vec<PyEvaluationData>>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight = sample_weight.as_slice()?;

        let evaluation_data_: Option<Vec<EvaluationData>> = match evaluation_data.as_ref() {
            None => None,
            Some(values) => {
                let mut eval_data = Vec::new();
                for (a, r, c, y_, w_) in values.iter() {
                    eval_data.push((
                        Matrix::new(a.as_slice()?, *r, *c),
                        y_.as_slice()?,
                        w_.as_slice()?,
                    ));
                }
                Some(eval_data)
            }
        };
        match self.booster.fit(&data, y, sample_weight, evaluation_data_) {
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
        method: &str,
        parallel: Option<bool>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        let method_ = to_value_error(serde_plain::from_str(method))?;
        Ok(self
            .booster
            .predict_contributions(&data, method_, parallel)
            .into_pyarray(py))
    }

    pub fn calculate_feature_importance(
        &self,
        method: &str,
        normalize: bool,
    ) -> PyResult<HashMap<usize, f32>> {
        let method_ = to_value_error(serde_plain::from_str(method))?;
        Ok(self
            .booster
            .calculate_feature_importance(method_, normalize))
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

    pub fn insert_metadata(&mut self, key: String, value: String) -> PyResult<()> {
        self.booster.insert_metadata(key, value);
        Ok(())
    }

    pub fn get_metadata(&self, key: String) -> PyResult<String> {
        match self.booster.get_metadata(&key) {
            Some(m) => Ok(m),
            None => Err(PyKeyError::new_err(format!(
                "No value associated with provided key {}",
                key
            ))),
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
        let objective_ = to_value_error(serde_plain::to_string::<ObjectiveType>(
            &self.booster.objective_type,
        ))?;
        let sample_method_: Option<String> = match self.booster.sample_method {
            SampleMethod::None => None,
            _ => serde_plain::to_string::<SampleMethod>(&self.booster.sample_method).ok(),
        };
        let grow_policy_: Option<String> =
            serde_plain::to_string::<GrowPolicy>(&self.booster.grow_policy).ok();

        let evaluation_metric_ = match self.booster.evaluation_metric {
            None => None,
            Some(v) => serde_plain::to_string::<Metric>(&v).ok(),
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
        let missing_node_treatment_ = to_value_error(
            serde_plain::to_string::<MissingNodeTreatment>(&self.booster.missing_node_treatment),
        )?;
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
            ("subsample", self.booster.subsample.to_object(py)),
            ("top_rate", self.booster.top_rate.to_object(py)),
            ("other_rate", self.booster.other_rate.to_object(py)),
            ("seed", self.booster.seed.to_object(py)),
            ("missing", self.booster.missing.to_object(py)),
            (
                "create_missing_branch",
                self.booster.create_missing_branch.to_object(py),
            ),
            ("sample_method", sample_method_.to_object(py)),
            ("grow_policy", grow_policy_.to_object(py)),
            ("evaluation_metric", evaluation_metric_.to_object(py)),
            (
                "early_stopping_rounds",
                self.booster.early_stopping_rounds.to_object(py),
            ),
            (
                "initialize_base_score",
                self.booster.initialize_base_score.to_object(py),
            ),
            (
                "terminate_missing_features",
                self.booster.terminate_missing_features.to_object(py),
            ),
            (
                "missing_node_treatment",
                missing_node_treatment_.to_object(py),
            ),
        ];
        let dict = key_vals.into_py_dict(py);
        Ok(dict.to_object(py))
    }

    pub fn get_evaluation_history<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, &'py PyArray1<f64>)>> {
        if let Some(data) = &self.booster.evaluation_history {
            let d = data.data.to_owned().into_pyarray(py);
            return Ok(Some((data.rows, data.cols, d)));
        }
        Ok(None)
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
