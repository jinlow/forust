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
use pyo3::types::PyDict;
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
        l1,
        l2,
        gamma,
        max_delta_step,
        min_leaf_weight,
        base_score,
        nbins,
        parallel,
        allow_missing_splits,
        monotone_constraints,
        subsample,
        top_rate,
        other_rate,
        colsample_bytree,
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
        log_iterations,
        force_children_to_bound_parent,
        num_classes=1,
        quantile_alpha=0.5,
    ))]
    pub fn new(
        objective_type: &str,
        iterations: usize,
        learning_rate: f32,
        max_depth: usize,
        max_leaves: usize,
        l1: f32,
        l2: f32,
        gamma: f32,
        max_delta_step: f32,
        min_leaf_weight: f32,
        base_score: f64,
        nbins: u16,
        parallel: bool,
        allow_missing_splits: bool,
        monotone_constraints: HashMap<usize, i8>,
        subsample: f32,
        top_rate: f64,
        other_rate: f64,
        colsample_bytree: f64,
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
        log_iterations: usize,
        force_children_to_bound_parent: bool,
        num_classes: usize,
        quantile_alpha: f64,
    ) -> PyResult<Self> {
        let constraints = int_map_to_constraint_map(monotone_constraints)?;
        let objective_ = if objective_type == "QuantileLoss" {
            ObjectiveType::QuantileLoss {
                alpha: quantile_alpha,
            }
        } else {
            to_value_error(serde_plain::from_str(objective_type))?
        };
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
            l1,
            l2,
            gamma,
            max_delta_step,
            min_leaf_weight,
            base_score,
            nbins,
            parallel,
            allow_missing_splits,
            Some(constraints),
            subsample,
            top_rate,
            other_rate,
            colsample_bytree,
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
            log_iterations,
            force_children_to_bound_parent,
        );
        let mut booster = to_value_error(booster)?;
        booster.num_classes = num_classes;
        Ok(GradientBooster { booster })
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

    #[setter]
    fn set_num_classes(&mut self, value: usize) -> PyResult<()> {
        if !self.booster.trees.is_empty() && self.booster.num_classes != value {
            return Err(PyValueError::new_err(
                "cannot change num_classes after fit/load; it would reinterpret the stored tree layout",
            ));
        }
        self.booster.num_classes = value;
        Ok(())
    }

    #[getter]
    fn prediction_iteration(&self) -> PyResult<Option<usize>> {
        Ok(self.booster.prediction_iteration_limit())
    }

    #[getter]
    fn num_classes(&self) -> PyResult<usize> {
        Ok(self.booster.num_classes)
    }

    #[getter]
    fn best_iteration(&self) -> PyResult<Option<usize>> {
        Ok(self.booster.best_iteration)
    }

    #[getter]
    fn base_score(&self) -> PyResult<f64> {
        Ok(self.booster.base_score)
    }

    #[getter]
    fn number_of_trees(&self) -> PyResult<usize> {
        Ok(self.booster.trees.len())
    }

    #[pyo3(signature = (flat_data, rows, cols, y, sample_weight, evaluation_data=None))]
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

    #[pyo3(signature = (flat_data, rows, cols, parallel=None))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        to_value_error(self.booster.validate_prediction_state())?;
        to_value_error(self.booster.validate_prediction_data(&data))?;
        Ok(self.booster.predict(&data, parallel).into_pyarray(py))
    }

    #[pyo3(signature = (flat_data, rows, cols, parallel=None))]
    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !matches!(
            self.booster.objective_type,
            ObjectiveType::SoftmaxMultiClass
        ) || self.booster.num_classes < 2
        {
            return Err(PyValueError::new_err(
                "predict_proba is only available when objective_type is SoftmaxMultiClass and num_classes >= 2",
            ));
        }

        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        to_value_error(self.booster.validate_prediction_state())?;
        to_value_error(self.booster.validate_prediction_data(&data))?;
        Ok(self.booster.predict_proba(&data, parallel).into_pyarray(py))
    }

    #[pyo3(signature = (flat_data, rows, cols, method, parallel=None))]
    pub fn predict_contributions<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        method: &str,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if matches!(
            self.booster.objective_type,
            ObjectiveType::SoftmaxMultiClass
        ) {
            return Err(PyValueError::new_err(
                "predict_contributions is not supported for SoftmaxMultiClass; explanations are only implemented for scalar-output objectives",
            ));
        }
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        to_value_error(self.booster.validate_prediction_state())?;
        to_value_error(self.booster.validate_prediction_data(&data))?;
        let method_ = to_value_error(serde_plain::from_str(method))?;
        Ok(self
            .booster
            .predict_contributions(&data, method_, parallel)
            .into_pyarray(py))
    }

    pub fn predict_leaf_indices<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        to_value_error(self.booster.validate_prediction_state())?;
        to_value_error(self.booster.validate_prediction_data(&data))?;
        Ok(self.booster.predict_leaf_indices(&data).into_pyarray(py))
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
        if matches!(
            self.booster.objective_type,
            ObjectiveType::SoftmaxMultiClass
        ) {
            return Err(PyValueError::new_err(
                "value_partial_dependence is not supported for SoftmaxMultiClass; partial dependence is only implemented for scalar-output objectives",
            ));
        }
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
    pub fn load_booster(_: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let booster = match CrateGradientBooster::load_booster(path.as_str()) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(GradientBooster { booster })
    }

    #[classmethod]
    pub fn from_json(_: &Bound<'_, PyType>, json_str: &str) -> PyResult<Self> {
        let booster = match CrateGradientBooster::from_json(json_str) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(GradientBooster { booster })
    }

    pub fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        let (objective_, quantile_alpha_) = match &self.booster.objective_type {
            ObjectiveType::QuantileLoss { alpha } => ("QuantileLoss".to_string(), *alpha),
            objective_type => (
                to_value_error(serde_plain::to_string::<ObjectiveType>(objective_type))?,
                0.5,
            ),
        };
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
            .unwrap_or(&HashMap::new())
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
        let dict = PyDict::new(py);
        dict.set_item("objective_type", objective_)?;
        dict.set_item("quantile_alpha", quantile_alpha_)?;
        dict.set_item("iterations", self.booster.iterations)?;
        dict.set_item("learning_rate", self.booster.learning_rate)?;
        dict.set_item("max_depth", self.booster.max_depth)?;
        dict.set_item("max_leaves", self.booster.max_leaves)?;
        dict.set_item("l1", self.booster.l1)?;
        dict.set_item("l2", self.booster.l2)?;
        dict.set_item("gamma", self.booster.gamma)?;
        dict.set_item("max_delta_step", self.booster.max_delta_step)?;
        dict.set_item("min_leaf_weight", self.booster.min_leaf_weight)?;
        dict.set_item("base_score", self.booster.base_score)?;
        dict.set_item("nbins", self.booster.nbins)?;
        dict.set_item("parallel", self.booster.parallel)?;
        dict.set_item("allow_missing_splits", self.booster.allow_missing_splits)?;
        dict.set_item("monotone_constraints", constraints)?;
        dict.set_item("subsample", self.booster.subsample)?;
        dict.set_item("top_rate", self.booster.top_rate)?;
        dict.set_item("other_rate", self.booster.other_rate)?;
        dict.set_item("colsample_bytree", self.booster.colsample_bytree)?;
        dict.set_item("seed", self.booster.seed)?;
        dict.set_item("missing", self.booster.missing)?;
        dict.set_item("create_missing_branch", self.booster.create_missing_branch)?;
        dict.set_item("sample_method", sample_method_)?;
        dict.set_item("grow_policy", grow_policy_)?;
        dict.set_item("evaluation_metric", evaluation_metric_)?;
        dict.set_item("early_stopping_rounds", self.booster.early_stopping_rounds)?;
        dict.set_item("initialize_base_score", self.booster.initialize_base_score)?;
        dict.set_item("num_classes", self.booster.num_classes)?;
        dict.set_item(
            "terminate_missing_features",
            self.booster.terminate_missing_features.clone(),
        )?;
        dict.set_item("missing_node_treatment", missing_node_treatment_)?;
        dict.set_item("log_iterations", self.booster.log_iterations)?;
        dict.set_item(
            "force_children_to_bound_parent",
            self.booster.force_children_to_bound_parent,
        )?;
        Ok(dict.into_any().unbind())
    }

    pub fn get_evaluation_history<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(usize, usize, Bound<'py, PyArray1<f64>>)>> {
        if let Some(data) = &self.booster.evaluation_history {
            let d = data.data.to_owned().into_pyarray(py);
            return Ok(Some((data.rows, data.cols, d)));
        }
        Ok(None)
    }
}

#[pymodule]
mod forust {
    use super::*;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v_ = v.as_slice()?;
        let sample_weight_ = sample_weight.as_slice()?;
        let percentiles_ = percentiles.as_slice()?;
        let p = crate_percentiles(v_, sample_weight_, percentiles_);
        Ok(p.into_pyarray(py))
    }

    #[pymodule_export]
    use super::GradientBooster;
}
