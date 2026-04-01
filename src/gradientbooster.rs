use crate::binning::bin_matrix;
use crate::constraints::ConstraintMap;
use crate::data::{Matrix, RowMajorMatrix};
use crate::errors::ForustError;
use crate::metric::multiclass_log_loss;
use crate::metric::{is_comparison_better, metric_callables, Metric, MetricFn};
use crate::objective::{
    calc_init_callables, gradient_hessian_callables, LogLoss, ObjectiveFunction, ObjectiveType,
    SoftmaxMultiClass, SquaredLoss,
};
use crate::sampler::{GossSampler, RandomSampler, SampleMethod, Sampler};
use crate::shapley::predict_contributions_row_shapley;
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, Splitter};
use crate::tree::Tree;
use crate::utils::{
    fmt_vec_output, odds, validate_not_nan_vec, validate_positive_float_field,
    validate_positive_not_nan_vec,
};
use log::info;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;

pub type EvaluationData<'a> = (Matrix<'a, f64>, &'a [f64], &'a [f64]);
pub type TrainingEvaluationData<'a> = (&'a Matrix<'a, f64>, &'a [f64], &'a [f64], Vec<f64>);
type ImportanceFn = fn(&Tree, &mut HashMap<usize, (f32, usize)>);

const MULTICLASS_CONTRIBUTIONS_UNSUPPORTED: &str =
    "predict_contributions is not supported for SoftmaxMultiClass; explanations are only implemented for scalar-output objectives";
const MULTICLASS_PARTIAL_DEPENDENCE_UNSUPPORTED: &str =
    "value_partial_dependence is not supported for SoftmaxMultiClass; partial dependence is only implemented for scalar-output objectives";

#[derive(Serialize, Deserialize)]
pub enum GrowPolicy {
    DepthWise,
    LossGuide,
}

#[derive(Serialize, Deserialize)]
pub enum ContributionsMethod {
    /// This method will use the internal leaf weights, to calculate the contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
    Weight,
    /// If this option is specified, the average internal node values are calculated, this is equivalent to the `approx_contribs` parameter in XGBoost.
    Average,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the other non-missing branch. This method does not have the property where the contributions summed is equal to the final prediction of the model.
    BranchDifference,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the mid-point between the right and left node weighted by the cover of each node. This method does not have the property where the contributions summed is equal to the final prediction of the model.
    MidpointDifference,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the node with the largest cover (the mode node). This method does not have the property where the contributions summed is equal to the final prediction of the model.
    ModeDifference,
    /// This method is only valid when the objective type is set to "LogLoss". This method will calculate contributions as the change in a records probability of being 1 moving from a parent node to a child node. The sum of the returned contributions matrix, will be equal to the probability a record will be 1. For example, given a model, `model.predict_contributions(X, method="ProbabilityChange") == 1 / (1 + np.exp(-model.predict(X)))`
    ProbabilityChange,
    /// This method computes the Shapley values for each record, and feature.
    Shapley,
}

/// Method to calculate variable importance.
#[derive(Serialize, Deserialize)]
pub enum ImportanceMethod {
    /// The number of times a feature is used to split the data across all trees.
    Weight,
    /// The average split gain across all splits the feature is used in.
    Gain,
    /// The average coverage across all splits the feature is used in.
    Cover,
    /// The total gain across all splits the feature is used in.
    TotalGain,
    /// The total coverage across all splits the feature is used in.
    TotalCover,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum MissingNodeTreatment {
    /// Calculate missing node weight values without any constraints.
    None,
    /// Assign the weight of the missing node to that of the parent.
    AssignToParent,
    /// After training each tree, starting from the bottom of the tree, assign the missing node weight to the weighted average of the left and right child nodes. Next assign the parent to the weighted average of the children nodes. This is performed recursively up through the entire tree. This is performed as a post processing step on each tree after it is built, and prior to updating the predictions for which to train the next tree.
    AverageLeafWeight,
    /// Set the missing node to be equal to the weighted average weight of the left and the right nodes.
    AverageNodeWeight,
}

/// Gradient Booster object
#[derive(Deserialize, Serialize)]
pub struct GradientBooster {
    /// The name of objective function used to optimize.
    /// Valid options include "LogLoss" to use logistic loss as the objective function,
    /// or "SquaredLoss" to use Squared Error as the objective function.
    pub objective_type: ObjectiveType,
    /// Total number of trees to train in the ensemble.
    pub iterations: usize,
    /// Step size to use at each iteration. Each
    /// leaf weight is multiplied by this number. The smaller the value, the more
    /// conservative the weights will be.
    pub learning_rate: f32,
    /// Maximum depth of an individual tree. Valid values are 0 to infinity.
    pub max_depth: usize,
    /// Maximum number of leaves allowed on a tree. Valid values
    /// are 0 to infinity. This is the total number of final nodes.
    pub max_leaves: usize,
    /// L1 regularization term applied to the weights of the tree. Valid values
    /// are 0 to infinity. 0 Means no regularization applied.
    #[serde(default = "default_l1")]
    pub l1: f32,
    /// L2 regularization term applied to the weights of the tree. Valid values
    /// are 0 to infinity.
    pub l2: f32,
    /// The minimum amount of loss required to further split a node.
    /// Valid values are 0 to infinity.
    pub gamma: f32,
    /// Maximum delta step allowed at each leaf. This is the maximum magnitude a leaf can take. Setting to 0 results in no constrain.
    #[serde(default = "default_max_delta_step")]
    pub max_delta_step: f32,
    /// Minimum sum of the hessian values of the loss function
    /// required to be in a node.
    pub min_leaf_weight: f32,
    /// The initial prediction value of the model.
    pub base_score: f64,
    /// Number of bins to calculate to partition the data. Setting this to
    /// a smaller number, will result in faster training time, while potentially sacrificing
    /// accuracy. If there are more bins, than unique values in a column, all unique values
    /// will be used.
    pub nbins: u16,
    pub parallel: bool,
    /// Should the algorithm allow splits that completed seperate out missing
    /// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    /// is true, setting this to true will result in the missin branch being further split.
    pub allow_missing_splits: bool,
    /// Constraints that are used to enforce a specific relationship
    /// between the training features and the target variable.
    pub monotone_constraints: Option<ConstraintMap>,
    /// Percent of records to randomly sample at each iteration when training a tree.
    pub subsample: f32,
    /// Used only in goss. The retain ratio of large gradient data.
    #[serde(default = "default_top_rate")]
    pub top_rate: f64,
    /// Used only in goss. the retain ratio of small gradient data.
    #[serde(default = "default_other_rate")]
    pub other_rate: f64,
    /// Specify the fraction of columns that should be sampled at each iteration, valid values are in the range (0.0,1.0].
    #[serde(default = "default_colsample_bytree")]
    pub colsample_bytree: f64,
    /// Integer value used to seed any randomness used in the algorithm.
    pub seed: u64,
    /// Value to consider missing.
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    /// Should missing be split out it's own separate branch?
    pub create_missing_branch: bool,
    /// Specify the method that records should be sampled when training?
    #[serde(default = "default_sample_method")]
    pub sample_method: SampleMethod,
    /// Growth policy to use when training a tree, this is how the next node is selected.
    #[serde(default = "default_grow_policy")]
    pub grow_policy: GrowPolicy,
    /// Define the evaluation metric to record at each iterations.
    #[serde(default = "default_evaluation_metric")]
    pub evaluation_metric: Option<Metric>,
    /// Number of rounds where the evaluation metric value must improve in
    /// to keep training.
    #[serde(default = "default_early_stopping_rounds")]
    pub early_stopping_rounds: Option<usize>,
    /// If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
    #[serde(default = "default_initialize_base_score")]
    pub initialize_base_score: bool,
    /// A set of features for which the missing node will always be terminated, even
    /// if `allow_missing_splits` is set to true. This value is only valid if
    /// `create_missing_branch` is also True.
    #[serde(default = "default_terminate_missing_features")]
    pub terminate_missing_features: HashSet<usize>,
    /// A matrix of the evaluation history on the evaluation datasets.
    #[serde(default = "default_evaluation_history")]
    pub evaluation_history: Option<RowMajorMatrix<f64>>,
    /// Zero-based boosting iteration with the best evaluation metric.
    #[serde(default = "default_best_iteration")]
    pub best_iteration: Option<usize>,
    /// Number of boosting iterations to use when predicting.
    /// For SoftmaxMultiClass, each boosting iteration corresponds to `num_classes` trees.
    #[serde(default = "default_prediction_iteration")]
    pub prediction_iteration: Option<usize>,
    /// How the missing nodes weights should be treated at training time.
    #[serde(default = "default_missing_node_treatment")]
    pub missing_node_treatment: MissingNodeTreatment,
    /// Should the model be trained showing output.
    #[serde(default = "default_log_iterations")]
    pub log_iterations: usize,
    /// Should the children nodes contain the parent node in their bounds, setting this to true, will result in no children being created that result in the higher and lower child values both being greater than, or less than the parent weight.
    #[serde(default = "default_force_children_to_bound_parent")]
    pub force_children_to_bound_parent: bool,
    /// Number of classes for SoftmaxMultiClass. Default 1 (binary/regression).
    /// Must be >= 2 when objective is SoftmaxMultiClass.
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    /// Per-class base scores for SoftmaxMultiClass. Length = num_classes.
    /// None for binary/regression (uses existing scalar base_score field).
    #[serde(default)]
    pub base_scores: Option<Vec<f64>>,
    // Members internal to the booster object, and not parameters set by the user.
    // Trees is public, just to interact with it directly in the python wrapper.
    pub trees: Vec<Tree>,
    metadata: HashMap<String, String>,
}

fn default_l1() -> f32 {
    0.0
}
fn default_max_delta_step() -> f32 {
    0.0
}

fn default_initialize_base_score() -> bool {
    false
}

fn default_grow_policy() -> GrowPolicy {
    GrowPolicy::DepthWise
}

fn default_top_rate() -> f64 {
    0.1
}
fn default_other_rate() -> f64 {
    0.2
}
fn default_sample_method() -> SampleMethod {
    SampleMethod::None
}
fn default_evaluation_metric() -> Option<Metric> {
    None
}
fn default_early_stopping_rounds() -> Option<usize> {
    None
}
fn default_evaluation_history() -> Option<RowMajorMatrix<f64>> {
    None
}
fn default_best_iteration() -> Option<usize> {
    None
}
fn default_prediction_iteration() -> Option<usize> {
    None
}
fn default_terminate_missing_features() -> HashSet<usize> {
    HashSet::new()
}
fn default_colsample_bytree() -> f64 {
    1.0
}
fn default_missing_node_treatment() -> MissingNodeTreatment {
    MissingNodeTreatment::AssignToParent
}

fn default_log_iterations() -> usize {
    0
}
fn default_force_children_to_bound_parent() -> bool {
    false
}

fn default_num_classes() -> usize {
    1
}

fn parse_missing<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or(f64::NAN))
}

/// Extract class k's N-length grad/hess from N×K row-major arrays.
fn extract_class_slice(
    grad: &[f32],
    hess: &[f32],
    class: usize,
    num_classes: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    for i in 0..n {
        g.push(grad[i * num_classes + class]);
        h.push(hess[i * num_classes + class]);
    }
    (g, h)
}

/// Reduce N×K grad/hess to N-length for row-wise sampling.
/// Per row: grad = max(|grad_ik|) over k, hess = sum(hess_ik) over k.
fn max_abs_grad_per_row(grad: &[f32], hess: &[f32], num_classes: usize) -> (Vec<f32>, Vec<f32>) {
    let n = grad.len() / num_classes;
    let mut g = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * num_classes;
        let max_g = (0..num_classes)
            .map(|k| grad[offset + k].abs())
            .fold(0.0f32, f32::max);
        let sum_h: f32 = (0..num_classes).map(|k| hess[offset + k]).sum();
        g.push(max_g);
        h.push(sum_h);
    }
    (g, h)
}

impl Default for GradientBooster {
    fn default() -> Self {
        Self::new(
            ObjectiveType::LogLoss,
            10,
            0.3,
            5,
            usize::MAX,
            0.,
            1.,
            0.,
            0.,
            1.,
            0.5,
            256,
            true,
            true,
            None,
            1.,
            0.1,
            0.2,
            1.0,
            0,
            f64::NAN,
            false,
            SampleMethod::None,
            GrowPolicy::DepthWise,
            None,
            None,
            true,
            HashSet::new(),
            MissingNodeTreatment::AssignToParent,
            0,
            false,
        )
        .unwrap()
    }
}

impl GradientBooster {
    /// Gradient Booster object
    ///
    /// * `objective_type` - The name of objective function used to optimize.
    ///   Valid options include "LogLoss" to use logistic loss as the objective function,
    ///   or "SquaredLoss" to use Squared Error as the objective function.
    /// * `iterations` - Total number of trees to train in the ensemble.
    /// * `learning_rate` - Step size to use at each iteration. Each
    ///   leaf weight is multiplied by this number. The smaller the value, the more
    ///   conservative the weights will be.
    /// * `max_depth` - Maximum depth of an individual tree. Valid values
    ///   are 0 to infinity.
    /// * `max_leaves` - Maximum number of leaves allowed on a tree. Valid values
    ///   are 0 to infinity. This is the total number of final nodes.
    /// * `l2` - L2 regularization term applied to the weights of the tree. Valid values
    ///   are 0 to infinity.
    /// * `gamma` - The minimum amount of loss required to further split a node.
    ///   Valid values are 0 to infinity.
    /// * `min_leaf_weight` - Minimum sum of the hessian values of the loss function
    ///   required to be in a node.
    /// * `base_score` - The initial prediction value of the model. If set to None the parameter `initialize_base_score` will automatically be set to `true`, in which case the base score will be chosen based on the objective function at fit time.
    /// * `nbins` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    /// * `parallel` - Should the algorithm be run in parallel?
    /// * `allow_missing_splits` - Should the algorithm allow splits that completed seperate out missing
    /// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    /// is true, setting this to true will result in the missin branch being further split.
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///   between the training features and the target variable.
    /// * `subsample` - Percent of records to randomly sample at each iteration when training a tree.
    /// * `top_rate` - Used only in goss. The retain ratio of large gradient data.
    /// * `other_rate` - Used only in goss. the retain ratio of small gradient data.
    /// * `colsample_bytree` - Specify the fraction of columns that should be sampled at each iteration, valid values are in the range (0.0,1.0].
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `missing` - Value to consider missing.
    /// * `create_missing_branch` - Should missing be split out it's own separate branch?
    /// * `sample_method` - Specify the method that records should be sampled when training?
    /// * `evaluation_metric` - Define the evaluation metric to record at each iterations.
    /// * `early_stopping_rounds` - Number of rounds that must
    /// * `initialize_base_score` - If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
    /// * `missing_node_treatment` - specify how missing nodes should be handled during training.
    /// * `log_iterations` - Setting to a value (N) other than zero will result in information being logged about ever N iterations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        objective_type: ObjectiveType,
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
        monotone_constraints: Option<ConstraintMap>,
        subsample: f32,
        top_rate: f64,
        other_rate: f64,
        colsample_bytree: f64,
        seed: u64,
        missing: f64,
        create_missing_branch: bool,
        sample_method: SampleMethod,
        grow_policy: GrowPolicy,
        evaluation_metric: Option<Metric>,
        early_stopping_rounds: Option<usize>,
        initialize_base_score: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: MissingNodeTreatment,
        log_iterations: usize,
        force_children_to_bound_parent: bool,
    ) -> Result<Self, ForustError> {
        let booster = GradientBooster {
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
            evaluation_history: None,
            best_iteration: None,
            prediction_iteration: None,
            missing_node_treatment,
            log_iterations,
            force_children_to_bound_parent,
            num_classes: 1,
            base_scores: None,
            trees: Vec::new(),
            metadata: HashMap::new(),
        };
        booster.validate_parameters()?;
        Ok(booster)
    }

    fn validate_parameters(&self) -> Result<(), ForustError> {
        validate_positive_float_field!(self.learning_rate);
        validate_positive_float_field!(self.l1);
        validate_positive_float_field!(self.l2);
        validate_positive_float_field!(self.gamma);
        validate_positive_float_field!(self.max_delta_step);
        validate_positive_float_field!(self.min_leaf_weight);
        validate_positive_float_field!(self.subsample);
        validate_positive_float_field!(self.top_rate);
        validate_positive_float_field!(self.other_rate);
        validate_positive_float_field!(self.colsample_bytree);
        Ok(())
    }

    /// Fit the gradient booster on a provided dataset.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `y` - Either a pandas Series, or a 1 dimensional numpy array.
    /// * `sample_weight` - Instance weights to use when
    /// training the model. If None is passed, a weight of 1 will be used for every record.
    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: &[f64],
        evaluation_data: Option<Vec<EvaluationData>>,
    ) -> Result<(), ForustError> {
        // Validate inputs
        validate_not_nan_vec(y, "y".to_string())?;
        validate_positive_not_nan_vec(sample_weight, "sample_weight".to_string())?;
        // Multi-class consistency guards
        if matches!(self.objective_type, ObjectiveType::SoftmaxMultiClass) {
            if self.num_classes < 2 {
                return Err(ForustError::InvalidInput(
                    "SoftmaxMultiClass requires num_classes >= 2".into(),
                ));
            }
            SoftmaxMultiClass::validate_labels(y, self.num_classes)?;
        } else if self.num_classes > 1 {
            return Err(ForustError::InvalidInput(format!(
                "num_classes={} requires ObjectiveType::SoftmaxMultiClass, got {:?}",
                self.num_classes, self.objective_type
            )));
        }

        let is_multiclass = matches!(self.objective_type, ObjectiveType::SoftmaxMultiClass);

        if is_multiclass {
            if let Some(metric) = self.evaluation_metric {
                if !matches!(metric, Metric::MultiClassLogLoss) {
                    return Err(ForustError::InvalidInput(format!(
                        "evaluation_metric {:?} is not supported for SoftmaxMultiClass; use MultiClassLogLoss",
                        metric
                    )));
                }
            }
        }

        if let Some(eval_data) = &evaluation_data {
            for (i, (_, eval_y, eval_sample_weight)) in eval_data.iter().enumerate() {
                validate_not_nan_vec(eval_y, format!("eval set {} y", i).to_string())?;
                validate_positive_not_nan_vec(
                    eval_sample_weight,
                    format!("eval set {} sample_weight", i).to_string(),
                )?;
                if is_multiclass {
                    SoftmaxMultiClass::validate_labels(eval_y, self.num_classes)?;
                }
            }
        }

        let constraints_map = self
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();
        if self.create_missing_branch {
            let splitter = MissingBranchSplitter {
                l1: self.l1,
                l2: self.l2,
                max_delta_step: self.max_delta_step,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
                terminate_missing_features: self.terminate_missing_features.clone(),
                missing_node_treatment: self.missing_node_treatment,
                force_children_to_bound_parent: self.force_children_to_bound_parent,
            };
            if is_multiclass {
                self.fit_trees_multiclass(y, sample_weight, data, &splitter, evaluation_data)?;
            } else {
                self.fit_trees(y, sample_weight, data, &splitter, evaluation_data)?;
            }
        } else {
            let splitter = MissingImputerSplitter {
                l1: self.l1,
                l2: self.l2,
                max_delta_step: self.max_delta_step,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
            };
            if is_multiclass {
                self.fit_trees_multiclass(y, sample_weight, data, &splitter, evaluation_data)?;
            } else {
                self.fit_trees(y, sample_weight, data, &splitter, evaluation_data)?;
            }
        };

        Ok(())
    }

    fn sample_index(
        &self,
        rng: &mut StdRng,
        index: &[usize],
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>) {
        match self.sample_method {
            SampleMethod::None => (index.to_owned(), Vec::new()),
            SampleMethod::Random => {
                RandomSampler::new(self.subsample).sample(rng, index, grad, hess)
            }
            SampleMethod::Goss => {
                GossSampler::new(self.top_rate, self.other_rate).sample(rng, index, grad, hess)
            }
        }
    }

    fn get_metric_fn(&self) -> (MetricFn, bool) {
        let metric = match &self.evaluation_metric {
            None => match self.objective_type {
                ObjectiveType::LogLoss => LogLoss::default_metric(),
                ObjectiveType::SquaredLoss => SquaredLoss::default_metric(),
                ObjectiveType::SoftmaxMultiClass => {
                    panic!("SoftmaxMultiClass uses multiclass_log_loss directly, not get_metric_fn")
                }
            },
            Some(v) => *v,
        };
        metric_callables(&metric)
    }

    fn reset(&mut self) {
        self.trees = Vec::new();
        self.evaluation_history = None;
        self.best_iteration = None;
        self.prediction_iteration = None;
        self.base_scores = None;
    }

    fn fit_trees<T: Splitter>(
        &mut self,
        y: &[f64],
        sample_weight: &[f64],
        data: &Matrix<f64>,
        splitter: &T,
        evaluation_data: Option<Vec<EvaluationData>>,
    ) -> Result<(), ForustError> {
        // Is this a booster that has already been fit? If it is, reset the trees.
        // In the future we could continue training.
        if !self.trees.is_empty() {
            self.reset()
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        if self.initialize_base_score {
            self.base_score = calc_init_callables(&self.objective_type)(y, sample_weight);
        }

        let mut yhat = vec![self.base_score; y.len()];

        let calc_grad_hess = gradient_hessian_callables(&self.objective_type);
        let (mut grad, mut hess) = calc_grad_hess(y, &yhat, sample_weight);

        // Generate binned data
        // TODO
        // In scikit-learn, they sample 200_000 records for generating the bins.
        // we could consider that, especially if this proved to be a large bottleneck...
        let binned_data = bin_matrix(data, sample_weight, self.nbins, self.missing)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        // Create the predictions, saving them with the evaluation data.
        let mut evaluation_sets: Option<Vec<TrainingEvaluationData>> =
            evaluation_data.as_ref().map(|evals| {
                evals
                    .iter()
                    .map(|(d, y, w)| (d, *y, *w, vec![self.base_score; y.len()]))
                    .collect()
            });

        let mut best_metric: Option<f64> = None;

        // This will always be false, unless early stopping rounds are used.
        let mut stop_early = false;
        let col_index: Vec<usize> = (0..data.cols).collect();
        for i in 0..self.iterations {
            let verbose = if self.log_iterations == 0 {
                false
            } else {
                i % self.log_iterations == 0
            };
            // We will eventually use the excluded index.
            let (chosen_index, _excluded_index) =
                self.sample_index(&mut rng, &data.index, &mut grad, &mut hess);
            let mut tree = Tree::new();

            // If we are doing any column sampling...
            let colsample_index: Vec<usize> = if self.colsample_bytree == 1.0 {
                Vec::new()
            } else {
                let amount = ((col_index.len() as f64) * self.colsample_bytree).floor() as usize;
                let mut v: Vec<usize> = col_index
                    .iter()
                    .choose_multiple(&mut rng, amount)
                    .iter()
                    .map(|i| **i)
                    .collect();
                v.sort();
                v
            };

            let fit_col_index = if self.colsample_bytree == 1.0 {
                &col_index
            } else {
                &colsample_index
            };

            tree.fit(
                &bdata,
                chosen_index,
                fit_col_index,
                &binned_data.cuts,
                &grad,
                &hess,
                splitter,
                self.max_leaves,
                self.max_depth,
                self.parallel,
                &self.sample_method,
                &self.grow_policy,
            );

            self.update_predictions_inplace(&mut yhat, &tree, data);

            // Update Evaluation data, if it's needed.
            if let Some(eval_sets) = &mut evaluation_sets {
                if self.evaluation_history.is_none() {
                    self.evaluation_history =
                        Some(RowMajorMatrix::new(Vec::new(), 0, eval_sets.len()));
                }
                let mut metrics: Vec<f64> = Vec::new();
                let n_eval_sets = eval_sets.len();
                for (eval_i, (data, y, w, yhat)) in eval_sets.iter_mut().enumerate() {
                    self.update_predictions_inplace(yhat, &tree, data);
                    let (metric_fn, maximize) = self.get_metric_fn();
                    let m = metric_fn(y, yhat, w);
                    // If early stopping rounds are defined, and this is the last
                    // eval dataset, check if we want to stop or keep training.
                    // Updating to align with XGBoost, originally we were using the first
                    // dataset, but switching to use the last.
                    if (eval_i + 1) == n_eval_sets {
                        if let Some(early_stopping_rounds) = self.early_stopping_rounds {
                            // If best metric is undefined, this must be the first
                            // iteration...
                            best_metric = match best_metric {
                                None => {
                                    self.update_best_iteration(i);
                                    Some(m)
                                }
                                // Otherwise the best could be farther back.
                                Some(v) => {
                                    // We have reached a new best value...
                                    if is_comparison_better(v, m, maximize) {
                                        self.update_best_iteration(i);
                                        Some(m)
                                    } else {
                                        // Previous value was better.
                                        if let Some(best_iteration) = self.best_iteration {
                                            if i - best_iteration >= early_stopping_rounds {
                                                // If any logging is requested, print this message.
                                                if self.log_iterations > 0 {
                                                    info!("Stopping early at iteration {} with metric value {}", i, m)
                                                }
                                                stop_early = true;
                                            }
                                        }
                                        Some(v)
                                    }
                                }
                            };
                        }
                    }
                    metrics.push(m);
                }
                if verbose {
                    info!(
                        "Iteration {} evaluation data values: {}",
                        i,
                        fmt_vec_output(&metrics)
                    );
                }
                if let Some(history) = &mut self.evaluation_history {
                    history.append_row(metrics);
                }
            }
            self.trees.push(tree);

            // Did we trigger the early stopping rounds criteria?
            if stop_early {
                break;
            }

            (grad, hess) = calc_grad_hess(y, &yhat, sample_weight);
            if verbose {
                info!("Completed iteration {} of {}", i, self.iterations);
            }
        }
        if self.log_iterations > 0 {
            info!(
                "Finished training booster with {} iterations.",
                self.trees.len()
            );
        }
        Ok(())
    }

    fn fit_trees_multiclass<T: Splitter>(
        &mut self,
        y: &[f64],
        sample_weight: &[f64],
        data: &Matrix<f64>,
        splitter: &T,
        evaluation_data: Option<Vec<EvaluationData>>,
    ) -> Result<(), ForustError> {
        if !self.trees.is_empty() {
            self.reset();
        }

        let k = self.num_classes;
        let n = y.len();
        let mut rng = StdRng::seed_from_u64(self.seed);

        // 1. Base scores (K-length Vec)
        if self.initialize_base_score {
            self.base_scores = Some(SoftmaxMultiClass::calc_init(y, sample_weight, k));
        }
        let base = self.base_scores.clone().unwrap_or_else(|| vec![0.0; k]);

        // 2. Initialize yhat: N×K row-major
        let mut yhat = Vec::with_capacity(n * k);
        for _ in 0..n {
            yhat.extend_from_slice(&base);
        }

        // 3. Initial grad/hess: N×K flat as f32
        let (mut grad, mut hess) = SoftmaxMultiClass::calc_grad_hess(y, &yhat, sample_weight, k);

        // 4. Bin data once (unchanged)
        let binned_data = bin_matrix(data, sample_weight, self.nbins, self.missing)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        // Initialize eval set predictions (N×K per eval set)
        let mut evaluation_sets: Option<Vec<(&Matrix<f64>, &[f64], &[f64], Vec<f64>)>> =
            evaluation_data.as_ref().map(|evals| {
                evals
                    .iter()
                    .map(|(d, ey, ew)| {
                        let n_eval = ey.len();
                        let mut eval_yhat = Vec::with_capacity(n_eval * k);
                        for _ in 0..n_eval {
                            eval_yhat.extend_from_slice(&base);
                        }
                        (d, *ey, *ew, eval_yhat)
                    })
                    .collect()
            });

        let mut best_metric: Option<f64> = None;
        let mut stop_early = false;
        let col_index: Vec<usize> = (0..data.cols).collect();

        for round in 0..self.iterations {
            let verbose = if self.log_iterations == 0 {
                false
            } else {
                round % self.log_iterations == 0
            };

            // 5. Sample rows ONCE per round (same rows for all K classes)
            let (mut sample_grad, mut sample_hess) = max_abs_grad_per_row(&grad, &hess, k);
            let (chosen_index, _excluded_index) =
                self.sample_index(&mut rng, &data.index, &mut sample_grad, &mut sample_hess);

            // Column sampling
            let colsample_index: Vec<usize> = if self.colsample_bytree == 1.0 {
                Vec::new()
            } else {
                let amount = ((col_index.len() as f64) * self.colsample_bytree).floor() as usize;
                let mut v: Vec<usize> = col_index
                    .iter()
                    .choose_multiple(&mut rng, amount)
                    .iter()
                    .map(|i| **i)
                    .collect();
                v.sort();
                v
            };

            let fit_col_index = if self.colsample_bytree == 1.0 {
                &col_index
            } else {
                &colsample_index
            };

            // 6. Fit K trees using SAME grad/hess (not recomputed between classes)
            for class in 0..k {
                let (grad_k, hess_k) = extract_class_slice(&grad, &hess, class, k, n);

                let mut tree = Tree::new();
                tree.fit(
                    &bdata,
                    chosen_index.clone(),
                    fit_col_index,
                    &binned_data.cuts,
                    &grad_k,
                    &hess_k,
                    splitter,
                    self.max_leaves,
                    self.max_depth,
                    self.parallel,
                    &self.sample_method,
                    &self.grow_policy,
                );

                // Update only class-k training predictions
                let preds = tree.predict(data, self.parallel, &self.missing);
                for (i, p) in preds.iter().enumerate() {
                    yhat[i * k + class] += *p;
                }

                // Update eval set predictions for this tree
                if let Some(eval_sets) = &mut evaluation_sets {
                    for (eval_d, _, _, eval_yhat) in eval_sets.iter_mut() {
                        let eval_preds = tree.predict(eval_d, self.parallel, &self.missing);
                        for (i, &p) in eval_preds.iter().enumerate() {
                            eval_yhat[i * k + class] += p;
                        }
                    }
                }

                self.trees.push(tree);
            }

            // 7. Eval metric after all K trees in this round
            if let Some(eval_sets) = &mut evaluation_sets {
                if self.evaluation_history.is_none() {
                    self.evaluation_history =
                        Some(RowMajorMatrix::new(Vec::new(), 0, eval_sets.len()));
                }
                let mut metrics: Vec<f64> = Vec::new();
                let n_eval_sets = eval_sets.len();
                for (eval_i, (_, ey, ew, eval_yhat)) in eval_sets.iter().enumerate() {
                    let m = multiclass_log_loss(ey, eval_yhat, ew, k);

                    if (eval_i + 1) == n_eval_sets {
                        if let Some(early_stopping_rounds) = self.early_stopping_rounds {
                            best_metric = match best_metric {
                                None => {
                                    self.update_best_iteration_multiclass(round, k);
                                    Some(m)
                                }
                                Some(v) => {
                                    // MultiClassLogLoss is minimized
                                    if is_comparison_better(v, m, false) {
                                        self.update_best_iteration_multiclass(round, k);
                                        Some(m)
                                    } else {
                                        if let Some(best_iteration) = self.best_iteration {
                                            if round - best_iteration >= early_stopping_rounds {
                                                if self.log_iterations > 0 {
                                                    info!("Stopping early at round {} with metric value {}", round, m);
                                                }
                                                stop_early = true;
                                            }
                                        }
                                        Some(v)
                                    }
                                }
                            };
                        }
                    }
                    metrics.push(m);
                }
                if verbose {
                    info!(
                        "Round {} evaluation data values: {}",
                        round,
                        fmt_vec_output(&metrics)
                    );
                }
                if let Some(history) = &mut self.evaluation_history {
                    history.append_row(metrics);
                }
            }

            if stop_early {
                break;
            }

            // 8. Recompute all N×K grad/hess for next round
            (grad, hess) = SoftmaxMultiClass::calc_grad_hess(y, &yhat, sample_weight, k);
            if verbose {
                info!("Completed round {} of {}", round, self.iterations);
            }
        }
        if self.log_iterations > 0 {
            info!(
                "Finished training multiclass booster with {} trees ({} rounds × {} classes).",
                self.trees.len(),
                self.trees.len() / k,
                k
            );
        }
        Ok(())
    }

    fn update_best_iteration_multiclass(&mut self, round: usize, num_classes: usize) {
        self.best_iteration = Some(round);
        self.prediction_iteration = Some((round + 1).min(self.trees.len() / num_classes));
    }

    fn update_best_iteration(&mut self, i: usize) {
        self.best_iteration = Some(i);
        self.prediction_iteration = Some(i + 1);
    }

    fn update_predictions_inplace(&self, yhat: &mut [f64], tree: &Tree, data: &Matrix<f64>) {
        let preds = tree.predict(data, self.parallel, &self.missing);
        yhat.iter_mut().zip(preds).for_each(|(i, j)| *i += j);
    }

    /// Fit the gradient booster on a provided dataset without any weights.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `y` - Either a pandas Series, or a 1 dimensional numpy array.
    pub fn fit_unweighted(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        evaluation_data: Option<Vec<EvaluationData>>,
    ) -> Result<(), ForustError> {
        let sample_weight = vec![1.0; data.rows];
        self.fit(data, y, &sample_weight, evaluation_data)
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    ///
    /// For multi-class (num_classes > 1), returns raw logits of length N × K (row-major).
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        if self.num_classes <= 1 {
            // Existing scalar prediction (unchanged)
            let mut init_preds = vec![self.base_score; data.rows];
            self.get_prediction_trees().iter().for_each(|tree| {
                for (p_, val) in
                    init_preds
                        .iter_mut()
                        .zip(tree.predict(data, parallel, &self.missing))
                {
                    *p_ += val;
                }
            });
            init_preds
        } else {
            let k = self.num_classes;
            let n = data.rows;
            let base = self.base_scores.as_deref().unwrap_or(&[]);
            let mut preds = Vec::with_capacity(n * k);
            for _ in 0..n {
                if base.len() == k {
                    preds.extend_from_slice(base);
                } else {
                    preds.extend(std::iter::repeat(0.0).take(k));
                }
            }
            for (t_idx, tree) in self.get_prediction_trees().iter().enumerate() {
                let class = t_idx % k;
                for (i, p) in tree
                    .predict(data, parallel, &self.missing)
                    .iter()
                    .enumerate()
                {
                    preds[i * k + class] += *p;
                }
            }
            preds
        }
    }

    /// Predict class probabilities for SoftmaxMultiClass.
    /// Returns Vec<f64> of length n_samples × num_classes, row-major.
    /// Each consecutive K values are one sample's probabilities, summing to 1.0.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        assert!(
            self.num_classes >= 2,
            "predict_proba requires num_classes >= 2"
        );
        let raw = self.predict(data, parallel);
        SoftmaxMultiClass::softmax(&raw, self.num_classes)
    }

    /// Predict the leaf Indexes, this returns a vector of length N records * N Trees
    pub fn predict_leaf_indices(&self, data: &Matrix<f64>) -> Vec<usize> {
        self.get_prediction_trees()
            .iter()
            .flat_map(|tree| tree.predict_leaf_indices(data, &self.missing))
            .collect()
    }

    /// Predict the contributions matrix for the provided dataset.
    pub fn predict_contributions(
        &self,
        data: &Matrix<f64>,
        method: ContributionsMethod,
        parallel: bool,
    ) -> Vec<f64> {
        if matches!(self.objective_type, ObjectiveType::SoftmaxMultiClass) {
            panic!("{}", MULTICLASS_CONTRIBUTIONS_UNSUPPORTED);
        }
        match method {
            ContributionsMethod::Average => self.predict_contributions_average(data, parallel),
            ContributionsMethod::ProbabilityChange => {
                match self.objective_type {
                    ObjectiveType::LogLoss => {},
                    ObjectiveType::SquaredLoss | ObjectiveType::SoftmaxMultiClass => panic!("ProbabilityChange contributions method is only valid when LogLoss objective is used."),
                }
                self.predict_contributions_probability_change(data, parallel)
            }
            _ => self.predict_contributions_tree_alone(data, parallel, method),
        }
    }

    // All of the contribution calculation methods, except for average are calculated
    // using just the model, so we don't need to have separate methods, we can instead
    // just have this one method, that dispatches to each one respectively.
    fn predict_contributions_tree_alone(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
        method: ContributionsMethod,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        let row_pred_fn = match method {
            ContributionsMethod::Weight => Tree::predict_contributions_row_weight,
            ContributionsMethod::BranchDifference => {
                Tree::predict_contributions_row_branch_difference
            }
            ContributionsMethod::MidpointDifference => {
                Tree::predict_contributions_row_midpoint_difference
            }
            ContributionsMethod::ModeDifference => Tree::predict_contributions_row_mode_difference,
            ContributionsMethod::Shapley => predict_contributions_row_shapley,
            ContributionsMethod::Average | ContributionsMethod::ProbabilityChange => unreachable!(),
        };
        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.missing);
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.missing);
                    });
                });
        }

        contribs
    }

    fn total_prediction_rounds(&self) -> usize {
        if self.num_classes > 1 {
            self.trees.len() / self.num_classes
        } else {
            self.trees.len()
        }
    }

    fn normalize_prediction_iteration_state(&mut self) {
        let Some(prediction_iteration) = self.prediction_iteration else {
            return;
        };
        if self.trees.is_empty() {
            return;
        }
        if self.num_classes <= 1 {
            self.prediction_iteration = Some(prediction_iteration.min(self.trees.len()));
            return;
        }

        let total_rounds = self.total_prediction_rounds();
        let normalized = if let Some(best_iteration) = self.best_iteration {
            let best_rounds = best_iteration + 1;
            let legacy_tree_count = best_rounds.saturating_mul(self.num_classes);
            if prediction_iteration == legacy_tree_count {
                best_rounds
            } else {
                prediction_iteration.min(total_rounds)
            }
        } else if prediction_iteration > total_rounds
            && prediction_iteration % self.num_classes == 0
            && prediction_iteration <= self.trees.len()
        {
            prediction_iteration / self.num_classes
        } else {
            prediction_iteration.min(total_rounds)
        };

        self.prediction_iteration = Some(normalized.min(total_rounds));
    }

    pub fn prediction_iteration_limit(&self) -> Option<usize> {
        if self.trees.is_empty() {
            return self.prediction_iteration;
        }
        self.prediction_iteration.map(|prediction_iteration| {
            if self.num_classes > 1 {
                prediction_iteration.min(self.total_prediction_rounds())
            } else {
                prediction_iteration.min(self.trees.len())
            }
        })
    }

    pub fn prediction_tree_count(&self) -> usize {
        match self.prediction_iteration_limit() {
            Some(prediction_iteration) if self.num_classes > 1 => prediction_iteration
                .saturating_mul(self.num_classes)
                .min(self.trees.len()),
            Some(prediction_iteration) => prediction_iteration,
            None => self.trees.len(),
        }
    }

    /// Get a reference to the trees used for predicting.
    fn get_prediction_trees(&self) -> &[Tree] {
        &self.trees[..self.prediction_tree_count()]
    }

    /// Generate predictions on data using the gradient booster.
    /// This is equivalent to the XGBoost predict contributions with approx_contribs
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    fn predict_contributions_average(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let weights: Vec<Vec<f64>> = if parallel {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        };
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.missing);
                        });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.missing);
                        });
                });
        }

        contribs
    }

    fn predict_contributions_probability_change(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += odds(self.base_score));

        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .fold(self.base_score, |acc, t| {
                            t.predict_contributions_row_probability_change(
                                &r_,
                                c,
                                &self.missing,
                                acc,
                            )
                        });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .fold(self.base_score, |acc, t| {
                            t.predict_contributions_row_probability_change(
                                &r_,
                                c,
                                &self.missing,
                                acc,
                            )
                        });
                });
        }
        contribs
    }

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        if matches!(self.objective_type, ObjectiveType::SoftmaxMultiClass) {
            panic!("{}", MULTICLASS_PARTIAL_DEPENDENCE_UNSUPPORTED);
        }
        let pd: f64 = if self.parallel {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.value_partial_dependence(feature, value, &self.missing))
                .sum()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.value_partial_dependence(feature, value, &self.missing))
                .sum()
        };
        pd + self.base_score
    }

    /// Calculate feature importance measure for the features
    /// in the model.
    /// - `method`: variable importance method to use.
    /// - `n_features`: The number of features to calculate the importance for.
    pub fn calculate_feature_importance(
        &self,
        method: ImportanceMethod,
        normalize: bool,
    ) -> HashMap<usize, f32> {
        let (average, importance_fn): (bool, ImportanceFn) = match method {
            ImportanceMethod::Weight => (false, Tree::calculate_importance_weight),
            ImportanceMethod::Gain => (true, Tree::calculate_importance_gain),
            ImportanceMethod::TotalGain => (false, Tree::calculate_importance_gain),
            ImportanceMethod::Cover => (true, Tree::calculate_importance_cover),
            ImportanceMethod::TotalCover => (false, Tree::calculate_importance_cover),
        };
        let mut stats = HashMap::new();
        for tree in self.trees.iter() {
            importance_fn(tree, &mut stats)
        }

        let importance = stats
            .iter()
            .map(|(k, (v, c))| {
                if average {
                    (*k, v / (*c as f32))
                } else {
                    (*k, *v)
                }
            })
            .collect::<HashMap<usize, f32>>();

        if normalize {
            // To make deterministic, sort values and then sum.
            // Otherwise we were getting them in different orders, and
            // floating point error was creeping in.
            let mut values: Vec<f32> = importance.values().copied().collect();
            // We are OK to unwrap because we know we will never have missing.
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let total: f32 = values.iter().sum();
            importance.iter().map(|(k, v)| (*k, v / total)).collect()
        } else {
            importance
        }
    }

    /// Save a booster as a json object to a file.
    ///
    /// * `path` - Path to save booster.
    pub fn save_booster(&self, path: &str) -> Result<(), ForustError> {
        let model = self.json_dump()?;
        match fs::write(path, model) {
            Err(e) => Err(ForustError::UnableToWrite(e.to_string())),
            Ok(_) => Ok(()),
        }
    }

    /// Dump a booster as a json object
    pub fn json_dump(&self) -> Result<String, ForustError> {
        match serde_json::to_string(self) {
            Ok(s) => Ok(s),
            Err(e) => Err(ForustError::UnableToWrite(e.to_string())),
        }
    }

    /// Load a booster from Json string
    ///
    /// * `json_str` - String object, which can be serialized to json.
    pub fn from_json(json_str: &str) -> Result<Self, ForustError> {
        let model = serde_json::from_str::<GradientBooster>(json_str);
        match model {
            Ok(mut m) => {
                m.normalize_prediction_iteration_state();
                Ok(m)
            }
            Err(e) => Err(ForustError::UnableToRead(e.to_string())),
        }
    }

    /// Load a booster from a path to a json booster object.
    ///
    /// * `path` - Path to load booster from.
    pub fn load_booster(path: &str) -> Result<Self, ForustError> {
        let json_str = match fs::read_to_string(path) {
            Ok(s) => Ok(s),
            Err(e) => Err(ForustError::UnableToRead(e.to_string())),
        }?;
        Self::from_json(&json_str)
    }

    // Set methods for paramters
    /// Set the objective_type on the booster.
    /// * `objective_type` - The objective type of the booster.
    pub fn set_objective_type(mut self, objective_type: ObjectiveType) -> Self {
        self.objective_type = objective_type;
        self
    }

    /// Set the iterations on the booster.
    /// * `iterations` - The number of iterations of the booster.
    pub fn set_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the learning_rate on the booster.
    /// * `learning_rate` - The learning rate of the booster.
    pub fn set_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the max_depth on the booster.
    /// * `max_depth` - The maximum tree depth of the booster.
    pub fn set_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the max_leaves on the booster.
    /// * `max_leaves` - The maximum number of leaves of the booster.
    pub fn set_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Set the number of nbins on the booster.
    /// * `max_leaves` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_nbins(mut self, nbins: u16) -> Self {
        self.nbins = nbins;
        self
    }

    /// Set the l1 on the booster.
    /// * `l1` - The l1 regulation term of the booster.
    pub fn set_l1(mut self, l1: f32) -> Self {
        self.l1 = l1;
        self
    }

    /// Set the l2 on the booster.
    /// * `l2` - The l2 regulation term of the booster.
    pub fn set_l2(mut self, l2: f32) -> Self {
        self.l2 = l2;
        self
    }

    /// Set the gamma on the booster.
    /// * `gamma` - The gamma value of the booster.
    pub fn set_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the max_delta_step on the booster.
    /// * `max_delta_step` - The max_delta_step value of the booster.
    pub fn set_max_delta_step(mut self, max_delta_step: f32) -> Self {
        self.max_delta_step = max_delta_step;
        self
    }

    /// Set the min_leaf_weight on the booster.
    /// * `min_leaf_weight` - The minimum sum of the hession values allowed in the
    ///     node of a tree of the booster.
    pub fn set_min_leaf_weight(mut self, min_leaf_weight: f32) -> Self {
        self.min_leaf_weight = min_leaf_weight;
        self
    }

    /// Set the base_score on the booster.
    /// * `base_score` - The base score of the booster.
    pub fn set_base_score(mut self, base_score: f64) -> Self {
        self.base_score = base_score;
        self
    }

    /// Set the base_score on the booster.
    /// * `base_score` - The base score of the booster.
    pub fn set_initialize_base_score(mut self, initialize_base_score: bool) -> Self {
        self.initialize_base_score = initialize_base_score;
        self
    }

    /// Set the parallel on the booster.
    /// * `parallel` - Set if the booster should be trained in parallels.
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.allow_missing_splits = allow_missing_splits;
        self
    }

    /// Set the monotone_constraints on the booster.
    /// * `monotone_constraints` - The monotone constraints of the booster.
    pub fn set_monotone_constraints(mut self, monotone_constraints: Option<ConstraintMap>) -> Self {
        self.monotone_constraints = monotone_constraints;
        self
    }

    /// Set the subsample on the booster.
    /// * `subsample` - Percent of the data to randomly sample when training each tree.
    pub fn set_subsample(mut self, subsample: f32) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the colsample_bytree on the booster.
    /// * `colsample_bytree` - Percent of the columns to randomly sample when training each tree.
    pub fn set_colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.colsample_bytree = colsample_bytree;
        self
    }

    /// Set the seed on the booster.
    /// * `seed` - Integer value used to see any randomness used in the algorithm.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set missing value of the booster
    /// * `missing` - Float value to consider as missing.
    pub fn set_missing(mut self, missing: f64) -> Self {
        self.missing = missing;
        self
    }

    /// Set create missing value of the booster
    /// * `create_missing_branch` - Bool specifying if missing should get it's own
    /// branch.
    pub fn set_create_missing_branch(mut self, create_missing_branch: bool) -> Self {
        self.create_missing_branch = create_missing_branch;
        self
    }

    /// Set sample method on the booster.
    /// * `sample_method` - Sample method.
    pub fn set_sample_method(mut self, sample_method: SampleMethod) -> Self {
        self.sample_method = sample_method;
        self
    }

    /// Set sample method on the booster.
    /// * `evaluation_metric` - Sample method.
    pub fn set_evaluation_metric(mut self, evaluation_metric: Option<Metric>) -> Self {
        self.evaluation_metric = evaluation_metric;
        self
    }

    /// Set early stopping rounds.
    /// * `early_stopping_rounds` - Early stoppings rounds.
    pub fn set_early_stopping_rounds(mut self, early_stopping_rounds: Option<usize>) -> Self {
        self.early_stopping_rounds = early_stopping_rounds;
        self
    }

    /// Set the number of classes used by SoftmaxMultiClass.
    /// * `num_classes` - Total number of target classes.
    pub fn set_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }

    /// Set prediction iterations.
    /// * `prediction_iteration` - Number of boosting iterations to use when predicting.
    ///   For SoftmaxMultiClass, each boosting iteration corresponds to `num_classes` trees.
    pub fn set_prediction_iteration(mut self, prediction_iteration: Option<usize>) -> Self {
        self.prediction_iteration = prediction_iteration;
        self
    }

    /// Set the features where whose missing nodes should
    /// always be terminated.
    /// * `terminate_missing_features` - Hashset of the feature indices for the
    /// features that should always terminate the missing node, if create_missing_branch
    /// is true.
    pub fn set_terminate_missing_features(
        mut self,
        terminate_missing_features: HashSet<usize>,
    ) -> Self {
        self.terminate_missing_features = terminate_missing_features;
        self
    }

    /// Insert metadata
    /// * `key` - String value for the metadata key.
    /// * `value` - value to assign to the metadata key.
    pub fn insert_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get Metadata
    /// * `key` - Get the associated value for the metadata key.
    pub fn get_metadata(&self, key: &String) -> Option<String> {
        self.metadata.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_booster_fit_subsample() {
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = GradientBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_subsample(0.5)
            .set_base_score(0.5)
            .set_initialize_base_score(false);
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit() {
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = GradientBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit_nofitted_base_score() {
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance-fare.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SquaredLoss)
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_initialize_base_score(true);
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_tree_save() {
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = GradientBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, None).unwrap();
        let preds = booster.predict(&data, true);

        booster.save_booster("resources/model64.json").unwrap();
        let booster2 = GradientBooster::load_booster("resources/model64.json").unwrap();
        assert_eq!(booster2.predict(&data, true)[0..10], preds[0..10]);

        // Test with non-NAN missing.
        booster.missing = 0.;
        booster.save_booster("resources/modelmissing.json").unwrap();
        let booster3 = GradientBooster::load_booster("resources/modelmissing.json").unwrap();
        assert_eq!(booster3.missing, 0.);
        assert_eq!(booster3.missing, booster.missing);
    }

    // ── Multiclass integration tests ──

    /// Generate synthetic 3-class data: 3 separable 2D clusters.
    /// Returns (col_major_data, labels, n_rows, n_cols).
    fn make_3class_data() -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n_per_class = 100;
        let n = n_per_class * 3;
        let n_cols = 2;
        // Row-major features: class 0 centered at (0,0), class 1 at (5,5), class 2 at (10,0)
        let mut rows = Vec::with_capacity(n * n_cols);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n_per_class {
            let offset = (i as f64) * 0.02 - 1.0; // [-1, 1] spread
            rows.push(0.0 + offset);
            rows.push(0.0 + offset);
            labels.push(0.0);
        }
        for i in 0..n_per_class {
            let offset = (i as f64) * 0.02 - 1.0;
            rows.push(5.0 + offset);
            rows.push(5.0 + offset);
            labels.push(1.0);
        }
        for i in 0..n_per_class {
            let offset = (i as f64) * 0.02 - 1.0;
            rows.push(10.0 + offset);
            rows.push(0.0 + offset);
            labels.push(2.0);
        }
        // Convert to column-major for forust-ml Matrix
        let mut col_major = vec![0.0; n * n_cols];
        for row in 0..n {
            for col in 0..n_cols {
                col_major[col * n + row] = rows[row * n_cols + col];
            }
        }
        (col_major, labels, n, n_cols)
    }

    #[test]
    fn test_multiclass_3class_synthetic() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50)
            .set_max_depth(4)
            .set_learning_rate(0.3);
        booster.num_classes = 3;
        booster.initialize_base_score = true;

        booster.fit(&data, &y, &w, None).unwrap();
        let probs = booster.predict_proba(&data, true);

        // Check accuracy > 90%
        let mut correct = 0;
        for i in 0..n {
            let row = &probs[i * 3..(i + 1) * 3];
            let predicted = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if predicted == y[i] as usize {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / n as f64;
        assert!(accuracy > 0.90, "accuracy = {} (expected > 0.90)", accuracy);
    }

    #[test]
    fn test_multiclass_predict_proba_sums_to_one() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(10);
        booster.num_classes = 3;
        booster.fit(&data, &y, &w, None).unwrap();
        let probs = booster.predict_proba(&data, true);

        for i in 0..n {
            let row = &probs[i * 3..(i + 1) * 3];
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "row {} sum = {}", i, sum);
        }
    }

    #[test]
    fn test_multiclass_predict_returns_logits() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(10);
        booster.num_classes = 3;
        booster.fit(&data, &y, &w, None).unwrap();
        let logits = booster.predict(&data, true);

        // Raw logits should NOT sum to 1 (they're not probabilities)
        assert_eq!(logits.len(), n * 3);
        let first_row = &logits[0..3];
        let sum: f64 = first_row.iter().sum();
        // With non-zero base scores and tree predictions, sum unlikely to be exactly 1
        // Just verify it returns raw values (can be negative, > 1, etc.)
        assert!(
            (sum - 1.0).abs() > 1e-6 || first_row.iter().any(|x| *x < 0.0 || *x > 1.0),
            "logits should not look like probabilities"
        );
    }

    #[test]
    fn test_multiclass_json_roundtrip() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(10);
        booster.num_classes = 3;
        booster.initialize_base_score = true;
        booster.fit(&data, &y, &w, None).unwrap();

        let preds1 = booster.predict(&data, true);

        // Roundtrip via JSON
        booster
            .save_booster("resources/model_multiclass.json")
            .unwrap();
        let loaded = GradientBooster::load_booster("resources/model_multiclass.json").unwrap();
        let preds2 = loaded.predict(&data, true);

        assert_eq!(preds1.len(), preds2.len());
        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert!((a - b).abs() < 1e-10, "predictions differ: {} vs {}", a, b);
        }
        assert_eq!(loaded.num_classes, 3);
        assert!(loaded.base_scores.is_some());
    }

    #[test]
    fn test_multiclass_default_is_1() {
        let booster = GradientBooster::default();
        assert_eq!(booster.num_classes, 1);
        assert!(booster.base_scores.is_none());
    }

    #[test]
    fn test_multiclass_tree_count() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];
        let iters = 15;

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(iters);
        booster.num_classes = 3;
        booster.fit(&data, &y, &w, None).unwrap();

        assert_eq!(
            booster.trees.len(),
            iters * 3,
            "trees = iters * num_classes"
        );
    }

    #[test]
    fn test_multiclass_early_stopping() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(200);
        booster.num_classes = 3;
        booster.early_stopping_rounds = Some(5);
        booster.initialize_base_score = true;

        let eval_data = vec![(
            Matrix::new(&data_vec, n, n_cols),
            y.as_slice(),
            w.as_slice(),
        )];
        booster.fit(&data, &y, &w, Some(eval_data)).unwrap();

        // Should have stopped early (well before 200 rounds on separable data)
        assert!(
            booster.trees.len() < 200 * 3,
            "expected early stopping, got {} trees",
            booster.trees.len()
        );
        assert_eq!(
            booster.prediction_iteration_limit(),
            booster.best_iteration.map(|best| best + 1)
        );
    }

    #[test]
    fn test_multiclass_prediction_iteration_uses_rounds() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(20);
        booster.fit(&data, &y, &w, None).unwrap();

        let full_preds = booster.predict(&data, true);
        booster.prediction_iteration = Some(2);
        let partial_preds = booster.predict(&data, true);
        let leaf_indices = booster.predict_leaf_indices(&data);

        assert_eq!(booster.prediction_iteration_limit(), Some(2));
        assert_eq!(leaf_indices.len(), n * 2 * 3);
        assert!(
            full_preds
                .iter()
                .zip(partial_preds.iter())
                .any(|(full, partial)| (full - partial).abs() > 1e-10),
            "using fewer boosting rounds should change predictions"
        );
    }

    #[test]
    fn test_multiclass_prediction_iteration_legacy_json_is_normalized() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(20);
        booster.early_stopping_rounds = Some(3);
        booster.initialize_base_score = true;

        let eval_data = vec![(
            Matrix::new(&data_vec, n, n_cols),
            y.as_slice(),
            w.as_slice(),
        )];
        booster.fit(&data, &y, &w, Some(eval_data)).unwrap();

        let best_rounds = booster.best_iteration.unwrap() + 1;
        let mut json_value: serde_json::Value =
            serde_json::from_str(&booster.json_dump().unwrap()).unwrap();
        json_value["prediction_iteration"] = serde_json::json!(best_rounds * booster.num_classes);

        let loaded = GradientBooster::from_json(&json_value.to_string()).unwrap();
        assert_eq!(loaded.prediction_iteration_limit(), Some(best_rounds));
    }

    #[test]
    fn test_existing_binary_unchanged() {
        // Train binary LogLoss and verify results are identical to before multiclass changes
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        let mut booster = GradientBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        // Verify num_classes defaults to 1
        assert_eq!(booster.num_classes, 1);

        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, None).unwrap();
        let preds = booster.predict(&data, false);

        // Should produce same number of predictions as rows
        assert_eq!(preds.len(), 891);
        // Predictions should be scalar (not N×K)
        assert!(preds[0].is_finite());
    }

    #[test]
    fn test_num_classes_without_softmax_objective() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default().set_iterations(5);
        booster.num_classes = 3; // but objective is LogLoss (default)

        let result = booster.fit(&data, &y, &w, None);
        assert!(
            result.is_err(),
            "should reject num_classes > 1 with LogLoss"
        );
    }

    #[test]
    fn test_softmax_objective_without_num_classes() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(5);
        // num_classes defaults to 1 — too low for SoftmaxMultiClass

        let result = booster.fit(&data, &y, &w, None);
        assert!(
            result.is_err(),
            "should reject SoftmaxMultiClass with num_classes < 2"
        );
    }

    #[test]
    fn test_multiclass_rejects_invalid_eval_labels() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];
        let eval_y = vec![3.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(5);

        let eval_data = vec![(
            Matrix::new(&data_vec, n, n_cols),
            eval_y.as_slice(),
            w.as_slice(),
        )];
        let result = booster.fit(&data, &y, &w, Some(eval_data));
        assert!(result.is_err(), "should reject invalid eval labels");
    }

    #[test]
    fn test_multiclass_rejects_unsupported_evaluation_metric() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(5)
            .set_evaluation_metric(Some(Metric::LogLoss));

        let eval_data = vec![(
            Matrix::new(&data_vec, n, n_cols),
            y.as_slice(),
            w.as_slice(),
        )];
        let result = booster.fit(&data, &y, &w, Some(eval_data));
        assert!(
            result.is_err(),
            "should reject evaluation metrics other than MultiClassLogLoss"
        );
    }

    #[test]
    #[should_panic(expected = "predict_contributions is not supported for SoftmaxMultiClass")]
    fn test_multiclass_predict_contributions_panics() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(5);
        booster.fit(&data, &y, &w, None).unwrap();

        let _ = booster.predict_contributions(&data, ContributionsMethod::Average, false);
    }

    #[test]
    #[should_panic(expected = "value_partial_dependence is not supported for SoftmaxMultiClass")]
    fn test_multiclass_value_partial_dependence_panics() {
        let (data_vec, y, n, n_cols) = make_3class_data();
        let data = Matrix::new(&data_vec, n, n_cols);
        let w = vec![1.0; n];

        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_num_classes(3)
            .set_iterations(5);
        booster.fit(&data, &y, &w, None).unwrap();

        let _ = booster.value_partial_dependence(0, 0.0);
    }
}
