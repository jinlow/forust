use crate::binning::bin_matrix;
use crate::constraints::ConstraintMap;
use crate::data::{Matrix, RowMajorMatrix};
use crate::errors::ForustError;
use crate::metric::{is_comparison_better, metric_callables, Metric, MetricFn};
use crate::objective::{
    calc_init_callables, gradient_hessian_callables, LogLoss, ObjectiveFunction, ObjectiveType,
    SquaredLoss,
};
use crate::sampler::{GossSampler, RandomSampler, SampleMethod, Sampler};
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, Splitter};
use crate::tree::Tree;
use crate::utils::validate_positive_float_field;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fs;
pub type EvaluationData<'a> = (Matrix<'a, f64>, &'a [f64], &'a [f64]);
pub type TrainingEvaluationData<'a> = (&'a Matrix<'a, f64>, &'a [f64], &'a [f64], Vec<f64>);
type ImportanceFn = fn(&Tree, &mut HashMap<usize, (f32, usize)>);

#[derive(Serialize, Deserialize)]
pub enum GrowPolicy {
    DepthWise,
    LossGuide,
}

#[derive(Serialize, Deserialize)]
pub enum ContributionsMethod {
    Weight,
    Average,
    BranchDifference,
    MidpointDifference,
    ModeDifference,
}

#[derive(Serialize, Deserialize)]
pub enum ImportanceMethod {
    Weight,
    Gain,
    Cover,
    TotalGain,
    TotalCover,
}

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
/// * `base_score` - The initial prediction value of the model.
/// * `nbins` - Number of bins to calculate to partition the data. Setting this to
///   a smaller number, will result in faster training time, while potentially sacrificing
///   accuracy. If there are more bins, than unique values in a column, all unique values
///   will be used.
/// * `allow_missing_splits` - Should the algorithm allow splits that completed seperate out missing
/// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
/// is true, setting this to true will result in the missin branch being further split.
/// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
///   between the training features and the target variable.
/// * `subsample` - Percent of records to randomly sample at each iteration when training a tree.
/// * `top_rate` - Used only in goss. The retain ratio of large gradient data.
/// * `other_rate` - Used only in goss. the retain ratio of small gradient data.
/// * `seed` - Integer value used to seed any randomness used in the algorithm.
/// * `missing` - Value to consider missing.
/// * `create_missing_branch` - Should missing be split out it's own separate branch?
/// * `sample_method` - Specify the method that records should be sampled when training?
/// * `evaluation_metric` - Define the evaluation metric to record at each iterations.
/// * `early_stopping_rounds` - Number of rounds where the evaluation metric value must improve in
///    to keep training.
/// * `initialize_base_score` - If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
#[derive(Deserialize, Serialize)]
pub struct GradientBooster {
    pub objective_type: ObjectiveType,
    pub iterations: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
    pub max_leaves: usize,
    pub l2: f32,
    pub gamma: f32,
    pub min_leaf_weight: f32,
    pub base_score: f64,
    pub nbins: u16,
    pub parallel: bool,
    pub allow_missing_splits: bool,
    pub monotone_constraints: Option<ConstraintMap>,
    pub subsample: f32,
    #[serde(default = "default_top_rate")]
    pub top_rate: f64,
    #[serde(default = "default_other_rate")]
    pub other_rate: f64,
    pub seed: u64,
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    pub create_missing_branch: bool,
    #[serde(default = "default_sample_method")]
    pub sample_method: SampleMethod,
    #[serde(default = "default_grow_policy")]
    pub grow_policy: GrowPolicy,
    #[serde(default = "default_evaluation_metric")]
    pub evaluation_metric: Option<Metric>,
    #[serde(default = "default_early_stopping_rounds")]
    pub early_stopping_rounds: Option<usize>,
    #[serde(default = "default_initialize_base_score")]
    pub initialize_base_score: bool,
    #[serde(default = "default_evaluation_history")]
    pub evaluation_history: Option<RowMajorMatrix<f64>>,
    #[serde(default = "default_best_iteration")]
    pub best_iteration: Option<usize>,
    /// number of trees to use when predicting,
    /// defaults to best_iteration if this is defined.
    #[serde(default = "default_prediction_iteration")]
    pub prediction_iteration: Option<usize>,
    // Members internal to the booster object, and not parameters set by the user.
    // Trees is public, just to interact with it directly in the python wrapper.
    pub trees: Vec<Tree>,
    metadata: HashMap<String, String>,
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

fn parse_missing<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or(f64::NAN))
}

impl Default for GradientBooster {
    fn default() -> Self {
        Self::new(
            ObjectiveType::LogLoss,
            10,
            0.3,
            5,
            usize::MAX,
            1.,
            0.,
            1.,
            None,
            256,
            true,
            true,
            None,
            1.,
            0.1,
            0.2,
            0,
            f64::NAN,
            false,
            SampleMethod::None,
            GrowPolicy::DepthWise,
            None,
            None,
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
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `missing` - Value to consider missing.
    /// * `create_missing_branch` - Should missing be split out it's own separate branch?
    /// * `sample_method` - Specify the method that records should be sampled when training?
    /// * `evaluation_metric` - Define the evaluation metric to record at each iterations.
    /// * `early_stopping_rounds` - Number of rounds that must
    /// * `initialize_base_score` - If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        objective_type: ObjectiveType,
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
        monotone_constraints: Option<ConstraintMap>,
        subsample: f32,
        top_rate: f64,
        other_rate: f64,
        seed: u64,
        missing: f64,
        create_missing_branch: bool,
        sample_method: SampleMethod,
        grow_policy: GrowPolicy,
        evaluation_metric: Option<Metric>,
        early_stopping_rounds: Option<usize>,
        initialize_base_score: bool,
    ) -> Result<Self, ForustError> {
        let (base_score_, initialize_base_score_) = match base_score {
            Some(v) => (v, initialize_base_score),
            None => (0.5, true),
        };
        let booster = GradientBooster {
            objective_type,
            iterations,
            learning_rate,
            max_depth,
            max_leaves,
            l2,
            gamma,
            min_leaf_weight,
            base_score: base_score_,
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
            initialize_base_score: initialize_base_score_,
            evaluation_history: None,
            best_iteration: None,
            prediction_iteration: None,
            trees: Vec::new(),
            metadata: HashMap::new(),
        };
        booster.validate_parameters()?;
        Ok(booster)
    }

    fn validate_parameters(&self) -> Result<(), ForustError> {
        validate_positive_float_field!(self.learning_rate);
        validate_positive_float_field!(self.l2);
        validate_positive_float_field!(self.gamma);
        validate_positive_float_field!(self.min_leaf_weight);
        validate_positive_float_field!(self.subsample);
        validate_positive_float_field!(self.top_rate);
        validate_positive_float_field!(self.other_rate);
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
        let constraints_map = self
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();
        if self.create_missing_branch {
            let splitter = MissingBranchSplitter {
                l2: self.l2,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
            };
            self.fit_trees(y, sample_weight, data, &splitter, evaluation_data)?;
        } else {
            let splitter = MissingImputerSplitter {
                l2: self.l2,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
            };
            self.fit_trees(y, sample_weight, data, &splitter, evaluation_data)?;
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
            },
            Some(v) => *v,
        };
        metric_callables(&metric)
    }

    fn fit_trees<T: Splitter>(
        &mut self,
        y: &[f64],
        sample_weight: &[f64],
        data: &Matrix<f64>,
        splitter: &T,
        evaluation_data: Option<Vec<EvaluationData>>,
    ) -> Result<(), ForustError> {
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

        for i in 0..self.iterations {
            // We will eventually use the excluded index.
            let (chosen_index, _excluded_index) =
                self.sample_index(&mut rng, &data.index, &mut grad, &mut hess);
            let mut tree = Tree::new();

            tree.fit(
                &bdata,
                chosen_index,
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
                for (eval_i, (data, y, w, yhat)) in eval_sets.iter_mut().enumerate() {
                    self.update_predictions_inplace(yhat, &tree, data);
                    let (metric_fn, maximize) = self.get_metric_fn();
                    let m = metric_fn(y, yhat, w);
                    // If early stopping rounds are defined, and this is the first
                    // eval dataset, check if we want to stop
                    // or keep training.
                    if eval_i == 0 {
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
                                                break;
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
                if let Some(history) = &mut self.evaluation_history {
                    history.append_row(metrics);
                }
            }
            self.trees.push(tree);
            (grad, hess) = calc_grad_hess(y, &yhat, sample_weight);
        }
        Ok(())
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
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.get_prediction_trees().iter().for_each(|tree| {
            for (p_, val) in init_preds
                .iter_mut()
                .zip(tree.predict(data, parallel, &self.missing))
            {
                *p_ += val;
            }
        });
        init_preds
    }

    /// Predict the contributions matrix for the provided dataset.
    pub fn predict_contributions(
        &self,
        data: &Matrix<f64>,
        method: ContributionsMethod,
        parallel: bool,
    ) -> Vec<f64> {
        match method {
            ContributionsMethod::Average => self.predict_contributions_average(data, parallel),
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
            ContributionsMethod::Average => unreachable!(),
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

    /// Get the a reference to the trees for predicting, ensureing that the right number of
    /// trees are used.
    fn get_prediction_trees(&self) -> &[Tree] {
        let n_iterations = self.prediction_iteration.unwrap_or(self.trees.len());
        &self.trees[..n_iterations]
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

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
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
            let total: f32 = importance.values().sum();
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
            Ok(m) => Ok(m),
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

    /// Set prediction iterations.
    /// * `early_stopping_rounds` - Early stoppings rounds.
    pub fn set_prediction_iteration(mut self, prediction_iteration: Option<usize>) -> Self {
        self.prediction_iteration = prediction_iteration.map(|i| i + 1);
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
}
