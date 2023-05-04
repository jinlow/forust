use crate::binning::bin_matrix;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::ForustError;
use crate::objective::{gradient_hessian_callables, ObjectiveType};
use crate::splitter::MissingImputerSplitter;
use crate::tree::Tree;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fs;
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
/// * `allow_missing_splits` - Allow for splits to be made such that all missing values go
///   down one branch, and all non-missing values go down the other, if this results
///   in the greatest reduction of loss. If this is false, splits will only be made on non
///   missing values.
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
    pub seed: u64,
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    pub trees: Vec<Tree>,
    metadata: HashMap<String, String>,
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
            0.5,
            256,
            true,
            true,
            None,
            1.,
            0,
            f64::NAN,
        )
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
    /// * `base_score` - The initial prediction value of the model.
    /// * `nbins` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///   between the training features and the target variable.
    /// * `subsample` - Percent of records to randomly sample at each iteration when training a tree.
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `missing` - Value to consider missing.
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
        base_score: f64,
        nbins: u16,
        parallel: bool,
        allow_missing_splits: bool,
        monotone_constraints: Option<ConstraintMap>,
        subsample: f32,
        seed: u64,
        missing: f64,
    ) -> Self {
        GradientBooster {
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
            seed,
            missing,
            trees: Vec::new(),
            metadata: HashMap::new(),
        }
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
    ) -> Result<(), ForustError> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let constraints_map = self
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();
        let splitter = MissingImputerSplitter {
            l2: self.l2,
            gamma: self.gamma,
            min_leaf_weight: self.min_leaf_weight,
            learning_rate: self.learning_rate,
            allow_missing_splits: self.allow_missing_splits,
            constraints_map,
        };
        let mut yhat = vec![self.base_score; y.len()];
        let (calc_grad, calc_hess) = gradient_hessian_callables(&self.objective_type);
        let mut grad = calc_grad(y, &yhat, sample_weight);
        let mut hess = calc_hess(y, &yhat, sample_weight);

        // Generate binned data
        // TODO
        // In scikit-learn, they sample 200_000 records for generating the bins.
        // we could consider that, especially if this proved to be a large bottleneck...
        let binned_data = bin_matrix(data, sample_weight, self.nbins, self.missing)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        for _ in 0..self.iterations {
            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                &binned_data.cuts,
                &grad,
                &hess,
                sample_weight,
                &splitter,
                self.max_leaves,
                self.max_depth,
                self.parallel,
                self.subsample,
                &mut rng,
            );
            let preds = tree.predict(data, self.parallel, &self.missing);
            yhat = yhat.iter().zip(preds).map(|(i, j)| *i + j).collect();
            self.trees.push(tree);
            grad = calc_grad(y, &yhat, sample_weight);
            hess = calc_hess(y, &yhat, sample_weight);
        }
        Ok(())
    }

    /// Fit the gradient booster on a provided dataset without any weights.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `y` - Either a pandas Series, or a 1 dimensional numpy array.
    pub fn fit_unweighted(&mut self, data: &Matrix<f64>, y: &[f64]) -> Result<(), ForustError> {
        let w = vec![1.0; data.rows];
        self.fit(data, y, &w)
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.trees.iter().for_each(|tree| {
            for (p_, val) in init_preds
                .iter_mut()
                .zip(tree.predict(data, parallel, &self.missing))
            {
                *p_ += val;
            }
        });
        init_preds
    }

    // pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
    //     // After we disconvered it's faster materializing the row once, and then
    //     // Passing that to each tree, let's see if we can do the same with the booster
    //     // prediction...
    //     // Clean this up..
    //     let mut init_preds = vec![self.base_score; data.rows];
    //     if parallel {
    //         init_preds.par_iter_mut().enumerate().for_each(|(i, p)| {
    //             let pred_row = data.get_row(i);
    //             for t in &self.trees {
    //                 *p += t.predict_row_from_row_slice(&pred_row);
    //             }
    //         });
    //     } else {
    //         init_preds.iter_mut().enumerate().for_each(|(i, p)| {
    //             let pred_row = data.get_row(i);
    //             for t in &self.trees {
    //                 *p += t.predict_row_from_row_slice(&pred_row);
    //             }
    //         });
    //     }
    //     init_preds
    // }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    pub fn predict_contributions(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let weights: Vec<Vec<f64>> = if parallel {
            self.trees
                .par_iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        } else {
            self.trees
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
                    self.trees.iter().zip(weights.iter()).for_each(|(t, w)| {
                        t.predict_contributions_row(&r_, c, w, &self.missing);
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.trees.iter().zip(weights.iter()).for_each(|(t, w)| {
                        t.predict_contributions_row(&r_, c, w, &self.missing);
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
        let pd: f64 = self
            .trees
            .iter()
            .map(|t| t.value_partial_dependence(feature, value, &self.missing))
            .sum();
        pd + self.base_score
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

    /// Set the nbins on the booster.
    /// * `nbins` - The nummber of bins used for partitioning the data of the booster.
    pub fn set_nbins(mut self, nbins: u16) -> Self {
        self.nbins = nbins;
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
        let mut booster = GradientBooster::default();
        booster.iterations = 10;
        booster.nbins = 300;
        booster.max_depth = 3;
        booster.subsample = 0.5;
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, false);
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
        let mut booster = GradientBooster::default();
        booster.iterations = 10;
        booster.nbins = 300;
        booster.max_depth = 3;
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, false);
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
        let mut booster = GradientBooster::default();
        booster.iterations = 10;
        booster.nbins = 300;
        booster.max_depth = 3;
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight).unwrap();
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
