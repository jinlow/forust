use crate::binning::bin_matrix;
use crate::data::Matrix;
use crate::errors::ForustError;
use crate::histsplitter::HistogramSplitter;
use crate::objective::{gradient_hessian_callables, ObjectiveType};
use crate::tree::Tree;
use serde::{Deserialize, Serialize};
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
    pub trees: Vec<Tree>,
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
            trees: Vec::new(),
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
        let splitter = HistogramSplitter {
            l2: self.l2,
            gamma: self.gamma,
            min_leaf_weight: self.min_leaf_weight,
            learning_rate: self.learning_rate,
        };
        let mut yhat = vec![self.base_score; y.len()];
        let (calc_grad, calc_hess) = gradient_hessian_callables(&self.objective_type);
        let mut grad = calc_grad(y, &yhat, sample_weight);
        let mut hess = calc_hess(y, &yhat, sample_weight);

        // Generate binned data
        let binned_data = bin_matrix(data, sample_weight, self.nbins)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        for _ in 0..self.iterations {
            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                &binned_data.cuts,
                &grad,
                &hess,
                &splitter,
                self.max_leaves,
                self.max_depth,
                self.parallel,
            );
            let preds = tree.predict(data, self.parallel);
            yhat = yhat.iter().zip(preds).map(|(i, j)| *i + j).collect();
            self.trees.push(tree);
            grad = calc_grad(y, &yhat, sample_weight);
            hess = calc_hess(y, &yhat, sample_weight);
        }
        Ok(())
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.trees.iter().for_each(|tree| {
            for (p_, val) in init_preds.iter_mut().zip(tree.predict(data, parallel)) {
                *p_ += val;
            }
        });
        init_preds
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_tree_fit() {
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
    }

    #[test]
    fn test_big_one() {
        let file =
            fs::read_to_string("../forust-profile/resources/contiguous_no_missing_no_sample.csv")
                .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("../forust-profile/resources/performance_no_sample.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let n_cols = data_vec.len() / y.len();

        let data = Matrix::new(&data_vec, y.len(), n_cols);
        // let data = Matrix::new(data.get_col(88), y.len(), 1);
        let sample_weight = vec![1.; y.len()];
        let mut booster = GradientBooster::default();
        booster.iterations = 1;
        booster.max_depth = 1;
        booster.fit(&data, &y, &sample_weight).unwrap();
        // let preds = booster.predict(&data, false);
        // println!("{}", booster.trees[0]);
        // println!("{}", booster.trees[0].nodes.len());
        // println!("{}", booster.trees.last().unwrap().nodes.len());
        // println!("{:?}", &preds[0..10]);
        println!("{}", booster.trees[0]);
    }
}
