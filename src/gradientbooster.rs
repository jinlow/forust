use crate::binning::bin_matrix;
use crate::data::{Matrix, MatrixData};
use crate::errors::ForustError;
use crate::histsplitter::HistogramSplitter;
use crate::objective::{gradient_hessian_callables, ObjectiveType};
use crate::tree::Tree;

pub struct GradientBooster<T: MatrixData<T>> {
    pub objective_type: ObjectiveType,
    pub iterations: usize,
    pub learning_rate: T,
    pub max_depth: usize,
    pub max_leaves: usize,
    pub l2: T,
    pub gamma: T,
    pub min_leaf_weight: T,
    pub base_score: T,
    pub nbins: u16,
    pub parallel: bool,
    pub trees: Vec<Tree<T>>,
}

impl Default for GradientBooster<f64> {
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

impl Default for GradientBooster<f32> {
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

impl<T> GradientBooster<T>
where
    T: MatrixData<T>,
{
    pub fn new(
        objective_type: ObjectiveType,
        iterations: usize,
        learning_rate: T,
        max_depth: usize,
        max_leaves: usize,
        l2: T,
        gamma: T,
        min_leaf_weight: T,
        base_score: T,
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

    pub fn fit(
        &mut self,
        data: &Matrix<T>,
        y: &[T],
        sample_weight: &[T],
        parallel: bool,
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
        let mut index = data.index.to_owned();

        // Generate binned data
        let binned_data = bin_matrix(&data, sample_weight, self.nbins)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        let index = index.as_mut();
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
                index,
                self.parallel,
            );
            yhat = yhat
                .iter()
                .zip(tree.predict(data, parallel))
                .map(|(i, j)| *i + j)
                .collect();
            self.trees.push(tree);
            grad = calc_grad(y, &yhat, sample_weight);
            hess = calc_hess(y, &yhat, sample_weight);
        }
        Ok(())
    }

    pub fn predict(&self, data: &Matrix<T>, parallel: bool) -> Vec<T> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.trees.iter().for_each(|tree| {
            for (p_, val) in init_preds.iter_mut().zip(tree.predict(data, parallel)) {
                *p_ += val;
            }
        });
        init_preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn test_tree_fit() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        let mut booster = GradientBooster::default();
        booster.iterations = 10;
        booster.nbins = 300;
        booster.parallel = false;
        let sample_weight = vec![1.; y.len()];
        booster.fit(&data, &y, &sample_weight, true).unwrap();
        let preds = booster.predict(&data, false);
        assert_eq!(39, booster.trees[0].nodes.len());
        assert_eq!(23, booster.trees.last().unwrap().nodes.len());
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }
}
