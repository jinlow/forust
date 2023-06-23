use crate::data::{JaggedMatrix, Matrix};
use crate::gradientbooster::GrowPolicy;
use crate::grower::Grower;
use crate::histogram::HistogramMatrix;
use crate::node::{Node, SplittableNode};
use crate::partial_dependence::tree_partial_dependence;
use crate::sampler::SampleMethod;
use crate::splitter::Splitter;
use crate::utils::fast_f64_sum;
use crate::utils::{gain, weight};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt::{self, Display};

#[derive(Deserialize, Serialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Tree { nodes: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit<T: Splitter>(
        &mut self,
        data: &Matrix<u16>,
        mut index: Vec<usize>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        splitter: &T,
        max_leaves: usize,
        max_depth: usize,
        parallel: bool,
        sample_method: &SampleMethod,
        grow_policy: &GrowPolicy,
    ) {
        // Recreating the index for each tree, ensures that the tree construction is faster
        // for the root node. This also ensures that sorting the records is always fast,
        // because we are starting from a nearly sorted array.
        let (gradient_sum, hessian_sum, sort) = match sample_method {
            // We don't need to sort, if we are not sampling. This is because
            // the data is already sorted.
            SampleMethod::None => (fast_f64_sum(grad), fast_f64_sum(hess), false),
            _ => {
                // Accumulate using f64 for numeric fidelity.
                let mut gs: f64 = 0.;
                let mut hs: f64 = 0.;
                for i in index.iter() {
                    let i_ = *i;
                    gs += grad[i_] as f64;
                    hs += hess[i_] as f64;
                }
                (gs as f32, hs as f32, true)
            }
        };

        let mut n_nodes = 1;
        let root_gain = gain(&splitter.get_l2(), gradient_sum, hessian_sum);
        let root_weight = weight(&splitter.get_l2(), gradient_sum, hessian_sum);
        // Calculate the histograms for the root node.
        let root_hists = HistogramMatrix::new(data, cuts, grad, hess, &index, parallel, sort);
        let root_node = SplittableNode::new(
            0,
            root_hists,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            0,
            0,
            index.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        // Add the first node to the tree nodes.
        self.nodes.push(root_node.as_node());
        let mut n_leaves = 1;

        let mut growable: Box<dyn Grower> = match grow_policy {
            GrowPolicy::DepthWise => Box::<VecDeque<SplittableNode>>::default(),
            GrowPolicy::LossGuide => Box::<BinaryHeap<SplittableNode>>::default(),
        };

        growable.add_node(root_node);
        while !growable.is_empty() {
            if n_leaves >= max_leaves {
                break;
            }

            // We know there is a value here, because of how the
            // while loop is setup.
            // Grab a splitable node from the stack
            // If we can split it, and update the corresponding
            // tree nodes children.
            let mut node = growable.get_next_node();
            let n_idx = node.num;
            // This will only be splittable nodes

            let depth = node.depth + 1;

            // If we have hit max depth, skip this node
            // but keep going, because there may be other
            // valid shallower nodes.
            if depth > max_depth {
                // self.nodes[n_idx] = node.as_leaf_node();
                continue;
            }

            // For max_leaves, subtract 1 from the n_leaves
            // every time we pop from the growable stack
            // then, if we can add two children, add two to
            // n_leaves. If we can't split the node any
            // more, then just add 1 back to n_leaves
            n_leaves -= 1;

            let new_nodes = splitter.split_node(
                &n_nodes, &mut node, &mut index, data, cuts, grad, hess, parallel,
            );

            let n_new_nodes = new_nodes.len();
            if n_new_nodes == 0 {
                n_leaves += 1;
            } else {
                self.nodes[n_idx].make_parent_node(node);
                n_leaves += n_new_nodes;
                n_nodes += n_new_nodes;
                for n in new_nodes {
                    self.nodes.push(n.as_node());
                    if !n.is_missing_leaf {
                        growable.add_node(n)
                    }
                }
            }
        }
    }

    // Branch average difference predictions
    pub fn predict_contributions_row_midpoint_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let child = &self.nodes[child_idx];
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[node.right_child]
            } else {
                &self.nodes[node.left_child]
            };
            let mid = (child.weight_value * child.hessian_sum
                + other_child.weight_value * other_child.hessian_sum)
                / (child.hessian_sum + other_child.hessian_sum);
            let delta = child.weight_value - mid;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // Branch difference predictions.
    pub fn predict_contributions_row_branch_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[node.right_child]
            } else {
                &self.nodes[node.left_child]
            };
            let delta = self.nodes[child_idx].weight_value - other_child.weight_value;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // How does the travelled childs weight change relative to the
    // mode branch.
    pub fn predict_contributions_row_mode_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let left_node = &self.nodes[node.left_child];
            let right_node = &self.nodes[node.right_child];
            let child_weight = self.nodes[child_idx].weight_value;

            let delta = if left_node.hessian_sum == right_node.hessian_sum {
                0.
            } else if left_node.hessian_sum > right_node.hessian_sum {
                child_weight - left_node.weight_value
            } else {
                child_weight - right_node.weight_value
            };
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    pub fn predict_contributions_row_weight(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += self.nodes[0].weight_value as f64;
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = self.nodes[node_idx].weight_value as f64;
            let child_weight = self.nodes[child_idx].weight_value as f64;
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_weight(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row_weight(&data.get_row(*row), contribs, missing)
            })
    }

    /// This is the method that XGBoost uses.
    pub fn predict_contributions_row_average(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        weights: &[f64],
        missing: &f64,
    ) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += weights[0];
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = weights[node_idx];
            let child_weight = weights[child_idx];
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_average(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        weights: &[f64],
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row_average(
                    &data.get_row(*row),
                    contribs,
                    weights,
                    missing,
                )
            })
    }

    fn predict_row(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.weight_value as f64;
            } else {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
            }
        }
    }

    pub fn predict_row_from_row_slice(&self, row: &[f64], missing: &f64) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.weight_value as f64;
            } else {
                node_idx = node.get_child_idx(&row[node.split_feature], missing);
            }
        }
    }

    fn predict_single_threaded(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index
            .iter()
            .map(|i| self.predict_row(data, *i, missing))
            .collect()
    }

    fn predict_parallel(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index
            .par_iter()
            .map(|i| self.predict_row(data, *i, missing))
            .collect()
    }

    pub fn predict(&self, data: &Matrix<f64>, parallel: bool, missing: &f64) -> Vec<f64> {
        if parallel {
            self.predict_parallel(data, missing)
        } else {
            self.predict_single_threaded(data, missing)
        }
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64, missing: &f64) -> f64 {
        tree_partial_dependence(self, 0, feature, value, 1.0, missing)
    }
    fn distribute_node_leaf_weights(&self, i: usize, weights: &mut [f64]) -> f64 {
        let node = &self.nodes[i];
        let mut w = node.weight_value as f64;
        if !node.is_leaf {
            let left_node = &self.nodes[node.left_child];
            let right_node = &self.nodes[node.right_child];
            w = left_node.hessian_sum as f64
                * self.distribute_node_leaf_weights(node.left_child, weights);
            w += right_node.hessian_sum as f64
                * self.distribute_node_leaf_weights(node.right_child, weights);
            // If this a tree with a missing branch.
            if node.has_missing_branch() {
                let missing_node = &self.nodes[node.missing_node];
                w += missing_node.hessian_sum as f64
                    * self.distribute_node_leaf_weights(node.missing_node, weights);
            }
            w /= node.hessian_sum as f64;
        }
        weights[i] = w;
        w
    }
    pub fn distribute_leaf_weights(&self) -> Vec<f64> {
        let mut weights = vec![0.; self.nodes.len()];
        self.distribute_node_leaf_weights(0, &mut weights);
        weights
    }

    fn calc_feature_node_stats<F>(
        &self,
        calc_stat: &F,
        node: &Node,
        stats: &mut HashMap<usize, (f32, usize)>,
    ) where
        F: Fn(&Node) -> f32,
    {
        if node.is_leaf {
            return;
        }
        stats
            .entry(node.split_feature)
            .and_modify(|(v, c)| {
                *v += calc_stat(node);
                *c += 1;
            })
            .or_insert((calc_stat(node), 1));
        self.calc_feature_node_stats(calc_stat, &self.nodes[node.left_child], stats);
        self.calc_feature_node_stats(calc_stat, &self.nodes[node.right_child], stats);
        if node.has_missing_branch() {
            self.calc_feature_node_stats(calc_stat, &self.nodes[node.missing_node], stats);
        }
    }

    fn get_node_stats<F>(&self, calc_stat: &F, stats: &mut HashMap<usize, (f32, usize)>)
    where
        F: Fn(&Node) -> f32,
    {
        self.calc_feature_node_stats(calc_stat, &self.nodes[0], stats);
    }

    pub fn calculate_importance_weight(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|_: &Node| 1., stats);
    }

    pub fn calculate_importance_gain(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.split_gain, stats);
    }

    pub fn calculate_importance_cover(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.hessian_sum, stats);
    }
}

impl Display for Tree {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut print_buffer: Vec<usize> = vec![0];
        let mut r = String::new();
        while !print_buffer.is_empty() {
            // This will always be populated, because we confirm
            // that the buffer is not empty.
            let idx = print_buffer.pop().unwrap();
            let node = &self.nodes[idx];
            if node.is_leaf {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
            } else {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
                print_buffer.push(node.right_child);
                print_buffer.push(node.left_child);
                if node.has_missing_branch() {
                    print_buffer.push(node.missing_node);
                }
            }
        }
        write!(f, "{}", r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::sampler::{RandomSampler, Sampler};
    use crate::splitter::MissingImputerSplitter;
    use crate::utils::precision_round;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::fs;
    #[test]
    fn test_tree_fit_with_subsample() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let mut g = LogLoss::calc_grad(&y, &yhat, &w);
        let mut h = LogLoss::calc_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let mut rng = StdRng::seed_from_u64(0);
        let (index, excluded) =
            RandomSampler::new(0.5).sample(&mut rng, &data.index, &mut g, &mut h);
        assert!(excluded.len() > 0);
        tree.fit(
            &bdata,
            index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::Random,
            &GrowPolicy::DepthWise,
        );
    }

    #[test]
    fn test_tree_fit() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::None,
            &GrowPolicy::DepthWise,
        );

        // println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_monotone() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);
        println!("GRADIENT -- {:?}", h);

        let data_ = Matrix::new(&data_vec, 891, 5);
        let data = Matrix::new(data_.get_col(1), 891, 1);
        let map = ConstraintMap::from([(0, Constraint::Negative)]);
        let splitter = MissingImputerSplitter {
            l2: 1.0,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: map,
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        tree.fit(
            &bdata,
            data.index.to_owned(),
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::None,
            &GrowPolicy::DepthWise,
        );

        // println!("{}", tree);
        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pred_data_vec.dedup();
        let pred_data = Matrix::new(&pred_data_vec, pred_data_vec.len(), 1);

        let preds = tree.predict(&pred_data, false, &f64::NAN);
        let increasing = preds.windows(2).all(|a| a[0] >= a[1]);
        assert!(increasing);

        let weights = tree.distribute_leaf_weights();

        // Average contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_lossguide() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: false,
            constraints_map: ConstraintMap::new(),
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            usize::MAX,
            true,
            &SampleMethod::None,
            &GrowPolicy::LossGuide,
        );

        println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        // assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }
}
