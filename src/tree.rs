use crate::data::{JaggedMatrix, Matrix};
use crate::histogram::HistogramMatrix;
use crate::node::{Node, SplittableNode};
use crate::partial_dependence::tree_partial_dependence;
use crate::splitter::Splitter;
use crate::utils::fast_f64_sum;
use crate::utils::{gain, weight};
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        sample_weight: &[f64],
        splitter: &T,
        max_leaves: usize,
        max_depth: usize,
        parallel: bool,
        subsample: f32,
        rng: &mut StdRng,
    ) {
        // Recreating the index for each tree, ensures that the tree construction is faster
        // for the root node. This also ensures that sorting the records is always fast,
        // because we are starting from a nearly sorted array.
        let (mut index, gradient_sum, hessian_sum, sort) = if subsample == 1. {
            // We don't need to sort, if we are not sampling. This is because
            // the data is already sorted.
            (
                data.index.to_owned(),
                fast_f64_sum(grad),
                fast_f64_sum(hess),
                false,
            )
        } else {
            // Accumulate using f64 for numeric fidelity.
            let mut gs: f64 = 0.;
            let mut hs: f64 = 0.;
            let mut index = Vec::new();
            for ((i, g), h) in data
                .index
                .iter()
                .zip(grad)
                .zip(hess)
                .filter(|_| rng.gen_range(0.0..1.0) < subsample)
            {
                index.push(*i);
                gs += *g as f64;
                hs += *h as f64;
            }
            (index, gs as f32, hs as f32, true)
        };

        let mut n_nodes = 1;
        let root_gain = gain(&splitter.get_l2(), gradient_sum, hessian_sum);
        let root_weight = weight(&splitter.get_l2(), gradient_sum, hessian_sum);
        // Calculate the histograms for the root node.
        let root_hists = HistogramMatrix::new(data, cuts, grad, hess, sample_weight, &index, parallel, sort);
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
        let mut growable: VecDeque<SplittableNode> = VecDeque::new();
        growable.push_front(root_node);
        while !growable.is_empty() {
            if n_leaves >= max_leaves {
                break;
            }

            // We know there is a value here, because of how the
            // while loop is setup.
            // Grab a splitable node from the stack
            // If we can split it, and update the corresponding
            // tree nodes children.
            let mut node = growable
                .pop_back()
                .expect("Growable buffer should not be empty.");
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
                &n_nodes, &mut node, &mut index, data, cuts, grad, hess, sample_weight, parallel,
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
                    growable.push_front(n)
                }
            }
        }
    }
    pub fn predict_contributions_row(
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

    fn predict_contributions_single_threaded(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        weights: &[f64],
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .iter()
            .zip(contribs.chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row(&data.get_row(*row), contribs, weights, missing)
            })
    }

    fn predict_contributions_parallel(
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
                self.predict_contributions_row(&data.get_row(*row), contribs, weights, missing)
            })
    }

    pub fn predict_contributions(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        weights: &[f64],
        parallel: bool,
        missing: &f64,
    ) {
        if parallel {
            self.predict_contributions_parallel(data, contribs, weights, missing)
        } else {
            self.predict_contributions_single_threaded(data, contribs, weights, missing)
        }
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
            w = ((left_node.hessian_sum as f64
                * self.distribute_node_leaf_weights(node.left_child, weights))
                + (right_node.hessian_sum as f64
                    * self.distribute_node_leaf_weights(node.right_child, weights)))
                / (node.hessian_sum as f64);
        }
        weights[i] = w;
        w
    }
    pub fn distribute_leaf_weights(&self) -> Vec<f64> {
        let mut weights = vec![0.; self.nodes.len()];
        self.distribute_node_leaf_weights(0, &mut weights);
        weights
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
        let mut rng = StdRng::seed_from_u64(0);
        tree.fit(
            &bdata,
            &b.cuts,
            &g,
            &h,
            &w,
            &splitter,
            usize::MAX,
            5,
            true,
            0.5,
            &mut rng,
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
        let mut rng = StdRng::seed_from_u64(0);
        tree.fit(
            &bdata,
            &b.cuts,
            &g,
            &h,
            &w,
            &splitter,
            usize::MAX,
            5,
            true,
            1.,
            &mut rng,
        );

        // println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions(&data, &mut contribs, &weights, false, &f64::NAN);
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

        let mut rng = StdRng::seed_from_u64(0);
        tree.fit(
            &bdata,
            &b.cuts,
            &g,
            &h,
            &w,
            &splitter,
            usize::MAX,
            5,
            true,
            1.,
            &mut rng,
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

        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions(&data, &mut contribs, &weights, false, &f64::NAN);
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
}
