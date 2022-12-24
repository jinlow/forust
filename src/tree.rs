use crate::data::{JaggedMatrix, Matrix};
use crate::histogram::HistogramMatrix;
use crate::node::{SplittableNode, TreeNode};
use crate::partial_dependence::tree_partial_dependence;
use crate::splitter::Splitter;
use crate::utils::fast_f64_sum;
use crate::utils::{gain, weight};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::{self, Display};

#[derive(Deserialize, Serialize)]
pub struct Tree {
    pub nodes: Vec<TreeNode>,
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
        splitter: &T,
        max_leaves: usize,
        max_depth: usize,
        parallel: bool,
    ) {
        // Recreating the index for each tree, ensures that the tree construction is faster
        // for the root node. This also ensures that sorting the records is always fast,
        // because we are starting from a nearly sorted array.
        let mut index = data.index.to_owned();
        let mut n_nodes = 1;
        let gradient_sum = fast_f64_sum(grad);
        let hessian_sum = fast_f64_sum(hess);
        let root_gain = gain(&splitter.get_l2(), gradient_sum, hessian_sum);
        let root_weight = weight(&splitter.get_l2(), gradient_sum, hessian_sum);
        // Calculate the histograms for the root node.
        let root_hists = HistogramMatrix::new(data, cuts, grad, hess, &index, parallel, true);
        let root_node = SplittableNode::new(
            0,
            root_hists,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            0,
            0,
            data.rows,
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        // Add the first node to the tree nodes.
        self.nodes.push(TreeNode::Splittable(root_node));
        let mut n_leaves = 1;
        let mut growable = VecDeque::new();
        growable.push_front(0);
        while !growable.is_empty() {
            if n_leaves >= max_leaves {
                // Clear the rest of the node idxs that
                // are not needed.
                for i in growable.iter() {
                    let n = &self.nodes[*i];
                    if let TreeNode::Splittable(node) = n {
                        self.nodes[*i] = node.as_leaf_node();
                    }
                }
                break;
            }

            // We know there is a value here, because of how the
            // while loop is setup.
            let n_idx = growable
                .pop_back()
                .expect("Growable buffer should not be empty.");

            let n = self.nodes.get_mut(n_idx);
            // This will only be splittable nodes
            if let Some(TreeNode::Splittable(node)) = n {
                let depth = node.depth + 1;

                // If we have hit max depth, skip this node
                // but keep going, because there may be other
                // valid shallower nodes.
                if depth > max_depth {
                    self.nodes[n_idx] = node.as_leaf_node();
                    continue;
                }

                // For max_leaves, subtract 1 from the n_leaves
                // every time we pop from the growable stack
                // then, if we can add two children, add two to
                // n_leaves. If we can't split the node any
                // more, then just add 1 back to n_leaves
                n_leaves -= 1;

                let new_nodes = splitter.split_node(
                    &n_nodes,
                    node,
                    &mut index,
                    data,
                    cuts,
                    grad,
                    hess,
                    parallel,
                    &mut growable,
                );

                let n_new_nodes = new_nodes.len();
                if n_new_nodes == 0 {
                    n_leaves += 1;
                    self.nodes[n_idx] = node.as_leaf_node();
                } else {
                    self.nodes[n_idx] = node.as_parent_node();
                    n_leaves += n_new_nodes;
                    n_nodes += n_new_nodes;
                    self.nodes.extend(new_nodes);
                }
            }
        }
    }

    pub fn predict_row(&self, data: &Matrix<f64>, row: usize) -> f64 {
        let mut node_idx = 0;
        loop {
            let n = &self.nodes[node_idx];
            match n {
                TreeNode::Leaf(node) => {
                    return node.weight_value as f64;
                }
                TreeNode::Parent(node) => {
                    let v = data.get(row, node.split_feature);
                    if v.is_nan() {
                        node_idx = node.missing_node;
                    } else if v < &node.split_value {
                        node_idx = node.left_child;
                    } else if v >= &node.split_value {
                        node_idx = node.right_child;
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    fn predict_single_threaded(&self, data: &Matrix<f64>) -> Vec<f64> {
        data.index
            .iter()
            .map(|i| self.predict_row(data, *i))
            .collect()
    }

    fn predict_parallel(&self, data: &Matrix<f64>) -> Vec<f64> {
        data.index
            .par_iter()
            .map(|i| self.predict_row(data, *i))
            .collect()
    }

    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        if parallel {
            self.predict_parallel(data)
        } else {
            self.predict_single_threaded(data)
        }
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        tree_partial_dependence(self, 0, feature, value, 1.0)
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
            let n = &self.nodes[idx];
            match n {
                TreeNode::Leaf(node) => {
                    r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), n).as_str();
                }
                TreeNode::Parent(node) => {
                    r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), n).as_str();
                    print_buffer.push(node.right_child);
                    print_buffer.push(node.left_child);
                }
                TreeNode::Splittable(node) => {
                    r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), n).as_str();
                    print_buffer.push(node.right_child);
                    print_buffer.push(node.left_child);
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
    use crate::splitter::MissingImputerSplitter;
    use std::fs;
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

        let b = bin_matrix(&data, &w, 300).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        tree.fit(&bdata, &b.cuts, &g, &h, &splitter, usize::MAX, 5, true);

        println!("{}", tree);
        let preds = tree.predict(&data, false);
        println!("{:?}", &preds[0..10]);
        assert_eq!(25, tree.nodes.len())
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

        let b = bin_matrix(&data, &w, 300).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        tree.fit(&bdata, &b.cuts, &g, &h, &splitter, usize::MAX, 5, true);

        // println!("{}", tree);
        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pred_data_vec.dedup();
        let pred_data = Matrix::new(&pred_data_vec, pred_data_vec.len(), 1);

        let preds = tree.predict(&pred_data, false);
        let increasing = preds.windows(2).all(|a| a[0] >= a[1]);
        assert!(increasing);
    }
}
