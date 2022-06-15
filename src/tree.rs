use crate::data::{JaggedMatrix, Matrix};
use crate::histogram::HistogramMatrix;
use crate::histsplitter::HistogramSplitter;
use crate::node::{SplittableNode, TreeNode};
use crate::utils::pivot_on_split;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

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
    pub fn fit(
        &mut self,
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        splitter: &HistogramSplitter,
        max_leaves: usize,
        max_depth: usize,
        parallel: bool,
    ) {
        let mut index = data.index.to_owned();
        let mut n_nodes = 1;
        let grad_sum = grad.iter().sum();
        let hess_sum = hess.iter().sum();
        let root_gain = splitter.gain(grad_sum, hess_sum);
        let root_weight = splitter.weight(grad_sum, hess_sum);
        println!("Grad Sum: {}, Hess Sum: {}", grad_sum, hess_sum);
        println!("{}", hess.len());
        println!("{}", &hess[0..10].iter().sum::<f32>());
        let mut h_copy = hess.to_vec();
        h_copy.sort_by(|a,b| a.partial_cmp(b).unwrap());
        h_copy.dedup();
        println!("All of hess: {:?}", h_copy);
        println!("Sum Hess: {}", (hess.len() as f32) * h_copy[0]);
        println!("Sum Hess: {}", hess.iter().fold(0., |sum, val| sum + val));
        
        
        // Calculate the histograms for the root node.
        let root_hists = HistogramMatrix::new(data, cuts, grad, hess, &index, parallel, true);

        println!("{:?}", root_hists.0.get_col(20));

        let root_node = SplittableNode::new(
            0,
            root_hists,
            root_weight,
            root_gain,
            grad_sum,
            hess_sum,
            0,
            false,
            0,
            data.rows,
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

                // Try to find a valid split for this node.
                let split_info = splitter.best_split(node);

                // If this is None, this means there
                // are no more valid nodes.
                match split_info {
                    // If the split info is None, we can't split
                    // this node any further, make a leaf, and keep going.
                    None => {
                        n_leaves += 1;
                        self.nodes[n_idx] = node.as_leaf_node();
                        continue;
                    }
                    Some(info) => {
                        // If we can add two more leaves
                        // add two.
                        n_leaves += 2;
                        let left_idx = n_nodes;
                        let right_idx = left_idx + 1;

                        // We need to move all of the index's above and bellow our
                        // split value.
                        // pivot the sub array that this node has on our split value
                        let mut split_idx = pivot_on_split(
                            &mut index[node.start_idx..node.stop_idx],
                            data.get_col(info.split_feature),
                            info.split_bin,
                            info.missing_right,
                        );

                        // Calculate histograms
                        let total_recs = node.stop_idx - node.start_idx;
                        let n_right = total_recs - split_idx - 1;
                        let n_left = total_recs - n_right;

                        // Now that we have calculated the number of records
                        // add the start index, to make the split_index
                        // relative to the entire index array
                        split_idx += node.start_idx;

                        // Build the histograms for the smaller node.
                        let left_histograms: HistogramMatrix;
                        let right_histograms: HistogramMatrix;
                        if n_left < n_right {
                            left_histograms = HistogramMatrix::new(
                                data,
                                cuts,
                                grad,
                                hess,
                                &index[node.start_idx..split_idx],
                                parallel,
                                false,
                            );
                            right_histograms = HistogramMatrix::from_parent_child(
                                &node.histograms,
                                &left_histograms,
                            );
                        } else {
                            right_histograms = HistogramMatrix::new(
                                data,
                                cuts,
                                grad,
                                hess,
                                &index[split_idx..node.stop_idx],
                                parallel,
                                false,
                            );
                            left_histograms = HistogramMatrix::from_parent_child(
                                &node.histograms,
                                &right_histograms,
                            );
                        }

                        node.update_children(left_idx, right_idx, &info);

                        let left_node = SplittableNode::new(
                            left_idx,
                            left_histograms,
                            info.left_weight,
                            info.left_gain,
                            info.left_grad,
                            info.left_cover,
                            depth,
                            false,
                            node.start_idx,
                            split_idx,
                        );
                        let right_node = SplittableNode::new(
                            right_idx,
                            right_histograms,
                            info.right_weight,
                            info.right_gain,
                            info.right_grad,
                            info.right_cover,
                            depth,
                            false,
                            split_idx,
                            node.stop_idx,
                        );
                        growable.push_front(left_idx);
                        growable.push_front(right_idx);
                        // It has children, so we know it's going to be a parent node
                        self.nodes[n_idx] = node.as_parent_node();
                        self.nodes.push(TreeNode::Splittable(left_node));
                        self.nodes.push(TreeNode::Splittable(right_node));
                        n_nodes += 2;
                    }
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
                        if node.missing_right {
                            node_idx = node.right_child;
                        } else {
                            node_idx = node.left_child;
                        }
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
}

impl fmt::Display for Tree {
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
    use crate::objective::{LogLoss, ObjectiveFunction};
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
        let splitter = HistogramSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
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
    fn test_tree_fit_another() {
        let file =
            fs::read_to_string("../forust-profile/resources/contiguous_no_missing_no_sample.csv")
                .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("../forust-profile/resources/performance_no_sample.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);

        let n_cols = data_vec.len() / y.len();

        let data = Matrix::new(&data_vec, y.len(), n_cols);
        let data = Matrix::new(data.get_col(88), y.len(), 1);

        let splitter = HistogramSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        tree.fit(&bdata, &b.cuts, &g, &h, &splitter, usize::MAX, 5, true);

        println!("{}", tree);
        let preds = tree.predict(&data, false);
        println!("{:?}", &preds[0..10]);
    }
}
