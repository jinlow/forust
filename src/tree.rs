use crate::data::{Matrix, MatrixData};
use crate::node::{SplittableNode, TreeNode};
use crate::splitter::Splitter;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;

pub struct Tree<T: MatrixData<T>> {
    pub nodes: Vec<TreeNode<T>>,
}

impl<T: MatrixData<T>> Default for Tree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: MatrixData<T>> Tree<T> {
    pub fn new() -> Self {
        Tree { nodes: Vec::new() }
    }

    pub fn fit<S: Splitter<T>>(
        &mut self,
        data: &Matrix<T>,
        grad: &[T],
        hess: &[T],
        splitter: &S,
        max_leaves: usize,
        max_depth: usize,
        index: &mut [usize],
    ) {
        let mut n_nodes = 1;
        let grad_sum: T = grad.iter().copied().sum();
        let hess_sum: T = hess.iter().copied().sum();
        let root_gain = splitter.gain(grad_sum, hess_sum);
        let root_weight = splitter.weight(grad_sum, hess_sum);
        let root_node = SplittableNode::new(
            0,
            // data.index.to_owned(),
            root_weight,
            root_gain,
            grad_sum,
            hess_sum,
            0,
            0,
            data.rows,
        );
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
            // This should only be splittable nodes
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
                let split_info = splitter.best_split(node, data, grad, hess, index);

                // If this is None, this means there
                // are no more valid nodes.
                match split_info {
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
                        node.update_children(left_idx, right_idx, &info);
                        let left_node = SplittableNode::new(
                            left_idx,
                            info.left_weight,
                            info.left_gain,
                            info.left_grad,
                            info.left_cover,
                            depth,
                            info.left_start_idx,
                            info.left_stop_idx,
                        );
                        let right_node = SplittableNode::new(
                            right_idx,
                            info.right_weight,
                            info.right_gain,
                            info.right_grad,
                            info.right_cover,
                            depth,
                            info.right_start_idx,
                            info.right_stop_idx,
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

    pub fn predict_row(&self, data: &Matrix<T>, row: usize) -> T {
        let mut node_idx = 0;
        loop {
            let n = &self.nodes[node_idx];
            match n {
                TreeNode::Leaf(node) => {
                    return node.weight_value;
                }
                TreeNode::Parent(node) => {
                    if data.get(row, node.split_feature) < &node.split_value {
                        node_idx = node.left_child;
                    } else {
                        node_idx = node.right_child;
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    fn predict_single_threaded(&self, data: &Matrix<T>) -> Vec<T> {
        data.index
            .iter()
            .map(|i| self.predict_row(data, *i))
            .collect()
    }

    fn predict_parallel(&self, data: &Matrix<T>) -> Vec<T> {
        data.index
            .par_iter()
            .map(|i| self.predict_row(data, *i))
            .collect()
    }

    pub fn predict(&self, data: &Matrix<T>, parallel: bool) -> Vec<T> {
        if parallel {
            self.predict_parallel(data)
        } else {
            self.predict_single_threaded(data)
        }
    }
}

impl<T> fmt::Display for Tree<T>
where
    T: FromStr + std::fmt::Display + MatrixData<T>,
    <T as FromStr>::Err: 'static + std::error::Error,
{
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
    use crate::exactsplitter::ExactSplitter;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use std::fs;
    #[test]
    fn test_tree_fit() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat, &w);
        let h = LogLoss::calc_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = ExactSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
        };
        let mut tree = Tree::new();
        let mut index = data.index.to_owned();
        let index = index.as_mut();
        tree.fit(&data, &g, &h, &splitter, usize::MAX, 5, index);
        println!("{}", tree);
        let preds = tree.predict(&data, false);
        println!("{:?}", &preds[0..10]);
        assert_eq!(25, tree.nodes.len())
    }
}
