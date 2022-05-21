use crate::data::{Matrix, MatrixData};
use crate::node::Node;
use crate::splitter::Splitter;
use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;

struct Tree<T: MatrixData<T>, S: Splitter<T>> {
    splitter: S,
    max_leaves: usize,
    max_depth: usize,
    nodes: Vec<Node<T>>,
    n_nodes: usize,
}

impl<T: MatrixData<T>, S: Splitter<T>> Tree<T, S> {
    pub fn new(splitter: S, max_leaves: usize, max_depth: usize) -> Self {
        Tree {
            splitter,
            max_leaves,
            max_depth,
            nodes: Vec::new(),
            n_nodes: 1,
        }
    }

    pub fn fit(&mut self, data: &Matrix<T>, grad: &[T], hess: &[T]) {
        let grad_sum: T = grad.iter().copied().sum();
        let hess_sum: T = hess.iter().copied().sum();
        let root_gain = self.splitter.gain(grad_sum, hess_sum);
        let root_weight = self.splitter.weight(grad_sum, hess_sum);
        let root_node = Node::new(
            0,
            (0..(data.rows)).collect(),
            root_weight,
            root_gain,
            grad_sum,
            hess_sum,
            0,
        );
        self.nodes.push(root_node);
        let mut n_leaves = 1;
        let mut growable = VecDeque::new();
        growable.push_front(0);
        while !growable.is_empty() {
            if n_leaves >= self.max_leaves {
                // Clear the rest of the node idxs that
                // are not needed.
                for i in growable.iter() {
                    self.nodes[*i].node_idxs = Vec::new();
                }
                break;
            }

            // We know there is a value here, because of how the
            // while loop is setup.
            let n_idx = growable.pop_back().unwrap();

            let n = &mut self.nodes[n_idx];
            let depth = n.depth + 1;

            // If we have hit max depth, skip this node
            // but keep going, because there may be other
            // valid shallower nodes.
            if depth > self.max_depth {
                n.node_idxs = Vec::new();
                continue;
            }

            // For max_leaves, subtract 1 from the n_leaves
            // everytime we pop from the growable stack
            // then, if we can add two children, add two to
            // n_leaves. If we can't split the node any
            // more, then just add 1 back to n_leaves
            n_leaves -= 1;

            // Try to find a valid split for this node.
            let split_info = self.splitter.best_split(n, data, grad, hess);

            // If this is None, this means there
            // are no more valid nodes.
            match split_info {
                None => {
                    n_leaves += 1;
                    continue;
                }
                Some(info) => {
                    // If we can add two more leaves
                    // add two.
                    n_leaves += 2;
                    let left_idx = self.n_nodes;
                    let right_idx = left_idx + 1;
                    n.update_children(left_idx, right_idx, &info);
                    let left_node = Node::new(
                        left_idx,
                        info.left_idxs,
                        info.left_weight,
                        info.left_gain,
                        info.left_grad,
                        info.left_cover,
                        depth,
                    );
                    let right_node = Node::new(
                        right_idx,
                        info.right_idxs.to_vec(),
                        info.right_weight,
                        info.right_gain,
                        info.right_grad,
                        info.right_cover,
                        depth,
                    );
                    growable.push_front(left_idx);
                    growable.push_front(right_idx);
                    self.nodes.push(left_node);
                    self.nodes.push(right_node);
                    self.n_nodes += 2;
                }
            }
        }
    }

    pub fn predict_row(&self, x_row: &[T]) -> T {
        let mut node_idx = 0;
        loop {
            let n = &self.nodes[node_idx];
            if n.is_leaf() {
                return n.weight_value;
            }
            if x_row[n.split_feature_.unwrap()] < n.split_value_.unwrap() {
                node_idx = n.left_child_.unwrap();
            } else {
                node_idx = n.right_child_.unwrap();
            }
        }
    }
}

impl<T, S> fmt::Display for Tree<T, S>
where
    T: FromStr + std::fmt::Display + MatrixData<T>,
    <T as FromStr>::Err: 'static + std::error::Error,
    S: Splitter<T>,
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
            r += format!("{}{}\n", "      ".repeat(n.depth).as_str(), n).as_str();
            if !n.is_leaf() {
                print_buffer.push(n.right_child_.unwrap());
                print_buffer.push(n.left_child_.unwrap());
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
        let mut data_vec: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<i64>().unwrap() as f64)
            .collect();
        let yhat = vec![0.5; y.len()];
        let g = LogLoss::calc_grad(&y, &yhat);
        let h = LogLoss::calc_hess(&y, &yhat);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = ExactSplitter {
            l2: 1.0,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            min_split_gain: 0.0,
        };
        let mut tree = Tree::new(splitter, usize::MAX, 5);
        tree.fit(&data, &g, &h);
        println!("{}", tree);
        assert_eq!(25, tree.nodes.len())
    }
}
