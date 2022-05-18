use crate::data::{Matrix, MatrixData};
use crate::node::Node;
use crate::splitter::{SplitInfo, Splitter};
use std::collections::VecDeque;

struct Tree<T: MatrixData<T>, S: Splitter<T>> {
    splitter: S,
    max_leaves: usize,
    max_depth: usize,
    nodes: Vec<Node<T>>,
    n_nodes: usize,
}

impl<T: MatrixData<T>, S: Splitter<T>> Tree<T, S> {
    fn new(splitter: S, max_leaves: usize, max_depth: usize) -> Self {
        Tree {
            splitter,
            max_leaves,
            max_depth,
            nodes: Vec::new(),
            n_nodes: 0,
        }
    }
    fn fit(&mut self, data: &Matrix<T>, grad: &[T], hess: &[T]) {
        let grad_sum: T = grad.iter().copied().sum();
        let hess_sum: T = grad.iter().copied().sum();
        let root_gain = self.splitter.gain(grad_sum, hess_sum);
        let root_weight = self.splitter.weight(grad_sum, hess_sum);
        let root_node = Node::new(
            0,
            (0..data.rows).collect(),
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
        while growable.len() > 0 {
            if n_leaves >= self.max_leaves {
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
                        info.left_gain,
                        info.left_cover,
                        depth,
                    );
                    let right_node = Node::new(
                        right_idx,
                        info.right_idxs.to_vec(),
                        info.right_weight,
                        info.right_gain,
                        info.right_gain,
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
}
