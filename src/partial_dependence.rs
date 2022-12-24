use crate::node::TreeNode;
use crate::tree::Tree;

/// Partial Dependence Calculator
// struct PDCalculator {
//     partial_dependence: f32,
//     base_score: f64,
//     tree_prediction: f64,

// }

fn get_node_cover(tree: &Tree, node_idx: usize) -> f32 {
    match &tree.nodes[node_idx] {
        TreeNode::Leaf(n) => n.hessian_sum,
        TreeNode::Parent(n) => n.hessian_sum,
        TreeNode::Splittable(_) => unreachable!(),
    }
}

pub fn tree_partial_dependence(
    tree: &Tree,
    node_idx: usize,
    feature: usize,
    value: f64,
    proportion: f32,
) -> f64 {
    let node = &tree.nodes[node_idx];
    match node {
        TreeNode::Leaf(n) => f64::from(proportion * n.weight_value),
        TreeNode::Parent(n) => {
            if n.split_feature == feature {
                let child = if value.is_nan() {
                    n.missing_node
                } else if value < n.split_value {
                    n.left_child
                } else {
                    n.right_child
                };
                tree_partial_dependence(tree, child, feature, value, proportion)
            } else {
                let left_cover = get_node_cover(tree, n.left_child);
                let right_cover = get_node_cover(tree, n.right_child);
                let total_cover = left_cover + right_cover;

                tree_partial_dependence(
                    tree,
                    n.left_child,
                    feature,
                    value,
                    proportion * (left_cover / total_cover),
                ) + tree_partial_dependence(
                    tree,
                    n.right_child,
                    feature,
                    value,
                    proportion * (right_cover / total_cover),
                )
            }
        }
        TreeNode::Splittable(_) => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::constraints::ConstraintMap;
    use crate::data::Matrix;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::splitter::MissingImputerSplitter;
    use crate::tree::Tree;
    use std::fs;
    #[test]
    fn test_partial_dependence() {
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
        let pdp1 = tree_partial_dependence(&tree, 0, 0, 1.0, 1.0);
        let pdp2 = tree_partial_dependence(&tree, 0, 0, 2.0, 1.0);
        let pdp3 = tree_partial_dependence(&tree, 0, 0, 3.0, 1.0);
        println!("{}, {}, {}", pdp1, pdp2, pdp3);
    }
}
