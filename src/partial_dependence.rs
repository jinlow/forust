use crate::{tree::Tree, utils::is_missing};

/// Partial Dependence Calculator
// struct PDCalculator {
//     partial_dependence: f32,
//     base_score: f64,
//     tree_prediction: f64,

// }

fn get_node_cover(tree: &Tree, node_idx: usize) -> f32 {
    tree.nodes[node_idx].hessian_sum
}

pub fn tree_partial_dependence(
    tree: &Tree,
    node_idx: usize,
    feature: usize,
    value: f64,
    proportion: f32,
    missing: &f64,
) -> f64 {
    let n = &tree.nodes[node_idx];
    if n.is_leaf {
        f64::from(proportion * n.weight_value)
    } else if n.split_feature == feature {
        let child = if is_missing(&value, missing) {
            n.missing_node
        } else if value < n.split_value {
            n.left_child
        } else {
            n.right_child
        };
        tree_partial_dependence(tree, child, feature, value, proportion, missing)
    } else {
        let left_cover = get_node_cover(tree, n.left_child);
        let right_cover = get_node_cover(tree, n.right_child);
        let missing_cover = if n.has_missing_branch() {
            get_node_cover(tree, n.missing_node)
        } else {
            0.0
        };
        let total_cover = left_cover + right_cover + missing_cover;
        let missing_pd = if n.has_missing_branch() {
            tree_partial_dependence(
                tree,
                n.missing_node,
                feature,
                value,
                proportion * (missing_cover / total_cover),
                missing,
            )
        } else {
            0.
        };
        tree_partial_dependence(
            tree,
            n.left_child,
            feature,
            value,
            proportion * (left_cover / total_cover),
            missing,
        ) + tree_partial_dependence(
            tree,
            n.right_child,
            feature,
            value,
            proportion * (right_cover / total_cover),
            missing,
        ) + missing_pd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::constraints::ConstraintMap;
    use crate::data::Matrix;
    use crate::gradientbooster::GrowPolicy;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::sampler::SampleMethod;
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
            &GrowPolicy::LossGuide,
        );
        let pdp1 = tree_partial_dependence(&tree, 0, 0, 1.0, 1.0, &f64::NAN);
        let pdp2 = tree_partial_dependence(&tree, 0, 0, 2.0, 1.0, &f64::NAN);
        let pdp3 = tree_partial_dependence(&tree, 0, 0, 3.0, 1.0, &f64::NAN);
        println!("{}, {}, {}", pdp1, pdp2, pdp3);
    }
}
