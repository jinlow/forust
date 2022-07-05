use crate::node::TreeNode;

/// Partial Dependence Calculator
// struct PDCalculator {
//     partial_dependence: f32,
//     base_score: f64,
//     tree_prediction: f64,

// }

fn tree_partial_dependence(node: TreeNode, feature: usize, value: f64, proportion: f32) -> f32 {
    match node {
        TreeNode::Leaf(n) => return proportion * n.weight_value,
        TreeNode::Parent(n) => {
            if n.split_feature == feature {
                0.0
            }
        },
        TreeNode::Splittable(_) => unreachable!()
    }
}
