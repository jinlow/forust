use crate::data::FloatData;
use crate::histogram::HistogramMatrix;
use crate::histsplitter::SplitInfo;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

#[derive(Debug, Deserialize, Serialize)]
pub struct SplittableNode {
    pub num: usize,
    pub histograms: HistogramMatrix,
    pub weight_value: f32,
    pub gain_value: f32,
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub depth: usize,
    pub split_value: f64,
    pub split_feature: usize,
    pub split_gain: f32,
    pub missing_right: bool,
    pub left_child: usize,
    pub right_child: usize,
    pub start_idx: usize,
    pub stop_idx: usize,
}

#[derive(Deserialize, Serialize)]
pub struct ParentNode {
    num: usize,
    pub weight_value: f32,
    hess_sum: f32,
    pub depth: usize,
    pub split_value: f64,
    pub split_feature: usize,
    split_gain: f32,
    pub missing_right: bool,
    pub left_child: usize,
    pub right_child: usize,
}

#[derive(Deserialize, Serialize)]
pub struct LeafNode {
    pub num: usize,
    pub weight_value: f32,
    pub hess_sum: f32,
    pub depth: usize,
}

#[derive(Deserialize, Serialize)]
pub enum TreeNode {
    Parent(ParentNode),
    Leaf(LeafNode),
    Splittable(SplittableNode),
}

impl SplittableNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num: usize,
        histograms: HistogramMatrix,
        weight_value: f32,
        gain_value: f32,
        grad_sum: f32,
        hess_sum: f32,
        depth: usize,
        missing_right: bool,
        start_idx: usize,
        stop_idx: usize,
    ) -> Self {
        SplittableNode {
            num,
            histograms,
            weight_value,
            gain_value,
            grad_sum,
            hess_sum,
            depth,
            split_value: f64::ZERO,
            split_feature: 0,
            split_gain: f32::ZERO,
            missing_right,
            left_child: 0,
            right_child: 0,
            start_idx,
            stop_idx,
        }
    }

    pub fn update_children(
        &mut self,
        left_child: usize,
        right_child: usize,
        split_info: &SplitInfo,
    ) {
        self.left_child = left_child;
        self.right_child = right_child;
        self.split_feature = split_info.split_feature;
        self.split_gain = split_info.left_gain + split_info.right_gain - self.gain_value;
        self.split_value = split_info.split_value;
        self.missing_right = split_info.missing_right;
    }
    pub fn as_leaf_node(&self) -> TreeNode {
        TreeNode::Leaf(LeafNode {
            num: self.num,
            weight_value: self.weight_value,
            hess_sum: self.hess_sum,
            depth: self.depth,
        })
    }
    pub fn as_parent_node(&self) -> TreeNode {
        TreeNode::Parent(ParentNode {
            num: self.num,
            weight_value: self.weight_value,
            hess_sum: self.hess_sum,
            depth: self.depth,
            missing_right: self.missing_right,
            split_value: self.split_value,
            split_feature: self.split_feature,
            split_gain: self.split_gain,
            left_child: self.left_child,
            right_child: self.right_child,
        })
    }
}

impl fmt::Display for TreeNode {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TreeNode::Leaf(leaf) => write!(
                f,
                "{}:leaf={},cover={}",
                leaf.num, leaf.weight_value, leaf.hess_sum
            ),
            TreeNode::Parent(parent) => {
                let missing = if parent.missing_right {
                    parent.right_child
                } else {
                    parent.left_child
                };
                write!(
                    f,
                    "{}:[{} < {}] yes={},no={},missing={},gain={},cover={}",
                    parent.num,
                    parent.split_feature,
                    parent.split_value,
                    parent.left_child,
                    parent.right_child,
                    missing,
                    parent.split_gain,
                    parent.hess_sum
                )
            }
            TreeNode::Splittable(node) => {
                let missing = if node.missing_right {
                    node.right_child
                } else {
                    node.left_child
                };
                write!(
                    f,
                    "SPLITTABLE - {}:[{} < {}] yes={},no={},missing={},gain={},cover={}",
                    node.num,
                    node.split_feature,
                    node.split_value,
                    missing,
                    node.left_child,
                    node.right_child,
                    node.split_gain,
                    node.hess_sum
                )
            }
        }
    }
}
