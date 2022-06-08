use crate::data::MatrixData;
use crate::histogram::Histograms;
use crate::histsplitter::SplitInfo;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::str::FromStr;

#[derive(Debug, Deserialize, Serialize)]
pub struct SplittableNode<T> {
    pub num: usize,
    pub histograms: Histograms<T>,
    pub weight_value: T,
    pub gain_value: T,
    pub grad_sum: T,
    pub hess_sum: T,
    pub depth: usize,
    pub split_value: T,
    pub split_feature: usize,
    pub split_gain: T,
    pub missing_right: bool,
    pub left_child: usize,
    pub right_child: usize,
    pub start_idx: usize,
    pub stop_idx: usize,
}

#[derive(Deserialize, Serialize)]
pub struct ParentNode<T> {
    num: usize,
    pub weight_value: T,
    hess_sum: T,
    pub depth: usize,
    pub split_value: T,
    pub split_feature: usize,
    split_gain: T,
    pub missing_right: bool,
    pub left_child: usize,
    pub right_child: usize,
}

#[derive(Deserialize, Serialize)]
pub struct LeafNode<T> {
    pub num: usize,
    pub weight_value: T,
    pub hess_sum: T,
    pub depth: usize,
}

#[derive(Deserialize, Serialize)]
pub enum TreeNode<T> {
    Parent(ParentNode<T>),
    Leaf(LeafNode<T>),
    Splittable(SplittableNode<T>),
}

impl<'a, T> SplittableNode<T>
where
    T: MatrixData<T>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num: usize,
        histograms: Histograms<T>,
        weight_value: T,
        gain_value: T,
        grad_sum: T,
        hess_sum: T,
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
            split_value: T::ZERO,
            split_feature: 0,
            split_gain: T::ZERO,
            missing_right: false,
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
        split_info: &SplitInfo<T>,
    ) {
        self.left_child = left_child;
        self.right_child = right_child;
        self.split_feature = split_info.split_feature;
        self.split_gain = split_info.left_gain + split_info.right_gain - self.gain_value;
        self.split_value = split_info.split_value;
        self.missing_right = split_info.missing_right;
    }
    pub fn as_leaf_node(&self) -> TreeNode<T> {
        TreeNode::Leaf(LeafNode {
            num: self.num,
            weight_value: self.weight_value,
            hess_sum: self.hess_sum,
            depth: self.depth,
        })
    }
    pub fn as_parent_node(&self) -> TreeNode<T> {
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

impl<'a, T> fmt::Display for TreeNode<T>
where
    T: FromStr + std::fmt::Display + MatrixData<T>,
    <T as FromStr>::Err: 'static + std::error::Error,
{
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
