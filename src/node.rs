use crate::data::MatrixData;
use crate::splitter::SplitInfo;
use std::fmt::{self, Debug};
use std::str::FromStr;

#[derive(Debug)]
pub struct SplittableNode<T> {
    pub num: usize,
    // pub node_idxs: Vec<usize>,
    pub weight_value: T,
    pub gain_value: T,
    pub grad_sum: T,
    pub hess_sum: T,
    pub depth: usize,
    pub split_value: T,
    pub split_feature: usize,
    pub split_gain: T,
    pub left_child: usize,
    pub right_child: usize,
    pub start_idx: usize,
    pub stop_idx: usize,
}

pub struct ParentNode<T> {
    num: usize,
    pub weight_value: T,
    hess_sum: T,
    pub depth: usize,
    pub split_value: T,
    pub split_feature: usize,
    split_gain: T,
    pub left_child: usize,
    pub right_child: usize,
}

pub struct LeafNode<T> {
    pub num: usize,
    pub weight_value: T,
    pub hess_sum: T,
    pub depth: usize,
}

pub enum TreeNode<T> {
    Parent(ParentNode<T>),
    Leaf(LeafNode<T>),
    Splittable(SplittableNode<T>),
}

impl<'a, T> SplittableNode<T>
where
    T: MatrixData<T>,
{
    pub fn new(
        num: usize,
        weight_value: T,
        gain_value: T,
        grad_sum: T,
        hess_sum: T,
        depth: usize,
        start_idx: usize,
        stop_idx: usize,
    ) -> Self {
        SplittableNode {
            num,
            weight_value,
            gain_value,
            grad_sum,
            hess_sum,
            depth,
            split_value: T::zero(),
            split_feature: 0,
            split_gain: T::zero(),
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
            TreeNode::Parent(parent) => write!(
                f,
                "{}:[{} < {}] yes={},no={},gain={},cover={}",
                parent.num,
                parent.split_feature,
                parent.split_value,
                parent.left_child,
                parent.right_child,
                parent.split_gain,
                parent.hess_sum
            ),
            TreeNode::Splittable(node) => write!(
                f,
                "SPLITTABLE - {}:[{} < {}] yes={},no={},gain={},cover={}",
                node.num,
                node.split_feature,
                node.split_value,
                node.left_child,
                node.right_child,
                node.split_gain,
                node.hess_sum
            ),
        }
    }
}
