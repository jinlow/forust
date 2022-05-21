use crate::data::MatrixData;
use crate::splitter::SplitInfo;
use std::fmt::{self, Debug};
use std::str::FromStr;

#[derive(Debug)]
pub struct Node<T> {
    pub num: usize,
    pub node_idxs: Vec<usize>,
    pub weight_value: T,
    pub gain_value: T,
    pub grad_sum: T,
    pub hess_sum: T,
    pub depth: usize,
    pub split_value_: Option<T>,
    pub split_feature_: Option<usize>,
    pub split_gain_: Option<T>,
    pub left_child_: Option<usize>,
    pub right_child_: Option<usize>,
}

pub struct NodeInfo<'a, T> {
    pub split_value_: &'a T,
    pub split_feature_: &'a usize,
    pub split_gain_: &'a T,
    pub left_child_: &'a usize,
    pub right_child_: &'a usize,
}

// impl<'a, T> NodeInfo<'a, T> {
//     // Generate node info from a node, this method will
//     // panic if this Node is a leaf node
//     pub fn from_node(
//         Node {
//             split_value_,
//             split_feature_,
//             split_gain_,
//             left_child_,
//             right_child_,
//             ..
//         }: Node<T>
//     ) -> Self {
//         let msg = "Leaf value passed to from_node method.";
//         NodeInfo {
//             split_value_: &split_value_.expect(msg),
//             split_feature_: &split_feature_.expect(msg),
//             split_gain_: &split_gain_.expect(msg),
//             left_child_: &left_child_.expect(msg),
//             right_child_: &right_child_.expect(msg),
//         }
//     }
// }

impl<T> Node<T>
where
    T: MatrixData<T>,
{
    pub fn new(
        num: usize,
        node_idxs: Vec<usize>,
        weight_value: T,
        gain_value: T,
        grad_sum: T,
        hess_sum: T,
        depth: usize,
    ) -> Self {
        Node {
            num,
            node_idxs,
            weight_value,
            gain_value,
            grad_sum,
            hess_sum,
            depth,
            split_value_: None,
            split_feature_: None,
            split_gain_: None,
            left_child_: None,
            right_child_: None,
        }
    }
    pub fn is_leaf(&self) -> bool {
        self.split_feature_.is_none()
    }

    pub fn update_children(
        &mut self,
        left_child: usize,
        right_child: usize,
        split_info: &SplitInfo<T>,
    ) {
        self.left_child_ = Some(left_child);
        self.right_child_ = Some(right_child);
        self.split_feature_ = Some(split_info.split_feature);
        self.split_gain_ = Some(split_info.left_gain + split_info.right_gain - self.gain_value);
        self.split_value_ = Some(split_info.split_value);
    }
}

impl<T> fmt::Display for Node<T>
where
    T: FromStr + std::fmt::Display + MatrixData<T>,
    <T as FromStr>::Err: 'static + std::error::Error,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_leaf() {
            write!(
                f,
                "{}:leaf={},cover={}",
                self.num, self.weight_value, self.hess_sum
            )
        } else {
            write!(
                f,
                "{}:[{} < {}] yes={},no={},gain={},cover={}",
                self.num,
                self.split_feature_.unwrap(),
                self.split_value_.unwrap(),
                self.left_child_.unwrap(),
                self.right_child_.unwrap(),
                self.split_gain_.unwrap(),
                self.hess_sum
            )
        }
    }
}
