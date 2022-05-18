use crate::data::MatrixData;
use crate::splitter::SplitInfo;

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
