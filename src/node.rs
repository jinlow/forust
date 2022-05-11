struct SplitInfo {
    split_gain: T,
    split_feature: usize,
    split_value: T,
    left_gain: T,
    left_cover: T,
    left_weight: T,
    left_idxs: Vec<usize>,
    right_gain: T,
    right_cover: T,
    right_weight: T,
    right_idxs: Vec<usize>,
}

struct Node<T> {
    num: usize,
    node_idxs: Vec<usize>,
    weight_value: T,
    gain_value: T,
    cover_value: T,
    depth: usize,
    split_value_: Option<T>,
    split_feature_: Option<T>,
    split_gain_: Option<T>,
    left_child_: Option<T>,
    right_child_: Option<T>,
}

impl<T> Node<T> {
    pub fn new(
        num: usize,
        node_idxs: Vec<usize>,
        weight_value: T,
        gain_value: T,
        cover_value: T,
        depth: usize,
    ) -> Self {
        Node {
            num,
            node_idxs,
            weight_value,
            gain_value,
            cover_value,
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
}
