use std::collections::HashSet;

use crate::constraints::{Constraint, ConstraintMap};
use crate::data::{JaggedMatrix, Matrix};
use crate::gradientbooster::MissingNodeTreatment;
use crate::histogram::HistogramMatrix;
use crate::node::SplittableNode;
use crate::tree::Tree;
use crate::utils::{
    between, bound_to_parent, constrained_weight, cull_gain, gain_given_weight, pivot_on_split,
    pivot_on_split_exclude_missing,
};

#[derive(Debug)]
pub struct SplitInfo {
    pub split_gain: f32,
    pub split_feature: usize,
    pub split_value: f64,
    pub split_bin: u16,
    pub left_node: NodeInfo,
    pub right_node: NodeInfo,
    pub missing_node: MissingInfo,
}

#[derive(Debug)]
pub struct NodeInfo {
    pub grad: f32,
    pub gain: f32,
    pub cover: f32,
    pub weight: f32,
    pub bounds: (f32, f32),
}

#[derive(Debug)]
pub enum MissingInfo {
    Left,
    Right,
    Leaf(NodeInfo),
    Branch(NodeInfo),
}

pub trait Splitter {
    /// When a split happens, how many leaves will the tree increase by?
    /// For example, if a binary split happens, the split will increase the
    /// number of leaves by 1, if a ternary split happens, the number of leaves will
    /// increase by 2.
    fn new_leaves_added(&self) -> usize {
        1
    }
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint>;
    // fn get_allow_missing_splits(&self) -> bool;
    fn get_gamma(&self) -> f32;
    fn get_l1(&self) -> f32;
    fn get_l2(&self) -> f32;
    fn get_max_delta_step(&self) -> f32;
    fn get_learning_rate(&self) -> f32;

    /// Perform any post processing on the tree that is
    /// relevant for the specific splitter, empty default
    /// implementation so that it can be called even if it's
    /// not used.
    fn clean_up_splits(&self, _tree: &mut Tree) {}

    /// Find the best possible split, considering all feature histograms.
    /// If we wanted to add Column sampling, this is probably where
    /// we would need to do it, otherwise, it would be at the tree level.
    fn best_split(&self, node: &SplittableNode, col_index: &[usize]) -> Option<SplitInfo> {
        let mut best_split_info = None;
        let mut best_gain = 0.0;
        for (idx, feature) in col_index.iter().enumerate() {
            let split_info = self.best_feature_split(node, *feature, idx);
            match split_info {
                Some(info) => {
                    if info.split_gain > best_gain {
                        best_gain = info.split_gain;
                        best_split_info = Some(info);
                    }
                }
                None => continue,
            }
        }
        best_split_info
    }

    /// Evaluate a split, returning the node info for the left, and right splits,
    /// as well as the node info the missing data of a feature.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        right_gradient: f32,
        right_hessian: f32,
        missing_gradient: f32,
        missing_hessian: f32,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)>;

    /// The idx is the index of the feature in the histogram data, whereas feature
    /// is the index of the actual feature in the data.
    fn best_feature_split(
        &self,
        node: &SplittableNode,
        feature: usize,
        idx: usize,
    ) -> Option<SplitInfo> {
        let mut split_info: Option<SplitInfo> = None;
        let mut max_gain: Option<f32> = None;

        let HistogramMatrix(histograms) = &node.histograms;
        let histogram = histograms.get_col(idx);

        // We also know we will have a missing bin.
        let missing = &histogram[0];
        let mut cuml_grad = 0.0; // first_bin.gradient_sum;
        let mut cuml_hess = 0.0; // first_bin.hessian_sum;
        let constraint = self.get_constraint(&feature);

        let elements = histogram.len();
        assert!(elements == histogram.len());

        for (i, bin) in histogram[1..].iter().enumerate() {
            let left_gradient = cuml_grad;
            let left_hessian = cuml_hess;
            let right_gradient = node.gradient_sum - cuml_grad - missing.gradient_sum;
            let right_hessian = node.hessian_sum - cuml_hess - missing.hessian_sum;
            cuml_grad += bin.gradient_sum;
            cuml_hess += bin.hessian_sum;

            let (mut left_node_info, mut right_node_info, missing_info) = match self.evaluate_split(
                left_gradient,
                left_hessian,
                right_gradient,
                right_hessian,
                missing.gradient_sum,
                missing.hessian_sum,
                node.lower_bound,
                node.upper_bound,
                node.weight_value,
                constraint,
            ) {
                None => {
                    continue;
                }
                Some(v) => v,
            };

            let split_gain = node.get_split_gain(
                &left_node_info,
                &right_node_info,
                &missing_info,
                self.get_gamma(),
            );

            // Check monotonicity holds
            let split_gain = cull_gain(
                split_gain,
                left_node_info.weight,
                right_node_info.weight,
                constraint,
            );

            if split_gain <= 0.0 {
                continue;
            }

            let mid = (left_node_info.weight + right_node_info.weight) / 2.0;
            let (left_bounds, right_bounds) = match constraint {
                None | Some(Constraint::Unconstrained) => (
                    (node.lower_bound, node.upper_bound),
                    (node.lower_bound, node.upper_bound),
                ),
                Some(Constraint::Negative) => ((mid, node.upper_bound), (node.lower_bound, mid)),
                Some(Constraint::Positive) => ((node.lower_bound, mid), (mid, node.upper_bound)),
            };
            left_node_info.bounds = left_bounds;
            right_node_info.bounds = right_bounds;

            // If split gain is NaN, one of the sides is empty, do not allow
            // this split.
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };
            if max_gain.is_none() || split_gain > max_gain.unwrap() {
                max_gain = Some(split_gain);
                split_info = Some(SplitInfo {
                    split_gain,
                    split_feature: feature,
                    split_value: bin.cut_value,
                    split_bin: (i + 1) as u16,
                    left_node: left_node_info,
                    right_node: right_node_info,
                    missing_node: missing_info,
                });
            }
        }
        split_info
    }

    /// Handle the split info, creating the children nodes, this function
    /// will return a vector of new splitable nodes, that can be added to the
    /// growable stack, and further split, or converted to leaf nodes.
    #[allow(clippy::too_many_arguments)]
    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        parallel: bool,
    ) -> Vec<SplittableNode>;

    /// Split the node, if we cant find a best split, we will need to
    /// return an empty vector, this node is a leaf.
    #[allow(clippy::too_many_arguments)]
    fn split_node(
        &self,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        parallel: bool,
    ) -> Vec<SplittableNode> {
        match self.best_split(node, col_index) {
            Some(split_info) => self.handle_split_info(
                split_info, n_nodes, node, index, col_index, data, cuts, grad, hess, parallel,
            ),
            None => Vec::new(),
        }
    }
}

/// Missing branch splitter
/// Always creates a separate branch for the missing values of a feature.
/// This results, in every node having a specific "missing", direction.
/// If this node is able, it will be split further, otherwise it will
/// a leaf node will be generated.
pub struct MissingBranchSplitter {
    pub l1: f32,
    pub l2: f32,
    pub max_delta_step: f32,
    pub gamma: f32,
    pub min_leaf_weight: f32,
    pub learning_rate: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
    pub terminate_missing_features: HashSet<usize>,
    pub missing_node_treatment: MissingNodeTreatment,
    pub force_children_to_bound_parent: bool,
}

impl MissingBranchSplitter {
    pub fn new_leaves_added(&self) -> usize {
        2
    }
    pub fn update_average_missing_nodes(tree: &mut Tree, node_idx: usize) -> f64 {
        let node = &tree.nodes[node_idx];

        if node.is_leaf {
            return node.weight_value as f64;
        }

        let right = node.right_child;
        let left = node.left_child;
        let current_node = node.num;
        let missing = node.missing_node;

        let right_hessian = tree.nodes[right].hessian_sum as f64;
        let right_avg_weight = Self::update_average_missing_nodes(tree, right);

        let left_hessian = tree.nodes[left].hessian_sum as f64;
        let left_avg_weight = Self::update_average_missing_nodes(tree, left);

        // This way this process supports missing branches that terminate (and will neutralize)
        // and then if missing is split further those values will have non-zero contributions.
        let (missing_hessian, missing_avg_weight, missing_leaf) = if tree.nodes[missing].is_leaf {
            (0., 0., true)
        } else {
            (
                tree.nodes[missing].hessian_sum as f64,
                Self::update_average_missing_nodes(tree, missing),
                false,
            )
        };

        let update = (right_avg_weight * right_hessian
            + left_avg_weight * left_hessian
            + missing_avg_weight * missing_hessian)
            / (left_hessian + right_hessian + missing_hessian);

        // Update current node, and the missing value
        if let Some(n) = tree.nodes.get_mut(current_node) {
            n.weight_value = update as f32;
        }
        // Only update the missing node if it's a leaf, otherwise we will auto-update
        // them via the recursion called earlier.
        if missing_leaf {
            if let Some(m) = tree.nodes.get_mut(missing) {
                m.weight_value = update as f32;
            }
        }

        update
    }
}

impl Splitter for MissingBranchSplitter {
    fn clean_up_splits(&self, tree: &mut Tree) {
        if let MissingNodeTreatment::AverageLeafWeight = self.missing_node_treatment {
            MissingBranchSplitter::update_average_missing_nodes(tree, 0);
        }
    }

    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }

    fn get_gamma(&self) -> f32 {
        self.gamma
    }

    fn get_l1(&self) -> f32 {
        self.l1
    }

    fn get_l2(&self) -> f32 {
        self.l2
    }
    fn get_max_delta_step(&self) -> f32 {
        self.max_delta_step
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        right_gradient: f32,
        right_hessian: f32,
        missing_gradient: f32,
        missing_hessian: f32,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, there is nothing to split on,
        // and so we should continue.
        if (left_gradient == 0.0) && (left_hessian == 0.0)
            || (right_gradient == 0.0) && (right_hessian == 0.0)
        {
            return None;
        }

        let mut left_weight = constrained_weight(
            &self.l1,
            &self.l2,
            &self.max_delta_step,
            left_gradient,
            left_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );
        let mut right_weight = constrained_weight(
            &self.l1,
            &self.l2,
            &self.max_delta_step,
            right_gradient,
            right_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );

        if self.force_children_to_bound_parent {
            (left_weight, right_weight) = bound_to_parent(parent_weight, left_weight, right_weight);
            assert!(between(lower_bound, upper_bound, left_weight));
            assert!(between(lower_bound, upper_bound, right_weight));
        }

        let left_gain = gain_given_weight(&self.l2, left_gradient, left_hessian, left_weight);
        let right_gain = gain_given_weight(&self.l2, right_gradient, right_hessian, right_weight);

        // Check the min_hessian constraint first
        if (right_hessian < self.min_leaf_weight) || (left_hessian < self.min_leaf_weight) {
            // Update for new value
            return None;
        }

        // We have not considered missing at all up until this point, we could if we wanted
        // to give more predictive power probably to missing.
        // If we don't want to allow the missing branch to be split further,
        // we will default to creating an empty branch.

        // Set weight based on the missing node treatment.
        let missing_weight = match self.missing_node_treatment {
            MissingNodeTreatment::AssignToParent => constrained_weight(
                &self.get_l1(),
                &self.get_l2(),
                &self.max_delta_step,
                missing_gradient + left_gradient + right_gradient,
                missing_hessian + left_hessian + right_hessian,
                lower_bound,
                upper_bound,
                constraint,
            ),
            // Calculate the local leaf average for now, after training the tree.
            // Recursively assign to the leaf weights underneath.
            MissingNodeTreatment::AverageLeafWeight | MissingNodeTreatment::AverageNodeWeight => {
                (right_weight * right_hessian + left_weight * left_hessian)
                    / (right_hessian + left_hessian)
            }
            MissingNodeTreatment::None => {
                // If there are no missing records, just default
                // to the parent weight.
                if missing_hessian == 0. || missing_gradient == 0. {
                    parent_weight
                } else {
                    constrained_weight(
                        &self.get_l1(),
                        &self.get_l2(),
                        &self.max_delta_step,
                        missing_gradient,
                        missing_hessian,
                        lower_bound,
                        upper_bound,
                        constraint,
                    )
                }
            }
        };
        let missing_gain = gain_given_weight(
            &self.get_l2(),
            missing_gradient,
            missing_hessian,
            missing_weight,
        );
        let missing_info = NodeInfo {
            grad: missing_gradient,
            gain: missing_gain,
            cover: missing_hessian,
            weight: missing_weight,
            // Constrain to the same bounds as the parent.
            // This will ensure that splits further down in the missing only
            // branch are monotonic.
            bounds: (lower_bound, upper_bound),
        };
        let missing_node = // Check Missing direction
        if ((missing_gradient != 0.0) || (missing_hessian != 0.0)) && self.allow_missing_splits {
            MissingInfo::Branch(
                missing_info
            )
        } else {
            MissingInfo::Leaf(
                missing_info
            )
        };

        if (right_hessian < self.min_leaf_weight) || (left_hessian < self.min_leaf_weight) {
            // Update for new value
            return None;
        }
        Some((
            NodeInfo {
                grad: left_gradient,
                gain: left_gain,
                cover: left_hessian,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                grad: right_gradient,
                gain: right_gain,
                cover: right_hessian,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_node,
        ))
    }

    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        parallel: bool,
    ) -> Vec<SplittableNode> {
        let missing_child = *n_nodes;
        let left_child = missing_child + 1;
        let right_child = missing_child + 2;
        node.update_children(missing_child, left_child, right_child, &split_info);

        let (mut missing_is_leaf, mut missing_info) = match split_info.missing_node {
            MissingInfo::Branch(i) => {
                if self
                    .terminate_missing_features
                    .contains(&split_info.split_feature)
                {
                    (true, i)
                } else {
                    (false, i)
                }
            }
            MissingInfo::Leaf(i) => (true, i),
            _ => unreachable!(),
        };
        // Set missing weight to parent weight value...
        // This essentially neutralizes missing.
        // Manually calculating it, was leading to some small numeric
        // rounding differences...
        if let MissingNodeTreatment::AssignToParent = self.missing_node_treatment {
            missing_info.weight = node.weight_value;
        }
        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Missing all falls to the bottom.
        let (mut missing_split_idx, mut split_idx) = pivot_on_split_exclude_missing(
            &mut index[node.start_idx..node.stop_idx],
            data.get_col(split_info.split_feature),
            split_info.split_bin,
        );
        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right - missing_split_idx;
        let n_missing = total_recs - (n_right + n_left);
        let max_ = match [n_missing, n_left, n_right]
            .iter()
            .enumerate()
            .max_by(|(_, i), (_, j)| i.cmp(j))
        {
            Some((i, _)) => i,
            // if we can't compare them, it doesn't
            // really matter, build the histogram on
            // any of them.
            None => 0,
        };

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;
        missing_split_idx += node.start_idx;

        // Build the histograms for the smaller node.
        let left_histograms: HistogramMatrix;
        let right_histograms: HistogramMatrix;
        let missing_histograms: HistogramMatrix;
        if n_missing == 0 {
            // If there are no missing records, we know the missing value
            // will be a leaf, assign this node as a leaf.
            missing_is_leaf = true;
            if max_ == 1 {
                missing_histograms = HistogramMatrix::empty();
                right_histograms = HistogramMatrix::new(
                    data,
                    cuts,
                    grad,
                    hess,
                    &index[split_idx..node.stop_idx],
                    col_index,
                    parallel,
                    true,
                );
                left_histograms =
                    HistogramMatrix::from_parent_child(&node.histograms, &right_histograms);
            } else {
                missing_histograms = HistogramMatrix::empty();
                left_histograms = HistogramMatrix::new(
                    data,
                    cuts,
                    grad,
                    hess,
                    &index[missing_split_idx..split_idx],
                    col_index,
                    parallel,
                    true,
                );
                right_histograms =
                    HistogramMatrix::from_parent_child(&node.histograms, &left_histograms);
            }
        } else if max_ == 0 {
            // Max is missing, calculate the other two
            // levels histograms.
            left_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[missing_split_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            right_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            missing_histograms = HistogramMatrix::from_parent_two_children(
                &node.histograms,
                &left_histograms,
                &right_histograms,
            )
        } else if max_ == 1 {
            missing_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..missing_split_idx],
                col_index,
                parallel,
                true,
            );
            right_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            left_histograms = HistogramMatrix::from_parent_two_children(
                &node.histograms,
                &missing_histograms,
                &right_histograms,
            )
        } else {
            // right is the largest
            missing_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..missing_split_idx],
                col_index,
                parallel,
                true,
            );
            left_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[missing_split_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            right_histograms = HistogramMatrix::from_parent_two_children(
                &node.histograms,
                &missing_histograms,
                &left_histograms,
            )
        }

        let mut missing_node = SplittableNode::from_node_info(
            missing_child,
            missing_histograms,
            node.depth + 1,
            node.start_idx,
            missing_split_idx,
            missing_info,
        );
        missing_node.is_missing_leaf = missing_is_leaf;
        let left_node = SplittableNode::from_node_info(
            left_child,
            left_histograms,
            node.depth + 1,
            missing_split_idx,
            split_idx,
            split_info.left_node,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            right_histograms,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            split_info.right_node,
        );
        vec![missing_node, left_node, right_node]
    }
}

/// Missing imputer splitter
/// Splitter that imputes missing values, by sending
/// them down either the right or left branch, depending
/// on which results in a higher increase in gain.
pub struct MissingImputerSplitter {
    pub l1: f32,
    pub l2: f32,
    pub max_delta_step: f32,
    pub gamma: f32,
    pub min_leaf_weight: f32,
    pub learning_rate: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
}

impl MissingImputerSplitter {
    /// Generate a new missing imputer splitter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l1: f32,
        l2: f32,
        max_delta_step: f32,
        gamma: f32,
        min_leaf_weight: f32,
        learning_rate: f32,
        allow_missing_splits: bool,
        constraints_map: ConstraintMap,
    ) -> Self {
        MissingImputerSplitter {
            l1,
            l2,
            max_delta_step,
            gamma,
            min_leaf_weight,
            learning_rate,
            allow_missing_splits,
            constraints_map,
        }
    }
}

impl Splitter for MissingImputerSplitter {
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }

    fn get_gamma(&self) -> f32 {
        self.gamma
    }

    fn get_l1(&self) -> f32 {
        self.l1
    }

    fn get_l2(&self) -> f32 {
        self.l2
    }
    fn get_max_delta_step(&self) -> f32 {
        self.max_delta_step
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        right_gradient: f32,
        right_hessian: f32,
        missing_gradient: f32,
        missing_hessian: f32,
        lower_bound: f32,
        upper_bound: f32,
        _parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, we will possibly lead to a missing only
        // split, if we don't want this, bomb.
        if ((left_gradient == 0.0) && (left_hessian == 0.0)
            || (right_gradient == 0.0) && (right_hessian == 0.0))
            && !self.allow_missing_splits
        {
            return None;
        }

        // By default missing values will go into the right node.
        let mut missing_info = MissingInfo::Right;

        let mut left_gradient = left_gradient;
        let mut left_hessian = left_hessian;

        let mut right_gradient = right_gradient;
        let mut right_hessian = right_hessian;

        let mut left_weight = constrained_weight(
            &self.l1,
            &self.l2,
            &self.max_delta_step,
            left_gradient,
            left_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );
        let mut right_weight = constrained_weight(
            &self.l1,
            &self.l2,
            &self.max_delta_step,
            right_gradient,
            right_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );

        let mut left_gain = gain_given_weight(&self.l2, left_gradient, left_hessian, left_weight);
        let mut right_gain =
            gain_given_weight(&self.l2, right_gradient, right_hessian, right_weight);

        if !self.allow_missing_splits {
            // Check the min_hessian constraint first, if we do not
            // want to allow missing only splits.
            if (right_hessian < self.min_leaf_weight) || (left_hessian < self.min_leaf_weight) {
                // Update for new value
                return None;
            }
        }

        // Check Missing direction
        // Don't even worry about it, if there are no missing values
        // in this bin.
        if (missing_gradient != 0.0) || (missing_hessian != 0.0) {
            // If
            // TODO: Consider making this safer, by casting to f64, summing, and then
            // back to f32...
            // The weight if missing went left
            let missing_left_weight = constrained_weight(
                &self.l1,
                &self.l2,
                &self.max_delta_step,
                left_gradient + missing_gradient,
                left_hessian + missing_hessian,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain if missing went left
            let missing_left_gain = gain_given_weight(
                &self.l2,
                left_gradient + missing_gradient,
                left_hessian + missing_hessian,
                missing_left_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_left_gain = cull_gain(
                missing_left_gain,
                missing_left_weight,
                right_weight,
                constraint,
            );

            // The gain if missing went right
            let missing_right_weight = constrained_weight(
                &self.l1,
                &self.l2,
                &self.max_delta_step,
                right_gradient + missing_gradient,
                right_hessian + missing_hessian,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain is missing went right
            let missing_right_gain = gain_given_weight(
                &self.l2,
                right_gradient + missing_gradient,
                right_hessian + missing_hessian,
                missing_right_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_right_gain = cull_gain(
                missing_right_gain,
                left_weight,
                missing_right_weight,
                constraint,
            );

            if (missing_right_gain - right_gain) < (missing_left_gain - left_gain) {
                // Missing goes left
                left_gradient += missing_gradient;
                left_hessian += missing_hessian;
                left_gain = missing_left_gain;
                left_weight = missing_left_weight;
                missing_info = MissingInfo::Left;
            } else {
                // Missing goes right
                right_gradient += missing_gradient;
                right_hessian += missing_hessian;
                right_gain = missing_right_gain;
                right_weight = missing_right_weight;
                missing_info = MissingInfo::Right;
            }
        }

        if (right_hessian < self.min_leaf_weight) || (left_hessian < self.min_leaf_weight) {
            // Update for new value
            return None;
        }
        Some((
            NodeInfo {
                grad: left_gradient,
                gain: left_gain,
                cover: left_hessian,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                grad: right_gradient,
                gain: right_gain,
                cover: right_hessian,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_info,
        ))
    }

    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        parallel: bool,
    ) -> Vec<SplittableNode> {
        let left_child = *n_nodes;
        let right_child = left_child + 1;

        let missing_right = match split_info.missing_node {
            MissingInfo::Left => false,
            MissingInfo::Right => true,
            _ => unreachable!(),
        };

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Here we assign missing to a specific direction.
        // This will need to be refactored once we add a
        // separate missing branch.
        let mut split_idx = pivot_on_split(
            &mut index[node.start_idx..node.stop_idx],
            data.get_col(split_info.split_feature),
            split_info.split_bin,
            missing_right,
        );
        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right;

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;

        // Build the histograms for the smaller node.
        let left_histograms: HistogramMatrix;
        let right_histograms: HistogramMatrix;
        if n_left < n_right {
            left_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            right_histograms =
                HistogramMatrix::from_parent_child(&node.histograms, &left_histograms);
        } else {
            right_histograms = HistogramMatrix::new(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            left_histograms =
                HistogramMatrix::from_parent_child(&node.histograms, &right_histograms);
        }
        let missing_child = if missing_right {
            right_child
        } else {
            left_child
        };
        node.update_children(missing_child, left_child, right_child, &split_info);

        let left_node = SplittableNode::from_node_info(
            left_child,
            left_histograms,
            node.depth + 1,
            node.start_idx,
            split_idx,
            split_info.left_node,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            right_histograms,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            split_info.right_node,
        );
        vec![left_node, right_node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::data::Matrix;
    use crate::node::SplittableNode;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::utils::gain;
    use crate::utils::weight;
    use std::fs;
    #[test]
    fn test_best_feature_split() {
        let d = vec![4., 2., 3., 4., 5., 1., 4.];
        let data = Matrix::new(&d, 7, 1);
        let y = vec![0., 0., 0., 1., 1., 0., 1.];
        let yhat = vec![0.; 7];
        let w = vec![1.; y.len()];
        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        let b = bin_matrix(&data, &w, 10, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists = HistogramMatrix::new(&bdata, &b.cuts, &grad, &hess, &index, &[0], true, true);
        let splitter = MissingImputerSplitter {
            l1: 0.0,
            l2: 0.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f32>(),
            hess.iter().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_feature_split(&mut n, 0, 0).unwrap();
        assert_eq!(s.split_value, 4.0);
        assert_eq!(s.left_node.cover, 0.75);
        assert_eq!(s.right_node.cover, 1.0);
        assert_eq!(s.left_node.gain, 3.0);
        assert_eq!(s.right_node.gain, 1.0);
        assert_eq!(s.split_gain, 3.86);
    }

    #[test]
    fn test_best_split() {
        let d: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 0., 4., 2., 3., 4., 5., 1., 4.];
        let data = Matrix::new(&d, 7, 2);
        let y = vec![0., 0., 0., 1., 1., 0., 1.];
        let yhat = vec![0.; 7];
        let w = vec![1.; y.len()];
        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);

        let b = bin_matrix(&data, &w, 10, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let hists =
            HistogramMatrix::new(&bdata, &b.cuts, &grad, &hess, &index, &[0, 1], true, true);
        println!("{:?}", hists);
        let splitter = MissingImputerSplitter {
            l1: 0.0,
            l2: 0.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 0.0,
            learning_rate: 1.0,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let mut n = SplittableNode::new(
            0,
            // vec![0, 1, 2, 3, 4, 5, 6],
            hists,
            0.0,
            0.14,
            grad.iter().sum::<f32>(),
            hess.iter().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_split(&mut n, &[0, 1]).unwrap();
        println!("{:?}", s);
        assert_eq!(s.split_feature, 1);
        assert_eq!(s.split_value, 4.);
        assert_eq!(s.left_node.cover, 0.75);
        assert_eq!(s.right_node.cover, 1.);
        assert_eq!(s.left_node.gain, 3.);
        assert_eq!(s.right_node.gain, 1.);
        assert_eq!(s.split_gain, 3.86);
    }

    #[test]
    fn test_data_split() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);

        let splitter = MissingImputerSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let gradient_sum = grad.iter().copied().sum();
        let hessian_sum = hess.iter().copied().sum();
        let root_weight = weight(
            &splitter.l1,
            &splitter.l2,
            &splitter.max_delta_step,
            gradient_sum,
            hessian_sum,
        );
        let root_gain = gain(&splitter.l2, gradient_sum, hessian_sum);
        let data = Matrix::new(&data_vec, 891, 5);

        let b = bin_matrix(&data, &w, 10, f64::NAN).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();
        let hists = HistogramMatrix::new(
            &bdata, &b.cuts, &grad, &hess, &index, &col_index, true, false,
        );

        let mut n = SplittableNode::new(
            0,
            // (0..(data.rows - 1)).collect(),
            hists,
            root_weight,
            root_gain,
            grad.iter().copied().sum::<f32>(),
            hess.iter().copied().sum::<f32>(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        let s = splitter.best_split(&mut n, &col_index).unwrap();
        println!("{:?}", s);
        n.update_children(2, 1, 2, &s);
        assert_eq!(0, s.split_feature);
    }
}
