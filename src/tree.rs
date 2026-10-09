use crate::binning::PREDICT_NAN_BIN;
use crate::data::{JaggedMatrix, Matrix};
use crate::gradientbooster::GrowPolicy;
use crate::grower::Grower;
use crate::histogram::HistogramMatrix;
use crate::node::{Node, SplittableNode};
use crate::partial_dependence::tree_partial_dependence;
use crate::sampler::SampleMethod;
use crate::splitter::Splitter;
use crate::utils::{fast_f64_sum, is_missing};
use crate::utils::{gain, odds, weight};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt::{self, Display};

#[derive(Deserialize, Serialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

/// The training rows in each node of a tree, as left by `Tree::fit`.
pub struct TreeRows {
    /// The rows the tree was fit on, ordered so each node's rows are contiguous.
    pub index: Vec<usize>,
    /// The `(start, stop)` range of `index` for each node, by node number.
    pub ranges: Vec<(usize, usize)>,
}

impl TreeRows {
    /// Each leaf's rows and weight.
    pub fn leaves<'a>(&'a self, tree: &'a Tree) -> impl Iterator<Item = (&'a [usize], f64)> + 'a {
        tree.nodes.iter().filter(|n| n.is_leaf).map(|n| {
            let (start, stop) = self.ranges[n.num];
            (&self.index[start..stop], n.weight_value as f64)
        })
    }
}

/// The split bin of each node, from its split value: values below the split value
/// fall in the bins below it. Leaves get 0.
pub fn split_bins(tree: &Tree, cuts: &JaggedMatrix<f64>) -> Vec<u16> {
    tree.nodes
        .iter()
        .map(|n| {
            if n.is_leaf {
                return 0;
            }
            let feature_cuts = cuts.get_col(n.split_feature);
            let bin = feature_cuts.partition_point(|c| *c < n.split_value) + 1;
            debug_assert_eq!(feature_cuts[bin - 1], n.split_value);
            bin as u16
        })
        .collect()
}

/// Does the right child of a split hold rows that go right? The partition functions
/// are only wrong when no row goes right, and then the right child holds a row that
/// goes left, or none. Checking its first row is enough.
fn right_child_rows_exact(
    parent: &SplittableNode,
    children: &[SplittableNode],
    index: &[usize],
    data: &Matrix<u16>,
    cuts: &JaggedMatrix<f64>,
) -> bool {
    let right = children.last().expect("a split has children");
    if right.start_idx == right.stop_idx {
        return false;
    }
    let feature_cuts = cuts.get_col(parent.split_feature);
    let split_bin = (feature_cuts.partition_point(|c| *c < parent.split_value) + 1) as u16;
    let bin = *data.get(index[right.start_idx], parent.split_feature);
    if bin == 0 {
        parent.missing_node == right.num
    } else {
        bin >= split_bin
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Tree { nodes: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit<T: Splitter>(
        &mut self,
        data: &Matrix<u16>,
        mut index: Vec<usize>,
        col_index: &[usize],
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        splitter: &T,
        max_leaves: usize,
        max_depth: usize,
        parallel: bool,
        sample_method: &SampleMethod,
        grow_policy: &GrowPolicy,
    ) -> Option<TreeRows> {
        // Recreating the index for each tree, ensures that the tree construction is faster
        // for the root node. This also ensures that sorting the records is always fast,
        // because we are starting from a nearly sorted array.
        let (gradient_sum, hessian_sum, sort) = match sample_method {
            // We don't need to sort, if we are not sampling. This is because
            // the data is already sorted.
            SampleMethod::None => (fast_f64_sum(grad), fast_f64_sum(hess), false),
            _ => {
                // Accumulate using f64 for numeric fidelity.
                let mut gs: f64 = 0.;
                let mut hs: f64 = 0.;
                for i in index.iter() {
                    let i_ = *i;
                    gs += grad[i_] as f64;
                    hs += hess[i_] as f64;
                }
                (gs as f32, hs as f32, true)
            }
        };

        let mut n_nodes = 1;
        let mut ranges = vec![(0, index.len())];
        // False if a split's partition could disagree with predicting the rows.
        let mut rows_exact = true;
        let root_gain = gain(&splitter.get_l2(), gradient_sum, hessian_sum);
        let root_weight = weight(
            &splitter.get_l1(),
            &splitter.get_l2(),
            &splitter.get_max_delta_step(),
            gradient_sum,
            hessian_sum,
        );
        // Calculate the histograms for the root node.
        let root_hists =
            HistogramMatrix::new(data, cuts, grad, hess, &index, col_index, parallel, sort);
        let root_node = SplittableNode::new(
            0,
            root_hists,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            0,
            0,
            index.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
        );
        // Add the first node to the tree nodes.
        self.nodes
            .push(root_node.as_node(splitter.get_learning_rate()));
        let mut n_leaves = 1;

        let mut growable: Box<dyn Grower> = match grow_policy {
            GrowPolicy::DepthWise => Box::<VecDeque<SplittableNode>>::default(),
            GrowPolicy::LossGuide => Box::<BinaryHeap<SplittableNode>>::default(),
        };

        growable.add_node(root_node);
        while !growable.is_empty() {
            // If this will push us over the max leaves parameter, break.
            if (n_leaves + splitter.new_leaves_added()) > max_leaves {
                break;
            }
            // We know there is a value here, because of how the
            // while loop is setup.
            // Grab a splitable node from the stack
            // If we can split it, and update the corresponding
            // tree nodes children.
            let mut node = growable.get_next_node();
            let n_idx = node.num;

            let depth = node.depth + 1;

            // If we have hit max depth, skip this node
            // but keep going, because there may be other
            // valid shallower nodes.
            if depth > max_depth {
                continue;
            }

            // For max_leaves, subtract 1 from the n_leaves
            // every time we pop from the growable stack
            // then, if we can add two children, add two to
            // n_leaves. If we can't split the node any
            // more, then just add 1 back to n_leaves
            n_leaves -= 1;

            // Children at max_depth are never split, so skip their histograms.
            let new_nodes = splitter.split_node(
                &n_nodes,
                &mut node,
                &mut index,
                col_index,
                data,
                cuts,
                grad,
                hess,
                parallel,
                depth < max_depth,
            );

            let n_new_nodes = new_nodes.len();
            if n_new_nodes == 0 {
                n_leaves += 1;
            } else {
                rows_exact &= right_child_rows_exact(&node, &new_nodes, &index, data, cuts);
                for n in new_nodes.iter() {
                    if ranges.len() <= n.num {
                        ranges.resize(n.num + 1, (0, 0));
                    }
                    ranges[n.num] = (n.start_idx, n.stop_idx);
                }
                self.nodes[n_idx].make_parent_node(node);
                n_leaves += n_new_nodes;
                n_nodes += n_new_nodes;
                for n in new_nodes {
                    self.nodes.push(n.as_node(splitter.get_learning_rate()));
                    if !n.is_missing_leaf {
                        growable.add_node(n)
                    }
                }
            }
        }

        // Any final post processing required.
        splitter.clean_up_splits(self);
        rows_exact.then_some(TreeRows { index, ranges })
    }

    /// Predict a row from its bins. `bin(feature)` gives the row's bin for a feature,
    /// `split_bins` comes from `split_bins`, and `missing_bin` is the bin of missing
    /// values. A bin of `PREDICT_NAN_BIN` is a NAN when `missing` isn't, which panics
    /// like predicting the value would.
    #[inline]
    pub fn predict_row_from_bins(
        &self,
        bin: impl Fn(usize) -> u16,
        split_bins: &[u16],
        missing_bin: u16,
        missing: &f64,
    ) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.weight_value as f64;
            }
            let b = bin(node.split_feature);
            node_idx = if b == missing_bin {
                node.missing_node
            } else if b < split_bins[node_idx] {
                node.left_child
            } else {
                if b == PREDICT_NAN_BIN {
                    is_missing(&f64::NAN, missing);
                }
                node.right_child
            };
        }
    }

    pub fn predict_contributions_row_probability_change(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
        current_logodds: f64,
    ) -> f64 {
        contribs[contribs.len() - 1] +=
            odds(current_logodds + self.nodes[0].weight_value as f64) - odds(current_logodds);
        let mut node_idx = 0;
        let mut lo = current_logodds;
        loop {
            let node = &self.nodes[node_idx];
            let node_odds = odds(node.weight_value as f64 + current_logodds);
            if node.is_leaf {
                lo += node.weight_value as f64;
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let child_odds = odds(self.nodes[child_idx].weight_value as f64 + current_logodds);
            let delta = child_odds - node_odds;
            contribs[node.split_feature] += delta;
            node_idx = child_idx;
        }
        lo
    }

    // Branch average difference predictions
    pub fn predict_contributions_row_midpoint_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let child = &self.nodes[child_idx];
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[node.right_child]
            } else {
                &self.nodes[node.left_child]
            };
            let mid = (child.weight_value * child.hessian_sum
                + other_child.weight_value * other_child.hessian_sum)
                / (child.hessian_sum + other_child.hessian_sum);
            let delta = child.weight_value - mid;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // Branch difference predictions.
    pub fn predict_contributions_row_branch_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[node.right_child]
            } else {
                &self.nodes[node.left_child]
            };
            let delta = self.nodes[child_idx].weight_value - other_child.weight_value;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // How does the travelled childs weight change relative to the
    // mode branch.
    pub fn predict_contributions_row_mode_difference(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Bias term is left as 0.
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let left_node = &self.nodes[node.left_child];
            let right_node = &self.nodes[node.right_child];
            let child_weight = self.nodes[child_idx].weight_value;

            let delta = if left_node.hessian_sum == right_node.hessian_sum {
                0.
            } else if left_node.hessian_sum > right_node.hessian_sum {
                child_weight - left_node.weight_value
            } else {
                child_weight - right_node.weight_value
            };
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    pub fn predict_contributions_row_weight(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += self.nodes[0].weight_value as f64;
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = self.nodes[node_idx].weight_value as f64;
            let child_weight = self.nodes[child_idx].weight_value as f64;
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_weight(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row_weight(&data.get_row(*row), contribs, missing)
            })
    }

    /// This is the method that XGBoost uses.
    pub fn predict_contributions_row_average(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        weights: &[f64],
        missing: &f64,
    ) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += weights[0];
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = weights[node_idx];
            let child_weight = weights[child_idx];
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_average(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        weights: &[f64],
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row_average(
                    &data.get_row(*row),
                    contribs,
                    weights,
                    missing,
                )
            })
    }

    fn predict_leaf(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> &Node {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node;
            } else {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
            }
        }
    }

    pub fn predict_row_from_row_slice(&self, row: &[f64], missing: &f64) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.weight_value as f64;
            } else {
                node_idx = node.get_child_idx(&row[node.split_feature], missing);
            }
        }
    }

    fn predict_single_threaded(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index
            .iter()
            .map(|i| self.predict_leaf(data, *i, missing).weight_value as f64)
            .collect()
    }

    fn predict_parallel(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index
            .par_iter()
            .map(|i| self.predict_leaf(data, *i, missing).weight_value as f64)
            .collect()
    }

    pub fn predict(&self, data: &Matrix<f64>, parallel: bool, missing: &f64) -> Vec<f64> {
        if parallel {
            self.predict_parallel(data, missing)
        } else {
            self.predict_single_threaded(data, missing)
        }
    }

    pub fn predict_leaf_indices(&self, data: &Matrix<f64>, missing: &f64) -> Vec<usize> {
        data.index
            .par_iter()
            .map(|i| self.predict_leaf(data, *i, missing).num)
            .collect()
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64, missing: &f64) -> f64 {
        tree_partial_dependence(self, 0, feature, value, 1.0, missing)
    }
    fn distribute_node_leaf_weights(&self, i: usize, weights: &mut [f64]) -> f64 {
        let node = &self.nodes[i];
        let mut w = node.weight_value as f64;
        if !node.is_leaf {
            let left_node = &self.nodes[node.left_child];
            let right_node = &self.nodes[node.right_child];
            w = left_node.hessian_sum as f64
                * self.distribute_node_leaf_weights(node.left_child, weights);
            w += right_node.hessian_sum as f64
                * self.distribute_node_leaf_weights(node.right_child, weights);
            // If this a tree with a missing branch.
            if node.has_missing_branch() {
                let missing_node = &self.nodes[node.missing_node];
                w += missing_node.hessian_sum as f64
                    * self.distribute_node_leaf_weights(node.missing_node, weights);
            }
            w /= node.hessian_sum as f64;
        }
        weights[i] = w;
        w
    }
    pub fn distribute_leaf_weights(&self) -> Vec<f64> {
        let mut weights = vec![0.; self.nodes.len()];
        self.distribute_node_leaf_weights(0, &mut weights);
        weights
    }

    pub fn get_average_leaf_weights(&self, i: usize) -> f64 {
        let node = &self.nodes[i];
        let mut w = node.weight_value as f64;
        if node.is_leaf {
            w
        } else {
            let left_node = &self.nodes[node.left_child];
            let right_node = &self.nodes[node.right_child];
            w = left_node.hessian_sum as f64 * self.get_average_leaf_weights(node.left_child);
            w += right_node.hessian_sum as f64 * self.get_average_leaf_weights(node.right_child);
            // If this a tree with a missing branch.
            if node.has_missing_branch() {
                let missing_node = &self.nodes[node.missing_node];
                w += missing_node.hessian_sum as f64
                    * self.get_average_leaf_weights(node.missing_node);
            }
            w /= node.hessian_sum as f64;
            w
        }
    }

    fn calc_feature_node_stats<F>(
        &self,
        calc_stat: &F,
        node: &Node,
        stats: &mut HashMap<usize, (f32, usize)>,
    ) where
        F: Fn(&Node) -> f32,
    {
        if node.is_leaf {
            return;
        }
        stats
            .entry(node.split_feature)
            .and_modify(|(v, c)| {
                *v += calc_stat(node);
                *c += 1;
            })
            .or_insert((calc_stat(node), 1));
        self.calc_feature_node_stats(calc_stat, &self.nodes[node.left_child], stats);
        self.calc_feature_node_stats(calc_stat, &self.nodes[node.right_child], stats);
        if node.has_missing_branch() {
            self.calc_feature_node_stats(calc_stat, &self.nodes[node.missing_node], stats);
        }
    }

    fn get_node_stats<F>(&self, calc_stat: &F, stats: &mut HashMap<usize, (f32, usize)>)
    where
        F: Fn(&Node) -> f32,
    {
        self.calc_feature_node_stats(calc_stat, &self.nodes[0], stats);
    }

    pub fn calculate_importance_weight(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|_: &Node| 1., stats);
    }

    pub fn calculate_importance_gain(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.split_gain, stats);
    }

    pub fn calculate_importance_cover(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.hessian_sum, stats);
    }
}

impl Display for Tree {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut print_buffer: Vec<usize> = vec![0];
        let mut r = String::new();
        while let Some(idx) = print_buffer.pop() {
            let node = &self.nodes[idx];
            if node.is_leaf {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
            } else {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
                print_buffer.push(node.right_child);
                print_buffer.push(node.left_child);
                if node.has_missing_branch() {
                    print_buffer.push(node.missing_node);
                }
            }
        }
        write!(f, "{}", r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::sampler::{RandomSampler, Sampler};
    use crate::splitter::MissingImputerSplitter;
    use crate::utils::precision_round;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::fs;
    fn assert_subset_tree_matches<T: Splitter>(splitter: &T, col_index: &[usize], parallel: bool) {
        use crate::sampler::{GossSampler, RowSubset};
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        // Varied predictions, so the GOSS scores are mostly distinct.
        let yhat: Vec<f64> = (0..y.len())
            .map(|i| ((i * 7919) % 101) as f64 / 50. - 1.)
            .collect();
        let w = vec![1.; y.len()];
        let (mut g, mut h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        let data = Matrix::new(&data_vec, 891, 5);
        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let mut rng = StdRng::seed_from_u64(0);
        let (index, _) = GossSampler::new(0.2, 0.1).sample(&mut rng, &data.index, &mut g, &mut h);
        assert!(RowSubset::should_use(index.len(), data.rows));

        let fit = |data: &Matrix<u16>, index: Vec<usize>, g: &[f32], h: &[f32]| {
            let mut tree = Tree::new();
            tree.fit(
                data,
                index,
                col_index,
                &b.cuts,
                g,
                h,
                splitter,
                usize::MAX,
                5,
                parallel,
                &SampleMethod::Goss,
                &GrowPolicy::DepthWise,
            );
            serde_json::to_string(&tree).unwrap()
        };
        let full = fit(&bdata, index.clone(), &g, &h);
        let mut subset = RowSubset::default();
        subset.fill(&bdata, &index, col_index, &g, &h, parallel);
        let (subset_data, subset_index) = subset.matrix();
        let from_subset = fit(&subset_data, subset_index, &subset.grad, &subset.hess);
        assert!(full.matches("split_feature").count() > 3);
        assert_eq!(full, from_subset);
    }

    /// The rows `fit` leaves in each leaf must be the rows predicting reaches, and
    /// walking the tree on bins must match walking it on values.
    fn assert_tree_rows_match_predictions<T: Splitter>(
        splitter: &T,
        grow_policy: &GrowPolicy,
        sampled: bool,
    ) {
        use crate::sampler::GossSampler;
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat: Vec<f64> = (0..y.len())
            .map(|i| ((i * 7919) % 101) as f64 / 50. - 1.)
            .collect();
        let w = vec![1.; y.len()];
        let (mut g, mut h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        let data = Matrix::new(&data_vec, 891, 5);
        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let (index, sample_method) = if sampled {
            let mut rng = StdRng::seed_from_u64(0);
            let (index, _) =
                GossSampler::new(0.2, 0.1).sample(&mut rng, &data.index, &mut g, &mut h);
            (index, SampleMethod::Goss)
        } else {
            (data.index.to_owned(), SampleMethod::None)
        };
        let mut tree = Tree::new();
        let rows = tree
            .fit(
                &bdata,
                index.clone(),
                &[0, 1, 2, 3, 4],
                &b.cuts,
                &g,
                &h,
                splitter,
                24,
                6,
                true,
                &sample_method,
                grow_policy,
            )
            .expect("partition matches predictions");
        assert!(tree.nodes.len() > 9);
        let predictions = tree.predict(&data, false, &f64::NAN);
        let mut seen = vec![0; data.rows];
        for (leaf_rows, weight) in rows.leaves(&tree) {
            for &row in leaf_rows {
                seen[row] += 1;
                assert_eq!(predictions[row], weight);
            }
        }
        for (row, count) in seen.iter().enumerate() {
            assert_eq!(*count, index.contains(&row) as usize);
        }
        let split_bins = split_bins(&tree, &b.cuts);
        for (row, prediction) in predictions.iter().enumerate() {
            assert_eq!(
                tree.predict_row_from_bins(|f| *bdata.get(row, f), &split_bins, 0, &f64::NAN),
                *prediction
            );
        }
    }

    /// Predicting from `bin_for_prediction` bins must match predicting from values,
    /// including values outside the training range, on a cut, missing, and infinite.
    #[test]
    fn test_predict_from_bins_matches_values() {
        use crate::binning::{bin_for_prediction, PREDICT_MISSING_BIN};
        let file = fs::read_to_string("resources/contiguous_with_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file
            .lines()
            .map(|x| x.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &vec![0.5; y.len()], &w);
        let (rows, cols) = (891, 5);
        // Missing as NAN, and missing as a number (with the NANs replaced).
        for missing in [f64::NAN, -9999.0] {
            let train: Vec<f64> = data_vec
                .iter()
                .map(|v| if v.is_nan() { missing } else { *v })
                .collect();
            let data = Matrix::new(&train, rows, cols);
            let b = bin_matrix(&data, &w, 64, missing, false).unwrap();
            let bdata = Matrix::new(&b.binned_data, rows, cols);
            let splitter = MissingImputerSplitter {
                l1: 0.0,
                l2: 1.0,
                max_delta_step: 0.,
                gamma: 0.0,
                min_leaf_weight: 1.0,
                learning_rate: 0.3,
                allow_missing_splits: true,
                constraints_map: ConstraintMap::new(),
            };
            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                data.index.to_owned(),
                &[0, 1, 2, 3, 4],
                &b.cuts,
                &g,
                &h,
                &splitter,
                usize::MAX,
                6,
                false,
                &SampleMethod::None,
                &GrowPolicy::DepthWise,
            );
            assert!(tree.nodes.len() > 9);
            // Each column: every cut, values just around each cut, beyond the range,
            // infinities and missing.
            let mut columns: Vec<Vec<f64>> = (0..cols)
                .map(|c| {
                    let mut v = Vec::new();
                    for cut in b.cuts.get_col(c) {
                        v.extend([*cut, cut - 1e-9, cut + 1e-9, cut - 0.5, cut + 0.5]);
                    }
                    v.extend([f64::INFINITY, f64::NEG_INFINITY, f64::MIN, missing, -1e300]);
                    v
                })
                .collect();
            let n = columns.iter().map(|c| c.len()).max().unwrap();
            for c in columns.iter_mut() {
                let len = c.len();
                for k in len..n {
                    c.push(c[k % len]);
                }
            }
            // Pair each value with every other column's values at a few offsets.
            let mut eval_cols: Vec<Vec<f64>> = vec![Vec::new(); cols];
            for shift in 0..7 {
                for (c, col) in columns.iter().enumerate() {
                    eval_cols[c].extend((0..n).map(|k| col[(k + shift * (c + 1)) % n]));
                }
            }
            let eval_rows = eval_cols[0].len();
            let eval_vec: Vec<f64> = eval_cols.concat();
            let eval = Matrix::new(&eval_vec, eval_rows, cols);
            let expected = tree.predict(&eval, false, &missing);
            let bins = bin_for_prediction(&eval, &b.cuts, &missing, true);
            let split_bins = split_bins(&tree, &b.cuts);
            for (row, e) in expected.iter().enumerate() {
                let p = tree.predict_row_from_bins(
                    |f| bins[f * eval_rows + row],
                    &split_bins,
                    PREDICT_MISSING_BIN,
                    &missing,
                );
                assert_eq!(p, *e);
            }
        }
    }

    #[test]
    #[should_panic(expected = "NAN value found in data")]
    fn test_predict_from_bins_panics_on_nan_when_missing_is_a_number() {
        use crate::binning::{bin_for_prediction, PREDICT_MISSING_BIN};
        let tree: Tree = serde_json::from_str(
            r#"{"nodes":[
            {"num":0,"weight_value":0.0,"hessian_sum":2.0,"depth":0,"split_value":1.5,"split_feature":0,"split_gain":1.0,"missing_node":1,"left_child":1,"right_child":2,"is_leaf":false},
            {"num":1,"weight_value":-1.0,"hessian_sum":1.0,"depth":1,"split_value":0.0,"split_feature":0,"split_gain":0.0,"missing_node":0,"left_child":0,"right_child":0,"is_leaf":true},
            {"num":2,"weight_value":1.0,"hessian_sum":1.0,"depth":1,"split_value":0.0,"split_feature":0,"split_gain":0.0,"missing_node":0,"left_child":0,"right_child":0,"is_leaf":true}]}"#,
        )
        .unwrap();
        let mut cuts = crate::data::JaggedMatrix::new();
        cuts.data = vec![1.0, 1.5, f64::MAX];
        cuts.ends = vec![3];
        cuts.cols = 1;
        cuts.n_records = 3;
        let values = [f64::NAN];
        let data = Matrix::new(&values, 1, 1);
        let bins = bin_for_prediction(&data, &cuts, &-1.0, false);
        let split_bins = split_bins(&tree, &cuts);
        tree.predict_row_from_bins(|f| bins[f], &split_bins, PREDICT_MISSING_BIN, &-1.0);
    }

    #[test]
    fn test_tree_rows_match_predictions() {
        let imputer = MissingImputerSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let branch = crate::splitter::MissingBranchSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
            terminate_missing_features: std::collections::HashSet::new(),
            missing_node_treatment: crate::gradientbooster::MissingNodeTreatment::AverageLeafWeight,
            force_children_to_bound_parent: false,
        };
        for grow_policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
            for sampled in [false, true] {
                assert_tree_rows_match_predictions(&imputer, &grow_policy, sampled);
                assert_tree_rows_match_predictions(&branch, &grow_policy, sampled);
            }
        }
    }

    #[test]
    fn test_tree_fit_on_row_subset_matches_full_data() {
        let imputer = MissingImputerSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let branch = crate::splitter::MissingBranchSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
            terminate_missing_features: std::collections::HashSet::new(),
            missing_node_treatment: crate::gradientbooster::MissingNodeTreatment::AssignToParent,
            force_children_to_bound_parent: false,
        };
        for parallel in [false, true] {
            for col_index in [vec![0, 1, 2, 3, 4], vec![0, 2, 4]] {
                assert_subset_tree_matches(&imputer, &col_index, parallel);
                assert_subset_tree_matches(&branch, &col_index, parallel);
            }
        }
    }

    #[test]
    fn test_tree_fit_with_subsample() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (mut g, mut h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        // let mut h = LogLoss::calc_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
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
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let mut rng = StdRng::seed_from_u64(0);
        let (index, excluded) =
            RandomSampler::new(0.5).sample(&mut rng, &data.index, &mut g, &mut h);
        assert!(excluded.len() > 0);
        let col_index: Vec<usize> = (0..data.cols).collect();
        tree.fit(
            &bdata,
            index,
            &col_index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::Random,
            &GrowPolicy::DepthWise,
        );
    }

    #[test]
    fn test_tree_fit() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
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
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &col_index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::None,
            &GrowPolicy::DepthWise,
        );

        // println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_colsample() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
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
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = vec![1, 3];
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &col_index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            false,
            &SampleMethod::None,
            &GrowPolicy::DepthWise,
        );
        for n in tree.nodes {
            if !n.is_leaf {
                assert!((n.split_feature == 1) || (n.split_feature == 3))
            }
        }
    }

    #[test]
    fn test_tree_fit_monotone() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        println!("GRADIENT -- {:?}", h);

        let data_ = Matrix::new(&data_vec, 891, 5);
        let data = Matrix::new(data_.get_col(1), 891, 1);
        let map = ConstraintMap::from([(0, Constraint::Negative)]);
        let splitter = MissingImputerSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 0.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: true,
            constraints_map: map,
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &col_index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            5,
            true,
            &SampleMethod::None,
            &GrowPolicy::DepthWise,
        );

        // println!("{}", tree);
        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pred_data_vec.dedup();
        let pred_data = Matrix::new(&pred_data_vec, pred_data_vec.len(), 1);

        let preds = tree.predict(&pred_data, false, &f64::NAN);
        let increasing = preds.windows(2).all(|a| a[0] >= a[1]);
        assert!(increasing);

        let weights = tree.distribute_leaf_weights();

        // Average contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_lossguide() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv")
            .expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter {
            l1: 0.0,
            l2: 1.0,
            max_delta_step: 0.,
            gamma: 3.0,
            min_leaf_weight: 1.0,
            learning_rate: 0.3,
            allow_missing_splits: false,
            constraints_map: ConstraintMap::new(),
        };
        let mut tree = Tree::new();

        let b = bin_matrix(&data, &w, 300, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        tree.fit(
            &bdata,
            data.index.to_owned(),
            &col_index,
            &b.cuts,
            &g,
            &h,
            &splitter,
            usize::MAX,
            usize::MAX,
            true,
            &SampleMethod::None,
            &GrowPolicy::LossGuide,
        );

        println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        // assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs
            .chunks(data.cols + 1)
            .map(|i| i.iter().sum())
            .collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }
}
