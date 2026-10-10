use crate::binning::TiledBins;
use crate::data::{FloatData, JaggedMatrix, Matrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many rows, gathering gradients serially is cheaper than scheduling Rayon tasks.
const PARALLEL_GATHER_MIN_ROWS: usize = 16_384;
/// Bins per Rayon task when subtracting histograms in parallel.
const PARALLEL_SUBTRACT_MIN_BINS: usize = 4_096;
/// Rows in a segment of the tiled fill, at least.
const TILE_SEGMENT_MIN_ROWS: usize = 2_048;
/// Segments the tiled fill splits a node's rows into, at most.
const TILE_MAX_SEGMENTS: usize = 16;
/// Bytes of per-segment histograms the tiled fill may hold at once.
const TILE_PARTIALS_MAX_BYTES: usize = 64 << 20;
/// The tiled fill is only used when it has at least this many tasks; smaller
/// nodes use the column-wise fill, which has a task per column.
const TILE_MIN_TASKS: usize = 8;
/// Rows ahead the tiled fill prefetches, when a node's rows are scattered.
const TILE_PREFETCH_ROWS: usize = 16;

/// Hint that `p` will be read soon. Prefetching never faults, whatever the address.
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` only hints the cache; it doesn't dereference `p`.
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Struct to hold the information of a given bin. Bin `k > 0` of a feature's
/// histogram covers values below cut `k - 1` of that feature; bin 0 is missing.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct Bin<T> {
    /// The sum of the gradient for this bin.
    pub gradient_sum: T,
    /// The sum of the hession values for this bin.
    pub hessian_sum: T,
}

impl Bin<f32> {
    pub fn new_f32() -> Self {
        Bin {
            gradient_sum: f32::ZERO,
            hessian_sum: f32::ZERO,
        }
    }

    /// Calculate a new bin, using the subtraction trick on the parent bin,
    /// and the child bin.
    pub fn from_parent_child(root_bin: &Bin<f32>, child_bin: &Bin<f32>) -> Self {
        Bin {
            gradient_sum: root_bin.gradient_sum - child_bin.gradient_sum,
            hessian_sum: root_bin.hessian_sum - child_bin.hessian_sum,
        }
    }

    /// Calculate a new bin, using the subtraction trick when the parent node
    /// has three directions, left, right, and missing.
    pub fn from_parent_two_children(
        root_bin: &Bin<f32>,
        first_child_bin: &Bin<f32>,
        second_child_bin: &Bin<f32>,
    ) -> Self {
        Bin {
            gradient_sum: root_bin.gradient_sum
                - (first_child_bin.gradient_sum + second_child_bin.gradient_sum),
            hessian_sum: root_bin.hessian_sum
                - (first_child_bin.hessian_sum + second_child_bin.hessian_sum),
        }
    }
}

impl Bin<f64> {
    pub fn new_f64() -> Self {
        Bin {
            gradient_sum: f64::ZERO,
            hessian_sum: f64::ZERO,
        }
    }

    pub fn as_f32_bin(&self) -> Bin<f32> {
        Bin {
            gradient_sum: self.gradient_sum as f32,
            hessian_sum: self.hessian_sum as f32,
        }
    }
}

/// Histograms implemented as as jagged matrix.
#[derive(Debug, Deserialize, Serialize)]
pub struct HistogramMatrix(pub JaggedMatrix<Bin<f32>>);

/// Fill the histogram for a given feature. Sums accumulate in f64 so we don't
/// lose precision, but are stored as f32 values for memory efficiency and speed.
/// `out` must have one bin per cut value: the missing bin, then one per cut
/// excluding the final maximum.
pub fn fill_feature_histogram(
    out: &mut [Bin<f32>],
    feature: &[u16],
    sorted_grad: &[f32],
    sorted_hess: &[f32],
    index: &[usize],
) {
    let mut sums = vec![(f64::ZERO, f64::ZERO); out.len()];
    index
        .iter()
        .zip(sorted_grad)
        .zip(sorted_hess)
        .for_each(|((i, g), h)| {
            if let Some(v) = sums.get_mut(feature[*i] as usize) {
                v.0 += f64::from(*g);
                v.1 += f64::from(*h);
            }
        });
    for (bin, (g, h)) in out.iter_mut().zip(sums) {
        *bin = Bin {
            gradient_sum: g as f32,
            hessian_sum: h as f32,
        };
    }
}

/// The gradients and hessians of `index`'s rows, in its order.
fn gather(grad: &[f32], hess: &[f32], index: &[usize], parallel: bool) -> (Vec<f32>, Vec<f32>) {
    if parallel && index.len() >= PARALLEL_GATHER_MIN_ROWS {
        index.par_iter().map(|&i| (grad[i], hess[i])).unzip()
    } else {
        index.iter().map(|&i| (grad[i], hess[i])).unzip()
    }
}

/// The end of each column's bins in a histogram over `col_index`, and the total bins.
fn histogram_ends(
    n_cols: usize,
    cuts: &JaggedMatrix<f64>,
    col_index: &[usize],
) -> (Vec<usize>, usize) {
    // If we have sampled down the columns, we need to recalculate the ends.
    // we can do this by iterating over the cut's, as this will be the size
    // of the histograms.
    if col_index.len() == n_cols {
        return (cuts.ends.to_owned(), cuts.n_records);
    }
    let ends: Vec<usize> = col_index
        .iter()
        .scan(0_usize, |state, i| {
            *state += cuts.get_col(*i).len();
            Some(*state)
        })
        .collect();
    let n_records = ends.iter().sum();
    (ends, n_records)
}

/// The columns of one tile that a histogram uses.
struct TileWork {
    tile: usize,
    /// Each column's position in the tile's rows.
    positions: Vec<usize>,
    /// Where each column's bins start in a task's scratch histogram. A column has one
    /// slot past its bins, for values past the last cut (`+inf`), which are dropped
    /// like the column-wise fill drops them. This saves a check on every value.
    offsets: Vec<usize>,
    /// Each column's number of bins.
    bins: Vec<usize>,
    /// True if every column of the tile is used, in order.
    full: bool,
    /// Slots in a task's scratch histogram.
    scratch: usize,
}

/// How the tiled fill splits a histogram into tasks: one for each tile used and
/// segment of the node's rows. The split depends only on the node's size and the
/// columns used, so the sums are added in the same order for any number of threads.
struct TilePlan {
    work: Vec<TileWork>,
    segment_rows: usize,
    segments: usize,
}

impl TilePlan {
    /// `None` if the columns aren't in increasing order, or there's too little work.
    fn new(
        tiles: &TiledBins,
        cuts: &JaggedMatrix<f64>,
        rows: usize,
        col_index: &[usize],
    ) -> Option<Self> {
        if !col_index.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        let mut work = Vec::new();
        let mut cols = col_index.iter().peekable();
        for (tile, bounds) in tiles.starts.windows(2).enumerate() {
            let (mut positions, mut offsets, mut bins) = (Vec::new(), Vec::new(), Vec::new());
            let mut scratch = 0;
            while let Some(&&col) = cols.peek() {
                if col >= bounds[1] {
                    break;
                }
                let col_bins = cuts.get_col(col).len();
                positions.push(col - bounds[0]);
                offsets.push(scratch);
                bins.push(col_bins);
                scratch += col_bins + 1;
                cols.next();
            }
            if !positions.is_empty() {
                work.push(TileWork {
                    tile,
                    full: positions.len() == bounds[1] - bounds[0],
                    positions,
                    offsets,
                    bins,
                    scratch,
                });
            }
        }
        let scratch: usize = work.iter().map(|w| w.scratch).sum();
        let max_segments =
            (TILE_PARTIALS_MAX_BYTES / (16 * scratch.max(1))).clamp(1, TILE_MAX_SEGMENTS);
        let segment_rows = rows.div_ceil(max_segments).max(TILE_SEGMENT_MIN_ROWS);
        let segments = rows.div_ceil(segment_rows);
        (work.len() * segments >= TILE_MIN_TASKS).then_some(TilePlan {
            work,
            segment_rows,
            segments,
        })
    }
}

impl HistogramMatrix {
    /// Create an empty histogram matrix.
    pub fn empty() -> Self {
        HistogramMatrix(JaggedMatrix {
            data: Vec::new(),
            ends: Vec::new(),
            cols: 0,
            n_records: 0,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        col_index: &[usize],
        parallel: bool,
        sort: bool,
    ) -> Self {
        // Sort gradients and hessians to reduce cache hits.
        // This made a really sizeable difference on larger datasets
        // Bringing training time down from nearly 6 minutes, to 2 minutes.
        let gathered: (Vec<f32>, Vec<f32>);
        let (sorted_grad, sorted_hess): (&[f32], &[f32]) = if !sort {
            (grad, hess)
        } else {
            gathered = gather(grad, hess, index, parallel);
            (&gathered.0, &gathered.1)
        };

        let (ends, n_records) = histogram_ends(data.cols, cuts, col_index);

        let total_bins = ends.last().copied().unwrap_or(0);
        let mut histograms: Vec<Bin<f32>> = Vec::with_capacity(total_bins);
        if parallel {
            (0..total_bins)
                .into_par_iter()
                .map(|_| Bin::new_f32())
                .collect_into_vec(&mut histograms);
        } else {
            histograms.resize(total_bins, Bin::new_f32());
        }
        let mut column_bins: Vec<&mut [Bin<f32>]> = Vec::with_capacity(col_index.len());
        let mut rest = histograms.as_mut_slice();
        for col in col_index {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(cuts.get_col(*col).len());
            column_bins.push(head);
            rest = tail;
        }
        let fill = |(out, col): (&mut [Bin<f32>], &usize)| {
            fill_feature_histogram(out, data.get_col(*col), sorted_grad, sorted_hess, index)
        };
        if parallel {
            column_bins.into_par_iter().zip(col_index).for_each(fill);
        } else {
            column_bins.into_iter().zip(col_index).for_each(fill);
        }

        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends,
            cols: col_index.len(),
            n_records,
        })
    }

    /// Like `new`, but uses the tiled, row-wise fill when `tiles` is given and the
    /// node is big enough. The tiled fill adds up rows in a different order, so its
    /// sums can differ from `new`'s in the last bits.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        tiles: Option<&TiledBins>,
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        col_index: &[usize],
        parallel: bool,
        sort: bool,
    ) -> Self {
        if let Some(tiles) = tiles {
            if let Some(plan) = TilePlan::new(tiles, cuts, index.len(), col_index) {
                return Self::new_tiled(
                    tiles, &plan, data.cols, cuts, grad, hess, index, col_index, parallel, sort,
                );
            }
        }
        Self::new(data, cuts, grad, hess, index, col_index, parallel, sort)
    }

    /// Fill the histogram row by row, a tile of columns at a time. Each task sums one
    /// segment of the node's rows for one tile, then each tile's segment sums are added
    /// in order. Gradients are read by row, so unlike `new` they aren't gathered first.
    #[allow(clippy::too_many_arguments)]
    fn new_tiled(
        tiles: &TiledBins,
        plan: &TilePlan,
        n_cols: usize,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: &[f32],
        index: &[usize],
        col_index: &[usize],
        parallel: bool,
        sort: bool,
    ) -> Self {
        let (ends, n_records) = histogram_ends(n_cols, cuts, col_index);
        // Without `sort`, the rows are the whole data in order, which the hardware
        // prefetches on its own.
        let scattered = sort;

        let fill = |task: usize| {
            let work = &plan.work[task / plan.segments];
            let first = (task % plan.segments) * plan.segment_rows;
            let rows = &index[first..(first + plan.segment_rows).min(index.len())];
            let tile = &tiles.tiles[work.tile];
            let width = tiles.width(work.tile);
            let mut sums = vec![(f64::ZERO, f64::ZERO); work.scratch];
            let mut add = |slot: usize, g: f64, h: f64| {
                let v = &mut sums[slot];
                v.0 += g;
                v.1 += h;
            };
            for (k, &i) in rows.iter().enumerate() {
                if scattered {
                    if let Some(&ahead) = rows.get(k + TILE_PREFETCH_ROWS) {
                        let row = tile.as_ptr().wrapping_add(ahead * width);
                        prefetch(row);
                        prefetch(row.wrapping_add(width - 1));
                        prefetch(grad.as_ptr().wrapping_add(ahead));
                        prefetch(hess.as_ptr().wrapping_add(ahead));
                    }
                }
                let row = &tile[i * width..(i + 1) * width];
                let (g, h) = (f64::from(grad[i]), f64::from(hess[i]));
                if work.full {
                    for (&bin, &offset) in row.iter().zip(&work.offsets) {
                        add(offset + bin as usize, g, h);
                    }
                } else {
                    for (&position, &offset) in work.positions.iter().zip(&work.offsets) {
                        add(offset + row[position] as usize, g, h);
                    }
                }
            }
            sums
        };
        let n_tasks = plan.work.len() * plan.segments;
        let partials: Vec<Vec<(f64, f64)>> = if parallel {
            (0..n_tasks).into_par_iter().map(fill).collect()
        } else {
            (0..n_tasks).map(fill).collect()
        };

        let total_bins = ends.last().copied().unwrap_or(0);
        let mut histograms = vec![Bin::new_f32(); total_bins];
        let mut tile_bins: Vec<&mut [Bin<f32>]> = Vec::with_capacity(plan.work.len());
        let mut rest = histograms.as_mut_slice();
        for work in &plan.work {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(work.bins.iter().sum());
            tile_bins.push(head);
            rest = tail;
        }
        let add_segments = |(w, out): (usize, &mut [Bin<f32>])| {
            let mut segments = partials[w * plan.segments..(w + 1) * plan.segments].iter();
            let mut sums = segments.next().expect("a tile has a segment").clone();
            for segment in segments {
                for (s, v) in sums.iter_mut().zip(segment) {
                    s.0 += v.0;
                    s.1 += v.1;
                }
            }
            let work = &plan.work[w];
            let kept = work
                .offsets
                .iter()
                .zip(&work.bins)
                .flat_map(|(&offset, &bins)| &sums[offset..offset + bins]);
            for (bin, &(g, h)) in out.iter_mut().zip(kept) {
                *bin = Bin {
                    gradient_sum: g as f32,
                    hessian_sum: h as f32,
                };
            }
        };
        if parallel {
            tile_bins.into_par_iter().enumerate().for_each(add_segments);
        } else {
            tile_bins.into_iter().enumerate().for_each(add_segments);
        }

        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends,
            cols: col_index.len(),
            n_records,
        })
    }

    /// Calculate the histogram matrix, for a child, given the parent histogram
    /// matrix, and the other child histogram matrix. This should be used
    /// when the node has only two possible splits, left and right.
    pub fn from_parent_child(
        root_histogram: &HistogramMatrix,
        child_histogram: &HistogramMatrix,
        parallel: bool,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(child) = child_histogram;
        let histograms = if parallel {
            root.data
                .par_iter()
                .zip(child.data.par_iter())
                .with_min_len(PARALLEL_SUBTRACT_MIN_BINS)
                .map(|(root_bin, child_bin)| Bin::from_parent_child(root_bin, child_bin))
                .collect()
        } else {
            root.data
                .iter()
                .zip(child.data.iter())
                .map(|(root_bin, child_bin)| Bin::from_parent_child(root_bin, child_bin))
                .collect()
        };
        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends: child.ends.to_owned(),
            cols: child.cols,
            n_records: child.n_records,
        })
    }

    /// Calculate the histogram matrix for a child, given the parent histogram
    /// and two other child histograms. This should be used with the node has
    /// three possible split paths, right, left, and missing.
    pub fn from_parent_two_children(
        root_histogram: &HistogramMatrix,
        first_child_histogram: &HistogramMatrix,
        second_child_histogram: &HistogramMatrix,
        parallel: bool,
    ) -> Self {
        let HistogramMatrix(root) = root_histogram;
        let HistogramMatrix(first_child) = first_child_histogram;
        let HistogramMatrix(second_child) = second_child_histogram;
        let histograms = if parallel {
            root.data
                .par_iter()
                .zip(first_child.data.par_iter())
                .zip(second_child.data.par_iter())
                .with_min_len(PARALLEL_SUBTRACT_MIN_BINS)
                .map(|((root_bin, first_child_bin), second_child_bin)| {
                    Bin::from_parent_two_children(root_bin, first_child_bin, second_child_bin)
                })
                .collect()
        } else {
            root.data
                .iter()
                .zip(first_child.data.iter())
                .zip(second_child.data.iter())
                .map(|((root_bin, first_child_bin), second_child_bin)| {
                    Bin::from_parent_two_children(root_bin, first_child_bin, second_child_bin)
                })
                .collect()
        };
        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends: first_child.ends.to_owned(),
            cols: first_child.cols,
            n_records: first_child.n_records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use std::fs;
    #[test]
    fn test_single_histogram() {
        let file = fs::read_to_string("resources/contiguous_no_missing.csv")
            .expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let sample_weight = vec![1.; data.rows];
        let b = bin_matrix(&data, &sample_weight, 10, f64::NAN, false).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let w = vec![1.; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);
        let cuts = b.cuts.get_col(1);
        let mut hist = vec![Bin::new_f32(); cuts.len()];
        fill_feature_histogram(&mut hist, &bdata.get_col(1), &g, &h, &bdata.index);
        // println!("{:?}", hist);
        let mut f = bdata.get_col(1).to_owned();
        println!("{:?}", hist);
        f.sort();
        f.dedup();
        assert_eq!(f.len() + 1, hist.len());
    }

    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    /// Random column-major bins (0 is missing) with 2-300 bins a column, cuts with
    /// those bin counts, and gradients spanning many magnitudes. A bin one past a
    /// column's last is what `map_bin` gives `+inf`; histograms drop it.
    fn random_binned(
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> (Vec<u16>, JaggedMatrix<f64>, Vec<f32>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_bins: Vec<usize> = (0..cols).map(|_| rng.gen_range(2..=300)).collect();
        let mut bins = Vec::with_capacity(rows * cols);
        for &n in &n_bins {
            bins.extend((0..rows).map(|_| rng.gen_range(0..=n) as u16));
        }
        let ends: Vec<usize> = n_bins
            .iter()
            .scan(0, |total, n| {
                *total += n;
                Some(*total)
            })
            .collect();
        let total = *ends.last().unwrap();
        let cuts = JaggedMatrix {
            data: vec![0.0; total],
            ends,
            cols,
            n_records: total,
        };
        let grad = (0..rows)
            .map(|_| (rng.gen_range(-1.0..1.0f64) * 2f64.powi(-rng.gen_range(0..20))) as f32)
            .collect();
        let hess = (0..rows)
            .map(|_| 2f64.powi(-rng.gen_range(2..20)) as f32)
            .collect();
        (bins, cuts, grad, hess)
    }

    fn plan(
        tiles: &TiledBins,
        cuts: &JaggedMatrix<f64>,
        rows: usize,
        col_index: &[usize],
        segment_rows: usize,
    ) -> TilePlan {
        let mut plan = TilePlan::new(tiles, cuts, usize::MAX / 2, col_index).unwrap();
        plan.segment_rows = segment_rows;
        plan.segments = rows.div_ceil(segment_rows);
        plan
    }

    #[test]
    fn test_tiled_bins_layout() {
        let (bins, cuts, _, _) = random_binned(1000, 75, 0);
        let data = Matrix::new(&bins, 1000, 75);
        let tiles = TiledBins::new(&data, &cuts, true);
        assert_eq!(*tiles.starts.last().unwrap(), 75);
        for t in 0..tiles.tiles.len() {
            let width = tiles.width(t);
            let tile_bins: usize = (tiles.starts[t]..tiles.starts[t + 1])
                .map(|c| cuts.get_col(c).len())
                .sum();
            assert!(width <= 32 && (width == 1 || tile_bins <= 8_192));
            for r in 0..1000 {
                for j in 0..width {
                    assert_eq!(
                        tiles.tiles[t][r * width + j],
                        data.get_col(tiles.starts[t] + j)[r]
                    );
                }
            }
        }
    }

    #[test]
    fn test_tiled_matches_column_wise() {
        let (rows, cols) = (3000, 75);
        let (bins, cuts, grad, hess) = random_binned(rows, cols, 1);
        let data = Matrix::new(&bins, rows, cols);
        let tiles = TiledBins::new(&data, &cuts, true);
        let mut rng = StdRng::seed_from_u64(2);
        let all: Vec<usize> = (0..cols).collect();
        let mut some = all.clone();
        some.shuffle(&mut rng);
        some.truncate(20);
        some.sort();
        let mut sampled: Vec<usize> = (0..rows).filter(|_| rng.gen_bool(0.3)).collect();
        sampled.shuffle(&mut rng);
        // (rows, sort): the root in order, and a sampled node in any order.
        for (index, sort) in [(&data.index, false), (&sampled, true)] {
            for col_index in [&all, &some] {
                for segment_rows in [97, 1000, rows] {
                    let expected = HistogramMatrix::new(
                        &data, &cuts, &grad, &hess, index, col_index, true, sort,
                    );
                    let p = plan(&tiles, &cuts, index.len(), col_index, segment_rows);
                    let tiled = HistogramMatrix::new_tiled(
                        &tiles, &p, cols, &cuts, &grad, &hess, index, col_index, true, sort,
                    );
                    assert_eq!(expected.0.ends, tiled.0.ends);
                    assert_eq!(expected.0.n_records, tiled.0.n_records);
                    for (e, t) in expected.0.data.iter().zip(&tiled.0.data) {
                        for (a, b) in [
                            (e.gradient_sum, t.gradient_sum),
                            (e.hessian_sum, t.hessian_sum),
                        ] {
                            assert!((a - b).abs() <= 1e-6 * a.abs().max(1e-6), "{a} {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_tiled_same_for_any_thread_count() {
        let (rows, cols) = (5000, 40);
        let (bins, cuts, grad, hess) = random_binned(rows, cols, 3);
        let data = Matrix::new(&bins, rows, cols);
        let tiles = TiledBins::new(&data, &cuts, true);
        let all: Vec<usize> = (0..cols).collect();
        let p = plan(&tiles, &cuts, rows, &all, 311);
        let fill = |parallel| {
            HistogramMatrix::new_tiled(
                &tiles,
                &p,
                cols,
                &cuts,
                &grad,
                &hess,
                &data.index,
                &all,
                parallel,
                true,
            )
            .0
            .data
            .iter()
            .map(|b| (b.gradient_sum.to_bits(), b.hessian_sum.to_bits()))
            .collect::<Vec<_>>()
        };
        let serial = fill(false);
        for threads in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(serial, pool.install(|| fill(true)));
        }
    }

    #[test]
    fn test_build_falls_back_to_column_wise() {
        let (rows, cols) = (3000, 75);
        let (bins, cuts, grad, hess) = random_binned(rows, cols, 4);
        let data = Matrix::new(&bins, rows, cols);
        let tiles = TiledBins::new(&data, &cuts, true);
        let bits = |h: HistogramMatrix| {
            h.0.data
                .iter()
                .map(|b| (b.gradient_sum.to_bits(), b.hessian_sum.to_bits()))
                .collect::<Vec<_>>()
        };
        let all: Vec<usize> = (0..cols).collect();
        let unordered = vec![5, 3, 40];
        let small: Vec<usize> = (0..100).collect();
        for (index, col_index) in [(&data.index, &unordered), (&small, &all)] {
            let new =
                HistogramMatrix::new(&data, &cuts, &grad, &hess, index, col_index, true, true);
            let build = HistogramMatrix::build(
                Some(&tiles),
                &data,
                &cuts,
                &grad,
                &hess,
                index,
                col_index,
                true,
                true,
            );
            assert_eq!(bits(new), bits(build));
        }
    }
}
