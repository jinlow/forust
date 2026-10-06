# Optimization Results

Running log of each change in `plan.md` and its effect on speed. Raw data is in
`results.jsonl`. Tables are generated with `scripts/perf_compare.py`.

## How to read this

- **Tree ms:** steady-state milliseconds per tree (the first 10 iterations are
  skipped), median of 3 runs, from `scripts/perf_grid.py`.
- **Serial fraction:** share of training-loop wall time spent in serial code
  at 8 threads, from `scripts/perf_serial_share.py`. Lower is better.
- **Unchanged models:** after every change, `scripts/perf_golden.py check`
  confirms the trained trees are byte-identical to the baseline (5
  configurations, both splitters), and `cargo test` passes, including a test
  that serial and parallel training give identical trees.
- **Setup:** synthetic binary-classification data from
  `scripts/make_perf_data.py`, learning rate 0.1, 256 bins, evaluation set of
  1/4 the training rows. Azure VM with 8 physical cores (16 vCPU). Repeat runs
  vary by about 5%.
- **Serial timings also carry about ±6% from code placement.** Between item 2b
  and 2c, the serial histogram loop slowed 3-8% even though its instructions
  were byte-identical; only its position relative to 64-byte boundaries
  changed (details under item 2c). Read small serial differences as noise.

| Grid | What it isolates |
| --- | --- |
| G1 | 200 columns, depth 5, 25k-250k rows: fixed vs per-row cost |
| G2 | 200 columns, depth 8: many small nodes |
| G3 | 500 columns, depth 5: wide data |
| G4 | 1M rows x 200 columns, depth 5 |

## Summary

All four plan items are done. Trained models are byte-identical to the
baseline throughout, and the Python suite passes (112 tests, including the
XGBoost parity checks and a new `num_threads` test).

| Workload | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| 100-iteration `fit`, 1M x 200, 8 threads | 52.7 s | 26.1 s | 2.0x faster |
| 1,000 iterations, 100k x 200, default 16 threads | 80.8 s | 37.2 s | 2.2x faster |
| Tree, 8 threads, across G1-G3 | | | 39-81% less time |
| Binning, 8 threads | | | 91-93% less time |

Tree ms per change. Each column includes all earlier changes; "item2" is
items 2a-2c (2d was dropped). Per-step numbers are in the step log.

| Grid | Data | Rows | Depth | Mode | baseline | item1 | item2 | item3 | item4 | final vs baseline |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | w200 | 25k | 5 | serial | 69.5 | 69.8 | 57.9 | 59.4 | 60.2 | -13.4% |
| G1 | w200 | 25k | 5 | 8 threads | 52.7 | 35.9 | 20.0 | 16.5 | 13.1 | -75.1% |
| G1 | w200 | 100k | 5 | serial | 165.4 | 169.8 | 153.9 | 158.0 | 156.9 | -5.1% |
| G1 | w200 | 100k | 5 | 8 threads | 72.5 | 56.4 | 39.0 | 39.3 | 32.5 | -55.2% |
| G1 | w200 | 250k | 5 | serial | 340.8 | 340.1 | 332.3 | 333.5 | 333.0 | -2.3% |
| G1 | w200 | 250k | 5 | 8 threads | 108.3 | 89.2 | 73.8 | 73.4 | 66.1 | -39.0% |
| G2 | w200 | 25k | 8 | serial | 242.7 | 241.7 | 195.4 | 220.5 | 214.5 | -11.6% |
| G2 | w200 | 25k | 8 | 8 threads | 236.6 | 147.0 | 80.8 | 76.3 | 45.9 | -80.6% |
| G2 | w200 | 100k | 8 | serial | 560.6 | 528.6 | 498.9 | 509.7 | 495.4 | -11.6% |
| G2 | w200 | 100k | 8 | 8 threads | 358.1 | 245.1 | 149.4 | 152.3 | 98.2 | -72.6% |
| G3 | w500 | 25k | 5 | serial | 190.9 | 190.2 | 176.4 | 170.1 | 173.8 | -9.0% |
| G3 | w500 | 25k | 5 | 8 threads | 128.1 | 85.0 | 40.3 | 38.8 | 37.6 | -70.6% |
| G3 | w500 | 100k | 5 | serial | 433.9 | 434.2 | 410.2 | 411.9 | 416.6 | -4.0% |
| G3 | w500 | 100k | 5 | 8 threads | 174.9 | 127.7 | 86.2 | 84.0 | 79.5 | -54.6% |
| G4 | w200-1m | 1000k | 5 | 8 threads | 261.0 | 247.6 | 230.9 | 229.6 | 228.9 | -12.3% |

At 1M rows, per-row histogram work dominates and already scaled well, so the
tree time barely moved; the 1M fit gain comes mostly from binning (item 3).

8-thread speedup over serial:

| Case | baseline | item1 | item2 | final |
| --- | ---: | ---: | ---: | ---: |
| 25k x 200, depth 5 | 1.3x | 1.9x | 2.9x | 4.6x |
| 100k x 200, depth 5 | 2.3x | 3.0x | 3.9x | 4.8x |
| 250k x 200, depth 5 | 3.1x | 3.8x | 4.5x | 5.0x |
| 25k x 200, depth 8 | 1.0x | 1.6x | 2.4x | 4.7x |
| 100k x 200, depth 8 | 1.6x | 2.2x | 3.3x | 5.0x |
| 100k x 500, depth 5 | 2.5x | 3.4x | 4.8x | 5.2x |

Compared with XGBoost 1.7.6 `hist` on the same data and settings, 8 threads,
ms per tree. XGBoost times are full iterations; Forust's are tree-only, which
understates Forust's iteration time by 3-10%.

| Case | Forust baseline | Forust final | XGBoost |
| --- | ---: | ---: | ---: |
| 25k x 200, depth 5 | 52.7 | 13.1 | 10.1 |
| 100k x 200, depth 5 | 72.5 | 32.5 | 30.1 |
| 250k x 200, depth 5 | 108.3 | 66.1 | 63.8 |
| 25k x 200, depth 8 | 236.6 | 45.9 | 39.9 |
| 100k x 200, depth 8 | 358.1 | 98.2 | 87.6 |
| 25k x 500, depth 5 | 128.1 | 37.6 | 37.1 |
| 100k x 500, depth 5 | 174.9 | 79.5 | 107.1 |
| 1M x 200, depth 5 | 261.0 | 228.9 | 373.2 |

Serial fraction at 8 threads (item 4 moved training onto a pool thread, so
the main thread now only waits and this metric no longer applies):

| Case | baseline | item1 | item2a | item2b | item2c |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k x 200, depth 5 | 63% | 43% | 47% | 36% | 17% |
| 25k x 200, depth 8 | 81% | 57% | 54% | 35% | 12% |

Binning time per fit (seconds), after item 3:

| Data | Rows | Mode | baseline | item3 | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| w200 | 100k | serial | 1.94 | 1.18 | -39% |
| w200 | 100k | 8 threads | 1.91 | 0.16 | -92% |
| w200 | 250k | serial | 5.52 | 3.11 | -44% |
| w200 | 250k | 8 threads | 5.78 | 0.49 | -92% |
| w500 | 100k | serial | 5.30 | 3.39 | -36% |
| w500 | 100k | 8 threads | 5.14 | 0.41 | -92% |
| w200-1m | 1M | 8 threads | 24.73 | 1.79 | -93% |

End to end, a real 100-iteration `fit` on 1M x 200 with 8 threads went from
52.7 s (baseline) to 27.2 s (item 3) and 26.1 s (final), with bit-identical
evaluation log-loss.

## Step log

### baseline (`4a6d828`)

Analysis tooling, the determinism test, and the golden models. No library
changes. Serial and parallel training already produce identical trees.

Main thread at 100k x 200, depth 5, 8 threads: `evaluate_split` 31%, loops
inlined into the caller (mostly the split scan) 18%, unresolved (likely kernel)
12%, `HistogramMatrix::new` 10%, `free` 9%, histogram subtraction 8%.

### item1: parallel split search (`c111e36`)

`Splitter::best_split` scans features in parallel when `parallel` is set. Ties
go to the earliest feature, as in the serial loop. A new unit test checks
parallel against serial with many exact gain ties.

- Models unchanged; 44 Rust tests pass.
- Features per task: 1 (Rayon's default) beat 8 and 32. At 25k depth 8:
  149.8 / 156.2 / 170.0 ms; at 100k depth 5: 54.4 / 55.9 / 57.0 ms.
- 8 threads: 17-38% faster across the grid, largest for many small nodes
  (depth 8) and wide data. 1M rows: 5% faster, because histogram building
  dominates there.
- Serial: unchanged within noise (control).
- Main thread now: unresolved (likely kernel) 24%, `HistogramMatrix::new` 18%,
  `free` 18%, histogram subtraction 15%, row partitioning 11%.
  `evaluate_split` is gone from the main thread.

### item2a: borrow gradients instead of copying (`763e861`)

`HistogramMatrix::new` borrowed the full gradient/Hessian arrays at the root
(where no reordering is needed) instead of copying them every tree, and
gathers them in parallel for nodes with at least 16,384 rows.

- Models unchanged.
- Serial: 3-10% faster (no per-tree full copy). 8 threads: a further 1-9%.
- Serial fraction about the same (47% / 54%).

### item2b: write histograms in place (`18744aa`)

Each node's histogram is allocated once and split into one slice per column,
and the columns are filled in parallel. Previously every column produced its
own vector and Rayon concatenated them on the calling thread.

- Models unchanged.
- 8 threads: a further 15-33% faster than item2a (25k depth 8: 143.9 to
  99.2 ms). Serial: 3-20% faster (fewer allocations).
- Serial fraction: 36% / 35%. Histogram subtraction became the largest named
  item on the main thread.

### item2c: parallel histogram subtraction (`aefa101`)

`from_parent_child` and `from_parent_two_children` subtract in parallel
(4,096 bins per task) when `parallel` is set. Also applied `rustfmt` to code
added on this branch.

- Models unchanged.
- 8 threads: a further 1-28% faster than item2b (25k x 500: 56.3 to
  40.3 ms; 25k depth 8: 99.2 to 80.8 ms). 1M rows: 230.9 ms (-11.5% vs
  baseline).
- Serial fraction: 17% / 12%. The main thread now mostly runs row
  partitioning (`handle_split_info`) and unresolved (likely kernel) code.
- **Serial regression investigated:** serial runs were 3-8% slower than
  item2b, although this change doesn't touch the serial histogram loop. Item 2b
  and 2c builds timed back to back confirmed it (100k: ~145 vs ~155 ms), and
  `perf` placed the extra time in the histogram fill loop. Its disassembly is
  byte-identical in both builds; only the address differs. In item2b the hot
  31-byte inner loop sits within one 64-byte block; in item2c it starts
  48 bytes in and crosses a boundary. This is a code-placement effect that
  any change can trigger, not a cost of item 2c. Forcing loop alignment is
  a possible later experiment.

### item2d: small-node threshold (dropped)

Build histograms serially when a node's rows x columns is below a threshold.
Tested with a temporary environment variable, 8 threads, tree ms:

| Threshold (rows x columns) | 25k depth 8 | 25k depth 5 | 100k depth 5 |
| ---: | ---: | ---: | ---: |
| 0 (always parallel) | 80.7 | 19.1 | 35.9 |
| 50k | 92.8 | 19.0 | 41.0 |
| 200k | 99.3 | 20.3 | 48.9 |
| 1M | 107.2 | 28.6 | 72.7 |

Every threshold was slower. With 200+ columns even small nodes have enough
per-column work to keep the workers busy, so the change was not committed.

### item3: parallel binning, one sort per column (`86a77b3`)

- **3a:** `percentiles_or_value` sorted a copy of each column to count
  distinct values, then `percentiles` sorted an index array again. Now each
  column is sorted once. With uniform weights it sorts the values directly;
  tied values can't change the percentile walk when every weight is equal.
  Otherwise it does one index sort shared by both steps. The percentile walk
  moved to `utils::percentiles_of_sorted`; the public `percentiles` (used by
  Python) is unchanged.
- **3b:** `bin_matrix` takes a `parallel` argument (API change, approved).
  Column cuts and bin assignment run per column in parallel, which also
  removes a division per value.
- Models unchanged. New tests: the new `percentiles_or_value` matches the old
  algorithm on uniform and weighted, continuous and tied data; serial and
  parallel `bin_matrix` give identical cuts and bins with missing values and
  weights. 46 Rust tests pass.
- Binning: serial (3a only) 34-44% faster; 8 threads 91-93% faster (11-14x).
  1M x 200: 24.7 s to 1.8 s, beating the 5 s target.
- Tree times unchanged within noise.

### item3a': pair sort for weighted data (not adopted)

Sorting (value, weight) pairs instead of an index array would only help
weighted data (uniform weights already use a direct sort), and could change a
cut in its last digit when tied values carry different weights. Timed in a
scratch program on 1M-row columns: continuous 68.4 to 49.0 ms, 50 levels 23.9
to 15.0 ms per column. For a weighted 1M x 200 fit on 8 threads that's about
0.5 s of a 27 s fit (~2%), so it isn't worth the risk.

### item4: thread count control (`fcbc1c0`, `c1baff6`)

- **`num_threads` (`fcbc1c0`):** new optional setting on `GradientBooster`
  (Rust setter, Python constructor argument, `get_params`, save/load).
  When set, `fit`, `predict`, `predict_contributions`,
  `predict_leaf_indices`, and partial dependence run in a dedicated Rayon
  pool of that size. Defaults to `None` (today's behavior). Older saved
  models still load. New Rust and Python tests: identical trees and
  predictions for 1, 2, and 4 threads, and the setting survives save/load.
- **Unexpected finding:** `num_threads=8` was much faster than
  `RAYON_NUM_THREADS=8` (25k depth 8: 0.88 s vs 1.57 s per fit). Inside a pool,
  `fit` runs on a worker thread, so each parallel step is split among workers
  directly. Called from the main thread, every parallel step is a blocking
  hand-off to the pool, which adds up over thousands of small nodes.
- **Fix (`c1baff6`):** parallel `fit` and prediction now always run on a pool
  thread (`rayon::scope`), so the default path gets the same benefit. Serial
  mode is unchanged. The harness was updated to match.
- Models unchanged; 47 Rust and 112 Python tests pass.
- 8 threads, vs item 3: a further 4-40% faster; 25k depth 8: 76.3 to
  45.9 ms, 100k depth 8: 152.3 to 98.2 ms.
- `num_threads=8` now matches `RAYON_NUM_THREADS=8` within noise.
- Building a pool on every `predict` call adds no measurable cost: 1,000 calls
  on 1,000 rows with 100 trees took 2.46 ms per call with `num_threads=8`
  versus 3.1 ms on the default 16-thread pool, so no pool caching is needed.

## Final confirmation

- Full G1-G4 grid (table above), XGBoost rerun, and Python suite (112
  passed).
- 1,000-iteration run at 100k x 200, default 16 threads: 80.8 s to 37.2 s.
  Per-tree time is stable (39.6 ms early, 33 ms steady state) and tree sizes
  match the baseline run. Binning is now 0.14 s of the 37.2 s; trees are 91%.
