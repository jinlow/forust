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

Tree ms per change. Each column includes all earlier changes; "item2" is
items 2a-2c (2d was dropped). Per-step numbers are in the step log.

| Grid | Data | Rows | Depth | Mode | baseline | item1 | item2 | item1 vs baseline | item2 vs baseline |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| G1 | w200 | 25k | 5 | serial | 69.5 | 69.8 | 57.9 | +0.5% | -16.7% |
| G1 | w200 | 25k | 5 | 8 threads | 52.7 | 35.9 | 20.0 | -32.0% | -62.0% |
| G1 | w200 | 100k | 5 | serial | 165.4 | 169.8 | 153.9 | +2.7% | -6.9% |
| G1 | w200 | 100k | 5 | 8 threads | 72.5 | 56.4 | 39.0 | -22.3% | -46.3% |
| G1 | w200 | 250k | 5 | serial | 340.8 | 340.1 | 332.3 | -0.2% | -2.5% |
| G1 | w200 | 250k | 5 | 8 threads | 108.3 | 89.2 | 73.8 | -17.7% | -31.8% |
| G2 | w200 | 25k | 8 | serial | 242.7 | 241.7 | 195.4 | -0.4% | -19.5% |
| G2 | w200 | 25k | 8 | 8 threads | 236.6 | 147.0 | 80.8 | -37.9% | -65.9% |
| G2 | w200 | 100k | 8 | serial | 560.6 | 528.6 | 498.9 | -5.7% | -11.0% |
| G2 | w200 | 100k | 8 | 8 threads | 358.1 | 245.1 | 149.4 | -31.6% | -58.3% |
| G3 | w500 | 25k | 5 | serial | 190.9 | 190.2 | 176.4 | -0.4% | -7.6% |
| G3 | w500 | 25k | 5 | 8 threads | 128.1 | 85.0 | 40.3 | -33.7% | -68.6% |
| G3 | w500 | 100k | 5 | serial | 433.9 | 434.2 | 410.2 | +0.1% | -5.5% |
| G3 | w500 | 100k | 5 | 8 threads | 174.9 | 127.7 | 86.2 | -27.0% | -50.7% |
| G4 | w200-1m | 1000k | 5 | 8 threads | 261.0 | 247.6 | 230.9 | -5.2% | -11.5% |

8-thread speedup over serial:

| Case | baseline | item1 | item2 |
| --- | ---: | ---: | ---: |
| 25k x 200, depth 5 | 1.3x | 1.9x | 2.9x |
| 100k x 200, depth 5 | 2.3x | 3.0x | 3.9x |
| 250k x 200, depth 5 | 3.1x | 3.8x | 4.5x |
| 25k x 200, depth 8 | 1.0x | 1.6x | 2.4x |
| 100k x 200, depth 8 | 1.6x | 2.2x | 3.3x |
| 100k x 500, depth 5 | 2.5x | 3.4x | 4.8x |

Serial fraction at 8 threads:

| Case | baseline | item1 | item2a | item2b | item2c |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k x 200, depth 5 | 63% | 43% | 47% | 36% | 17% |
| 25k x 200, depth 8 | 81% | 57% | 54% | 35% | 12% |

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
