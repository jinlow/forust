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

| Grid | What it isolates |
| --- | --- |
| G1 | 200 columns, depth 5, 25k-250k rows: fixed vs per-row cost |
| G2 | 200 columns, depth 8: many small nodes |
| G3 | 500 columns, depth 5: wide data |
| G4 | 1M rows x 200 columns, depth 5 |

## Summary

Tree ms per change. Each column includes all earlier changes.

| Grid | Data | Rows | Depth | Mode | baseline | item1 | item1 vs baseline |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| G1 | w200 | 25k | 5 | serial | 69.5 | 69.8 | +0.5% |
| G1 | w200 | 25k | 5 | 8 threads | 52.7 | 35.9 | -32.0% |
| G1 | w200 | 100k | 5 | serial | 165.4 | 169.8 | +2.7% |
| G1 | w200 | 100k | 5 | 8 threads | 72.5 | 56.4 | -22.3% |
| G1 | w200 | 250k | 5 | serial | 340.8 | 340.1 | -0.2% |
| G1 | w200 | 250k | 5 | 8 threads | 108.3 | 89.2 | -17.7% |
| G2 | w200 | 25k | 8 | serial | 242.7 | 241.7 | -0.4% |
| G2 | w200 | 25k | 8 | 8 threads | 236.6 | 147.0 | -37.9% |
| G2 | w200 | 100k | 8 | serial | 560.6 | 528.6 | -5.7% |
| G2 | w200 | 100k | 8 | 8 threads | 358.1 | 245.1 | -31.6% |
| G3 | w500 | 25k | 5 | serial | 190.9 | 190.2 | -0.4% |
| G3 | w500 | 25k | 5 | 8 threads | 128.1 | 85.0 | -33.7% |
| G3 | w500 | 100k | 5 | serial | 433.9 | 434.2 | +0.1% |
| G3 | w500 | 100k | 5 | 8 threads | 174.9 | 127.7 | -27.0% |
| G4 | w200-1m | 1000k | 5 | 8 threads | 261.0 | 247.6 | -5.2% |

Serial fraction at 8 threads:

| Case | baseline | item1 |
| --- | ---: | ---: |
| 100k x 200, depth 5 | 63% | 43% |
| 25k x 200, depth 8 | 81% | 57% |

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
