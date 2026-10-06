# Forust Performance Recommendations

Date: 2026-10-05, updated 2026-10-06

Status: the four recommended changes are implemented on branch
`feat/optimize` (local commits, not pushed). Per-change measurements are in
`results.md`. The original analysis follows the final recommendations below.

Target workload: binary classification, 200+ columns, up to 1,000+ iterations,
and one evaluation set scored on every iteration.

## Final results

All changes leave trained models byte-identical (checked after every commit),
and the Python suite passes (112 tests, including the XGBoost parity checks).

| Workload | Before | After |
| --- | ---: | ---: |
| 100-iteration `fit`, 1M x 200, 8 threads | 52.7 s | 26.1 s (2.0x) |
| 1,000 iterations, 100k x 200, default threads | 80.8 s | 37.2 s (2.2x) |
| Tree time, 8 threads, 25k-250k rows | | 39-81% less |
| Binning, 8 threads | | 91-93% less (1M x 200: 24.7 s to 1.8 s) |
| 8-thread speedup over serial | 1.0-3.1x | 4.6-5.2x |

Versus XGBoost `hist` at 8 threads, Forust went from 1.6-6x slower to:
- within 1-30% on small and deep cases;
- faster on 100k x 500 (79.5 vs 107.1 ms) and 1M x 200 at depth 5
  (228.9 vs 373.2 ms);
- but 39% slower on 1M x 200 at depth 8 (633.7 vs 454.6 ms).

A matched 1M x 200, depth 6, 20-iteration `LossGuide`/`lossguide` fit with
one evaluation set was effectively tied: Forust 8.07 s median versus XGBoost
8.15 s. Forust's corrected tree-only phase was 310.0 ms versus XGBoost's
407.3 ms, but
the full-fit comparison is the meaningful one because it includes predictions
and evaluation in both libraries.

| Change | Commit | Main effect |
| --- | --- | --- |
| Parallel split search | `c111e36` | 17-38% faster at 8 threads |
| Borrow gradients instead of copying | `763e861` | 3-10% faster |
| Write histograms in place | `18744aa` | 15-33% faster at 8 threads |
| Parallel histogram subtraction | `aefa101` | Up to 28% faster at 8 threads |
| Parallel binning, one sort per column | `86a77b3` | Binning 11-14x faster |
| `num_threads` setting | `fcbc1c0` | Thread-count control |
| Run parallel work on a pool thread | `c1baff6` | 4-40% faster at 8 threads |

Tried and dropped: building small nodes serially (slower at every threshold),
and a faster pair sort for weighted binning (~2% gain, small risk of
last-digit cut changes).

## Recommendations

1. **Merge the branch.** Every change is model-neutral and measured. API
   changes to review:
   - `bin_matrix` takes a `parallel` argument (approved).
   - The `Splitter` trait now requires `Sync`, and `best_split` takes a
     `parallel` argument; this only affects code implementing its own splitter.
   - New optional `num_threads` setting in Rust and Python. Older saved models
     still load.

   The branch also includes the analysis tooling (`scripts/`,
   `examples/perf_phases.rs`), the Python packaging fix, and your `polars`
   removal.
2. **Set `num_threads` to the number of physical cores.** On this VM
   (8 cores, 16 vCPU), the default of 16 threads is 2-18% slower than 8 at
   depth 5 and 25-66% slower at depth 8. Before changing the default, measure
   on your production machines; detecting physical cores would need a new
   dependency or a heuristic.
3. **Next optimization candidates, if you want more.**
   - **Row partitioning, first.** After each split, `pivot_on_split` reorders
     the node's rows on one thread, so every tree level makes a serial pass
     over all rows. It was the top remaining serial item after item 2c, and
     is the likely reason Forust is 39% slower than XGBoost at 1M x 200,
     depth 8. Profile that case first to confirm.

   In the final profile (25k x 200, depth 8, 8 threads), the remaining time
   is mostly:
   - the split search itself, ~40% (`evaluate_split` and its loop): it
     computes weights and gains for every bin of every feature at every node;
     skipping bins that fail `min_leaf_weight` before computing gains would
     cut work without changing results;
   - histogram accumulation, ~17%: already parallel; fusing it with the split
     scan per column (one pass while the histogram is in cache) is the
     follow-up from the plan;
   - Rayon scheduling, ~15%: tasks are small with many tiny nodes at depth 8.

   This is where the remaining gap to XGBoost on deep, small-row trees comes
   from. The gains here are likely smaller than what's already done.
4. **Keep the tooling as a regression check.** `scripts/perf_golden.py` proves
   models are unchanged; `scripts/perf_grid.py` and `scripts/perf_compare.py`
   reproduce the timing tables. The serial-versus-parallel determinism test
   already runs in `cargo test`. The toy `wide-200-column` Criterion group was
   removed; `examples/perf_phases.rs` on generated data replaces it.
5. **Read small serial differences as noise.** Serial timings moved by up to
   ±6% between builds because a hot loop's position in memory changed, even
   with byte-identical instructions (see item 2c in `results.md`).

Housekeeping: `perf.data` in the repository root is the raw capture from your
flamegraph run and can be deleted. The benchmark datasets in `/tmp/forust-perf`
may not survive the VM being deallocated, but `scripts/make_perf_data.py`
regenerates them exactly.

---

# Original analysis (2026-10-05)

This is the analysis that led to the plan, before any library change.

## Summary

1. **Tree building is ~93% of training time.** In a 1,000-iteration run
   (100k rows x 200 columns), the other phases are small:

   | Phase | Share |
   | --- | ---: |
   | Binning | 2.4% |
   | Training-set predictions | 1.7% |
   | Evaluation predictions + metric | 1.6% |
   | Gradients/Hessians | 1.0% |

   Per-tree cost is stable across the run, falling from 84 ms early to about
   75 ms.
2. **Single-threaded, Forust is about as fast as XGBoost `hist`.** On the same
   data and settings, Forust's serial tree time is 0.64-1.5x XGBoost's.
3. **The gap is parallel scaling.** At 8 threads:
   - XGBoost speeds up 3.6-4.7x.
   - Forust speeds up 1.0-3.0x up to 250k rows, and 4.5x at 1M rows.
4. **Why Forust scales poorly:** only histogram accumulation runs in parallel.
   The rest of the per-node work stays serial on the main thread:
   - the split search
   - histogram subtraction
   - gathering gradients/Hessians
   - concatenating per-column histograms
   - row partitioning

   This serial work is about half of the tree-building time at 100k x 200 with
   8 threads. It grows with columns x bins x nodes, so deep trees suffer most:
   at depth 8, Forust gets 1.0-1.5x versus XGBoost's 3.6-3.8x.
5. **Binning is serial and slow at large row counts.** At 1M x 200 it takes
   24 s, versus 1.5 s for XGBoost's data preparation (`DMatrix`) on 8 threads.
   That is about 8% of a 1,000-iteration fit but almost half of a
   100-iteration fit.
6. **Thread count:** 8 threads (one per physical core) is best for both
   libraries. 16 threads (with hyperthreading) is 4-19% slower for Forust and
   55-380% slower for XGBoost.
7. **Several earlier findings were artifacts of the toy benchmark and are
   withdrawn** (details below): "parallel is slower than serial", "binning is
   the top priority", and "evaluation costs 61%".

## Method

- **Data** (`scripts/make_perf_data.py`): scikit-learn `make_classification`
  with 20% informative and 20% redundant features, 3 clusters per class, 5%
  label noise, and about 17% positives. Then, to mimic real data:
  - 25% of columns reduced to 2-50 levels
  - 15% zero-inflated "amount" columns
  - 3% near-constant columns
  - missing values in 30% of columns, at 1-30% rates

  56 of 200 columns end up with 256 or fewer distinct values. Forust reaches
  about 0.90 validation AUC (100k rows, 200 iterations).
- **Datasets** (local disk, `/tmp/forust-perf/`; the script is seeded, so they
  can be regenerated):

  | Name | Training rows | Evaluation rows | Columns |
  | --- | ---: | ---: | ---: |
  | `w200` | 250k | 62.5k | 200 |
  | `w500` | 100k | 25k | 500 |
  | `w200-1m` | 1M | 250k | 200 |

  Rows are shuffled, so the first N rows are a random sample.
- **Harness** (`examples/perf_phases.rs`): `--mode phases` replays the
  `GradientBooster::fit` loop with a timer on each phase; `--mode fit` times
  the real `fit`. Both modes give identical evaluation log-loss and total
  times within 1%.
- **Sweeps** (`scripts/perf_sweep.py`): steady-state ms per tree (skipping the
  first 10 iterations), serial versus `RAYON_NUM_THREADS` values.
- **XGBoost** (`scripts/perf_xgboost.py`): xgboost 1.7.6 `hist` with the same
  depth, bins, learning rate, `lambda`, and `min_child_weight`, and one
  evaluation set. XGBoost times are full iterations (tree, predictions, and
  evaluation); Forust's are tree-only, which understates Forust's iteration
  time by 3-10%.
- **Settings** unless noted: learning rate 0.1, depth 5, 256 bins, evaluation
  rows = training rows / 4.
- **Host:** 16 vCPU (8 physical cores with hyperthreading), Xeon 8272CL,
  62 GB RAM, Azure VM. Repeat runs vary by about 5%.
- **Profiles:** `perf` at 999-1999 Hz on a release build with debug info,
  split by thread ID.

## Results

### Where the time goes over 1,000 iterations

100k x 200, depth 5, default 16 Rayon threads, 80.8 s total:

| Phase | Time | Share |
| --- | ---: | ---: |
| Tree building | 75.4 s | 93.3% |
| Binning | 2.0 s | 2.4% |
| Training-set predictions | 1.3 s | 1.7% |
| Evaluation predictions + metric | 1.3 s | 1.6% |
| Gradients/Hessians | 0.8 s | 1.0% |

| Iterations | Mean ms per tree | Mean nodes per tree |
| --- | ---: | ---: |
| 0-10 | 84.5 | 62.8 |
| 10-100 | 79.1 | 62.3 |
| 100-300 | 75.0 | 59.5 |
| 300-600 | 75.4 | 59.5 |
| 600-1000 | 74.6 | 58.8 |

Trees stay nearly full (63 nodes is the maximum at depth 5), so 50-60
steady-state iterations are a valid stand-in for a full 1,000-iteration fit.

### Thread scaling

Milliseconds per tree, 200 columns, depth 5:

| Rows | Forust serial | 2 threads | 4 threads | 8 threads | 16 threads | XGBoost serial | XGBoost 8 threads | XGBoost 16 threads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25k | 63.8 | 55.1 | 48.1 | 52.6 | 60.6 | 49.1 | 12.9 | 62.5 |
| 50k | 92.2 | 72.4 | 59.5 | 56.7 | 67.3 | - | - | - |
| 100k | 152.0 | 98.9 | 74.7 | 68.9 | 77.3 | 123.6 | 27.1 | 42.0 |
| 250k | 305.4 | 186.2 | 124.0 | 100.5 | 104.2 | 268.8 | 56.6 | 95.3 |
| 1M | 1146.4 | - | - | 257.2 | - | 1787.3 | 404.4 | - |

At 1M rows, a full Forust iteration on 8 threads is about 285 ms (259 ms
tree, plus about 9 ms each for training predictions, evaluation, and
gradients), versus 404 ms for XGBoost.

### Depth and width

Milliseconds per tree:

| Case | Forust serial | Forust 8 threads | Speedup | XGBoost serial | XGBoost 8 threads | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25k x 200, depth 8 | 230.1 | 227.5 | 1.01x | 178.8 | 50.0 | 3.6x |
| 100k x 200, depth 8 | 514.7 | 348.6 | 1.48x | 341.2 | 90.4 | 3.8x |
| 25k x 500, depth 5 | 181.4 | 123.9 | 1.46x | 177.2 | 40.4 | 4.4x |
| 100k x 500, depth 5 | 404.6 | 165.5 | 2.44x | 495.0 | 105.7 | 4.7x |

### Fixed versus per-row cost

Reducing bins from 256 to 64 cut about 30 ms per tree at both 25k rows
(63.8 to 31.3 ms) and 250k rows (305.4 to 277.4 ms), so part of each tree's
cost depends on bins rather than rows. Fitting the 25k-250k results:

| Mode | Fixed cost per tree | Per-row cost |
| --- | ---: | ---: |
| Serial | ~40 ms | ~1.07 ms per 1k rows |
| 8 threads | ~47 ms | ~0.21 ms per 1k rows |

The per-row part (histogram accumulation) already speeds up about 5x at
8 threads. The fixed part (proportional to columns x bins x nodes) does not
speed up at all. At 100k rows with 8 threads it is about 68% of each tree's
time. With 64 bins, the 8-thread speedup at 250k rows rises from 3.0x to
3.8x.

### Profiles

**Serial, 100k x 200, 60 iterations** (share of all samples):

| Function | Share |
| --- | ---: |
| Histogram accumulation (`create_feature_histogram`) | 52% |
| Binning (`bin_matrix` + percentile sorts) | ~14% |
| `evaluate_split` | 6.6% |
| Concatenating per-column histograms (`flat_map`) | 3.5% |
| Tree prediction | 3.1% |
| Gradient/Hessian gather (`HistogramMatrix::new`) | 1.5% |
| Row partitioning (`handle_split_info`) | 1.4% |
| Histogram subtraction (`from_parent_child`) | 1.3% |

**8 threads, 100k x 200, 60 iterations** (6.45 s total, 4.3 s tree phase):
the main thread logged about 5.4 s of samples and each of the 8 Rayon workers
about 1.3 s. The main thread only runs serial code, because it blocks without
sampling while parallel tasks run.

| Serial work on the main thread | Time |
| --- | ---: |
| Binning and its sorts | ~1.9 s (outside the tree phase) |
| Split search (`evaluate_split` + loops inlined into the caller) | ~1.36 s |
| Allocator, memory copies, kernel (mostly loading the input) | ~0.75 s |
| Gradient/Hessian gather | ~0.25 s |
| Histogram subtraction | ~0.22 s |
| Row partitioning | ~0.15 s |

Roughly half of the 4.3 s tree phase is serial. `strace` logged 259,714
`sched_yield` calls (Rayon workers spinning idle while the main thread works)
but only 50 `mmap` calls, so histogram allocations are not going to the
kernel.

### Withdrawn earlier findings

| Earlier finding | Why it was wrong |
| --- | --- |
| Parallel training is slower than serial | Measured on 20k rows with an unlearnable target. On realistic data, 2-8 threads always win. |
| Binning is the top priority | Measured with 10 iterations and a profile that ran Criterion in test mode. At 1,000 iterations, binning is 2-8%. |
| One evaluation set adds 61% | Measured on 5 columns. Prediction cost doesn't grow with columns; at 200 columns it is 1.6%. |
| Flamegraphs at `/tmp/forust-*.svg` | Captured in Criterion test mode plus setup code; don't use them. |

## Recommendations (ranked)

None of these change what the algorithm computes, so models should be
identical before and after (tie-breaking is noted where it matters).

### 1. Parallelize the split search across features

`Splitter::best_split` (`src/splitter.rs`) scans every feature's histogram
serially. It is the largest serial block: about 1.36 s of the 4.3 s tree phase
above, or about 23 ms of the ~47 ms fixed cost per tree. Each feature's scan is
independent, so the scans can run in parallel and then be reduced by gain,
breaking ties by the lowest feature index. That reproduces the current serial
choice exactly: today the first feature with the strictly highest gain wins.

- Expected gain: at 100k x 200 on 8 threads, roughly 69 ms to 50 ms per tree
  (estimate). Larger at depth 8 and with small nodes.
- Risk: low. No floating-point order changes.
- Follow-up option: fuse histogram building and the split scan into one
  parallel task per column, so each histogram is scanned while still in cache.

### 2. Move the remaining per-node work into the parallel section

These run serially on every node:

- `HistogramMatrix::new` gathers gradients/Hessians for the node's rows, and
  copies the full arrays when no gather is needed.
- `from_parent_child` / `from_parent_two_children` subtract histograms for the
  sibling node.
- The parallel `flat_map` allocates one vector per column and then
  concatenates them.

Options: subtract in parallel per column (or inside the per-column task),
write each column's histogram straight into a preallocated buffer, and
parallelize the gather.

- Risk: low. Subtraction is element-wise, so results are unchanged.

### 3. Parallelize binning and sort each column once

`bin_matrix` processes columns one at a time. `percentiles_or_value` sorts and
deduplicates each column, then `percentiles` sorts an index array a second
time. Binning columns in parallel and sorting once should cut 24 s at 1M x 200
to a few seconds (estimate, not measured).

- Matters most for large data with fewer iterations: almost half of a
  100-iteration fit at 1M rows, about 8% of a 1,000-iteration fit.
- Risk: low. Each column is computed the same way, so bin boundaries are
  unchanged.

### 4. Thread count control

16 hyperthreaded threads were 4-19% slower than 8 threads for Forust (and
55-380% slower for XGBoost). Options: add a `num_threads` parameter (like
XGBoost's `nthread`) using a local Rayon pool, or document
`RAYON_NUM_THREADS`. Low priority, since users can already set the environment
variable.

### Not recommended now

- **Evaluation changes:** 1.6% of time.
- **Row-parallel histograms (XGBoost-style):** the per-row part already scales
  about 5x at 8 threads; it is not the bottleneck.
- **Lowering the default `nbins`:** faster, but changes models.

### Expected ceiling

If the fixed per-tree cost scaled like the per-row part (~5x at 8 threads),
100k x 200 at depth 5 would drop from about 69 ms per tree to about 30 ms,
comparable to XGBoost's 27 ms. This is a rough estimate to verify after items
1 and 2.

## Validation plan for implementation

For each change:

1. Run the Python suite (111 tests, including the XGBoost parity checks).
2. Check the model is unchanged: `perf_phases --mode fit` should report an
   identical evaluation log-loss before and after.
3. Re-measure with the same grid: `w200` at 25k/100k/250k rows, serial and
   8 threads, depth 5 and 8, and `w500`; then confirm on `w200-1m`.

The Criterion `wide-200-column` group in `benches/forust_benchmarks.rs` still
uses the old unlearnable data. It should be switched to the generated data or
removed; `perf_phases` is the better tool for this analysis.

## Reproducing

```sh
PY=/home/azureuser/localfiles/uv-environments/forust/bin/python
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w200 --rows 250000 --eval-rows 62500 --cols 200
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w500 --rows 100000 --eval-rows 25000 --cols 500
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w200-1m --rows 1000000 --eval-rows 250000 --cols 200

cargo build --release --example perf_phases
python3 scripts/perf_sweep.py --data /tmp/forust-perf/w200 \
  --rows 25000 100000 250000 --threads 8 --out /tmp/forust-perf/results/sweep.jsonl
python3 scripts/perf_sweep.py --data /tmp/forust-perf/w200 \
  --rows 25000 100000 --threads 8 --iterations 30 \
  --out /tmp/forust-perf/results/sweep_depth8.jsonl -- --max-depth 8
$PY scripts/perf_xgboost.py --data /tmp/forust-perf/w200 --rows 100000 --threads 1 8

# Profile (release build with debug info)
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --example perf_phases
RAYON_NUM_THREADS=8 perf record -F 999 -o /tmp/forust-perf/par.data -- \
  target/release/examples/perf_phases --rows 100000 --iterations 60
perf report -i /tmp/forust-perf/par.data --stdio -s pid
```

Raw results are in `/tmp/forust-perf/results/`. `/tmp` is local disk and may
not survive the VM being deallocated; everything can be regenerated with the
commands above.

## Working tree

- New: `scripts/make_perf_data.py`, `scripts/perf_sweep.py`,
  `scripts/perf_xgboost.py`, `examples/perf_phases.rs`.
- Earlier changes: the wide-data Criterion group in
  `benches/forust_benchmarks.rs`; the Python packaging fix
  (`py-forust/README.md`, SPDX license, `.gitignore`); `polars` and the Titanic
  example removed (by you).
- `perf.data` in the repository root is the raw capture from your flamegraph
  run and can be deleted.
- `cargo test --all-targets`: 42 passed with the new example. The Python suite
  last passed (111 tests) before these scripts were added; no library code has
  changed since.