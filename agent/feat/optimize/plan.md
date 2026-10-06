# Forust Performance Implementation Plan

Date: 2026-10-05

This plan implements the four recommendations in `recommendations.md`. It
also defines the speed and correctness checks for each one.

## Ground rules

- **One change per commit.** Each item, and each sub-item of item 2, is a
  separate commit. Every commit gets its own before/after measurement, so a
  regression can be traced to one change.
- **Models must be bit-identical.** None of these changes alter what the
  algorithm computes, so the saved model JSON must be byte-for-byte the same
  before and after, for both serial and parallel training. This is stricter
  than the Python suite's XGBoost tolerance checks, and catches subtle
  ordering bugs.
- **`parallel=false` stays fully single-threaded.** Every new parallel path is
  gated on the existing `parallel` flag.
- **Every commit passes:** `cargo test --all-targets` and the Python suite
  (111 tests, including the XGBoost parity checks).

## Step 0: Baseline and tooling (no library changes)

1. **Harness flags** (`examples/perf_phases.rs`):
   - `--save-model PATH` writes the trained booster with `save_booster`.
   - `--missing-branch true` trains with `create_missing_branch`, so the second
     splitter (`MissingBranchSplitter`) is covered.
2. **Golden models:** with the current code, save models for these
   configurations to `/tmp/forust-perf/golden/`:

   | Data | Rows | Depth | Iterations | Mode |
   | --- | ---: | ---: | ---: | --- |
   | `w200` | 100k | 5 | 50 | serial and 8 threads |
   | `w200` | 25k | 8 | 20 | 8 threads |
   | `w500` | 25k | 5 | 20 | 8 threads |
   | `w200` | 25k | 5 | 20 | 8 threads, missing branch |

   After each change, retrain and compare with `cmp`. First, confirm that
   today's serial and parallel models already match each other. If they don't,
   find out why before changing anything.
3. **Determinism test** (Rust, in `src/gradientbooster.rs` tests): train on the
   Titanic resource data with `parallel` true and false, for both splitters,
   and assert `json_dump()` is identical. This runs in CI and protects all
   four items.
4. **Sweep tooling:**
   - `scripts/perf_sweep.py`: add `--repeats 3` (report the median) and
     `--label` to tag result rows, such as `baseline` or `item1`.
   - New `scripts/perf_compare.py`: prints a before/after table by
     configuration from two labels.
5. **Baseline run of the core grid** (median of 3 repeats, about 8 minutes):

   | Grid | Data | Rows | Depth | Iterations | Modes | What it isolates |
   | --- | --- | --- | ---: | ---: | --- | --- |
   | G1 | `w200` | 25k, 100k, 250k | 5 | 60 | serial, 8 threads | Fixed vs per-row cost |
   | G2 | `w200` | 25k, 100k | 8 | 30 | serial, 8 threads | Many small nodes |
   | G3 | `w500` | 25k, 100k | 5 | 30 | serial, 8 threads | Wide data |
   | G4 | `w200-1m` | 1M | 5 | 20 | 8 threads | Final confirmation only |

   Also record the **main-thread share** of the tree phase from a `perf`
   profile at 100k x 200, depth 5, 8 threads (`perf report -s pid`). The
   baseline is about 50%. This is the most direct measure of serial work
   remaining.

## Item 1: Parallelize the split search across features

**Change** (`src/splitter.rs`):

- Add a `parallel` argument to `Splitter::best_split`; `split_node` passes its
  existing `parallel` value.
- Parallel branch: scan features with `par_iter`, keep only splits with gain
  > 0, and reduce to the highest gain, breaking ties by the lowest feature
  position. The serial branch is unchanged.
- Add `Sync` as a supertrait of `Splitter`. Both splitters only hold
  thread-safe data, so this needs no other changes.
- Batch features per task (`with_min_len`). Start at 8 and tune between 1, 8,
  and 32 using G2 at 25k rows, which is most sensitive to scheduling
  overhead.

**Why the model can't change:** each feature's best split is computed by the
same function as today. Today's loop keeps the first feature with a strictly
higher gain. The reduction reproduces that exactly: highest gain wins, and on
a tie the lowest feature position wins.

**Correctness:** a new unit test builds random histograms, including forced
gain ties between features, and checks that parallel and serial
`best_split` pick the same feature, bin, and gain. Then golden models and the
determinism test.

**Speed testing:**

| Check | Expected result |
| --- | --- |
| G1-G3, 8 threads | Fixed per-tree cost falls; 25k rows and depth 8 improve most |
| G1-G3, serial | Unchanged within noise (control) |
| Per-row cost (slope of G1) | Unchanged |
| Main-thread profile | `evaluate_split` moves to the worker threads; main-thread share falls from ~50% toward ~30% |

Targets (estimates; record actuals): 100k x 200 depth 5 on 8 threads from
69 ms to 55 ms or less per tree; 100k depth 8 from 349 ms to 260 ms or less.

## Item 2: Move the remaining per-node work into the parallel section

Four sub-items, each a separate commit (`src/histogram.rs`, `src/splitter.rs`):

- **2a. Stop copying gradients/Hessians when they don't need reordering.**
  `HistogramMatrix::new` copies the full arrays with `to_vec()` when
  `sort == false`; borrow them instead (`Cow<[f32]>`). When a reorder is
  needed, gather in parallel, preserving order.
- **2b. Write histograms in place.** Allocate the node's full histogram once.
  Split it into one slice per column and fill the slices in parallel. This
  replaces the parallel `flat_map`, which allocates a vector per column and
  then concatenates. Each column keeps the same f64 accumulation and f32
  conversion, so bins are identical.
- **2c. Parallel histogram subtraction.** `from_parent_child` and
  `from_parent_two_children` subtract element by element. Run them in
  parallel in large chunks when `parallel` is set; this needs a new argument
  at 7 call sites across both splitters.
- **2d. Small-node threshold (optional).** When a node's rows x columns is
  below a threshold, build its histogram serially even in parallel mode,
  because scheduling costs more than the work. Tune with G2 at 25k rows,
  trying thresholds of 0, 50k, 200k, and 1M elements.

**Not now:** row partitioning (`pivot_on_split`, about 3.5% of serial time).
Revisit only if it becomes the top item in the main-thread profile.

**Correctness:** a new unit test checks that serial and parallel
`HistogramMatrix::new` and subtraction give identical bins. Then golden
models and the determinism test.

**Speed testing:**

| Check | Expected result |
| --- | --- |
| G1-G3 after each of 2a-2d | Each step improves or is neutral; any step that regresses is dropped |
| Main-thread profile | Gather, subtraction, `memmove`, and `free` leave the main thread |
| Peak memory (`/usr/bin/time -v`, 100k x 200 depth 8) | No increase; 2b should reduce allocations |

Targets: 8-thread speedup at 100k x 200 depth 5 of 3x or more (from 2.2x);
depth 8 of 2.5x or more (from 1.5x); no regression versus serial at 25k rows.

## Item 3: Parallelize binning and sort each column once

**Change** (`src/binning.rs`):

- **3a. One sort per column.** Today `percentiles_or_value` sorts and
  deduplicates a copy of the column, then `percentiles` sorts an index array.
  Instead, sort the index array once, with the same comparator as today. Count
  distinct values while walking it. If there are at most `nbins + 1`, return
  them; otherwise run the existing percentile walk on the same sorted index.
  Same sort on the same input, so cuts are identical. The only possible
  difference is which of `-0.0` or `0.0` is kept, and the two compare equal,
  so bins don't change.
- **3a' (optional).** Sort (value, weight) pairs directly instead of an index
  array, which is more cache-friendly and faster. Tied values may be added in
  a different order, so with non-uniform weights a cut could differ in the
  last digit in rare cases. Measure it, and only adopt it with your approval
  and if golden models and weighted tests are unchanged.
- **3b. Parallel per column.** Compute each column's cuts in parallel
  (`into_par_iter().map().collect()` keeps column order), then assemble the
  cut matrix serially. Assign bins in parallel per column, which also removes
  the per-element `i / rows` division.
- **API:** `bin_matrix` is public. Option A: add a `parallel: bool` argument
  (breaks direct crate users; about 15 internal call sites, mostly tests and
  benches). Option B: keep the signature and add a parallel variant.
  `GradientBooster` passes `self.parallel`.

**Correctness:** the existing `test_bin_data`, plus a new test that serial and
parallel cuts and binned data are identical. Then golden models and the
weighted Python tests.

**Speed testing:**

| Check | Expected result |
| --- | --- |
| Binning time from `perf_phases` at 100k, 250k, 1M; serial and 8 threads | 3a: serial faster by roughly a third; 3b: 5-7x faster at 8 threads |
| 1M x 200, 100-iteration end-to-end fit | Total time falls by roughly the binning savings |

Target: 1M x 200 binning from 24 s to 5 s or less on 8 threads.

Also replace the toy `wide-200-column` Criterion group in
`benches/forust_benchmarks.rs`. It can load the generated data when an
environment variable points to it (skipping otherwise), or be removed in
favor of `perf_phases`.

## Item 4: Thread count control

**Change:**

- Add `num_threads: Option<usize>` to `GradientBooster`. Use a serde default
  of `None` so older saved models still load, and add a `set_num_threads`
  setter.
- When it is set and `parallel` is true, build a local Rayon thread pool and
  run `fit`, `predict`, `predict_contributions`, and partial dependence inside
  it (`pool.install`). `None` keeps today's behavior (the global pool).
- Python: add `num_threads: int | None = None` to `GradientBooster.__init__`
  (`py-forust/forust/__init__.py`), the pyo3 constructor and `get_params`
  (`py-forust/src/lib.rs`), and the docstrings.

**Correctness:** a Rust test that `num_threads` of 1, 2, and 4 give identical
models. A Python test that the parameter survives `get_params` and
save/load, and that predictions match.

**Speed testing:**

| Check | Expected result |
| --- | --- |
| G1, G2 with `num_threads` of 1, 2, 4, 8, 16 | Matches the same `RAYON_NUM_THREADS` runs within noise, confirming the wiring |
| `predict` on 1k rows x 200 columns, 1,000 calls, `num_threads=8` vs unset | Pool creation adds under ~5% per call; otherwise cache the pool on the booster |

## Final confirmation

After all four items:

- Run the full G1-G4 grid and the XGBoost comparison with the same scripts.
- Repeat the 1,000-iteration drift run on 100k x 200.
- Run the Python suite.
- Update `recommendations.md` with a before/after table.

## Decisions needed from you

1. **Bit-identical models** as the acceptance bar for items 1-3 (recommended)?
    Answer: Yes
2. **`bin_matrix` API:** add a `parallel` argument (Option A) or a separate
   parallel function (Option B)?
   Answer: No one uses that function API breaking is OK, add a prallel argument.
3. **`num_threads`:** is the name OK, and should the default stay `None`
   (today's behavior)? Defaulting to physical cores would need a new
   dependency or a heuristic, and should first be measured on your production
   machines.
   answer: I think its fine.
4. **3a':** try the faster pair sort, accepting a small risk of last-digit cut
   differences with non-uniform weights, or skip it?
   ansewr: We could test 3a last

## Item 5: Parallel row partitioning (not started)

Added after items 1-4 were done. This section is a self-contained hand-off:
items 1-4 are implemented and measured (see `results.md`), and the ground
rules above still apply. Byte-identical trees are required.

### Why

At 1M rows x 200 columns, depth 8, 8 threads, Forust takes 633.7 ms per tree
versus 454.6 ms for XGBoost (39% slower). From depth 5 to 8, Forust's tree
time grows 177% at 1M rows versus 22% for XGBoost. After every split, the
node's rows are reordered on one thread, so each tree level makes a serial pass
over all rows. It was the largest named serial item in the main-thread profile
after item 2c (`Splitter::handle_split_info`, which inlines the partition).

**First, confirm this with a profile** of 1M x 200, depth 8, 8 threads. If
partitioning isn't a large share there, stop and re-plan:

```sh
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --example perf_phases
RAYON_NUM_THREADS=8 perf record -F 999 --delay 25000 -o /tmp/forust-perf/rows.data -- \
  target/release/examples/perf_phases --data /tmp/forust-perf/w200-1m \
  --rows 1000000 --max-depth 8 --iterations 10
perf report -i /tmp/forust-perf/rows.data --stdio --no-children -s symbol --percent-limit 1
```

The delay skips data loading and binning (roughly 20-25 s at 1M rows; adjust if
the profile shows `read_f64` or `bin_matrix`). Look for `pivot_on_split`,
`pivot_on_split_exclude_missing`, or `handle_split_info`.

### Where the code is

- `src/utils.rs`: `pivot_on_split` (two-way split, used by
  `MissingImputerSplitter`) and `pivot_on_split_exclude_missing` (three-way:
  missing, left, right; used by `MissingBranchSplitter`).
- `src/splitter.rs`: both `handle_split_info` implementations call these on
  `&mut index[node.start_idx..node.stop_idx]`, then build the smaller child's
  histogram from its slice of `index`. Both receive the `parallel` flag.
- Existing tests: `test_pivot` and `test_pivot_missing` in `src/utils.rs`.

### The order requirement

Each child's histogram is built by summing gradients in the order of its rows
in `index` (f64 sums, then rounded to f32). Changing that order can change the
last bits of the sums, then split gains and tie-breaks, so trees stop being
byte-identical. A parallel version must produce **exactly the same `index`
order** as today's functions, not just the same set of rows on each side.

`pivot_on_split` is a Hoare-style partition. Its output order has a closed
form. A Python port of the loop matched this closed form on 200,000 random
inputs (lengths 1-40, random bins, missing rows, both `missing_right`
settings); still confirm it with Rust tests:
- `low` scans forward and stops at each row that belongs on the right;
  `high` scans backward and stops at each row that belongs on the left; the
  two are swapped.
- So the k-th right-side row from the front is swapped with the k-th left-side
  row from the back, for k = 1..m, where m is the number of misplaced pairs.
- The result is computable without the sequential loop:
  1. Let L be the number of rows that go left. The returned split index is L,
     **except** when every row goes left (L = n), where it returns n - 1.
     That case never happens in training (both sides of a split have rows),
     but match it anyway.
  2. Misplaced right rows are the right-side rows in positions `[0, L)`;
     misplaced left rows are the left-side rows in positions `[L, n)`. There
     are equally many, m.
  3. The k-th misplaced right row (counting forward from the start) swaps
     places with the k-th misplaced left row counting backward from the end.
     Every other row stays where it is.

This is parallelizable:
1. Classify every row and count left rows per chunk (parallel); a prefix sum
   gives L.
2. In parallel per chunk, number the misplaced right rows in `[0, L)` from the
   front and the misplaced left rows in `[L, n)` from the back (chunk counts
   plus prefix sums give each its rank).
3. Write the swaps into a copy of the slice, or swap in place since each swap
   pair is disjoint.

`pivot_on_split_exclude_missing` is harder: it also moves missing rows to the
front with extra swaps inside the same loop, so its output order interleaves
two processes. Either derive its closed form the same way and test it
exhaustively, or keep it serial and parallelize only `pivot_on_split` first.
`MissingImputerSplitter` is the default (`create_missing_branch=False`), so
`pivot_on_split` covers the common case.

Edge cases to preserve:
- `pivot_on_split` handles missing rows through `missing_compare` and the
  `missing_right` flag.
- Both functions index `index.len() - 1`, so they assume a non-empty slice.
- The returned split index (and the missing index for the three-way version)
  must match exactly.

### Implementation steps

1. **Reference tests first.** Add property tests in `src/utils.rs` that
   compare a new `pivot_on_split_parallel` against `pivot_on_split` on random
   inputs: lengths 1 to ~10,000, many bin values, values equal to the split
   value, all-left, all-right, `missing_right` both ways, and many missing
   rows. Assert the returned index **and the full `index` order** are equal.
2. **Implement the closed form serially** and make those tests pass. This
   proves the reasoning before adding threads.
3. **Parallelize it** with Rayon (chunks of a few thousand rows, chunk counts,
   prefix sums). Keep the property tests passing.
4. **Use it only for large nodes:** call the parallel version in
   `handle_split_info` when `parallel` is set and the node has more than a
   threshold of rows (start at 65,536 and tune). Small nodes keep the serial
   function, where Rayon overhead would dominate.
5. **Optional:** repeat for `pivot_on_split_exclude_missing` if the profile
   shows `MissingBranchSplitter` matters for your workloads.

### Correctness

- New property tests (above), plus `cargo test --all-targets`.
- `python scripts/perf_golden.py check`: all 5 golden models `same`. The
  `missing_branch` golden model covers the three-way function.
- The Python suite (`uv run pytest` in `py-forust`).

### Speed testing

Add a G5 grid to `scripts/perf_grid.py` for the case that motivated this item:

```python
"G5": [("w200-1m", 1_000_000, 8, 20, False)],
```

Regenerate the 1M data if needed:

```sh
python scripts/make_perf_data.py --out /tmp/forust-perf/w200-1m --rows 1000000 --eval-rows 250000 --cols 200
```

| Check | Expected result |
| --- | --- |
| G5 (1M x 200, depth 8, 8 threads), before and after | The main target; baseline 633.7 ms/tree |
| G4 (1M x 200, depth 5) | Improves, less than G5 |
| G1-G3 | Unchanged or better; small nodes shouldn't regress (threshold) |
| Threshold sweep: 16k, 65k, 262k rows | Pick the fastest across G1, G2, G5 |
| `scripts/perf_xgboost.py --data /tmp/forust-perf/w200-1m --rows 1000000 --threads 8 --iterations 20 --max-depth 8` | Rerun XGBoost (454.6 ms) for comparison |

Record results with `--label item5` and add a step-log entry and summary
column to `results.md`, as for items 1-4.
