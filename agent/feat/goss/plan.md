# Forust GOSS Implementation Plan

Date: 2026-10-06. Branch: `feat/goss` (based on `feat/optimize`).

Goal: make Gradient-based One-Side Sampling (GOSS) a supported, documented,
user-selectable setting, and show with benchmarks that it keeps eval logloss
close to the no-sampling result while reducing training time.

## Starting point

GOSS is already partly implemented, but not documented or tested:

- `SampleMethod::Goss` and `GossSampler` in `src/sampler.rs`.
- `top_rate` and `other_rate` on `GradientBooster`, wired through
  `sample_index` in `src/gradientbooster.rs`.
- Python: `GradientBooster(sample_method="goss", top_rate=..., other_rate=...)`.

Problems with the current version:

| # | Problem | Where |
| --- | --- | --- |
| 1 | Default rates are swapped: booster and Python use `top_rate=0.1, other_rate=0.2`; LightGBM and `GossSampler::default` use `0.2 / 0.1`. | `default_top_rate`, `default_other_rate`, `GradientBooster::default`, `py-forust/forust/__init__.py` |
| 2 | No warm-up: LightGBM skips GOSS for the first `int(1 / learning_rate)` iterations. | `fit_trees` |
| 3 | Rows are ranked by `\|g\|`; LightGBM ranks by `\|g * h\|`. | `GossSampler::sample` |
| 4 | Full `O(n log n)` sort of every row each iteration. | `GossSampler::sample` |
| 5 | Small-gradient rows are sampled independently at random, so the count varies; LightGBM samples an exact count. | `GossSampler::sample` |
| 6 | No validation of `top_rate` / `other_rate`; Python silently turns unknown `sample_method` strings into `"Random"`. | `validate_parameters`, `SAMPLE_METHODS.get(..., "Random")` |
| 7 | The only test is `assert True`. | `test_goss_sampling_method` |
| 8 | Not mentioned in the README or docs. | `README.md`, `py-forust/docs` |

## Decisions (agreed)

- Match LightGBM: warm-up, `|g * h|` ranking, defaults `top_rate=0.2`,
  `other_rate=0.1`, exact-count sampling, multiplier `(n - top_k) / other_k`.
- Invalid settings raise errors (Rust `ForustError`, Python `ValueError`).
- Benchmarks use synthetic data only (`make_perf_data.py`), one dataset per size.
- Accuracy is measured with eval logloss only (no AUC), matching XGBoost and
  LightGBM reporting.
- LightGBM (with and without GOSS) and XGBoost `hist` are reference columns.
- **Baseline is random row sampling with `subsample=0.8`** (`random80`), the
  common practical setting; no sampling is reported for information only.
  XGBoost uses `subsample=0.8` and LightGBM `bagging_fraction=0.8`,
  `bagging_freq=1`; LightGBM GOSS is compared with LightGBM bagging.
- Sampled runs use 5 model seeds (3 for the 1M early-stopping and 1-thread runs).

## Library changes

### 1. Sampler (`src/sampler.rs`)

Rewrite `GossSampler::sample` to follow LightGBM's `GOSSStrategy::Helper`:

1. Score each row by `|grad[i] * hess[i]|`.
2. `top_k = max(1, floor(n * top_rate))`, `other_k = floor(n * other_rate)`.
3. Find the `top_k`-th largest score with `select_nth_unstable_by` on a copy of
   the scores (`O(n)`, no full sort). Rows with score `>= threshold` are kept
   unchanged.
4. Walk the remaining rows in order and keep each with probability
   `rest_need / rest_all`, which yields exactly `other_k` rows. Scale the kept
   rows' `grad` and `hess` by `(n - top_k) / other_k`.
5. Return the chosen row indices in ascending order, so `tree.fit` sorts cheaply.

Fix the index handling: the sampler must return values from `index`, not
positions in it. (They are equal today because `data.index` is `0..rows`.)

### 2. Booster (`src/gradientbooster.rs`)

- Defaults: `default_top_rate() = 0.2`, `default_other_rate() = 0.1`, and the
  same in `GradientBooster::default()`.
- Warm-up: in `fit_trees`, use the full index for iterations
  `i < (1.0 / learning_rate) as usize` when `sample_method == Goss`; pass
  `SampleMethod::None` to `tree.fit` for those iterations so it skips the sort.
- Validation in `validate_parameters`, when `sample_method == Goss`:
  - `0 < top_rate`, `0 < other_rate`, `top_rate + other_rate <= 1`;
  - `subsample == 1.0`, matching LightGBM's "Cannot use bagging in GOSS".
- Rust doc comments for `sample_method`, `top_rate`, `other_rate` updated.

Saved models are unaffected: they store `top_rate` and `other_rate`
explicitly. The changed serde defaults only apply to JSON from versions
before those fields existed, and they only matter if that model is refit.

### 3. Python (`py-forust`)

- Defaults `top_rate=0.2`, `other_rate=0.1` in `GradientBooster.__init__`.
- Unknown `sample_method` raises `ValueError` listing valid options.
- Keep the existing rule that `subsample < 1` with no `sample_method` means
  `"random"`; `sample_method="goss"` with `subsample < 1` raises `ValueError`.
- Docstrings for `sample_method`, `top_rate`, `other_rate`, explaining the
  warm-up and that GOSS ignores `subsample`.
- Update `.pyi` stubs if they list these parameters.

### 4. Docs

- `README.md` and `py-forust/README.md`: short GOSS section with an example.
- `py-forust/docs`: parameter descriptions picked up from the docstrings.

### 5. Contiguous row subset (added after first benchmarks)

As LightGBM does when `top_rate + other_rate <= 0.5` (and for bagging), copy the
sampled rows' binned values, gradients, and hessians into reused contiguous
buffers (`RowSubset` in `src/sampler.rs`) and fit the tree on rows `0..m`.
Applies to GOSS and random sampling whenever at most 50% of rows are sampled.
Trees are byte-identical to fitting on the full data with the sampled index.

Measured at 1M x 200, 8 threads, GOSS 0.2/0.1, steady state per tree: tree
111.1 ms -> 89.2 ms, plus an 11.6 ms copy (bandwidth-bound). Row selection
(15 ms, serial) is now the largest sampling cost.

## Tests

Rust (`src/sampler.rs`, `src/gradientbooster.rs`):

- Sampler returns exactly `top_k + other_k` rows for distinct scores.
- All `top_k` highest-`|g * h|` rows are chosen and left unscaled.
- Sampled small-gradient rows are scaled by `(n - top_k) / other_k`; unsampled
  rows are untouched.
- Returned indices are sorted and come from `index`.
- Warm-up: the first `int(1 / learning_rate)` trees train on all rows (check
  root node cover/hessian sum equals the full-data value).
- Same seed gives identical models; different seeds differ.
- Validation errors for bad rates and for `subsample < 1` with GOSS.

Python (`py-forust/tests/test_booster.py`):

- Replace `assert True` with a check that GOSS eval logloss is within a loose
  tolerance of no sampling on the Titanic data.
- `ValueError` for an unknown `sample_method`, bad rates, and GOSS with
  `subsample < 1`.
- JSON save/load round-trip keeps `sample_method`, `top_rate`, `other_rate`.
- Full Python suite, including XGBoost parity checks, still passes. The default
  is `sample_method=None`, so parity tests should be unaffected.

## Benchmark

### Tooling changes

- `examples/perf_phases.rs`
  - New flags: `--sample-method none|random|goss`, `--top-rate`,
    `--other-rate`, `--subsample`, `--seed` (both `fit` and `phases` modes).
  - `phases` mode calls the sampler (with the same warm-up rule as `fit`) and
    records a new `sample` time in `phases_s`, plus rows used per tree.
  - Report eval logloss in both modes (already present).
- `scripts/perf_sweep.py`: no structural change; GOSS flags pass through after
  `--`, and `--label` tags each sampling setting.
- `scripts/perf_reference.py` (generalizes `perf_xgboost.py`):
  `--library xgboost|lightgbm`, `--goss`, `--top-rate`, `--other-rate`, same
  data, threads, depth, bins, learning rate and iterations; reports fit time
  and eval logloss. LightGBM uses `data_sample_strategy="goss"`.
  Keep `perf_xgboost.py` as a thin wrapper, or replace it if nothing else uses it.
- `scripts/perf_goss.py`: drives the runs below, writes
  `agent/feat/goss/results.jsonl`, and prints the pass/fail table.

### Environment

The previously used environment
`/home/azureuser/localfiles/uv-environments/forust` no longer exists. Recreate
it with `uv`, install `py-forust` (maturin, release), `xgboost`, `lightgbm`,
`scikit-learn`, `numpy`.

### Data

```sh
PY=/home/azureuser/localfiles/uv-environments/forust/bin/python
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w200    --rows 250000  --eval-rows 62500  --cols 200
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w500    --rows 100000  --eval-rows 25000  --cols 500
$PY scripts/make_perf_data.py --out /tmp/forust-perf/w200-1m --rows 1000000 --eval-rows 250000 --cols 200
```

### Sampling settings compared

| Label | Setting |
| --- | --- |
| `none` | `sample_method=None` (information only) |
| `random80` | `random`, `subsample=0.8` (baseline) |
| `random30` | `random`, `subsample=0.3` (same row budget as default GOSS) |
| `goss` | `goss`, `top_rate=0.2`, `other_rate=0.1` |
| `goss-small` | `goss`, `top_rate=0.1`, `other_rate=0.1` |

`random30` shows whether choosing rows by gradient beats choosing them at
random with the same number of rows.

### Runs

1. **Per-tree speed (phases):** `perf_sweep.py` on w200 (25k, 100k, 250k
   rows), w500 (100k), w200-1m (1M); depth 5 and 8; 8 threads; 3 repeats,
   median reported. Shows tree, sample, predict, eval and grad/hess time per
   setting.

   ```sh
   cargo build --release --example perf_phases
   for s in "none" "random --subsample 0.3" "goss" "goss --top-rate 0.1 --other-rate 0.1"; do
     python3 scripts/perf_sweep.py --data /tmp/forust-perf/w200 \
       --rows 25000 100000 250000 --threads 8 --repeats 3 --label "$s" \
       --out agent/feat/goss/results.jsonl -- --sample-method $s
   done
   ```

2. **Full fit, fixed iterations:** `perf_phases --mode fit`, 1 eval set,
   learning rate 0.1, 100 iterations (300 at 100k for a longer run), 8 threads,
   seeds 0-4 for sampled settings. Records total fit time and eval logloss.
3. **Full fit, early stopping:** same as run 2 with `early_stopping_rounds=20`
   and up to 1,000 iterations. Records iterations used, time, and best eval
   logloss. GOSS may need more trees, so this checks total time to reach the
   same quality.
4. **References:** `perf_reference.py` with XGBoost `hist`, LightGBM `gbdt`,
   and LightGBM GOSS on the same data, threads, and settings as run 2.
5. **Thread check:** run 2 on w200-1m at 1 and 8 threads, to see whether the
   GOSS gain differs between serial and parallel training.
6. **Profile:** release build with debug info, `perf record` of
   `perf_phases` on w200 100k for `none` and `goss`, to confirm histogram time
   drops and the sampler stays a small share.

   ```sh
   CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --example perf_phases
   for m in none goss; do
     RAYON_NUM_THREADS=8 perf record -F 999 -o /tmp/forust-perf/goss-$m.data -- \
       target/release/examples/perf_phases --rows 100000 --iterations 60 --sample-method $m
     perf report -i /tmp/forust-perf/goss-$m.data --stdio --no-children | head -60
   done
   ```

7. **Sampler micro-benchmark:** add a Criterion case to
   `benches/forust_benchmarks.rs` for `GossSampler::sample` at 100k and 1M rows.

### Pass/fail rule

For default GOSS (`top_rate=0.2`, `other_rate=0.1`) against `random80`, same
configuration:

- **Accuracy:** mean eval logloss over 5 seeds is no more than 1% above the
  baseline's mean, on every dataset, in both fixed-iteration and early-stopping runs.
- **Speed:** full `fit` (with an eval set) on w200-1m at 8 threads is at least
  1.5x faster at a fixed number of iterations.

Also reported, not gated: speedups at smaller sizes, `goss-small`, `random30`,
and how forust's GOSS gain compares with LightGBM's.

Expected limits: GOSS only shrinks per-row tree work. Binning, gradients,
training-set prediction updates, and eval scoring still cover all rows, and the
first `1 / learning_rate` trees use all rows. Gains should be largest at 1M rows
and deeper trees, and small at 25k rows with 8 threads.

### Outputs

- `agent/feat/goss/results.jsonl`: raw records.
- `agent/feat/goss/results.md`: tables for speed, logloss (mean and standard
  deviation over seeds), references, profile summary, and pass/fail.

## Order of work

1. Recreate the Python environment; build; run the current Python and Rust
   tests to confirm a clean baseline.
2. Benchmark tooling (`perf_phases` flags, `perf_reference.py`,
   `perf_goss.py`), and record the current, unfixed GOSS as a reference point.
3. Sampler rewrite and booster changes, with Rust tests.
4. Python changes, tests, and docs.
5. Full benchmark runs and `results.md`.
6. Final check: `cargo test`, `cargo clippy`, full Python suite.

Each step is a separate commit on `feat/goss`.

## Open points for review

- **Ties at the threshold:** LightGBM keeps every row with score `>= threshold`,
  so ties can push the count slightly above `top_k`. Plan: do the same.
- **Parallel sampling (done):** like LightGBM, rows are sampled in blocks, each
  with its own threshold and random stream. Blocks have a fixed target size
  (32,768 rows) rather than one per thread, so models don't depend on the thread
  count. 1M x 200, 8 threads: 15-17 ms -> 4.8 ms per tree.
- **Changed GOSS results:** the new sampler uses random numbers differently, so
  existing GOSS users get different models after upgrading. GOSS was
  undocumented, so this seems acceptable, but it should be in the release notes.
