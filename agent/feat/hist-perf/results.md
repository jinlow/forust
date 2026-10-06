# Histogram Performance Results (ideas 8 and 9)

Date: 2026-10-06. Branch: `feat/hist-perf` (off `feat/goss`). Ideas from
`agent/perf-ideas.md`. Machine: 8 physical cores (16 vCPU). All timings at 8
threads on `make_perf_data.py` data. Every kept change leaves models
byte-identical (`scripts/perf_golden.py`, 10 configurations including sampling,
LossGuide, missing branch and `nbins` 255/64), and the Python suite passes
(121 tests).

## Outcome

| Idea | Result |
| --- | --- |
| 8a. Skip histograms for children at `max_depth` | **Kept** (`6639d83`). 10-35% less tree time. |
| 8b. Drop `cut_value` from `Bin` | **Kept** (`b2a8687`). Up to 21% less tree time on small nodes and deep trees. |
| 8c. Reuse histogram buffers | **Dropped.** In-place sibling subtraction, with or without a buffer pool, was slower at depth 5 (250k: +6%, 1M: +7%) and not clearly better at depth 8. |
| 8d. Reuse the per-column `sums` buffer | **Dropped.** A thread-local buffer was slower (250k: +6-30%, 100k depth 8: +17%). |
| 9. `u8` bins | **Skipped.** Microbenchmark below: modest gain, only when `nbins <= 255`, and the generic plumbing is large. |
| 9. `u32` row index | **Rejected.** No gain in the microbenchmark. |

## End-to-end fit, 100 iterations, one evaluation set

Before is `feat/goss` (`agent/feat/goss/results.jsonl`, label `fixed`); after
is this branch (`fit.jsonl`). Median total seconds; GOSS and `random80` over
seeds 0-2. Eval logloss is identical before and after.

| Data | Rows | Setting | Before s | After s | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| w200 | 100k | none | 3.47 | 2.49 | -28.1% |
| w200 | 100k | random80 | 3.21 | 2.30 | -28.3% |
| w200 | 100k | goss | 2.31 | 1.94 | -16.0% |
| w200 | 250k | none | 7.91 | 5.60 | -29.2% |
| w200 | 250k | random80 | 7.13 | 5.18 | -27.4% |
| w200 | 250k | goss | 4.42 | 3.46 | -21.7% |
| w500 | 100k | none | 7.92 | 5.76 | -27.3% |
| w500 | 100k | random80 | 7.20 | 4.98 | -30.9% |
| w500 | 100k | goss | 4.88 | 3.57 | -26.8% |
| w200-1m | 1M | none | 26.74 | 19.53 | -27.0% |
| w200-1m | 1M | random80 | 24.10 | 18.35 | -23.8% |
| w200-1m | 1M | goss | 15.35 | 12.92 | -15.8% |

For reference, LightGBM with 80% bagging took 2.42 s (100k), 5.39 s (250k) and
16.28 s (1M) on the same data in the GOSS benchmark.

## Steady-state tree time per tree (`perf_sweep.py`, ms)

Raw records in `results.jsonl`, labels `baseline`, `8a`, `8b`.

| Configuration | Baseline | 8a | 8b |
| --- | ---: | ---: | ---: |
| w200 25k, depth 5 | 14.9 | 11.0 | 8.7 |
| w200 100k, depth 5 | 30.1 | 20.3 | 20.1 |
| w200 250k, depth 5 | 67.5 | 43.8 | 41.3 |
| w200 25k, depth 8 | 48.5 | 37.5 | 31.5 |
| w200 100k, depth 8 | 91.0 | 82.3 | 73.1 |
| w200 100k, random30 | 15.7 | 11.5 | 9.7 |
| w200 250k, random30 | 25.5 | 21.3 | 17.6 |
| w200-1m 1M, depth 5 | 224.7 | 158.0 | 152.8 |

Single runs vary by about 5-10%; 8c and 8d were judged with interleaved A/B runs
(`/tmp/forust-perf/ab.sh`), 3-4 rounds each.

## Item 9 microbenchmark

Histogram fill over 200 columns, parallel over columns, 8 threads, best of 7.
"30%" is a sorted index of 30% of rows.

| Rows | Index | u16 / usize (now) | u8 / usize | u16 / u32 | u8 / u32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100k | full | 3.80 ms | 3.47 ms | 3.75 ms | 3.49 ms |
| 100k | 30% | 1.39 ms | 1.11 ms | 1.16 ms | 0.99 ms |
| 1M | full | 31.03 ms | 28.95 ms | 33.23 ms | 30.99 ms |
| 1M | 30% | 12.27 ms | 9.86 ms | 11.95 ms | 9.71 ms |

`u8` bins make the fill 7-20% faster, which is likely 4-10% of tree time, and
only applies when every column has at most 255 non-missing bins (`nbins <= 255`
or low-cardinality data). The default `nbins=256` needs `u16`: bin 0 is missing
and bins 1-256 hold values.
