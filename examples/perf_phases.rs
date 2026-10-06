//! Per-phase timing of Forust training on data written by `scripts/make_perf_data.py`.
//!
//! `--mode phases` replays the `GradientBooster::fit` loop with timers around each phase;
//! `--mode fit` times the real `fit` so the replay can be checked against it.
use forust_ml::binning::bin_matrix;
use forust_ml::constraints::ConstraintMap;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::{GradientBooster, GrowPolicy};
use forust_ml::metric::log_loss;
use forust_ml::objective::{LogLoss, ObjectiveFunction};
use forust_ml::sampler::{GossSampler, RandomSampler, RowSubset, SampleMethod, Sampler};
use forust_ml::splitter::MissingImputerSplitter;
use forust_ml::tree::Tree;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        assert!(raw.len() % 2 == 0, "arguments must be `--key value` pairs");
        Args(
            raw.chunks(2)
                .map(|pair| {
                    (
                        pair[0].trim_start_matches("--").to_string(),
                        pair[1].clone(),
                    )
                })
                .collect(),
        )
    }

    fn get<T: FromStr>(&self, key: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.0
            .get(key)
            .map(|v| v.parse().unwrap())
            .unwrap_or(default)
    }
}

fn read_f64(path: &Path) -> Vec<f64> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn load_split(
    dir: &Path,
    name: &str,
    total_rows: usize,
    rows: usize,
    cols: usize,
) -> (Vec<f64>, Vec<f64>) {
    let x = read_f64(&dir.join(format!("X_{name}.f64")));
    let data = (0..cols)
        .flat_map(|c| x[c * total_rows..c * total_rows + rows].iter().copied())
        .collect();
    let mut y = read_f64(&dir.join(format!("y_{name}.f64")));
    y.truncate(rows);
    (data, y)
}

fn main() {
    let args = Args::parse();
    let dir = PathBuf::from(args.get("data", String::from("/tmp/forust-perf/w200")));
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("metadata.json")).unwrap()).unwrap();
    let total_rows = meta["rows"].as_u64().unwrap() as usize;
    let total_eval_rows = meta["eval_rows"].as_u64().unwrap() as usize;
    let total_cols = meta["cols"].as_u64().unwrap() as usize;

    let rows = args.get("rows", total_rows).min(total_rows);
    let eval_rows = args.get("eval-rows", rows / 4).min(total_eval_rows);
    let cols = args.get("cols", total_cols).min(total_cols);
    let iterations = args.get("iterations", 100usize);
    let max_depth = args.get("max-depth", 5usize);
    let learning_rate = args.get("learning-rate", 0.1f32);
    let nbins = args.get("nbins", 256u16);
    let grow_policy_name = args.get("grow-policy", String::from("DepthWise"));
    let grow_policy = match grow_policy_name.as_str() {
        "LossGuide" => GrowPolicy::LossGuide,
        "DepthWise" => GrowPolicy::DepthWise,
        value => panic!("unknown grow policy: {value}"),
    };
    let max_leaves = args.get("max-leaves", usize::MAX);
    let parallel = args.get("parallel", true);
    let missing_branch = args.get("missing-branch", false);
    let num_threads = args.get("num-threads", 0usize);
    let predict_calls = args.get("predict-calls", 0usize);
    let mode = args.get("mode", String::from("phases"));
    let sample_method_name = args
        .get("sample-method", String::from("none"))
        .to_lowercase();
    let sample_method = match sample_method_name.as_str() {
        "none" => SampleMethod::None,
        "random" => SampleMethod::Random,
        "goss" => SampleMethod::Goss,
        value => panic!("unknown sample method: {value}"),
    };
    let top_rate = args.get("top-rate", 0.2f64);
    let other_rate = args.get("other-rate", 0.1f64);
    let subsample = args.get("subsample", 1.0f32);
    let seed = args.get("seed", 0u64);
    let early_stopping_rounds = args.get("early-stopping-rounds", 0usize);
    // Phases mode only: copy sampled rows into a contiguous subset, as `fit` does.
    let use_subset = args.get("subset", true);

    let (values, y) = load_split(&dir, "train", total_rows, rows, cols);
    let (eval_values, eval_y) = load_split(&dir, "eval", total_eval_rows, eval_rows, cols);
    let data = Matrix::new(&values, rows, cols);
    let eval_data = Matrix::new(&eval_values, eval_rows, cols);
    let w = vec![1.0; rows];
    let eval_w = vec![1.0; eval_rows];

    let config = json!({
        "mode": mode,
        "data": dir.display().to_string(),
        "rows": rows,
        "eval_rows": eval_rows,
        "cols": cols,
        "iterations": iterations,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "nbins": nbins,
        "grow_policy": grow_policy_name,
        "max_leaves": max_leaves,
        "parallel": parallel,
        "missing_branch": missing_branch,
        "num_threads": num_threads,
        "rayon_threads": rayon::current_num_threads(),
        "sample_method": sample_method_name,
        "top_rate": top_rate,
        "other_rate": other_rate,
        "subsample": subsample,
        "seed": seed,
        "early_stopping_rounds": early_stopping_rounds,
        "subset": use_subset,
    });

    let result = if mode == "fit" {
        let mut booster = GradientBooster::default()
            .set_iterations(iterations)
            .set_learning_rate(learning_rate)
            .set_max_depth(max_depth)
            .set_nbins(nbins)
            .set_parallel(parallel)
            .set_create_missing_branch(missing_branch)
            .set_num_threads((num_threads > 0).then_some(num_threads))
            .set_sample_method(sample_method)
            .set_subsample(subsample)
            .set_seed(seed)
            .set_early_stopping_rounds(
                (early_stopping_rounds > 0).then_some(early_stopping_rounds),
            );
        booster.grow_policy = grow_policy;
        booster.max_leaves = max_leaves;
        booster.top_rate = top_rate;
        booster.other_rate = other_rate;
        let start = Instant::now();
        booster
            .fit(
                &data,
                &y,
                &w,
                Some(vec![(
                    Matrix::new(&eval_values, eval_rows, cols),
                    &eval_y[..],
                    &eval_w[..],
                )]),
            )
            .unwrap();
        let total = start.elapsed().as_secs_f64();
        let eval_logloss = log_loss(&eval_y, &booster.predict(&eval_data, parallel), &eval_w);
        // Trees only: the full model JSON also records the `parallel` setting.
        if let Some(path) = args.0.get("save-trees") {
            fs::write(path, serde_json::to_string(&booster.trees).unwrap()).unwrap();
        }
        // Time repeated predictions on a small batch, where per-call overhead shows up.
        let predict_ms = (predict_calls > 0).then(|| {
            let batch_rows = 1000.min(eval_rows);
            let batch: Vec<f64> = (0..cols)
                .flat_map(|c| eval_data.get_col(c)[..batch_rows].iter().copied())
                .collect();
            let batch = Matrix::new(&batch, batch_rows, cols);
            let start = Instant::now();
            for _ in 0..predict_calls {
                std::hint::black_box(booster.predict(&batch, parallel));
            }
            1000.0 * start.elapsed().as_secs_f64() / predict_calls as f64
        });
        json!({
            "config": config,
            "total_s": total,
            "eval_logloss": eval_logloss,
            "trees": booster.trees.len(),
            "best_iteration": booster.best_iteration,
            "predict_ms_per_call": predict_ms,
        })
    } else {
        let run = || {
            let start = Instant::now();
            let binned = bin_matrix(&data, &w, nbins, f64::NAN, parallel).unwrap();
            let bin_s = start.elapsed().as_secs_f64();
            let bdata = Matrix::new(&binned.binned_data, rows, cols);
            let col_index: Vec<usize> = (0..cols).collect();
            let splitter = MissingImputerSplitter {
                l1: 0.0,
                l2: 1.0,
                max_delta_step: 0.0,
                gamma: 0.0,
                min_leaf_weight: 1.0,
                learning_rate,
                allow_missing_splits: true,
                constraints_map: ConstraintMap::new(),
            };

            let base_score = LogLoss::calc_init(&y, &w);
            let mut yhat = vec![base_score; rows];
            let mut eval_yhat = vec![base_score; eval_rows];
            let (mut grad, mut hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);
            let (mut tree_s, mut train_predict_s, mut eval_s, mut grad_s) = (0.0, 0.0, 0.0, 0.0);
            let (mut sample_s, mut subset_s) = (0.0, 0.0);
            let mut row_subset = RowSubset::default();
            let mut subset_times = Vec::with_capacity(iterations);
            let mut rng = StdRng::seed_from_u64(seed);
            // Same warm-up rule as `GradientBooster::fit`.
            let warmup = match sample_method {
                SampleMethod::Goss => GossSampler::warmup_iterations(learning_rate),
                _ => 0,
            };
            let mut sample_times = Vec::with_capacity(iterations);
            let mut tree_rows = Vec::with_capacity(iterations);
            let mut tree_times = Vec::with_capacity(iterations);
            let mut tree_nodes = Vec::with_capacity(iterations);
            let mut eval_logloss = f64::NAN;

            for i in 0..iterations {
                let ts = Instant::now();
                let iteration_method = if i < warmup {
                    SampleMethod::None
                } else {
                    sample_method
                };
                let index = match iteration_method {
                    SampleMethod::None => data.index.to_owned(),
                    SampleMethod::Random => {
                        RandomSampler::new(subsample)
                            .sample(&mut rng, &data.index, &mut grad, &mut hess)
                            .0
                    }
                    SampleMethod::Goss => {
                        GossSampler::new(top_rate, other_rate)
                            .sample(&mut rng, &data.index, &mut grad, &mut hess)
                            .0
                    }
                };
                tree_rows.push(index.len());
                let tc = Instant::now();
                let subset = use_subset
                    && iteration_method != SampleMethod::None
                    && RowSubset::should_use(index.len(), rows);
                if subset {
                    row_subset.fill(&bdata, &index, &col_index, &grad, &hess, parallel);
                }
                let t0 = Instant::now();
                let mut tree = Tree::new();
                let fit_tree =
                    |tree: &mut Tree, data: &Matrix<u16>, index, g: &[f32], h: &[f32]| {
                        tree.fit(
                            data,
                            index,
                            &col_index,
                            &binned.cuts,
                            g,
                            h,
                            &splitter,
                            max_leaves,
                            max_depth,
                            parallel,
                            &iteration_method,
                            &grow_policy,
                        )
                    };
                if subset {
                    let (subset_data, subset_index) = row_subset.matrix();
                    fit_tree(
                        &mut tree,
                        &subset_data,
                        subset_index,
                        &row_subset.grad,
                        &row_subset.hess,
                    );
                } else {
                    fit_tree(&mut tree, &bdata, index, &grad, &hess);
                }
                let t1 = Instant::now();
                yhat.iter_mut()
                    .zip(tree.predict(&data, parallel, &f64::NAN))
                    .for_each(|(p, v)| *p += v);
                let t2 = Instant::now();
                eval_yhat
                    .iter_mut()
                    .zip(tree.predict(&eval_data, parallel, &f64::NAN))
                    .for_each(|(p, v)| *p += v);
                eval_logloss = log_loss(&eval_y, &eval_yhat, &eval_w);
                let t3 = Instant::now();
                (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, &w);
                let t4 = Instant::now();

                sample_s += (tc - ts).as_secs_f64();
                sample_times.push((tc - ts).as_secs_f64());
                subset_s += (t0 - tc).as_secs_f64();
                subset_times.push((t0 - tc).as_secs_f64());
                tree_s += (t1 - t0).as_secs_f64();
                train_predict_s += (t2 - t1).as_secs_f64();
                eval_s += (t3 - t2).as_secs_f64();
                grad_s += (t4 - t3).as_secs_f64();
                tree_times.push((t1 - t0).as_secs_f64());
                tree_nodes.push(tree.nodes.len());
            }
            json!({
                "config": config,
                "total_s": start.elapsed().as_secs_f64(),
                "eval_logloss": eval_logloss,
                "phases_s": {
                    "bin": bin_s,
                    "sample": sample_s,
                    "subset": subset_s,
                    "tree": tree_s,
                    "train_predict": train_predict_s,
                    "eval_predict_metric": eval_s,
                    "grad_hess": grad_s,
                },
                "tree_s": tree_times,
                "sample_s": sample_times,
                "subset_s": subset_times,
                "tree_rows": tree_rows,
                "tree_nodes": tree_nodes,
            })
        };
        // Match `GradientBooster::fit`, which runs parallel training on a pool thread.
        if parallel {
            rayon::scope(|_| run())
        } else {
            run()
        }
    };
    println!("{result}");
}
