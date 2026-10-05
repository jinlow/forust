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
use forust_ml::sampler::SampleMethod;
use forust_ml::splitter::MissingImputerSplitter;
use forust_ml::tree::Tree;
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
                .map(|pair| (pair[0].trim_start_matches("--").to_string(), pair[1].clone()))
                .collect(),
        )
    }

    fn get<T: FromStr>(&self, key: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.0.get(key).map(|v| v.parse().unwrap()).unwrap_or(default)
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
    let parallel = args.get("parallel", true);
    let missing_branch = args.get("missing-branch", false);
    let mode = args.get("mode", String::from("phases"));

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
        "parallel": parallel,
        "missing_branch": missing_branch,
        "rayon_threads": rayon::current_num_threads(),
    });

    let result = if mode == "fit" {
        let mut booster = GradientBooster::default()
            .set_iterations(iterations)
            .set_learning_rate(learning_rate)
            .set_max_depth(max_depth)
            .set_nbins(nbins)
            .set_parallel(parallel)
            .set_create_missing_branch(missing_branch);
        let start = Instant::now();
        booster
            .fit(
                &data,
                &y,
                &w,
                Some(vec![(Matrix::new(&eval_values, eval_rows, cols), &eval_y[..], &eval_w[..])]),
            )
            .unwrap();
        let total = start.elapsed().as_secs_f64();
        let eval_logloss = log_loss(&eval_y, &booster.predict(&eval_data, parallel), &eval_w);
        // Trees only: the full model JSON also records the `parallel` setting.
        if let Some(path) = args.0.get("save-trees") {
            fs::write(path, serde_json::to_string(&booster.trees).unwrap()).unwrap();
        }
        json!({"config": config, "total_s": total, "eval_logloss": eval_logloss})
    } else {
        let start = Instant::now();
        let binned = bin_matrix(&data, &w, nbins, f64::NAN).unwrap();
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
        let mut tree_times = Vec::with_capacity(iterations);
        let mut tree_nodes = Vec::with_capacity(iterations);
        let mut eval_logloss = f64::NAN;

        for _ in 0..iterations {
            let t0 = Instant::now();
            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                data.index.to_owned(),
                &col_index,
                &binned.cuts,
                &grad,
                &hess,
                &splitter,
                usize::MAX,
                max_depth,
                parallel,
                &SampleMethod::None,
                &GrowPolicy::DepthWise,
            );
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
                "tree": tree_s,
                "train_predict": train_predict_s,
                "eval_predict_metric": eval_s,
                "grad_hess": grad_s,
            },
            "tree_s": tree_times,
            "tree_nodes": tree_nodes,
        })
    };
    println!("{result}");
}
