use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forust_ml::binning::bin_matrix;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster;
use forust_ml::histogram::HistogramMatrix;
use forust_ml::histsplitter::HistogramSplitter;
use forust_ml::objective::{LogLoss, ObjectiveFunction};
use forust_ml::tree::Tree;
use forust_ml::utils::{fast_sum, naive_sum};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;

pub fn tree_benchmarks(c: &mut Criterion) {
    let file = fs::read_to_string("resources/contiguous_no_missing_100k_samp_seed0.csv")
        .expect("Something went wrong reading the file");
    let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let file = fs::read_to_string("resources/performance_100k_samp_seed0.csv")
        .expect("Something went wrong reading the file");
    let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let yhat = vec![0.5; y.len()];
    let w = vec![1.; y.len()];
    let g = LogLoss::calc_grad(&y, &yhat, &w);
    let h = LogLoss::calc_hess(&y, &yhat, &w);

    let data = Matrix::new(&data_vec, y.len(), 5);
    let splitter = HistogramSplitter {
        l2: 1.0,
        gamma: 3.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
    };
    let mut tree = Tree::new();

    let bindata = bin_matrix(&data, &w, 300).unwrap();
    let bdata = Matrix::new(&bindata.binned_data, data.rows, data.cols);

    let mut random_index: Vec<usize> = (0..g.len()).collect();
    random_index.shuffle(&mut thread_rng());

    c.bench_function("calc hist parallel", |b| {
        b.iter(|| {
            HistogramMatrix::new(
                black_box(&bdata),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&random_index),
                black_box(true),
                black_box(false),
            );
        })
    });

    c.bench_function("calc hist single", |b| {
        b.iter(|| {
            HistogramMatrix::new(
                black_box(&bdata),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&random_index),
                black_box(false),
                black_box(false),
            );
        })
    });

    c.bench_function("calc_grad", |b| {
        b.iter(|| LogLoss::calc_grad(black_box(&y), black_box(&yhat), black_box(&w)))
    });

    c.bench_function("calc_hess", |b| {
        b.iter(|| LogLoss::calc_hess(black_box(&y), black_box(&yhat), black_box(&w)))
    });

    tree.fit(
        &bdata,
        &bindata.cuts,
        &g,
        &h,
        &splitter,
        usize::MAX,
        5,
        true,
    );
    println!("{}", tree.nodes.len());
    c.bench_function("Train Tree", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();
            train_tree.fit(
                black_box(&bdata),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&splitter),
                black_box(usize::MAX),
                black_box(5),
                black_box(true),
            );
        })
    });
    c.bench_function("Tree Predict (Single Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(false)))
    });
    c.bench_function("Tree Predict (Multi Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(true)))
    });

    // Gradient Booster
    // Bench building
    c.bench_function("Train Booster", |b| {
        b.iter(|| {
            let mut booster = GradientBooster::default();
            booster
                .fit(black_box(&data), black_box(&y), black_box(&w))
                .unwrap();
        })
    });
    let mut booster = GradientBooster::default();
    booster.fit(&data, &y, &w).unwrap();
    c.bench_function("Predict Booster", |b| {
        b.iter(|| booster.predict(black_box(&data), true))
    });

    let v: Vec<f64> = vec![10.; 10000];
    c.bench_function("Niave Sum", |b| b.iter(|| naive_sum(black_box(&v))));
    c.bench_function("fast sum", |b| b.iter(|| fast_sum(black_box(&v))));
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
