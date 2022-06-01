use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forust::binning::bin_matrix;
use forust::data::Matrix;
use forust::gradientbooster::GradientBooster;
use forust::histsplitter::HistogramSplitter;
use forust::objective::{LogLoss, ObjectiveFunction};
use forust::tree::Tree;
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
    let mut index = data.index.to_owned();
    let index = index.as_mut();

    let bindata = bin_matrix(&data, &w, 300).unwrap();
    let bdata = Matrix::new(&bindata.binned_data, data.rows, data.cols);

    tree.fit(
        &bdata,
        &bindata.cuts,
        &g,
        &h,
        &splitter,
        usize::MAX,
        5,
        index,
        true,
    );
    println!("{}", tree.nodes.len());
    c.bench_function("Train Tree", |b| {
        b.iter(|| {
            let mut train_tree: Tree<f64> = Tree::new();
            train_tree.fit(
                black_box(&bdata),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&splitter),
                black_box(usize::MAX),
                black_box(5),
                black_box(index),
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
                .fit(
                    black_box(&data),
                    black_box(&y),
                    black_box(&w),
                    black_box(true),
                )
                .unwrap();
        })
    });
    let mut booster = GradientBooster::default();
    booster.fit(&data, &y, &w, true).unwrap();
    c.bench_function("Predict Booster", |b| {
        b.iter(|| booster.predict(&data, true))
    });
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
