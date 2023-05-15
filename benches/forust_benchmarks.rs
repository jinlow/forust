use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forust_ml::binning::bin_matrix;
use forust_ml::constraints::ConstraintMap;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster;
use forust_ml::objective::{LogLoss, ObjectiveFunction};
use forust_ml::sampler::SampleMethod;
use forust_ml::splitter::MissingImputerSplitter;
use forust_ml::tree::Tree;
use forust_ml::utils::{fast_f64_sum, fast_sum, naive_sum};
use std::fs;
use std::time::Duration;

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

    let v: Vec<f32> = vec![10.; 300000];
    c.bench_function("Niave Sum", |b| b.iter(|| naive_sum(black_box(&v))));
    c.bench_function("fast sum", |b| b.iter(|| fast_sum(black_box(&v))));
    c.bench_function("fast f64 sum", |b| b.iter(|| fast_f64_sum(black_box(&v))));

    c.bench_function("calc_grad", |b| {
        b.iter(|| LogLoss::calc_grad(black_box(&y), black_box(&yhat), black_box(&w)))
    });

    c.bench_function("calc_hess", |b| {
        b.iter(|| LogLoss::calc_hess(black_box(&y), black_box(&yhat), black_box(&w)))
    });

    let data = Matrix::new(&data_vec, y.len(), 5);
    let splitter = MissingImputerSplitter {
        l2: 1.0,
        gamma: 3.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
        allow_missing_splits: true,
        constraints_map: ConstraintMap::new(),
    };
    let mut tree = Tree::new();

    let bindata = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
    let bdata = Matrix::new(&bindata.binned_data, data.rows, data.cols);
    tree.fit(
        &bdata,
        data.index.to_owned(),
        &bindata.cuts,
        &g,
        &h,
        &splitter,
        usize::MAX,
        5,
        true,
        &SampleMethod::None,
    );
    println!("{}", tree.nodes.len());
    c.bench_function("Train Tree", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();
            train_tree.fit(
                black_box(&bdata),
                black_box(data.index.to_owned()),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&splitter),
                black_box(usize::MAX),
                black_box(10),
                black_box(false),
                black_box(&SampleMethod::None),
            );
        })
    });
    c.bench_function("Tree Predict (Single Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(false), black_box(&f64::NAN)))
    });
    c.bench_function("Tree Predict (Multi Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(true), black_box(&f64::NAN)))
    });

    // Gradient Booster
    // Bench building
    let mut booster_train = c.benchmark_group("train-booster");
    booster_train.warm_up_time(Duration::from_secs(10));
    booster_train.sample_size(50);
    // booster_train.sampling_mode(SamplingMode::Linear);
    booster_train.bench_function("Train Booster", |b| {
        b.iter(|| {
            let mut booster = GradientBooster::default().set_parallel(false);
            booster
                .fit(
                    black_box(&data),
                    black_box(&y),
                    black_box(&w),
                    black_box(None),
                )
                .unwrap();
        })
    });
    let mut booster = GradientBooster::default();
    booster.fit(&data, &y, &w, None).unwrap();
    booster_train.bench_function("Predict Booster", |b| {
        b.iter(|| booster.predict(black_box(&data), false))
    });
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
