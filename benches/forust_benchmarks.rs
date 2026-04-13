use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forust_ml::binning::bin_matrix;
use forust_ml::constraints::ConstraintMap;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::{GradientBooster, GrowPolicy};
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
    let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, &w);

    let v: Vec<f32> = vec![10.; 300000];
    c.bench_function("Niave Sum", |b| b.iter(|| naive_sum(black_box(&v))));
    c.bench_function("fast sum", |b| b.iter(|| fast_sum(black_box(&v))));
    c.bench_function("fast f64 sum", |b| b.iter(|| fast_f64_sum(black_box(&v))));

    c.bench_function("calc_grad_hess", |b| {
        b.iter(|| LogLoss::calc_grad_hess(black_box(&y), black_box(&yhat), black_box(&w)))
    });

    let data = Matrix::new(&data_vec, y.len(), 5);
    let splitter = MissingImputerSplitter {
        l1: 0.0,
        l2: 1.0,
        max_delta_step: 0.,
        gamma: 3.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
        allow_missing_splits: true,
        constraints_map: ConstraintMap::new(),
    };
    let mut tree = Tree::new();

    let bindata = bin_matrix(&data, &w, 300, f64::NAN).unwrap();
    let bdata = Matrix::new(&bindata.binned_data, data.rows, data.cols);
    let col_index: Vec<usize> = (0..data.cols).collect();
    tree.fit(
        &bdata,
        &mut data.index.to_owned(),
        &col_index,
        &bindata.cuts,
        &g,
        &h,
        &splitter,
        usize::MAX,
        5,
        true,
        &SampleMethod::None,
        &GrowPolicy::DepthWise,
    );
    println!("{}", tree.nodes.len());
    c.bench_function("Train Tree", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();
            train_tree.fit(
                black_box(&bdata),
                black_box(&mut data.index.to_owned()),
                black_box(&col_index),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&splitter),
                black_box(usize::MAX),
                black_box(10),
                black_box(false),
                black_box(&SampleMethod::None),
                black_box(&GrowPolicy::DepthWise),
            );
        })
    });
    c.bench_function("Train Tree - column subset", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();
            train_tree.fit(
                black_box(&bdata),
                black_box(&mut data.index.to_owned()),
                black_box(&[1, 3, 4]),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(&h),
                black_box(&splitter),
                black_box(usize::MAX),
                black_box(10),
                black_box(false),
                black_box(&SampleMethod::None),
                black_box(&GrowPolicy::DepthWise),
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
    booster_train.bench_function("Train Booster - Column Sampling", |b| {
        b.iter(|| {
            let mut booster = GradientBooster::default()
                .set_parallel(false)
                .set_colsample_bytree(0.5);
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
    booster_train.finish();

    // ---- Large realistic benchmarks: 500k rows x 200 columns ----
    let file = fs::read_to_string("resources/large_bench_500k_200col.csv")
        .expect("Something went wrong reading the file");
    let large_data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let file = fs::read_to_string("resources/large_bench_500k_200col_y.csv")
        .expect("Something went wrong reading the file");
    let large_y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let large_yhat = vec![0.5; large_y.len()];
    let large_w = vec![1.; large_y.len()];
    let (large_g, large_h) = LogLoss::calc_grad_hess(&large_y, &large_yhat, &large_w);

    let large_data = Matrix::new(&large_data_vec, large_y.len(), 200);
    let large_splitter = MissingImputerSplitter {
        l1: 0.0,
        l2: 1.0,
        max_delta_step: 0.,
        gamma: 3.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
        allow_missing_splits: true,
        constraints_map: ConstraintMap::new(),
    };

    let large_bindata = bin_matrix(&large_data, &large_w, 300, f64::NAN).unwrap();
    let large_bdata = Matrix::new(&large_bindata.binned_data, large_data.rows, large_data.cols);
    let large_col_index: Vec<usize> = (0..large_data.cols).collect();

    let mut large_group = c.benchmark_group("large-500k-200col");
    large_group.warm_up_time(Duration::from_secs(15));
    large_group.measurement_time(Duration::from_secs(30));
    large_group.sample_size(10);

    large_group.bench_function("Train Tree (large)", |b| {
        b.iter(|| {
            let mut train_tree = Tree::new();
            train_tree.fit(
                black_box(&large_bdata),
                black_box(&mut large_data.index.to_owned()),
                black_box(&large_col_index),
                black_box(&large_bindata.cuts),
                black_box(&large_g),
                black_box(&large_h),
                black_box(&large_splitter),
                black_box(usize::MAX),
                black_box(8),
                black_box(true),
                black_box(&SampleMethod::None),
                black_box(&GrowPolicy::DepthWise),
            );
        })
    });

    large_group.bench_function("Train Booster 100 iters (large)", |b| {
        b.iter(|| {
            let mut booster = GradientBooster::default()
                .set_iterations(100)
                .set_parallel(true);
            booster
                .fit(
                    black_box(&large_data),
                    black_box(&large_y),
                    black_box(&large_w),
                    black_box(None),
                )
                .unwrap();
        })
    });

    let mut large_booster = GradientBooster::default().set_iterations(100);
    large_booster
        .fit(&large_data, &large_y, &large_w, None)
        .unwrap();

    large_group.bench_function("Predict Booster (large)", |b| {
        b.iter(|| large_booster.predict(black_box(&large_data), false))
    });

    large_group.bench_function("Predict Booster parallel (large)", |b| {
        b.iter(|| large_booster.predict(black_box(&large_data), true))
    });

    large_group.finish();
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
