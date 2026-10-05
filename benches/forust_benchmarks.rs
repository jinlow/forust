use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
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

fn make_wide_data(rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let value = ((row * 31 + col * 17) % 1000) as f64 / 1000.0;
            data.push(value + col as f64 * 0.0001);
        }
    }

    let y = (0..rows).map(|row| (row % 2) as f64).collect();
    (data, y)
}

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
        data.index.to_owned(),
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
                black_box(data.index.to_owned()),
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
                black_box(data.index.to_owned()),
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
    drop(booster_train);

    let wide_rows = 20_000;
    let wide_cols = 200;
    let (wide_values, wide_y) = make_wide_data(wide_rows, wide_cols);
    let wide_data = Matrix::new(&wide_values, wide_rows, wide_cols);
    let wide_weights = vec![1.; wide_rows];
    let wide_yhat = vec![0.5; wide_rows];
    let (wide_grad, wide_hess) = LogLoss::calc_grad_hess(&wide_y, &wide_yhat, &wide_weights);
    let wide_binning = bin_matrix(&wide_data, &wide_weights, 64, f64::NAN).unwrap();
    let wide_bdata = Matrix::new(&wide_binning.binned_data, wide_rows, wide_cols);
    let wide_col_index: Vec<usize> = (0..wide_cols).collect();
    let wide_splitter = MissingImputerSplitter {
        l1: 0.0,
        l2: 1.0,
        max_delta_step: 0.,
        gamma: 0.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
        allow_missing_splits: true,
        constraints_map: ConstraintMap::new(),
    };
    let mut wide_benchmarks = c.benchmark_group("wide-200-column");
    wide_benchmarks.throughput(Throughput::Elements(wide_rows as u64));
    wide_benchmarks.warm_up_time(Duration::from_secs(3));
    wide_benchmarks.sample_size(10);

    wide_benchmarks.bench_function("Bin Matrix", |b| {
        b.iter(|| {
            bin_matrix(
                black_box(&wide_data),
                black_box(&wide_weights),
                black_box(64),
                black_box(f64::NAN),
            )
            .unwrap();
        })
    });

    for (label, parallel) in [("Serial", false), ("Parallel", true)] {
        wide_benchmarks.bench_function(format!("Train Tree - {}", label), |b| {
            b.iter(|| {
                let mut wide_tree = Tree::new();
                wide_tree.fit(
                    black_box(&wide_bdata),
                    black_box(wide_bdata.index.to_owned()),
                    black_box(&wide_col_index),
                    black_box(&wide_binning.cuts),
                    black_box(&wide_grad),
                    black_box(&wide_hess),
                    black_box(&wide_splitter),
                    black_box(usize::MAX),
                    black_box(6),
                    black_box(parallel),
                    black_box(&SampleMethod::None),
                    black_box(&GrowPolicy::DepthWise),
                );
            })
        });

        wide_benchmarks.bench_function(format!("Train Booster - {}", label), |b| {
            b.iter(|| {
                let mut wide_booster = GradientBooster::default()
                    .set_iterations(10)
                    .set_max_depth(6)
                    .set_nbins(64)
                    .set_parallel(parallel);
                wide_booster
                    .fit(
                        black_box(&wide_data),
                        black_box(&wide_y),
                        black_box(&wide_weights),
                        black_box(None),
                    )
                    .unwrap();
            })
        });

        wide_benchmarks.bench_function(
            format!("Train Booster - {} - Single Evaluation", label),
            |b| {
                b.iter(|| {
                    let mut wide_booster = GradientBooster::default()
                        .set_iterations(10)
                        .set_max_depth(6)
                        .set_nbins(64)
                        .set_parallel(parallel);
                    let evaluation_data = Some(vec![(
                        Matrix::new(&wide_values, wide_rows, wide_cols),
                        &wide_y[..],
                        &wide_weights[..],
                    )]);
                    wide_booster
                        .fit(
                            black_box(&wide_data),
                            black_box(&wide_y),
                            black_box(&wide_weights),
                            black_box(evaluation_data),
                        )
                        .unwrap();
                })
            },
        );
    }
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
