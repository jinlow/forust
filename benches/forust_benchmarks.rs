use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forust::data::Matrix;
use forust::exactsplitter::ExactSplitter;
use forust::objective::{LogLoss, ObjectiveFunction};
use forust::tree::Tree;
use std::fs;

pub fn predict_benchmarks(c: &mut Criterion) {
    let file = fs::read_to_string("resources/contiguous_no_missing_100k_samp_seed0.csv")
        .expect("Something went wrong reading the file");
    let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let file = fs::read_to_string("resources/performance_100k_samp_seed0.csv")
        .expect("Something went wrong reading the file");
    let y: Vec<f64> = file
        .lines()
        .map(|x| x.parse::<f64>().unwrap() as f64)
        .collect();
    let yhat = vec![0.5; y.len()];
    let g = LogLoss::calc_grad(&y, &yhat);
    let h = LogLoss::calc_hess(&y, &yhat);

    let data = Matrix::new(&data_vec, y.len(), 5);
    let splitter = ExactSplitter {
        l2: 1.0,
        gamma: 3.0,
        min_leaf_weight: 1.0,
        learning_rate: 0.3,
        min_split_gain: 0.0,
    };
    let mut tree = Tree::new();
    let mut index = data.index.to_owned();
    let index = index.as_mut();
    tree.fit(&data, &g, &h, &splitter, usize::MAX, 5, index);
    println!("{}", tree.nodes.len());
    c.bench_function("Tree Predict (Single Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(false)))
    });
    c.bench_function("Tree Predict (Multi Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(true)))
    });
}

criterion_group!(benches, predict_benchmarks);
criterion_main!(benches);
