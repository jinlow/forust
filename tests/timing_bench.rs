//! Timing benchmark for all multiclass tests.
//! Run with: cargo test --test timing_bench --release -- --nocapture

use forust_ml::objective::{ObjectiveType, SoftmaxMultiClass};
use forust_ml::sampler::SampleMethod;
use forust_ml::{GradientBooster, Matrix};
use std::time::Instant;

fn make_kclass_data(
    k: usize, n_per_class: usize, n_features: usize, separation: f64,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = k * n_per_class;
    let mut rows = Vec::with_capacity(n * n_features);
    let mut labels = Vec::with_capacity(n);
    for class in 0..k {
        for i in 0..n_per_class {
            let offset = (i as f64) * 0.02 - 1.0;
            for feat in 0..n_features {
                let center = if feat == 0 {
                    class as f64 * separation
                } else {
                    (class as f64 * separation * 0.5) + (feat as f64)
                };
                rows.push(center + offset);
            }
            labels.push(class as f64);
        }
    }
    let mut col_major = vec![0.0; n * n_features];
    for row in 0..n {
        for col in 0..n_features {
            col_major[col * n + row] = rows[row * n_features + col];
        }
    }
    (col_major, labels, n, n_features)
}

struct TimingResult {
    name: &'static str,
    duration_us: u128,
}

fn run_timed(name: &'static str, f: impl FnOnce()) -> TimingResult {
    let start = Instant::now();
    f();
    TimingResult { name, duration_us: start.elapsed().as_micros() }
}

#[test]
fn timing_all_tests() {
    let mut results: Vec<TimingResult> = Vec::new();

    // ── Softmax math ──
    results.push(run_timed("softmax K=3 (1 row)", || {
        for _ in 0..1000 {
            let _ = SoftmaxMultiClass::softmax(&[1.0, 2.0, 3.0], 3);
        }
    }));
    results.push(run_timed("softmax K=3 (1000 rows)", || {
        let scores: Vec<f64> = (0..3000).map(|i| (i as f64) * 0.001).collect();
        for _ in 0..100 {
            let _ = SoftmaxMultiClass::softmax(&scores, 3);
        }
    }));
    results.push(run_timed("softmax K=50 (100 rows)", || {
        let scores: Vec<f64> = (0..5000).map(|i| (i as f64) * 0.001).collect();
        for _ in 0..100 {
            let _ = SoftmaxMultiClass::softmax(&scores, 50);
        }
    }));
    results.push(run_timed("softmax extreme [-1000,0,1000]", || {
        for _ in 0..1000 {
            let _ = SoftmaxMultiClass::softmax(&[-1000.0, 0.0, 1000.0], 3);
        }
    }));

    // ── Grad/Hess ──
    results.push(run_timed("grad_hess K=3 N=1000", || {
        let y: Vec<f64> = (0..1000).map(|i| (i % 3) as f64).collect();
        let yhat = vec![0.0; 3000];
        let w = vec![1.0; 1000];
        for _ in 0..100 {
            let _ = SoftmaxMultiClass::calc_grad_hess(&y, &yhat, &w, 3);
        }
    }));
    results.push(run_timed("grad_hess K=10 N=1000", || {
        let y: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
        let yhat = vec![0.0; 10000];
        let w = vec![1.0; 1000];
        for _ in 0..100 {
            let _ = SoftmaxMultiClass::calc_grad_hess(&y, &yhat, &w, 10);
        }
    }));

    // ── CalcInit ──
    results.push(run_timed("calc_init K=3 N=10000", || {
        let y: Vec<f64> = (0..10000).map(|i| (i % 3) as f64).collect();
        let w = vec![1.0; 10000];
        for _ in 0..1000 {
            let _ = SoftmaxMultiClass::calc_init(&y, &w, 3);
        }
    }));

    // ── ValidateLabels ──
    results.push(run_timed("validate_labels N=10000", || {
        let y: Vec<f64> = (0..10000).map(|i| (i % 3) as f64).collect();
        for _ in 0..1000 {
            let _ = SoftmaxMultiClass::validate_labels(&y, 3);
        }
    }));

    // ── Training benchmarks ──
    results.push(run_timed("train K=3 N=300 iter=50", || {
        let (dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train K=3 N=5001 iter=20", || {
        let (dv, y, n, nc) = make_kclass_data(3, 1667, 4, 5.0);
        let n = 5001;
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(20).set_max_depth(5).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train K=10 N=500 iter=50", || {
        let (dv, y, n, nc) = make_kclass_data(10, 50, 3, 8.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 10; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train K=3 N=300 GOSS iter=30", || {
        let (dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(30).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.sample_method = SampleMethod::Goss; b.top_rate = 0.2; b.other_rate = 0.1;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train K=3 N=300 missing 10%", || {
        let (mut dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        for i in (0..dv.len()).step_by(10) { dv[i] = f64::NAN; }
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(30).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true; b.create_missing_branch = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    // ── Predict benchmarks ──
    results.push(run_timed("predict K=3 N=300 (50 trees)", || {
        let (dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
        for _ in 0..100 { let _ = b.predict(&d, true); }
    }));

    results.push(run_timed("predict_proba K=3 N=300", || {
        let (dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
        for _ in 0..100 { let _ = b.predict_proba(&d, true); }
    }));

    results.push(run_timed("predict K=10 N=500 (500 trees)", || {
        let (dv, y, n, nc) = make_kclass_data(10, 50, 3, 8.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 10; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
        for _ in 0..100 { let _ = b.predict(&d, true); }
    }));

    // ── JSON roundtrip ──
    results.push(run_timed("json roundtrip K=3 (150 trees)", || {
        let (dv, y, n, nc) = make_kclass_data(3, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(50).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
        for _ in 0..10 {
            let json = b.json_dump().unwrap();
            let _ = GradientBooster::from_json(&json).unwrap();
        }
    }));

    // ── Edge cases ──
    results.push(run_timed("train K=2 binary-via-softmax", || {
        let (dv, y, n, nc) = make_kclass_data(2, 100, 2, 5.0);
        let d = Matrix::new(&dv, n, nc);
        let w = vec![1.0; n];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(30).set_max_depth(4).set_learning_rate(0.3);
        b.num_classes = 2; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train single sample/class", || {
        let dv = vec![0.0, 5.0, 10.0, 0.0, 5.0, 0.0];
        let y = vec![0.0, 1.0, 2.0];
        let d = Matrix::new(&dv, 3, 2);
        let w = vec![1.0; 3];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(20).set_max_depth(2).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train all same class", || {
        let dv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0; 5];
        let d = Matrix::new(&dv, 5, 1);
        let w = vec![1.0; 5];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(10).set_max_depth(2).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    results.push(run_timed("train empty class (no label=2)", || {
        let dv = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let d = Matrix::new(&dv, 6, 1);
        let w = vec![1.0; 6];
        let mut b = GradientBooster::default()
            .set_objective_type(ObjectiveType::SoftmaxMultiClass)
            .set_iterations(20).set_max_depth(3).set_learning_rate(0.3);
        b.num_classes = 3; b.initialize_base_score = true;
        b.fit(&d, &y, &w, None).unwrap();
    }));

    // ── Print results ──
    println!("\n{}", "=".repeat(70));
    println!("  TIMING RESULTS (release build, single run)");
    println!("{}", "=".repeat(70));
    println!("{:<45} {:>10}", "Test", "Time");
    println!("{}", "-".repeat(57));
    for r in &results {
        let time_str = if r.duration_us < 1000 {
            format!("{} us", r.duration_us)
        } else if r.duration_us < 1_000_000 {
            format!("{:.2} ms", r.duration_us as f64 / 1000.0)
        } else {
            format!("{:.2} s", r.duration_us as f64 / 1_000_000.0)
        };
        println!("{:<45} {:>10}", r.name, time_str);
    }
    println!("{}", "-".repeat(57));
    let total: u128 = results.iter().map(|r| r.duration_us).sum();
    println!("{:<45} {:>10}", "TOTAL", format!("{:.2} ms", total as f64 / 1000.0));
}
