use forust::binning::bin_matrix;
use forust::data::Matrix;
use forust::gradientbooster::GradientBooster as CrateGradienBooster;
use forust::objective::ObjectiveType;
use forust::utils::percentiles as crate_percentiles;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// 32 bit implementation
#[pyclass(subclass)]
struct GradientBoosterF32 {
    booster: CrateGradienBooster<f32>,
}

#[pymethods]
impl GradientBoosterF32 {
    #[new]
    pub fn new(
        objective_type: &str,
        iterations: usize,
        learning_rate: f32,
        max_depth: usize,
        max_leaves: usize,
        l2: f32,
        gamma: f32,
        min_leaf_weight: f32,
        base_score: f32,
        nbins: u16,
        parallel: bool,
    ) -> Self {
        let objective_ = if objective_type == "LogLoss" {
            ObjectiveType::LogLoss
        } else if objective_type == "SquaredLoss" {
            ObjectiveType::SquaredLoss
        } else {
            panic!("Not a valid objective type provided.")
        };
        let booster = CrateGradienBooster::new(
            objective_,
            iterations,
            learning_rate,
            max_depth,
            max_leaves,
            l2,
            gamma,
            min_leaf_weight,
            base_score,
            nbins,
            parallel,
        );
        GradientBoosterF32 { booster }
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f32>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f32>,
        sample_weight: PyReadonlyArray1<f32>,
        parallel: Option<bool>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight = sample_weight.as_slice()?;
        let parallel = match parallel {
            None => true,
            Some(v) => v,
        };
        self.booster
            .fit(&data, &y, &sample_weight, parallel)
            .unwrap();
        Ok(())
    }
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f32>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = match parallel {
            None => true,
            Some(v) => v,
        };
        Ok(self.booster.predict(&data, parallel).into_pyarray(py))
    }

    pub fn text_dump(&self) -> PyResult<Vec<String>> {
        let mut trees = Vec::new();
        for t in &self.booster.trees {
            trees.push(format!("{}", t));
        }
        return Ok(trees);
    }
}

// 64 bit implementation
#[pyclass(subclass)]
struct GradientBoosterF64 {
    booster: CrateGradienBooster<f64>,
}

#[pymethods]
impl GradientBoosterF64 {
    #[new]
    pub fn new(
        objective_type: &str,
        iterations: usize,
        learning_rate: f64,
        max_depth: usize,
        max_leaves: usize,
        l2: f64,
        gamma: f64,
        min_leaf_weight: f64,
        base_score: f64,
        nbins: u16,
        parallel: bool,
    ) -> Self {
        let objective_ = if objective_type == "LogLoss" {
            ObjectiveType::LogLoss
        } else if objective_type == "SquaredLoss" {
            ObjectiveType::SquaredLoss
        } else {
            panic!("Not a valid objective type provided.")
        };
        let booster = CrateGradienBooster::new(
            objective_,
            iterations,
            learning_rate,
            max_depth,
            max_leaves,
            l2,
            gamma,
            min_leaf_weight,
            base_score,
            nbins,
            parallel,
        );
        GradientBoosterF64 { booster }
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        sample_weight: PyReadonlyArray1<f64>,
        parallel: Option<bool>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight = sample_weight.as_slice()?;
        let parallel = match parallel {
            None => true,
            Some(v) => v,
        };
        self.booster
            .fit(&data, &y, &sample_weight, parallel)
            .unwrap();
        Ok(())
    }
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = match parallel {
            None => true,
            Some(v) => v,
        };
        Ok(self.booster.predict(&data, parallel).into_pyarray(py))
    }

    pub fn text_dump(&self) -> PyResult<Vec<String>> {
        let mut trees = Vec::new();
        for t in &self.booster.trees {
            trees.push(format!("{}", t));
        }
        return Ok(trees);
    }
}

#[pyfunction]
fn rust_bin_matrix<'py>(
    py: Python<'py>,
    flat_data: PyReadonlyArray1<f32>,
    rows: usize,
    cols: usize,
    sample_weight: PyReadonlyArray1<f32>,
    nbins: u16,
) -> PyResult<(&'py PyArray1<u16>, Vec<Vec<f32>>, Vec<usize>)> {
    let flat_data = flat_data.as_slice()?;
    let sample_weight = sample_weight.as_slice()?;
    let data = Matrix::new(flat_data, rows, cols);
    let r = bin_matrix(&data, sample_weight, nbins).unwrap();
    Ok((r.binned_data.into_pyarray(py), r.cuts, r.nunique))
}

#[pyfunction]
fn print_matrix(x: PyReadonlyArray1<f32>, rows: usize, cols: usize) -> PyResult<()> {
    let m = Matrix::new(x.as_slice()?, rows, cols);
    println!("{}", m);
    Ok(())
}

#[pyfunction]
fn percentiles<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<f32>,
    sample_weight: PyReadonlyArray1<f32>,
    percentiles: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let v_ = v.as_slice()?;
    let sample_weight_ = sample_weight.as_slice()?;
    let percentiles_ = percentiles.as_slice()?;
    let p = crate_percentiles(v_, sample_weight_, percentiles_);
    Ok(p.into_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn forust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(rust_bin_matrix, m)?)?;
    m.add_class::<GradientBoosterF32>()?;
    m.add_class::<GradientBoosterF64>()?;
    Ok(())
}
