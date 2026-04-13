//! An example using the `titanic` dataset

use forust_ml::{GradientBooster, Matrix};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::error::Error;
use std::io::Cursor;

fn main() -> Result<(), Box<dyn Error>> {
    let data = Vec::from_iter(
        Client::new()
            .get("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
            .send()?
            .text()?
            .bytes(),
    );

    let df = CsvReader::new(Cursor::new(data))
        .has_header(true)
        .finish()?
        .select(["survived", "pclass", "age", "sibsp", "parch", "fare"])?;

    // Get data in column major format...
    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.melt(id_vars, ["pclass", "age", "sibsp", "parch", "fare"])?;

    let data = Vec::from_iter(
        mdf.select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    let y = Vec::from_iter(
        df.column("survived")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // Create Matrix from ndarray.
    let matrix = Matrix::new(&data, y.len(), 5);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = GradientBooster::default().set_learning_rate(0.3);
    model.fit_unweighted(&matrix, &y, None)?;

    println!(
        "Model prediction: {:?} ...",
        &model.predict(&matrix, true)[0..10]
    );

    Ok(())
}
