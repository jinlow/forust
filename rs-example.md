Cargo.toml
```toml
[package]
name = "forust-rust-example"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
forust-ml = {version="0.1.6", path="../forust/"}
polars = {version = "0.24.1", features = ["ndarray"]}
reqwest = { version = "0.11.11", features = ["blocking"] }
color-eyre = "0.6.2"
ndarray = "0.15.6"
```


Rust Code
```rust
use color_eyre::Result;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster;
use ndarray::Axis;
use polars::prelude::*;
use reqwest::blocking::Client;
use std::io::Cursor;

fn main() -> Result<()> {
    let data: Vec<u8> = Client::new()
        .get("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        .send()?
        .text()?
        .bytes()
        .collect();

    let df = CsvReader::new(Cursor::new(data))
        .has_header(true)
        .finish()?
        .select(["survived", "pclass", "age", "sibsp", "parch", "fare"])?;

    let mat = df.to_ndarray::<Float64Type>()?;

    let y = mat.select(Axis(1), &[0]).into_raw_vec();

    // Unravel data, into single vector of data, in columner format.
    let data = mat.select(Axis(1), &[1, 2, 3, 4, 5]).reversed_axes();
    let shape = data.shape();
    let data = data
        .to_shape((shape[0] * shape[1], 1))?
        .select(Axis(1), &[0])
        .into_raw_vec();

    // Create booster
    // To provide parameters generate a default booster, and then use
    // the relevant "set_" methods for any parameters you would like to
    // adjust.
    let mut gb = GradientBooster::default().set_learning_rate(0.3);
    let gb_matrix = Matrix::new(&data, shape[1], shape[0]);
    gb.fit_unweighted(&gb_matrix, &y)?;

    // Predict output.
    println!("{:?}", gb.predict(&gb_matrix, true));
    Ok(())
}
```