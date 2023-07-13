## A Complete Rust Example

To run this example, add the following code to your `Cargo.toml` file.
```toml
[dependencies]
forust-ml = "0.2.18"
polars = "0.28"
reqwest = { version = "0.11", features = ["blocking"] }
```

The following is a runable example using `polars` for data processing. The actual data manipulation can be performed with any tool, the only vital part, is the data be in column major format.
```rust
use forust_ml::{GradientBooster, Matrix};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::error::Error;
use std::io::Cursor;

fn main() -> Result<(), Box<dyn Error>> {
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

    // Get data in column major format...
    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.melt(id_vars, ["pclass", "age", "sibsp", "parch", "fare"])?;

    let data: Vec<f64> = mdf
        .select_at_idx(1)
        .expect("Invalid column")
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let y: Vec<f64> = df
        .column("survived")?
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    // Create Matrix from ndarray.
    let matrix = Matrix::new(&data, y.len(), 5);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = GradientBooster::default().set_learning_rate(0.3);
    model.fit_unweighted(&matrix, &y, None)?;

    // Predict output.
    println!("{:?} ...", &model.predict(&matrix, true)[0..10]);
    Ok(())
}
```
```
[-1.275806741323322, 0.9178487278986722, -1.4758225567638874, 1.0830510996747762, -1.7252372093498707, -1.4195771454833448, -0.27499967138282955, -0.9451315118931234, -0.08839774504303932, 1.374593096319586] ...
```

We first read in the data, and then, generate a contiguous matrix, that is used for training the booster. At this point, we can then instantiate out gradient booster, using the default parameters. These can be adjusted using the relevant `set_` methods, for any parameters of interest ([see here](src/gradientbooster.rs#L278) for all such methods).
