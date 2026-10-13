# Contributing to `forust`

## Development Setup
For development, it is assumed you have stable rust installed, and at least python 3.9. Additionally, in your python environment, you will need to install [`maturin`](https://github.com/PyO3/maturin).

### To run Rust tests
The rust test depend on some artifacts that are generated from a python script. You can either follow the steps in the python tests section, or run the following in an environment that is running python.

```sh
python -m pip install numpy pandas seaborn
python scripts/make_resources.py
```

If you have rust and the cargo package manager installed all you need to do to run the rust tests is in the root of the repository run the following command.
```sh
cargo test
```

### To run the python tests

Prior to running the tests, you should install `py-forust` in editable mode. To do this, from the project root directory you can run the following.

```sh
cd py-forust
# Install the project in editable mode and all development dependencies
python -m pip install -e .[dev]

# You can now return to the rood directory and run the tests...
cd ..

# Prior to running the tests, build all required test artifacts
python scripts/make_resources.py

# Now you can run the tests.
# on Linux...
source scripts/run-python-tests.sh
```

The test script can also be run from powershell.
```powershell
# on Windows (powershell)
.\scripts\run-python-tests.ps1
```

This script, builds the package in release mode, installs it, and then runs the test. Because of this, it is useful to run this whenever you want to test out a change in the python package.

## Preparing a Release

This repository has two Cargo packages and one Python package. The Python package gets its version dynamically from `py-forust/Cargo.toml`, so keep the versions synchronized:

- Set the Rust library version in the root `Cargo.toml`.
- Set the binding package version in `py-forust/Cargo.toml`.
- Set the `forust-ml` path dependency in `py-forust/Cargo.toml` to the same version.
- Leave `project.dynamic = ["version"]` in `py-forust/pyproject.toml`; maturin reads the version from Cargo.

Cargo and Python use different prerelease spellings. In Cargo manifests, use SemVer syntax such as `0.8.0-rc.1`. Maturin converts that to the PEP 440 Python version `0.8.0rc1` and produces wheel names such as `forust-0.8.0rc1-...whl`. Do not use `0.8.0rc1` directly in a Cargo manifest; Cargo rejects it.

Before tagging, validate the metadata and build a wheel:

```sh
cargo metadata --manifest-path py-forust/Cargo.toml --format-version 1
cargo check --manifest-path py-forust/Cargo.toml
maturin build --release --manifest-path py-forust/Cargo.toml --out dist
```

Create an annotated tag after committing and pushing the version changes:

```sh
git tag -a v0.8.0-rc.1 -m "Release 0.8.0rc1"
git push origin v0.8.0-rc.1
```

The CI workflow runs when a tag is pushed. It publishes the Rust crate and, after the platform builds pass, publishes the Python wheels and source distribution to PyPI. Published versions cannot be reused. If a tagged workflow fails after the Rust publish step, increment the candidate number (`rc.2`, `rc.3`, and so on) before trying again, even if the Python artifacts were never published.

## Benchmarking
Benchmarking is run using the [`criterion`](https://github.com/bheisler/criterion.rs) Rust crate.
To run the benchmarks, you can run the following command from your terminal.
```sh
cargo bench
```
specific benchmarks can be targeted by referring to them by name.
```sh
cargo bench "fast sum"
```

## Pre-commit
The [`pre-commit`](https://pre-commit.com/) framework should be installed and used to ensure all commits meet the required formatting, and linting checks prior to a commit being made to the repository.
```sh
# Install pre-commit, either right in your default python install
# or using a tool such as pipx (https://pypa.github.io/pipx/)
python -m pip install pre-commit

# In the root of the repository
pre-commit install
```

## Serialization
The saving and loading of the model is all handled by the [`serde`](https://docs.rs/serde/1.0.163/serde/) and [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) crates.

Because of this you will see the following attribue calls sprinkled throughout the package.
```rust
#[derive(Deserialize, Serialize)]
```

Additionally in order to not break backwards compatibility with models saved in previous versions, any new items added to the `Tree` or `GradientBooster` struts, should have a default value defined. This way models can be loaded, even if they were saved before the new fields was added.
A default value can be added for a fields using the `#[#[serde(default = "default_sample_method")]]` attribute. Where the string that default is referring to must be the name of a valid function, the following is a complete example of this.

```rust
use crate::sampler::SampleMethod, Sampler;

#[derive(Deserialize, Serialize)]
pub struct GradientBooster {
    // ...
    #[serde(default = "default_sample_method")]
    pub sample_method: SampleMethod,
    // ...
}

fn default_sample_method() -> SampleMethod {
    SampleMethod::None
}
```
