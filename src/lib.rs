//! HyperFuzz - High-performance string similarity algorithms
//!
//! This crate provides fast string similarity calculations implemented in Rust
//! with Python bindings via PyO3.

use pyo3::prelude::*;

mod distance;
mod fuzz;

/// HyperFuzz Python module
#[pymodule]
fn _hyperfuzz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Distance submodule
    let distance_module = PyModule::new(m.py(), "distance")?;
    distance::register_module(&distance_module)?;
    m.add_submodule(&distance_module)?;

    // Fuzz submodule
    let fuzz_module = PyModule::new(m.py(), "fuzz")?;
    fuzz::register_module(&fuzz_module)?;
    m.add_submodule(&fuzz_module)?;

    Ok(())
}
