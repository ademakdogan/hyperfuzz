//! Fuzz algorithms module
//!
//! Provides fuzzy string matching functions similar to RapidFuzz fuzz module.

use pyo3::prelude::*;

mod ratio;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Ratio functions
    m.add_function(wrap_pyfunction!(ratio::ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ratio::ratio_batch, m)?)?;

    Ok(())
}
