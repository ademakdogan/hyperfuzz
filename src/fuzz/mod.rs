//! Fuzz algorithms module
//!
//! Provides fuzzy string matching functions similar to RapidFuzz fuzz module.

use pyo3::prelude::*;

pub mod ratio;
pub mod partial_ratio;
pub mod token_ratio;
pub mod partial_token_ratio;
mod wratio;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Ratio
    m.add_function(wrap_pyfunction!(ratio::ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ratio::ratio_batch, m)?)?;

    // Partial Ratio
    m.add_function(wrap_pyfunction!(partial_ratio::partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_ratio::partial_ratio_batch, m)?)?;

    // Token Ratio variants
    m.add_function(wrap_pyfunction!(token_ratio::token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_ratio::token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_ratio::token_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_ratio::token_sort_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(token_ratio::token_set_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(token_ratio::token_ratio_batch, m)?)?;

    // Partial Token Ratio variants
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_sort_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_set_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(partial_token_ratio::partial_token_ratio_batch, m)?)?;

    // WRatio and QRatio
    m.add_function(wrap_pyfunction!(wratio::w_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(wratio::q_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(wratio::w_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(wratio::q_ratio_batch, m)?)?;

    Ok(())
}
