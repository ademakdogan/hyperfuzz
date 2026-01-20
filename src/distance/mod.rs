//! Distance algorithms module
//!
//! Provides various string distance metrics.

use pyo3::prelude::*;

mod levenshtein;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Levenshtein
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_normalized_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_normalized_similarity_batch, m)?)?;

    Ok(())
}
