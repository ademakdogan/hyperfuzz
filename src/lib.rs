//! HyperFuzz - High-performance string similarity algorithms
//!
//! This crate provides fast string similarity calculations implemented in Rust
//! with Python bindings via PyO3.

use pyo3::prelude::*;

mod distance;
mod fuzz;
mod set_based;
mod alignment;
mod vector;
mod lcs_core;

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

    // Set-based algorithms
    m.add_function(wrap_pyfunction!(set_based::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::jaccard_distance, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::jaccard_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::sorensen_dice_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::sorensen_dice_distance, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::sorensen_dice_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::tversky_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::tversky_distance, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::tversky_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::overlap_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::overlap_distance, m)?)?;
    m.add_function(wrap_pyfunction!(set_based::overlap_similarity_batch, m)?)?;

    // Alignment algorithms
    m.add_function(wrap_pyfunction!(alignment::smith_waterman_score, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::smith_waterman_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::smith_waterman_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::needleman_wunsch_score, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::needleman_wunsch_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::needleman_wunsch_score_batch, m)?)?;

    // Vector methods
    m.add_function(wrap_pyfunction!(vector::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(vector::cosine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(vector::cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(vector::soft_tfidf_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(vector::soft_tfidf_distance, m)?)?;
    m.add_function(wrap_pyfunction!(vector::soft_tfidf_similarity_batch, m)?)?;

    Ok(())
}
