//! Distance algorithms module
//!
//! Provides various string distance metrics.

use pyo3::prelude::*;

mod levenshtein;
mod damerau_levenshtein;
mod hamming;
pub mod jaro;
mod jaro_winkler;
mod indel;
mod lcs_seq;
mod osa;
mod prefix;
mod postfix;

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

    // Damerau-Levenshtein
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein::damerau_levenshtein_normalized_similarity_batch, m)?)?;

    // Hamming
    m.add_function(wrap_pyfunction!(hamming::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hamming::hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming::hamming_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hamming::hamming_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming::hamming_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hamming::hamming_normalized_similarity_batch, m)?)?;

    // Jaro
    m.add_function(wrap_pyfunction!(jaro::jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro::jaro_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro::jaro_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro::jaro_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro::jaro_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(jaro::jaro_distance_batch, m)?)?;

    // Jaro-Winkler
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_distance_batch, m)?)?;

    // Indel
    m.add_function(wrap_pyfunction!(indel::indel_distance, m)?)?;
    m.add_function(wrap_pyfunction!(indel::indel_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(indel::indel_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(indel::indel_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(indel::indel_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(indel::indel_normalized_similarity_batch, m)?)?;

    // LCSseq
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_seq::lcs_seq_normalized_similarity_batch, m)?)?;

    // OSA
    m.add_function(wrap_pyfunction!(osa::osa_distance, m)?)?;
    m.add_function(wrap_pyfunction!(osa::osa_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(osa::osa_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(osa::osa_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(osa::osa_distance_batch, m)?)?;
    m.add_function(wrap_pyfunction!(osa::osa_normalized_similarity_batch, m)?)?;

    // Prefix
    m.add_function(wrap_pyfunction!(prefix::prefix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(prefix::prefix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(prefix::prefix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(prefix::prefix_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(prefix::prefix_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prefix::prefix_normalized_similarity_batch, m)?)?;

    // Postfix
    m.add_function(wrap_pyfunction!(postfix::postfix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(postfix::postfix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(postfix::postfix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(postfix::postfix_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(postfix::postfix_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(postfix::postfix_normalized_similarity_batch, m)?)?;

    Ok(())
}
