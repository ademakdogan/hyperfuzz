//! Indel distance algorithm - Using lcs_core

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lcs_core::lcs_fast;

/// Get string length efficiently
#[inline(always)]
fn str_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// Calculate the Indel distance (insertions + deletions needed).
/// Formula: len(s1) + len(s2) - 2 * LCS(s1, s2)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 { return 0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let lcs = lcs_fast(s1, s2);
    let dist = len1 + len2 - 2 * lcs;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Indel similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 {
        return str_len(s1) * 2;
    }
    
    let lcs = lcs_fast(s1, s2);
    let sim = 2 * lcs;

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Indel distance (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 0.0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let total = len1 + len2;

    if total == 0 { return 0.0; }

    let lcs = lcs_fast(s1, s2);
    let dist = (len1 + len2 - 2 * lcs) as f64 / total as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Calculate normalized Indel similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = indel_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn indel_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| indel_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn indel_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| indel_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indel_distance() {
        assert_eq!(indel_distance("abcde", "ace", None), 2);
    }
}
