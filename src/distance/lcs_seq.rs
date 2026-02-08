//! LCSseq (Longest Common Subsequence) algorithm - Using lcs_core

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lcs_core::lcs_fast;

/// Get string length efficiently
#[inline(always)]
fn str_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// LCS distance: max(len1, len2) - LCS
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 { return 0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let lcs = lcs_fast(s1, s2);
    let dist = len1.max(len2) - lcs;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// LCS similarity: LCS length
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let lcs = lcs_fast(s1, s2);

    match score_cutoff {
        Some(cutoff) if lcs < cutoff => 0,
        _ => lcs,
    }
}

/// Normalized LCS distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 0.0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let max_len = len1.max(len2);

    if max_len == 0 { return 0.0; }

    let lcs = lcs_fast(s1, s2);
    let dist = (max_len - lcs) as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Normalized LCS similarity
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = lcs_seq_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_seq_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| lcs_seq_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_seq_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| lcs_seq_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_distance() {
        assert_eq!(lcs_seq_distance("abc", "abc", None), 0);
        assert_eq!(lcs_seq_distance("abc", "def", None), 3);
    }
}
