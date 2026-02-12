//! LCSstr (Longest Common Substring) algorithm
//!
//! Unlike LCSseq which finds non-contiguous subsequences,
//! LCSstr finds the longest contiguous substring.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lcs_core::lcsstr_fast;

/// Get string length efficiently
#[inline(always)]
fn str_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// LCS (substring) similarity: returns the length of the longest common substring
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_str_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let lcs = lcsstr_fast(s1, s2);

    match score_cutoff {
        Some(cutoff) if lcs < cutoff => 0,
        _ => lcs,
    }
}

/// LCS (substring) distance: len(s1) + len(s2) - 2 * LCSstr
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_str_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 { return 0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let lcs = lcsstr_fast(s1, s2);
    let dist = len1 + len2 - 2 * lcs;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Normalized LCS (substring) similarity: 2 * LCSstr / (len(s1) + len(s2))
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_str_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 1.0; }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let total = len1 + len2;
    
    if total == 0 { return 1.0; }
    
    let lcs = lcsstr_fast(s1, s2);
    let sim = (2.0 * lcs as f64) / total as f64;
    
    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0.0,
        _ => sim,
    }
}

/// Normalized LCS (substring) distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_str_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_sim = lcs_str_normalized_similarity(s1, s2, None);
    let norm_dist = 1.0 - norm_sim;
    
    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_str_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| lcs_str_similarity(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_str_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| lcs_str_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_str_similarity() {
        // "abcde" vs "bcd" -> longest common substring is "bcd" = 3
        assert_eq!(lcs_str_similarity("abcde", "bcd", None), 3);
        // "abc" vs "abc" -> "abc" = 3
        assert_eq!(lcs_str_similarity("abc", "abc", None), 3);
        // "abc" vs "def" -> no common substring = 0
        assert_eq!(lcs_str_similarity("abc", "def", None), 0);
    }
    
    #[test]
    fn test_lcs_str_vs_lcs_seq() {
        // LCSstr is contiguous: "ace" in "abcde" vs "ace" 
        // LCSseq would find 3 ("ace"), but LCSstr finds only 1 ('a', 'c', or 'e')
        assert_eq!(lcs_str_similarity("abcde", "ace", None), 1);
    }
}
