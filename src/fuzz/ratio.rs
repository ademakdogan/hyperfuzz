//! Ratio algorithm - Ultra Optimized using lcs_core
//!
//! Uses thread-local buffers from lcs_core for zero-allocation LCS computation.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lcs_core::{lcs_fast, ratio_from_lcs};

/// Calculate ratio using lcs_core optimizations
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast paths
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // Get lengths efficiently
    let (len1, len2) = if s1.is_ascii() && s2.is_ascii() {
        (s1.len(), s2.len())
    } else {
        (s1.chars().count(), s2.chars().count())
    };
    
    // Early rejection based on length ratio
    if let Some(cutoff) = score_cutoff {
        let max_possible = 100.0 * (2.0 * len1.min(len2) as f64) / ((len1 + len2) as f64);
        if max_possible < cutoff {
            return 0.0;
        }
    }

    let lcs = lcs_fast(s1, s2);
    let similarity = ratio_from_lcs(len1, len2, lcs);

    match score_cutoff {
        Some(cutoff) if similarity < cutoff => 0.0,
        _ => similarity,
    }
}

/// Re-export lcs_length for other modules
#[inline(always)]
pub fn lcs_length(s1: &str, s2: &str) -> usize {
    lcs_fast(s1, s2)
}

/// Calculate ratio for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("abcde", "ace"), 3);
        assert_eq!(lcs_length("abc", "abc"), 3);
        assert_eq!(lcs_length("abc", "def"), 0);
    }

    #[test]
    fn test_ratio() {
        let r = ratio("this is a test", "this is a test!", None);
        assert!((r - 96.55).abs() < 0.1);
    }

    #[test]
    fn test_ratio_identical() {
        assert!((ratio("test", "test", None) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_ratio_empty() {
        assert!((ratio("", "", None) - 100.0).abs() < 0.001);
    }
}
