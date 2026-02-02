//! Prefix similarity algorithm
//!
//! Measures the length of the common prefix between two strings.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};

/// Calculate common prefix length.
#[inline(always)]
fn prefix_length_internal(s1: &str, s2: &str) -> usize {
    s1.chars()
        .zip(s2.chars())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Calculate Prefix distance.
/// distance = max(len1, len2) - prefix_length
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn prefix_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let prefix = prefix_length_internal(s1, s2);
    let dist = max(len1, len2) - prefix;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Prefix similarity (common prefix length).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn prefix_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let sim = prefix_length_internal(s1, s2);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Prefix distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn prefix_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = max(len1, len2);

    if max_len == 0 {
        return 0.0;
    }

    let prefix = prefix_length_internal(s1, s2);
    let dist = (max_len - prefix) as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Calculate normalized Prefix similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn prefix_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = prefix_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn prefix_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| prefix_similarity(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn prefix_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| prefix_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix() {
        assert_eq!(prefix_length_internal("prefix", "pretest"), 3);
        assert_eq!(prefix_length_internal("abc", "xyz"), 0);
        assert_eq!(prefix_length_internal("hello", "hello"), 5);
    }
}
