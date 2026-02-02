//! Postfix (suffix) similarity algorithm
//!
//! Measures the length of the common suffix between two strings.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

/// Calculate common postfix (suffix) length.
#[inline(always)]
fn postfix_length_internal(s1: &str, s2: &str) -> usize {
    s1.chars()
        .rev()
        .zip(s2.chars().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Calculate Postfix distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn postfix_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let postfix = postfix_length_internal(s1, s2);
    let dist = max(len1, len2) - postfix;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Postfix similarity (common suffix length).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn postfix_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let sim = postfix_length_internal(s1, s2);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Postfix distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn postfix_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = max(len1, len2);

    if max_len == 0 {
        return 0.0;
    }

    let postfix = postfix_length_internal(s1, s2);
    let dist = (max_len - postfix) as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Calculate normalized Postfix similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn postfix_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = postfix_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn postfix_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| postfix_similarity(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn postfix_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| postfix_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postfix() {
        assert_eq!(postfix_length_internal("testing", "running"), 3); // "ing"
        assert_eq!(postfix_length_internal("abc", "xyz"), 0);
        assert_eq!(postfix_length_internal("hello", "hello"), 5);
    }
}
