//! Jaro-Winkler similarity algorithm
//!
//! Extension of Jaro similarity with prefix bonus.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;

use super::jaro::jaro_similarity_raw;

/// Calculate common prefix length (max 4 characters).
#[inline(always)]
fn common_prefix_length(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().take(4).collect();
    let s2_chars: Vec<char> = s2.chars().take(4).collect();
    
    let max_len = min(s1_chars.len(), s2_chars.len());
    let max_len = min(max_len, 4);

    let mut prefix_len = 0;
    for i in 0..max_len {
        if s1_chars[i] == s2_chars[i] {
            prefix_len += 1;
        } else {
            break;
        }
    }

    prefix_len
}

/// Calculate Jaro-Winkler similarity.
/// 
/// Adds a prefix bonus to Jaro similarity when the strings have a common prefix.
#[inline(always)]
fn jaro_winkler_similarity_internal(s1: &str, s2: &str, prefix_weight: f64) -> f64 {
    let jaro_sim = jaro_similarity_raw(s1, s2);
    
    if jaro_sim < 0.7 {
        return jaro_sim;
    }

    let prefix_len = common_prefix_length(s1, s2) as f64;
    
    jaro_sim + (prefix_len * prefix_weight * (1.0 - jaro_sim))
}

/// Calculate Jaro-Winkler similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_similarity(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let sim = jaro_winkler_similarity_internal(s1, s2, prefix_weight);
    
    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0.0,
        _ => sim,
    }
}

/// Calculate Jaro-Winkler distance (1 - similarity).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_distance(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let dist = 1.0 - jaro_winkler_similarity_internal(s1, s2, prefix_weight);
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Alias for jaro_winkler_similarity (normalized)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_normalized_similarity(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    jaro_winkler_similarity(s1, s2, prefix_weight, score_cutoff)
}

/// Alias for jaro_winkler_distance (normalized)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_normalized_distance(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    jaro_winkler_distance(s1, s2, prefix_weight, score_cutoff)
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_similarity_batch(
    pairs: Vec<(String, String)>,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| jaro_winkler_similarity(s1, s2, prefix_weight, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, prefix_weight=0.1, score_cutoff=None))]
pub fn jaro_winkler_distance_batch(
    pairs: Vec<(String, String)>,
    prefix_weight: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| jaro_winkler_distance(s1, s2, prefix_weight, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_winkler() {
        let sim = jaro_winkler_similarity_internal("MARTHA", "MARHTA", 0.1);
        assert!((sim - 0.961).abs() < 0.01);
    }

    #[test]
    fn test_common_prefix() {
        assert_eq!(common_prefix_length("prefix", "pretest"), 3);
        assert_eq!(common_prefix_length("abcdef", "abc"), 3);
        assert_eq!(common_prefix_length("abcdef", "abcdef"), 4); // Max 4
    }
}
