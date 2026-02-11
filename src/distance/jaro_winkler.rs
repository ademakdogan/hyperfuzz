//! Jaro-Winkler similarity algorithm - Optimized
//!
//! Extension of Jaro similarity with prefix bonus.
//! Optimizations: zero-allocation prefix calculation, early exits

use pyo3::prelude::*;
use rayon::prelude::*;

use super::jaro::jaro_similarity_raw;

/// Calculate common prefix length (max 4 characters) - zero allocation
#[inline(always)]
fn common_prefix_length(s1: &str, s2: &str) -> usize {
    // Fast path for ASCII
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        let max_len = b1.len().min(b2.len()).min(4);
        
        for i in 0..max_len {
            if b1[i] != b2[i] {
                return i;
            }
        }
        return max_len;
    }
    
    // Unicode path - iterate without allocation
    let mut prefix_len = 0;
    for (c1, c2) in s1.chars().zip(s2.chars()).take(4) {
        if c1 == c2 {
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
    // Fast path for identical strings
    if s1 == s2 {
        return if s1.is_empty() { 1.0 } else { 1.0 };
    }
    
    // Fast path for empty strings
    if s1.is_empty() || s2.is_empty() {
        return if s1.is_empty() && s2.is_empty() { 1.0 } else { 0.0 };
    }
    
    let jaro_sim = jaro_similarity_raw(s1, s2);
    
    // Only apply prefix bonus if Jaro similarity is high enough
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
    
    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler_similarity_internal("test", "test", 0.1) - 1.0).abs() < 0.001);
    }
}
