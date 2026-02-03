//! WRatio and QRatio algorithms
//!
//! WRatio: Weighted ratio with heuristics based on string lengths
//! QRatio: Quick ratio - simple full ratio

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

use super::ratio::ratio;
use super::partial_ratio::partial_ratio;
use super::token_ratio::{token_sort_ratio, token_set_ratio, token_ratio};
use super::partial_token_ratio::{partial_token_sort_ratio, partial_token_set_ratio};

/// QRatio - Quick ratio, same as ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn q_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    ratio(s1, s2, score_cutoff)
}

/// WRatio - Weighted ratio with heuristics.
/// 
/// Uses different algorithms based on the length ratio of the strings:
/// - If lengths are similar: uses ratio and token_sort_ratio
/// - If lengths differ significantly: uses partial_ratio variants
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn w_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    
    if len1 == 0 && len2 == 0 {
        return 100.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let (shorter_len, longer_len) = if len1 < len2 {
        (len1, len2)
    } else {
        (len2, len1)
    };

    let length_ratio = shorter_len as f64 / longer_len as f64;
    
    let mut best_score: f64;

    // Start with base ratio
    best_score = ratio(s1, s2, None);

    // If lengths are similar (ratio >= 0.5), try token-based
    if length_ratio >= 0.5 {
        let tsr = token_sort_ratio(s1, s2, None);
        best_score = best_score.max(tsr * 0.95); // Slight penalty for token_sort
        
        let tsetr = token_set_ratio(s1, s2, None);
        best_score = best_score.max(tsetr * 0.95);
    }

    // Try partial ratios with penalty based on length difference
    let partial = partial_ratio(s1, s2, None);
    let partial_weight = if length_ratio >= 0.6 { 0.9 } else { 0.6 };
    best_score = best_score.max(partial * partial_weight);
    
    // Try partial token ratios for very different lengths
    if length_ratio < 0.8 {
        let ptsr = partial_token_sort_ratio(s1, s2, None);
        let ptsetr = partial_token_set_ratio(s1, s2, None);
        let pt_weight = if length_ratio < 0.5 { 0.5 } else { 0.7 };
        best_score = best_score.max(ptsr * pt_weight);
        best_score = best_score.max(ptsetr * pt_weight);
    }

    match score_cutoff {
        Some(cutoff) if best_score < cutoff => 0.0,
        _ => best_score,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn q_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| q_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn w_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| w_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_ratio() {
        let r = q_ratio("this is a test", "this is a test!", None);
        assert!(r > 90.0);
    }

    #[test]
    fn test_w_ratio() {
        let r = w_ratio("this is a test", "this is a new test!!!", None);
        assert!(r > 80.0);
    }
}
