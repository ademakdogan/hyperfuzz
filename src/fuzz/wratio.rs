//! WRatio and QRatio algorithms
//!
//! WRatio: Weighted ratio with heuristics based on string lengths
//! QRatio: Quick ratio - simple full ratio

use pyo3::prelude::*;
use rayon::prelude::*;

use super::ratio::ratio;
use super::partial_ratio::partial_ratio;
use super::token_ratio::{token_sort_ratio, token_set_ratio};
use super::partial_token_ratio::{partial_token_sort_ratio, partial_token_set_ratio};

/// QRatio - Quick ratio, same as ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn q_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    ratio(s1, s2, score_cutoff)
}

/// WRatio - Weighted ratio with heuristics.
/// 
/// RapidFuzz's WRatio algorithm:
/// - Length ratio >= 0.95: Use ratio only (no partial)
/// - Length ratio >= 0.63: Partial ratios available but with weights
/// - Length ratio < 0.63: Only partial ratios with 0.9 weight
///
/// Key insight from RapidFuzz source: 
/// For length ratio > 0.63, it computes ratio first as baseline,
/// then partial_ratio is only used if length ratio < 0.95
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
    
    // Always start with base ratio
    let base = ratio(s1, s2, None);
    
    // RapidFuzz uses different thresholds
    // The key is that partial_ratio is only used when strings have very different lengths
    
    if length_ratio >= 0.95 {
        // Very similar lengths - just use ratio and token-based
        let tsr = token_sort_ratio(s1, s2, None) * 0.95;
        let tsetr = token_set_ratio(s1, s2, None) * 0.95;
        base.max(tsr).max(tsetr)
    } else if length_ratio > 0.63 {
        // Medium difference - ratio is still primary, partial_ratio not weighted as much
        // But for WRatio in RapidFuzz, partial_ratio must be > base / weight to contribute
        // This effectively means partial_ratio only wins if significantly better
        let tsr = token_sort_ratio(s1, s2, None) * 0.95;
        let tsetr = token_set_ratio(s1, s2, None) * 0.95;
        
        // Don't use partial_ratio in this range according to RapidFuzz behavior
        // Just use the base ratio and token ratios
        base.max(tsr).max(tsetr)
    } else {
        // Large length difference - use partial ratios with penalty
        let pr = partial_ratio(s1, s2, None) * 0.9;
        let ptsr = partial_token_sort_ratio(s1, s2, None) * 0.9;
        let ptsetr = partial_token_set_ratio(s1, s2, None) * 0.9;
        base.max(pr).max(ptsr).max(ptsetr)
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
    fn test_w_ratio_similar_lengths() {
        // For similar length strings (0.857 ratio), WRatio should return same as ratio
        let r = w_ratio("kitten", "sitting", None);
        // Should be 61.538... like ratio
        assert!((r - 61.54).abs() < 0.1);
    }
    
    #[test]
    fn test_w_ratio_different_lengths() {
        // For very different lengths (0.5 ratio), partial_ratio * 0.9 should be used
        let r = w_ratio("abc", "abcdef", None);
        // 100 * 0.9 = 90
        assert!((r - 90.0).abs() < 0.1);
    }
}
