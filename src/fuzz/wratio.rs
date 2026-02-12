//! WRatio and QRatio algorithms - RapidFuzz-compatible
//!
//! WRatio: Weighted ratio with heuristics based on string lengths
//! QRatio: Quick ratio - simple full ratio

use pyo3::prelude::*;
use rayon::prelude::*;

use super::ratio::ratio;
use super::partial_ratio::partial_ratio;
use super::token_ratio::token_ratio;
use super::partial_token_ratio::partial_token_ratio;

const UNBASE_SCALE: f64 = 0.95;

/// QRatio - Quick ratio, same as ratio but returns 0 for empty strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn q_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // RapidFuzz returns 0 for empty strings
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    ratio(s1, s2, score_cutoff)
}

/// WRatio - Weighted ratio with heuristics (RapidFuzz-compatible).
/// 
/// Uses score_cutoff optimization to avoid computing expensive ratios
/// when they can't improve the result.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn w_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // RapidFuzz returns 0 for empty strings
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    
    let score_cutoff_val = score_cutoff.unwrap_or(0.0);
    
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    
    let len_ratio = if len1 > len2 {
        len1 as f64 / len2 as f64
    } else {
        len2 as f64 / len1 as f64
    };

    // Start with base ratio
    let mut end_ratio = ratio(s1, s2, Some(score_cutoff_val));
    
    if len_ratio < 1.5 {
        // Similar length strings - use token_ratio with UNBASE_SCALE
        let token_cutoff = end_ratio.max(score_cutoff_val) / UNBASE_SCALE;
        let token_result = token_ratio(s1, s2, Some(token_cutoff)) * UNBASE_SCALE;
        end_ratio = end_ratio.max(token_result);
    } else {
        // Different length strings - use partial ratios
        let partial_scale = if len_ratio <= 8.0 { 0.9 } else { 0.6 };
        
        // Try partial_ratio
        let partial_cutoff = end_ratio.max(score_cutoff_val) / partial_scale;
        let partial_result = partial_ratio(s1, s2, Some(partial_cutoff)) * partial_scale;
        end_ratio = end_ratio.max(partial_result);
        
        // Try partial_token_ratio with combined scale
        let combined_scale = UNBASE_SCALE * partial_scale;
        let combined_cutoff = end_ratio.max(score_cutoff_val) / combined_scale;
        let partial_token_result = partial_token_ratio(s1, s2, Some(combined_cutoff)) * combined_scale;
        end_ratio = end_ratio.max(partial_token_result);
    }
    
    if end_ratio >= score_cutoff_val {
        end_ratio
    } else {
        0.0
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn q_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| q_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn w_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| w_ratio(s1, s2, score_cutoff)).collect()
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
    fn test_q_ratio_empty() {
        assert_eq!(q_ratio("", "test", None), 0.0);
        assert_eq!(q_ratio("test", "", None), 0.0);
    }

    #[test]
    fn test_w_ratio_similar_lengths() {
        let r = w_ratio("kitten", "sitting", None);
        assert!(r > 0.0);
    }
    
    #[test]
    fn test_w_ratio_different_lengths() {
        let r = w_ratio("abc", "abcdef", None);
        // partial_ratio should contribute
        assert!(r > 50.0);
    }
}
